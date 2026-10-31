"""Rejection-threshold transport -- the experiment the narrowed claim stands on.

Everything else in this repository asks "can the model tell these device types
apart?". This module asks a different question: once you have fixed the
rejection threshold on the data you had, does the OPERATING POINT it defines
survive being moved to a different network, a later month, a thinner protocol
visibility, or a different physical unit of the same product?

The distinction matters because the two failures look identical in a confusion
matrix and have opposite fixes:

  * ranking failure     the score no longer separates known from unknown.
                        AUROC falls. Nothing you do to the threshold helps.
  * calibration failure the score still separates them perfectly well, but the
                        whole score distribution has shifted, so a threshold
                        chosen at the 95th percentile of yesterday's knowns now
                        sits at their 60th. AUROC is unchanged. Recalibration
                        fixes it completely.

Published open-world IoT identification reports AUROC, AUPR and OSCR -- all
three are threshold-free, all three are blind to the second failure. A deployed
detector has one threshold and no labels with which to move it.

So the quantity reported here is the DISPLACEMENT of the realised operating
point:

    tau      is calibrated on source known-validation scores at the nominal
             acceptance level alpha (Config.target_known_tpr, 0.95).
    A(c)     is the REALISED known-acceptance on condition c at that same tau.
    delta_A  = alpha - A(c).  Zero means the operating point transported.
    R(c)     is the realised unknown-rejection on condition c.
    tau*(c)  is the threshold the condition would have needed, and
    R*(c)    the rejection it would then have achieved.

R*(c) - R(c) is the part of the loss that recalibration alone would recover,
and AUROC(c) says whether anything is left after that. Those three numbers
together are the finding; A(c) on its own is not.

The source control is not optional. It is held-out source data scored with the
same model and the same tau, so delta_A on it is the measurement noise floor
against which the shifted conditions are read.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

from .config import Config
from .metrics import oscr
from .openset import PrototypeOpenSet
from .views import Bundle
from .visibility import apply_profile

SCORERS = ("msp", "entropy", "energy", "proto")


# --------------------------------------------------------------------------
# Unknown-ness scores
# --------------------------------------------------------------------------
# A warning paid for in a wasted run. Two of these four scores are only defined
# for a model that emits UNNORMALISED logits and a learned embedding:
#
#   energy = -logsumexp(logits).  Feed it log-probabilities and logsumexp is
#            exactly 0 for every observation, so the score is the constant 0,
#            tau is 0, AUROC is chance, and the table looks like a finding
#            instead of like a bug. Measured: tau = -0.0000, AUROC 0.44.
#   proto  = distance to the nearest class prototype IN THE EMBEDDING SPACE.
#            A Random Forest has no embedding; `predict` returns log-posteriors
#            in its place, so the "prototypes" are corners of a simplex and the
#            distance carries almost nothing. Measured: AUROC 0.41-0.63 where
#            max-posterior on the same model reached 0.98.
#
# So both are gated behind `applicable_scorers`, which inspects the model output
# rather than trusting the caller, and a scorer that does not apply is recorded
# as degenerate rather than scored.

def msp_score(pr: dict) -> np.ndarray:
    """1 - max posterior. The baseline every open-set paper has to beat."""
    return 1.0 - pr["proba"].max(axis=1)


def entropy_score(pr: dict) -> np.ndarray:
    """Predictive entropy, normalised to [0, 1] by log K.

    Unlike energy this is well defined for any probabilistic classifier, which
    makes it the right second opinion when the model has no real logits.
    """
    p = np.clip(pr["proba"], 1e-12, 1.0)
    K = p.shape[1]
    return -(p * np.log(p)).sum(axis=1) / max(np.log(max(K, 2)), 1e-9)


def energy_score(pr: dict) -> np.ndarray:
    lg = np.asarray(pr["logits"], dtype=np.float64)
    m = lg.max(axis=1, keepdims=True)
    return -(m.squeeze(1) + np.log(np.exp(lg - m).sum(axis=1)))


def logits_are_normalised(pr: dict, tol: float = 1e-3) -> bool:
    """True when `logits` are really log-probabilities (logsumexp == 0)."""
    lg = pr.get("logits")
    if lg is None or not np.size(lg):
        return True
    m = lg.max(axis=1, keepdims=True)
    lse = m.squeeze(1) + np.log(np.exp(lg - m).sum(axis=1))
    return bool(np.nanmax(np.abs(lse)) < tol)


def embedding_is_posterior(pr: dict, tol: float = 1e-3) -> bool:
    """True when `z` is a (log-)posterior rather than a learned embedding."""
    z = pr.get("z")
    if z is None or not np.size(z):
        return True
    if z.shape[1] == pr["proba"].shape[1]:
        m = z.max(axis=1, keepdims=True)
        lse = m.squeeze(1) + np.log(np.exp(z - m).sum(axis=1))
        if np.nanmax(np.abs(lse)) < tol:
            return True
        s = z.sum(axis=1)
        if np.nanmax(np.abs(s - 1.0)) < tol:
            return True
    return False


def applicable_scorers(pr: dict, wanted=SCORERS) -> tuple[list[str], dict]:
    """Keep the scorers this model can actually support; explain the rest."""
    ok, why = [], {}
    for s in wanted:
        if s == "energy" and logits_are_normalised(pr):
            why[s] = ("model emits normalised log-probabilities, so "
                      "-logsumexp(logits) is identically 0")
        elif s == "proto" and embedding_is_posterior(pr):
            why[s] = ("model has no learned embedding; z is the posterior, so "
                      "prototype distance is not the proposed mechanism")
        else:
            ok.append(s)
    return ok, why


def score_of(name: str, pr: dict, proto: PrototypeOpenSet | None) -> np.ndarray:
    if name == "msp":
        return msp_score(pr)
    if name == "entropy":
        return entropy_score(pr)
    if name == "energy":
        return energy_score(pr)
    if name == "proto":
        if proto is None:
            raise ValueError("prototype scorer needs a fitted PrototypeOpenSet")
        return proto.score(pr["z"])
    raise ValueError(name)


# --------------------------------------------------------------------------
# Conditions
# --------------------------------------------------------------------------

@dataclass
class Condition:
    """One place a threshold might be asked to work.

    `known` and `unknown` index into the SAME materialised Bundle, so every
    condition is scored through one view space fitted on the source training
    windows -- a condition is a change of data, never a change of preprocessing.
    """
    name: str
    known: np.ndarray
    unknown: np.ndarray
    profile: str = "V2"
    note: str = ""
    shared_unknown_pool: bool = False   # True when the unknowns are source ones

    def __post_init__(self):
        self.known = np.asarray(self.known, dtype=int)
        self.unknown = np.asarray(self.unknown, dtype=int)


def _view(b: Bundle, sel: np.ndarray, profile: str, cfg: Config) -> Bundle:
    sub = b.subset(sel)
    return sub if profile in ("", "V2") else apply_profile(sub, profile, cfg)


# --------------------------------------------------------------------------
# The measurement
# --------------------------------------------------------------------------

def _operating_point(sk: np.ndarray, su: np.ndarray, tau: float, alpha: float,
                     y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Realised operating point on one condition, with the oracle beside it."""
    A = float(np.mean(sk <= tau)) if len(sk) else float("nan")
    R = float(np.mean(su > tau)) if len(su) else float("nan")

    tau_star = float(np.quantile(sk, alpha)) if len(sk) else float("nan")
    R_star = float(np.mean(su > tau_star)) if len(su) and np.isfinite(tau_star) \
        else float("nan")

    is_unk = np.concatenate([np.zeros(len(sk), bool), np.ones(len(su), bool)])
    s_all = np.concatenate([sk, su])
    auroc = (float(roc_auc_score(is_unk, s_all))
             if len(sk) and len(su) and np.ptp(s_all) > 0 else float("nan"))

    # known macro-F1 with rejected knowns scored as errors, at tau and at tau*
    def after_rejection(t: float) -> float:
        if not len(sk):
            return float("nan")
        p = np.where(sk <= t, y_pred, -1)
        lab = sorted(set(y_true.tolist()))
        return float(f1_score(y_true, p, average="macro", labels=lab,
                              zero_division=0))

    return {
        "n_known": int(len(sk)), "n_unknown": int(len(su)),
        "tau_used": float(tau),
        "known_acceptance": A,
        "delta_A": float(alpha - A) if np.isfinite(A) else float("nan"),
        "unknown_rejection": R,
        "tau_oracle": tau_star,
        "unknown_rejection_at_oracle": R_star,
        "rejection_recoverable_by_recalibration":
            float(R_star - R) if np.isfinite(R) and np.isfinite(R_star) else float("nan"),
        "auroc_unknown": auroc,
        "oscr": oscr(is_unk, s_all,
                     np.concatenate([y_true, np.full(len(su), -999)]),
                     np.concatenate([y_pred, np.full(len(su), -998)])),
        "known_macro_f1_after_rejection": after_rejection(tau),
        "known_macro_f1_after_rejection_at_oracle": after_rejection(tau_star),
        "closed_set_macro_f1": float(
            f1_score(y_true, y_pred, average="macro",
                     labels=sorted(set(y_true.tolist())), zero_division=0))
        if len(sk) else float("nan"),
        "score_median_known": float(np.median(sk)) if len(sk) else float("nan"),
        "score_median_unknown": float(np.median(su)) if len(su) else float("nan"),
    }


def run_transport(model, bundle: Bundle, cfg: Config,
                  calib_known: np.ndarray, conditions: list[Condition],
                  alpha: float | None = None, scorers=SCORERS,
                  proto_fit: np.ndarray | None = None) -> dict:
    """Calibrate on `calib_known`, then report the realised operating point on
    every condition.

    `calib_known` must be SOURCE known-class validation windows that the model
    did not train on. `proto_fit` is the training index used to place the class
    prototypes; it defaults to `calib_known`, but passing the training indices
    is closer to how a prototype network would be deployed.
    """
    alpha = cfg.target_known_tpr if alpha is None else alpha

    Bc = bundle.subset(np.asarray(calib_known, int))
    prc = model.predict(Bc)
    scorers, skipped = applicable_scorers(prc, tuple(scorers))
    if not scorers:
        raise ValueError(f"no usable unknown-ness score for this model: {skipped}")

    proto = None
    if "proto" in scorers:
        fit_idx = calib_known if proto_fit is None else np.asarray(proto_fit, int)
        Bf = bundle.subset(fit_idx)
        prf = model.predict(Bf)
        proto = PrototypeOpenSet("cosine").fit(prf["z"], Bf.y)

    taus = {s: float(np.quantile(score_of(s, prc, proto), alpha)) for s in scorers}

    out: dict = {"alpha": float(alpha), "tau": taus,
                 "scorers_used": list(scorers), "scorers_skipped": skipped,
                 "n_calibration_known": int(len(calib_known)),
                 "conditions": {}}
    for cond in conditions:
        Bk = _view(bundle, cond.known, cond.profile, cfg)
        prk = model.predict(Bk)
        if len(cond.unknown):
            Bu = _view(bundle, cond.unknown, cond.profile, cfg)
            pru = model.predict(Bu)
        else:
            pru = None
        rec = {"profile": cond.profile, "note": cond.note,
               "shared_unknown_pool": bool(cond.shared_unknown_pool),
               "scorers": {}}
        for s in scorers:
            sk = score_of(s, prk, proto)
            su = score_of(s, pru, proto) if pru is not None else np.array([])
            rec["scorers"][s] = _operating_point(sk, su, taus[s], alpha,
                                                 Bk.y, prk["y_pred"])
        out["conditions"][cond.name] = rec
    return out


# --------------------------------------------------------------------------
# Reading the result
# --------------------------------------------------------------------------

def transport_verdict(res: dict, control: str, shifted: str, scorer: str = "msp",
                      noise_mult: float = 3.0, rej_tol: float = 0.10,
                      min_useful_rejection: float = 0.5) -> dict:
    """Classify what happened between two conditions.

    Three questions, in order.

    0. Is there an operating point worth transporting? A score that rejects 20%
       of unknowns while accepting 95% of knowns has nothing to lose, so it will
       report a perfect transport and mean nothing by it. Measured here: free
       energy on the gated model held delta_A to -0.005 across networks while
       rejecting 17% of unknowns at the source. That is not robustness, it is a
       detector that was never detecting, and `min_useful_rejection` keeps it
       out of the headline.

    1. Did the operating point move? `noise_mult` x |delta_A on the control| is
       the band inside which a displacement is not distinguishable from split
       noise. Deliberately crude: the claim this experiment supports is an
       order-of-magnitude one, and a displacement that needs a subtle test is
       not the phenomenon.

    2. If it moved, could a practitioner with target labels put it back? That is
       NOT an AUROC question, which is why an earlier version of this function
       was wrong. AUROC can stay respectable while the score distributions
       overlap exactly where alpha falls. The honest test is the oracle
       threshold: set tau* so the target does accept alpha of its knowns, and
       look at the rejection R* you are left with. If R* is within `rej_tol` of
       the control's rejection, recalibration genuinely recovers the operating
       point. If R* has collapsed, no threshold on this condition reproduces the
       source operating point and recalibration is not the answer.
    """
    c = res["conditions"][control]["scorers"][scorer]
    t = res["conditions"][shifted]["scorers"][scorer]
    floor = max(0.02, noise_mult * abs(c["delta_A"]))
    moved = abs(t["delta_A"]) > floor
    R_ctrl = c["unknown_rejection"]
    R_star = t["unknown_rejection_at_oracle"]
    recoverable = bool(np.isfinite(R_star) and np.isfinite(R_ctrl)
                       and R_star >= R_ctrl - rej_tol)
    useful = bool(np.isfinite(R_ctrl) and R_ctrl >= min_useful_rejection)
    if not useful:
        v = (f"not testable: source rejection only {R_ctrl:.3f}, "
             f"no operating point to transport")
    elif not moved:
        v = "transports"
    elif recoverable:
        v = "calibration-only failure (recalibration recovers it)"
    else:
        v = "operating point unreachable (recalibration does not recover it)"
    return {"verdict": v, "noise_floor": float(floor), "moved": bool(moved),
            "source_operating_point_useful": useful,
            "delta_A_control": c["delta_A"], "delta_A_shifted": t["delta_A"],
            "auroc_control": c["auroc_unknown"], "auroc_shifted": t["auroc_unknown"],
            "rejection_control": R_ctrl,
            "rejection_shifted_at_stale_tau": t["unknown_rejection"],
            "rejection_shifted_at_oracle_tau": R_star,
            "closed_set_f1_control": c["closed_set_macro_f1"],
            "closed_set_f1_shifted": t["closed_set_macro_f1"],
            "recoverable_by_recalibration": recoverable}


def format_condition(name: str, rec: dict, scorer: str, alpha: float) -> list[str]:
    m = rec["scorers"][scorer]
    tag = f" [{rec['profile']}]" if rec["profile"] not in ("", "V2") else ""
    pool = "  (source unknown pool)" if rec["shared_unknown_pool"] else ""
    return [
        f"  {name}{tag}{pool}",
        f"      n = {m['n_known']:,} known / {m['n_unknown']:,} unknown",
        f"      known acceptance  {m['known_acceptance']:.3f}   "
        f"(nominal {alpha:.2f},  displacement {m['delta_A']:+.3f})",
        f"      unknown rejection {m['unknown_rejection']:.3f}   "
        f"-> {m['unknown_rejection_at_oracle']:.3f} if recalibrated "
        f"({m['rejection_recoverable_by_recalibration']:+.3f})",
        f"      AUROC {m['auroc_unknown']:.3f}   OSCR {m['oscr']:.3f}   "
        f"tau used {m['tau_used']:.4f} vs needed {m['tau_oracle']:.4f}",
        f"      closed-set macro-F1 {m['closed_set_macro_f1']:.3f}  "
        f"(the number a closed-set paper would report)",
    ]


# --------------------------------------------------------------------------
# Axis builders
# --------------------------------------------------------------------------
# Each builder returns pure index arrays. No features are touched here, because
# the view space has to be fitted on `fit` AFTER the split is decided -- fitting
# a scaler on windows that later serve as the shifted condition would hide the
# very distribution shift the experiment is trying to measure.

from .splits import _group_holdout   # noqa: E402  (placed here to keep the
                                     # narrative order of this file readable)


@dataclass
class TransportSpec:
    axis: str
    fit: np.ndarray
    calib: np.ndarray
    conditions: list = field(default_factory=list)
    known_classes: list = field(default_factory=list)
    unknown_classes: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)


def _pick_unknowns(candidates, n: int, seed: int) -> list[str]:
    c = sorted(candidates)
    if n <= 0 or not c:
        return []
    rng = np.random.default_rng(seed)
    rng.shuffle(c)
    return sorted(c[: min(n, max(0, len(c) - 2))])


def _three_way(groups, y, pool: np.ndarray, seed: int,
               ctrl_frac: float = 0.2, calib_frac: float = 0.2):
    """Split a source pool into fit / calibration / control, by whole group.

    Three partitions, not two: the threshold is calibrated on `calib` and the
    control is read on `ctrl`, so the noise floor is measured on windows that
    had no hand in placing the threshold.
    """
    pool = np.asarray(pool, int)
    rest, ctrl = _group_holdout(groups[pool], y[pool], ctrl_frac,
                                np.random.default_rng(seed))
    r = pool[rest]
    fit, calib = _group_holdout(groups[r], y[r], calib_frac,
                                np.random.default_rng(seed + 1))
    return r[fit], r[calib], pool[ctrl]


def _lab(idx) -> np.ndarray:
    return np.asarray(idx.labels)[idx.y]


def axis_network(idx, cfg: Config, source: str, target: str,
                 n_unknown: int = 6, seed: int = 0) -> TransportSpec:
    """Source network -> target network, shared classes only.

    Unknown classes are drawn from the SHARED set so that the unknown-rejection
    rate is comparable between the two conditions; a different unknown pool on
    each side would make R(target) - R(source) a statement about which classes
    were held out rather than about the networks.
    """
    lab = _lab(idx)
    src, tgt = idx.network == source, idx.network == target
    shared = sorted(set(lab[src].tolist()) & set(lab[tgt].tolist()))
    unk = _pick_unknowns(shared, n_unknown, seed)
    known = [c for c in shared if c not in set(unk)]

    k_src = np.where(src & np.isin(lab, known))[0]
    fit, calib, ctrl = _three_way(idx.group, idx.y, k_src, seed + 11)
    u_src = np.where(src & np.isin(lab, unk))[0]
    return TransportSpec(
        axis=f"network:{source}->{target}", fit=fit, calib=calib,
        known_classes=known, unknown_classes=unk,
        conditions=[
            Condition("source_control", ctrl, u_src,
                      note=f"held-out groups inside {source}"),
            Condition("target_network",
                      np.where(tgt & np.isin(lab, known))[0],
                      np.where(tgt & np.isin(lab, unk))[0],
                      note=f"all of {target}"),
        ],
        meta={"source": source, "target": target, "n_shared_classes": len(shared)})


def axis_time(idx, cfg: Config, network: str | None = None,
              n_unknown: int = 6, seed: int = 0) -> TransportSpec:
    """Earlier months -> later months, inside one capture environment.

    The split is on whole calendar months rather than a timestamp quantile, so
    no month contributes to both sides and the gap between the halves is a real
    interval rather than a cut through one day's activity.
    """
    lab = _lab(idx)
    m = np.ones(len(idx.y), bool) if network is None else (idx.network == network)
    months = sorted(set(idx.period[m].tolist()))
    if len(months) < 4:
        raise ValueError(f"need four distinct months, have {months}")
    half = len(months) // 2
    early, late = months[:half], months[len(months) - half:]
    e = m & np.isin(idx.period, early)
    l = m & np.isin(idx.period, late)
    both = sorted(set(lab[e].tolist()) & set(lab[l].tolist()))
    unk = _pick_unknowns(both, n_unknown, seed + 2)
    known = [c for c in both if c not in set(unk)]

    k_e = np.where(e & np.isin(lab, known))[0]
    fit, calib, ctrl = _three_way(idx.group, idx.y, k_e, seed + 12)
    return TransportSpec(
        axis=f"time:{early[0]}..{early[-1]} -> {late[0]}..{late[-1]}",
        fit=fit, calib=calib, known_classes=known, unknown_classes=unk,
        conditions=[
            Condition("source_control", ctrl,
                      np.where(e & np.isin(lab, unk))[0],
                      note=f"held-out groups in {early[0]}..{early[-1]}"),
            Condition("later_months",
                      np.where(l & np.isin(lab, known))[0],
                      np.where(l & np.isin(lab, unk))[0],
                      note=f"{late[0]}..{late[-1]}"),
        ],
        meta={"network": network, "early_months": early, "late_months": late,
              "classes_in_both": len(both)})


def axis_visibility(idx, cfg: Config, network: str | None = None,
                    profiles=("V4", "V5"), n_unknown: int = 6,
                    seed: int = 0) -> TransportSpec:
    """Same windows, same model, same threshold -- less of the handshake visible.

    This is the cleanest of the four axes: the control and the shifted
    conditions are literally the same observations, so nothing except the
    suppressed view can explain a displacement.
    """
    lab = _lab(idx)
    m = np.ones(len(idx.y), bool) if network is None else (idx.network == network)
    present = sorted(set(lab[m].tolist()))
    unk = _pick_unknowns(present, n_unknown, seed + 3)
    known = [c for c in present if c not in set(unk)]

    k = np.where(m & np.isin(lab, known))[0]
    fit, calib, ctrl = _three_way(idx.group, idx.y, k, seed + 13)
    u = np.where(m & np.isin(lab, unk))[0]
    conds = [Condition("source_control", ctrl, u, profile="V2",
                       note="full visibility")]
    for p in profiles:
        conds.append(Condition(f"visibility_{p}", ctrl, u, profile=p,
                               note="same windows, handshake view suppressed"))
    return TransportSpec(axis="visibility", fit=fit, calib=calib,
                         known_classes=known, unknown_classes=unk,
                         conditions=conds,
                         meta={"network": network, "profiles": list(profiles)})


def axis_instance(idx, cfg: Config, network: str, n_unknown: int = 2,
                  seed: int = 0) -> TransportSpec:
    """A held-out physical unit of a product the model already knows.

    The known evaluation is restricted to the classes whose unit was withheld,
    and the control is seen-unit windows of those SAME classes, so the two
    conditions differ only in which box produced the traffic. Unknown classes
    are drawn from the multi-unit set too and get a unit withheld as well, which
    is what makes R comparable; where that is impossible the condition is marked
    `shared_unknown_pool` and only the acceptance displacement should be read.
    """
    lab = _lab(idx)
    m = idx.network == network
    multi = []
    for cls in sorted(set(lab[m].tolist())):
        if len(set(idx.instance[m & (lab == cls)].tolist())) >= 2:
            multi.append(cls)
    if len(multi) < 3:
        raise ValueError(f"{network}: only {len(multi)} multi-unit classes")

    unk = _pick_unknowns(multi, min(n_unknown, len(multi) - 2), seed + 4)
    known_multi = [c for c in multi if c not in set(unk)]
    known = [c for c in sorted(set(lab[m].tolist())) if c not in set(unk)]

    rng = np.random.default_rng(seed + 14)
    held: list[str] = []
    for cls in known_multi + unk:
        units = sorted(set(idx.instance[m & (lab == cls)].tolist()))
        held.append(units[int(rng.integers(len(units)))])
    is_held = np.isin(idx.instance, held)

    # training: retained units of every known class (single-unit classes stay,
    # so the label space is the same on both sides of the comparison)
    k_seen = np.where(m & ~is_held & np.isin(lab, known))[0]
    fit, calib, ctrl_all = _three_way(idx.group, idx.y, k_seen, seed + 15)
    # control and target are both restricted to the withheld classes
    ctrl = ctrl_all[np.isin(lab[ctrl_all], known_multi)]
    u_seen = np.where(m & ~is_held & np.isin(lab, unk))[0]
    u_held = np.where(m & is_held & np.isin(lab, unk))[0]
    return TransportSpec(
        axis=f"instance:{network}", fit=fit, calib=calib,
        known_classes=known, unknown_classes=unk,
        conditions=[
            Condition("source_control", ctrl, u_seen,
                      note="seen units, same classes"),
            Condition("unseen_unit",
                      np.where(m & is_held & np.isin(lab, known_multi))[0],
                      u_held if len(u_held) else u_seen,
                      shared_unknown_pool=not len(u_held),
                      note="withheld physical units"),
        ],
        meta={"network": network, "multi_unit_classes": multi,
              "tested_classes": known_multi, "held_units": held})
