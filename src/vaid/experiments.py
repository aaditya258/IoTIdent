"""Experiment drivers (§6.5). Every driver returns a JSON-serialisable dict and
writes a CSV table into results/.

Guard rails that are part of the experiment, not decoration:

* the class list is split into trainable classes and a tail (fewer than
  `min_windows_per_class` windows). The tail is reported, and with
  `tail_classes_to_unknown` it becomes forced unknown material for E3 rather
  than a class with three training samples;
* feature scalers and vocabularies are fitted inside each fold, on that fold's
  training indices only;
* the forbidden-feature assertion runs on the materialised feature space before
  any model sees it;
* every result carries the number of test windows behind it, because a macro-F1
  over a class with one window is a number, not evidence.
"""
from __future__ import annotations

import json
import os
import platform
import time
from collections import Counter

import numpy as np

from .audit import assert_clean_feature_space, load_forbidden_spec, scan_feature_values
from .config import Config, Paths
from .dataset import (class_counts, filter_records, load_records, make_index,
                      materialise)
from .metrics import (closed_set_metrics, mean_ci, open_set_metrics,
                      paired_bootstrap, per_class_f1, wilcoxon_across_folds,
                      holm_bonferroni)
from .models_shallow import make_model
from .openset import EnergyOpenSet, PrototypeOpenSet, openness
from .perturb import LEVELS, perturb
from .splits import (e1_session_grouped, e2_chronological, e3_class_heldout,
                     e5_temporal_AB, trainable_classes)
from .views import all_feature_names
from .visibility import PRINCIPAL_PROFILES, apply_profile


# --------------------------------------------------------------------------
# shared plumbing
# --------------------------------------------------------------------------

def prepare(paths: Paths, cfg: Config, verbose: bool = True):
    recs = load_records(paths, cfg, verbose=verbose)
    counts = class_counts(recs)
    keep = [c for c, n in counts.items() if n >= cfg.min_windows_per_class]
    tail = [c for c, n in counts.items() if n < cfg.min_windows_per_class]
    if verbose:
        print(f"windows={len(recs)} classes={len(counts)} trainable={len(keep)} tail={len(tail)}")
        if tail:
            print("  tail (excluded from closed-set, routed to unknown pool): "
                  + ", ".join(f"{c}({counts[c]})" for c in tail))
    return recs, counts, sorted(keep), sorted(tail)


def _check_feature_space(b, cfg: Config, paths: Paths):
    spec = load_forbidden_spec(paths)
    names = all_feature_names(b)
    assert_clean_feature_space(names, spec, allow_endpoint=cfg.allow_endpoint_upper_bound)
    return {"n_features": len(names), **scan_feature_values(
        np.concatenate([b.Xagg, b.Xh, b.Xc], axis=1),
        [n for n in names if not n.startswith("seq__")])}


def _fit_predict(cfg, model_name, views, tr, va, te, seed, profiles=None):
    model = make_model(cfg, model_name, seed=seed, views=views)
    model.fit(tr, va)
    out = {}
    for prof in (profiles or ["V2"]):
        tep = apply_profile(te, prof, cfg)
        p = model.predict(tep)
        out[prof] = {"y_true": te.y, "y_pred": p["y_pred"], "proba": p["proba"],
                     "z": p.get("z"), "logits": p.get("logits"),
                     "alpha": p.get("alpha"),
                     "classes": getattr(model, "classes_", None)}
    return model, out


def _write(paths: Paths, name: str, payload: dict, table: list[dict] | None = None) -> str:
    os.makedirs(paths.results, exist_ok=True)
    p = os.path.join(paths.results, f"{name}.json")
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1, default=float)
    if table:
        import csv
        cp = os.path.join(paths.results, f"{name}.csv")
        keys = list({k for row in table for k in row})
        with open(cp, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(table)
    return p


# --------------------------------------------------------------------------
# Experiment 1 -- leakage-controlled closed-set baseline (Table 3)
# --------------------------------------------------------------------------

def experiment_1(paths: Paths, cfg: Config, methods=("rf", "gbm", "mlp", "cnn1d", "tcn"),
                 n_folds: int | None = None, verbose: bool = True) -> dict:
    recs, counts, keep, tail = prepare(paths, cfg, verbose)
    recs, _ = filter_records(recs, keep)
    idx = make_index(recs, cfg, labels=keep)
    n_folds = n_folds or cfg.n_repeats

    rows, per_method_folds = [], {m: [] for m in methods}
    fold_preds = {m: [] for m in methods}
    audit_note = None
    for f, (tr_i, va_i, te_i) in enumerate(e1_session_grouped(idx, n_folds, cfg.seed)):
        b, _ = materialise(recs, cfg, tr_i, labels=keep)
        if audit_note is None:
            audit_note = _check_feature_space(b, cfg, paths)
        tr, va, te = b.subset(tr_i), b.subset(va_i), b.subset(te_i)
        for m in methods:
            t0 = time.time()
            model, out = _fit_predict(cfg, m, ("F",), tr, va, te, cfg.seed + f)
            r = out["V2"]
            mt = closed_set_metrics(r["y_true"], r["y_pred"], r["proba"], r.get("classes"))
            mt.update({"method": m, "fold": f, "fit_seconds": round(time.time() - t0, 2),
                       "n_test": len(te.y)})
            rows.append(mt)
            per_method_folds[m].append(mt["macro_f1"])
            fold_preds[m].append((r["y_true"], r["y_pred"]))
        if verbose:
            print(f"  fold {f}: " + "  ".join(
                f"{m}={per_method_folds[m][-1]:.3f}" for m in methods))

    summary = {m: mean_ci(per_method_folds[m]) for m in methods}
    best = max(summary, key=lambda m: summary[m]["mean"])
    stats = {}
    for m in methods:
        if m == best:
            continue
        stats[f"{best}_vs_{m}"] = wilcoxon_across_folds(per_method_folds[best],
                                                        per_method_folds[m])
    payload = {"config": cfg.to_dict(), "view": "F only (§7.1)",
               "trainable_classes": keep, "tail_classes": tail,
               "class_window_counts": counts, "feature_audit": audit_note,
               "per_fold": rows, "summary": summary, "best": best,
               "statistics": stats,
               "holm": holm_bonferroni({k: v.get("p_wilcoxon", float("nan"))
                                        for k, v in stats.items()})}
    _write(paths, "exp1_closed_set", payload, rows)
    return payload


# --------------------------------------------------------------------------
# Experiment 2 -- contribution of individual views (Table 4)
# --------------------------------------------------------------------------

VIEW_SETS = {"flow": ("F",), "handshake": ("H",), "communication": ("C",),
             "flow+handshake": ("F", "H"), "flow+communication": ("F", "C"),
             "all": ("F", "H", "C")}


def experiment_2(paths: Paths, cfg: Config, n_folds: int | None = None,
                 verbose: bool = True) -> dict:
    recs, counts, keep, tail = prepare(paths, cfg, verbose)
    recs, _ = filter_records(recs, keep)
    idx = make_index(recs, cfg, labels=keep)
    n_folds = n_folds or max(3, cfg.n_repeats // 2)

    rows, folds = [], {k: [] for k in VIEW_SETS}
    for f, (tr_i, va_i, te_i) in enumerate(e1_session_grouped(idx, n_folds, cfg.seed)):
        b, _ = materialise(recs, cfg, tr_i, labels=keep)
        tr, va, te = b.subset(tr_i), b.subset(va_i), b.subset(te_i)
        for name, views in VIEW_SETS.items():
            _, out = _fit_predict(cfg, "gated", views, tr, va, te, cfg.seed + f)
            r = out["V2"]
            mt = closed_set_metrics(r["y_true"], r["y_pred"], r["proba"], r.get("classes"))
            mt.update({"views": name, "fold": f,
                       "handshake_available_frac": float(np.mean(te.mH)),
                       "n_test": len(te.y)})
            rows.append(mt)
            folds[name].append(mt["macro_f1"])
        if verbose:
            print(f"  fold {f}: " + "  ".join(f"{k}={folds[k][-1]:.3f}" for k in VIEW_SETS))

    payload = {"config": cfg.to_dict(), "per_fold": rows,
               "summary": {k: mean_ci(v) for k, v in folds.items()},
               "note": "A view that helps under full visibility is not thereby "
                       "essential; read this table together with Experiment 3."}
    _write(paths, "exp2_view_contribution", payload, rows)
    return payload


# --------------------------------------------------------------------------
# Experiment 3 -- visibility-aware fusion under V0..V5 (the central table)
# --------------------------------------------------------------------------

FUSIONS = ("gated", "concat", "late", "fixed", "attention")


def experiment_3(paths: Paths, cfg: Config, fusions=FUSIONS, n_folds: int | None = None,
                 profiles=tuple(PRINCIPAL_PROFILES), verbose: bool = True) -> dict:
    recs, counts, keep, tail = prepare(paths, cfg, verbose)
    recs, _ = filter_records(recs, keep)
    idx = make_index(recs, cfg, labels=keep)
    n_folds = n_folds or cfg.n_repeats

    rows = []
    scores = {m: {p: [] for p in profiles} for m in fusions}
    preds = {m: {p: [] for p in profiles} for m in fusions}
    for f, (tr_i, va_i, te_i) in enumerate(e1_session_grouped(idx, n_folds, cfg.seed)):
        b, _ = materialise(recs, cfg, tr_i, labels=keep)
        tr, va, te = b.subset(tr_i), b.subset(va_i), b.subset(te_i)
        for m in fusions:
            _, out = _fit_predict(cfg, m, ("F", "H", "C"), tr, va, te, cfg.seed + f,
                                  profiles=list(profiles))
            for p in profiles:
                r = out[p]
                mt = closed_set_metrics(r["y_true"], r["y_pred"], r["proba"], r.get("classes"))
                mt.update({"fusion": m, "profile": p, "fold": f, "n_test": len(te.y)})
                if r.get("alpha") is not None:
                    a = np.asarray(r["alpha"])
                    # mean fusion weight per view: shows WHETHER the gate is
                    # actually using the handshake, or achieving robustness by
                    # ignoring it.
                    for j, v in enumerate(("F", "H", "C")):
                        mt[f"alpha_{v}"] = float(a[:, j].mean())
                    mt["alpha_H_when_available"] = float(
                        a[te.mH > 0.5, 1].mean()) if (te.mH > 0.5).any() else float("nan")
                rows.append(mt)
                scores[m][p].append(mt["macro_f1"])
                preds[m][p].append((r["y_true"], r["y_pred"]))
        if verbose:
            print(f"  fold {f}: " + "  ".join(
                f"{m}:V2={scores[m]['V2'][-1]:.3f}/V5={scores[m]['V5'][-1]:.3f}"
                for m in fusions))

    # ΔF1 is the dependent variable of the claim: degradation from full
    # visibility (V2) to the reduced profile.
    degradation = {m: {p: mean_ci([scores[m]["V2"][i] - scores[m][p][i]
                                   for i in range(len(scores[m][p]))])
                       for p in profiles} for m in fusions}
    baselines = [m for m in fusions if m != "gated"]
    stats = {}
    for p in ("V4", "V5"):
        for bl in baselines:
            d_prop = [scores["gated"]["V2"][i] - scores["gated"][p][i]
                      for i in range(len(scores["gated"][p]))]
            d_base = [scores[bl]["V2"][i] - scores[bl][p][i] for i in range(len(scores[bl][p]))]
            stats[f"degradation_{p}_gated_vs_{bl}"] = wilcoxon_across_folds(d_base, d_prop)
        yt, yp_g = preds["gated"][p][0]
        _, yp_b = preds["concat"][p][0]
        stats[f"macroF1_{p}_gated_vs_concat_fold0"] = paired_bootstrap(
            yt, yp_g, yp_b, n=cfg.n_bootstrap, seed=cfg.seed)

    payload = {"config": cfg.to_dict(), "per_fold": rows,
               "summary": {m: {p: mean_ci(scores[m][p]) for p in profiles} for m in fusions},
               "degradation_from_V2": degradation, "statistics": stats,
               "holm": holm_bonferroni({k: v.get("p_wilcoxon", v.get("p_bootstrap", float("nan")))
                                        for k, v in stats.items()}),
               "claim": "supported only if gated shows statistically lower "
                        "degradation than every baseline across profiles"}
    _write(paths, "exp3_visibility_fusion", payload, rows)
    return payload


# --------------------------------------------------------------------------
# Experiment 4 -- open-set identification
# --------------------------------------------------------------------------

def experiment_4(paths: Paths, cfg: Config, opennesses=(0.10, 0.20, 0.30, 0.40),
                 n_repeats: int | None = None, verbose: bool = True) -> dict:
    recs_all, counts, keep, tail = prepare(paths, cfg, verbose)
    n_repeats = n_repeats or cfg.n_repeats
    all_labels = sorted(counts)
    rows = []
    for op in opennesses:
        for rep in range(n_repeats):
            seed = cfg.seed + 1000 * rep + int(op * 100)
            idx_all = make_index(recs_all, cfg, labels=all_labels)
            known, u_val, u_test = e3_class_heldout(
                idx_all, op, seed,
                forced_unknown=tail if cfg.tail_classes_to_unknown else None)
            known = [c for c in known if counts.get(c, 0) >= cfg.min_windows_per_class]
            if len(known) < 3 or not u_test:
                continue

            keep_lbl = known + u_val + u_test
            recs, _ = filter_records(recs_all, keep_lbl)
            idx = make_index(recs, cfg, labels=keep_lbl)
            lab_of = np.array([keep_lbl[i] for i in idx.y])
            known_mask = np.isin(lab_of, known)

            # known windows split by group; unknown classes never train
            k_idx = np.where(known_mask)[0]
            sub = idx.subset(k_idx)
            tr_s, va_s, te_s = next(iter(e1_session_grouped(sub, 4, seed)))
            tr_i, va_i, te_i = k_idx[tr_s], k_idx[va_s], k_idx[te_s]

            b, _ = materialise(recs, cfg, tr_i, labels=keep_lbl)
            tr, va = b.subset(tr_i), b.subset(va_i)
            uval_i = np.where(np.isin(lab_of, u_val))[0]
            utest_i = np.where(np.isin(lab_of, u_test))[0]
            te_known, te_unknown = b.subset(te_i), b.subset(utest_i)

            model = make_model(cfg, "gated", seed=seed, views=("F", "H", "C"))
            model.fit(tr, va)
            pv, pk, pu = model.predict(va), model.predict(te_known), model.predict(te_unknown)

            proto = PrototypeOpenSet(cfg.openset_metric).fit(
                model.predict(tr)["z"], tr.y)
            z_uval = (model.predict(b.subset(uval_i))["z"] if len(uval_i) else None)
            proto.calibrate(pv["z"], cfg.target_known_tpr, z_uval)
            energy = EnergyOpenSet()
            energy.calibrate(pv["logits"], cfg.target_known_tpr)

            z = np.concatenate([pk["z"], pu["z"]])
            lg = np.concatenate([pk["logits"], pu["logits"]])
            is_unknown = np.concatenate([np.zeros(len(pk["z"]), bool),
                                         np.ones(len(pu["z"]), bool)])
            y_true = np.concatenate([te_known.y, te_unknown.y])
            y_pred = np.concatenate([pk["y_pred"], pu["y_pred"]])

            for name, score, accept in (
                    ("prototype", proto.score(z), proto.accept(z)),
                    ("energy", energy.score(lg), energy.accept(lg))):
                mt = open_set_metrics(is_unknown, score, y_true, y_pred, accept)
                mt.update({"method": name, "openness_frac_classes": op,
                           "openness": openness(len(known), len(known) + len(u_test)),
                           "repeat": rep, "n_known_classes": len(known),
                           "n_unknown_classes_test": len(u_test),
                           "n_test_known": int((~is_unknown).sum()),
                           "n_test_unknown": int(is_unknown.sum()),
                           "u_val": ",".join(u_val), "u_test": ",".join(u_test)})
                rows.append(mt)
            if verbose:
                print(f"  openness={op:.2f} rep={rep} "
                      f"AUROC={rows[-2]['auroc_unknown']:.3f} OSCR={rows[-2]['oscr']:.3f}")

    summary = {}
    for name in ("prototype", "energy"):
        for op in opennesses:
            v = [r["auroc_unknown"] for r in rows if r["method"] == name
                 and r["openness_frac_classes"] == op and np.isfinite(r["auroc_unknown"])]
            summary[f"{name}@{op}"] = mean_ci(v)
    payload = {"config": cfg.to_dict(), "per_run": rows, "summary_auroc": summary,
               "note": "U_val and U_test are disjoint by construction; tail classes "
                       "are forced unknowns."}
    _write(paths, "exp4_open_set", payload, rows)
    return payload


# --------------------------------------------------------------------------
# Experiment 5 -- temporal generalisation (period A -> period B)
# --------------------------------------------------------------------------

def experiment_5(paths: Paths, cfg: Config, verbose: bool = True) -> dict:
    recs, counts, keep, tail = prepare(paths, cfg, verbose)
    recs, _ = filter_records(recs, keep)
    idx = make_index(recs, cfg, labels=keep)
    tr_i, va_i, te_i, devs = e5_temporal_AB(idx, seed=cfg.seed)

    shared_labels = sorted({recs[i].label for i in np.concatenate([tr_i, te_i])})
    b, _ = materialise(recs, cfg, tr_i, labels=keep)
    tr, va, te = b.subset(tr_i), b.subset(va_i), b.subset(te_i)

    rows = []
    for m in ("gated", "concat", "rf"):
        model, out = _fit_predict(cfg, m, ("F", "H", "C"), tr, va, te, cfg.seed)
        r = out["V2"]
        mt = closed_set_metrics(r["y_true"], r["y_pred"], r["proba"], r.get("classes"))
        mt.update({"method": m, "regime": "A->B (151 days)", "n_test": len(te.y)})
        rows.append(mt)
        # within-period control: same model family, held-out groups inside A
        sub = idx.subset(tr_i)
        a_tr, a_va, a_te = next(iter(e1_session_grouped(sub, 4, cfg.seed)))
        b2, _ = materialise(recs, cfg, tr_i[a_tr], labels=keep)
        _, out2 = _fit_predict(cfg, m, ("F", "H", "C"), b2.subset(tr_i[a_tr]),
                               b2.subset(tr_i[a_va]), b2.subset(tr_i[a_te]), cfg.seed)
        r2 = out2["V2"]
        mt2 = closed_set_metrics(r2["y_true"], r2["y_pred"], r2["proba"], r2.get("classes"))
        mt2.update({"method": m, "regime": "A->A (within period)", "n_test": len(a_te)})
        rows.append(mt2)
        if verbose:
            print(f"  {m}: within-A={mt2['macro_f1']:.3f}  A->B={mt['macro_f1']:.3f}"
                  f"  drop={mt2['macro_f1'] - mt['macro_f1']:+.3f}")

    per_class = {}
    for m in ("gated",):
        model, out = _fit_predict(cfg, m, ("F", "H", "C"), tr, va, te, cfg.seed)
        per_class[m] = per_class_f1(out["V2"]["y_true"], out["V2"]["y_pred"], keep)

    payload = {"config": cfg.to_dict(), "devices_in_both_periods": devs,
               "n_devices_shared": len(devs), "rows": rows,
               "per_class_f1_AB": per_class,
               "caveat": "Restricted to devices present in both sample periods; "
                         "the 9 single-period devices would confound drift with "
                         "class turnover."}
    _write(paths, "exp5_temporal_drift", payload, rows)
    return payload


# --------------------------------------------------------------------------
# Experiment 8 -- perturbation simulation
# --------------------------------------------------------------------------

def experiment_8(paths: Paths, cfg: Config, kinds=tuple(LEVELS), verbose: bool = True) -> dict:
    recs, counts, keep, tail = prepare(paths, cfg, verbose)
    recs, _ = filter_records(recs, keep)
    idx = make_index(recs, cfg, labels=keep)
    tr_i, va_i, te_i = next(iter(e1_session_grouped(idx, 4, cfg.seed)))
    b, vs = materialise(recs, cfg, tr_i, labels=keep)
    tr, va = b.subset(tr_i), b.subset(va_i)

    models = {}
    for m in ("gated", "concat"):
        models[m] = make_model(cfg, m, seed=cfg.seed, views=("F", "H", "C")).fit(tr, va)

    rng = np.random.default_rng(cfg.seed)
    rows = []
    for kind in kinds:
        for lvl in [0] + list(LEVELS[kind]):
            prec = recs if lvl == 0 else perturb(recs, kind, lvl, rng)
            bp = vs.transform(prec, labels=keep)          # train-fitted scalers reused
            tep = bp.subset(te_i)
            for m, model in models.items():
                p = model.predict(tep)
                mt = closed_set_metrics(tep.y, p["y_pred"], p["proba"], getattr(model, "classes_", None))
                mt.update({"perturbation": kind, "level": lvl, "method": m,
                           "n_test": len(tep.y)})
                rows.append(mt)
        if verbose:
            sel = [r for r in rows if r["perturbation"] == kind and r["method"] == "gated"]
            print(f"  {kind}: " + " ".join(f"{r['level']}={r['macro_f1']:.3f}" for r in sel))
    payload = {"config": cfg.to_dict(), "rows": rows,
               "note": "Sensitivity analysis. Perturbations are not optimised "
                       "against the model, so this is not adversarial robustness (§9.7)."}
    _write(paths, "exp8_perturbation", payload, rows)
    return payload


# --------------------------------------------------------------------------
# Experiment 9 -- ablation
# --------------------------------------------------------------------------

ABLATIONS = {
    "M0_full": dict(model="gated", views=("F", "H", "C"), dropout=True, supcon=True),
    "M1_no_visibility_gating": dict(model="attention", views=("F", "H", "C"), dropout=True, supcon=True),
    "M2_no_view_dropout": dict(model="gated", views=("F", "H", "C"), dropout=False, supcon=True),
    "M3_no_supcon": dict(model="gated", views=("F", "H", "C"), dropout=True, supcon=False),
    "M4_no_communication_view": dict(model="gated", views=("F", "H"), dropout=True, supcon=True),
    "M5_no_handshake_view": dict(model="gated", views=("F", "C"), dropout=True, supcon=True),
    "M6_cnn_instead_of_tcn": dict(model="gated", views=("F", "H", "C"), dropout=True,
                                  supcon=True, encoder="cnn1d"),
}


def experiment_9(paths: Paths, cfg: Config, n_folds: int = 3, verbose: bool = True) -> dict:
    from .models_torch import TORCH_AVAILABLE, NeuralClassifier
    recs, counts, keep, tail = prepare(paths, cfg, verbose)
    recs, _ = filter_records(recs, keep)
    idx = make_index(recs, cfg, labels=keep)

    rows, folds = [], {k: {"V2": [], "V5": []} for k in ABLATIONS}
    for f, (tr_i, va_i, te_i) in enumerate(e1_session_grouped(idx, n_folds, cfg.seed)):
        b, _ = materialise(recs, cfg, tr_i, labels=keep)
        tr, va, te = b.subset(tr_i), b.subset(va_i), b.subset(te_i)
        for name, spec in ABLATIONS.items():
            c2 = Config(**{**cfg.to_dict(), "supcon_lambda": cfg.supcon_lambda if spec["supcon"] else 0.0})
            if TORCH_AVAILABLE and cfg.backend in ("auto", "torch"):
                model = NeuralClassifier(c2, fusion=spec["model"],
                                         seq_encoder=spec.get("encoder", "tcn"),
                                         use_views=spec["views"],
                                         view_dropout=spec["dropout"], seed=cfg.seed + f)
            else:
                model = make_model(c2, spec["model"], seed=cfg.seed + f, views=spec["views"])
            model.fit(tr, va)
            for prof in ("V2", "V5"):
                p = model.predict(apply_profile(te, prof, cfg))
                mt = closed_set_metrics(te.y, p["y_pred"], p["proba"], getattr(model, "classes_", None))
                mt.update({"ablation": name, "profile": prof, "fold": f})
                rows.append(mt)
                folds[name][prof].append(mt["macro_f1"])
        if verbose:
            print(f"  fold {f}: " + "  ".join(
                f"{k.split('_')[0]}={folds[k]['V2'][-1]:.3f}" for k in ABLATIONS))

    summary = {k: {"V2": mean_ci(v["V2"]), "V5": mean_ci(v["V5"]),
                   "robustness_gap": mean_ci([a - b for a, b in zip(v["V2"], v["V5"])])}
               for k, v in folds.items()}
    payload = {"config": cfg.to_dict(), "rows": rows, "summary": summary,
               "key_comparison": "M0 vs M1 vs M2 -- the visibility-aware claim"}
    _write(paths, "exp9_ablation", payload, rows)
    return payload


# --------------------------------------------------------------------------
# Experiment 10 -- computational evaluation
# --------------------------------------------------------------------------

def experiment_10(paths: Paths, cfg: Config, verbose: bool = True) -> dict:
    import resource
    recs, counts, keep, tail = prepare(paths, cfg, verbose)
    recs, _ = filter_records(recs, keep)
    idx = make_index(recs, cfg, labels=keep)
    tr_i, va_i, te_i = next(iter(e1_session_grouped(idx, 4, cfg.seed)))

    t0 = time.time()
    b, vs = materialise(recs, cfg, tr_i, labels=keep)
    feat_seconds = time.time() - t0
    tr, va, te = b.subset(tr_i), b.subset(va_i), b.subset(te_i)

    rows = []
    for m in ("gated", "tcn", "rf"):
        model = make_model(cfg, m, seed=cfg.seed, views=("F", "H", "C"))
        t0 = time.time(); model.fit(tr, va); fit_s = time.time() - t0

        lat = []
        for i in range(min(200, len(te.y))):
            one = te.subset(np.array([i]))
            t1 = time.perf_counter(); model.predict(one); lat.append((time.perf_counter() - t1) * 1000)
        t0 = time.time(); model.predict(te); batch_s = time.time() - t0

        n_par = None
        size_mb = None
        if hasattr(model, "model") and hasattr(model.model, "parameters"):
            n_par = int(sum(p.numel() for p in model.model.parameters()))
            size_mb = round(n_par * 4 / 1e6, 3)
        rows.append({
            "method": m, "n_parameters": n_par, "model_size_mb": size_mb,
            "fit_seconds": round(fit_s, 2),
            "latency_ms_mean": round(float(np.mean(lat)), 3),
            "latency_ms_p95": round(float(np.percentile(lat, 95)), 3),
            "throughput_windows_per_s": round(len(te.y) / max(batch_s, 1e-9), 1),
            "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        })
        if verbose:
            print(f"  {m}: {rows[-1]['latency_ms_mean']} ms/window, "
                  f"{rows[-1]['throughput_windows_per_s']} windows/s")

    payload = {"config": cfg.to_dict(), "rows": rows,
               "feature_extraction_seconds_for_all_windows": round(feat_seconds, 2),
               "windows": len(recs), "platform": platform.platform(),
               "python": platform.python_version(),
               "note": "Measured on this machine, not on gateway hardware. §7.9 "
                       "requires the same-hardware comparison, so re-run on the "
                       "target device before quoting these numbers."}
    _write(paths, "exp10_compute", payload, rows)
    return payload
