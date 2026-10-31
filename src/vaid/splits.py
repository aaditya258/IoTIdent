"""Leakage-resistant partitions E1-E5 (§6.1).

E1 session-grouped   every window inside one device x adjacency block stays in
                     one partition, so temporally adjacent windows of the same
                     activity cannot straddle the split.
E2 chronological     earlier windows train, later windows test, per device.
E3 class-held-out    K_train / U_val / U_test are three DISJOINT class sets, so
                     the rejection threshold is never calibrated on the unknown
                     classes it is later scored against (§5.9).
E4 cross-network     requires a second environment; raises until one is present.
E5 temporal drift    train on period A, test on period B, restricted to devices
                     present in both, otherwise drift is confounded with class
                     turnover.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from .views import Bundle


def trainable_classes(b: Bundle, min_windows: int) -> tuple[list[str], list[str]]:
    """Split classes into trainable and tail. The tail is not silently dropped:
    it is returned so the caller can route it into the unknown pool."""
    cnt = Counter(b.labels[i] for i in b.y)
    keep = [c for c in b.labels if cnt.get(c, 0) >= min_windows]
    tail = [c for c in b.labels if cnt.get(c, 0) < min_windows]
    return keep, tail


def _group_holdout(groups: np.ndarray, y: np.ndarray, frac: float,
                   rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Hold out whole groups, trying to keep every class represented."""
    uniq = np.unique(groups)
    rng.shuffle(uniq)
    target = max(1, int(round(frac * len(uniq))))
    chosen: list = []
    covered: set = set()
    classes = set(y.tolist())
    for g in uniq:
        idx = np.where(groups == g)[0]
        if len(chosen) < target or not classes.issubset(covered):
            chosen.append(g)
            covered.update(y[idx].tolist())
        if len(chosen) >= target and classes.issubset(covered):
            break
    hold = np.isin(groups, chosen)
    return np.where(~hold)[0], np.where(hold)[0]


def e1_session_grouped(b: Bundle, n_splits: int = 5, seed: int = 0,
                       val_frac: float = 0.2):
    """Yield (train, val, test) index arrays with grouped, class-aware folds."""
    rng = np.random.default_rng(seed)
    groups = b.group
    uniq = np.unique(groups)
    rng.shuffle(uniq)
    folds = np.array_split(uniq, n_splits)
    for k in range(n_splits):
        test_groups = set(folds[k].tolist())
        test = np.array([i for i, g in enumerate(groups) if g in test_groups])
        rest = np.array([i for i, g in enumerate(groups) if g not in test_groups])
        if len(test) == 0 or len(rest) == 0:
            continue
        sub_tr, sub_va = _group_holdout(groups[rest], b.y[rest], val_frac,
                                        np.random.default_rng(seed + k))
        yield rest[sub_tr], rest[sub_va], test


def e2_chronological(b: Bundle, train_frac: float = 0.7, val_frac: float = 0.1):
    """Per-device time-ordered split. Guarantees every test window is strictly
    later than every training window of the same device."""
    tr, va, te = [], [], []
    for dev in np.unique(b.device):
        idx = np.where(b.device == dev)[0]
        idx = idx[np.argsort(b.t_start[idx])]
        n = len(idx)
        a = int(n * train_frac)
        c = int(n * (train_frac + val_frac))
        tr += idx[:a].tolist()
        va += idx[a:c].tolist()
        te += idx[c:].tolist()
    return np.array(tr), np.array(va), np.array(te)


def shared_period_devices(b: Bundle) -> list[str]:
    out = []
    for dev in np.unique(b.device):
        per = set(b.period[b.device == dev].tolist())
        if {"A", "B"}.issubset(per):
            out.append(dev)
    return out


def e5_temporal_AB(b: Bundle, val_frac: float = 0.15, seed: int = 0):
    """Train on period A, test on period B, devices present in both only."""
    devs = shared_period_devices(b)
    if not devs:
        raise ValueError("no device appears in both sample periods")
    m = np.isin(b.device, devs)
    tr_all = np.where(m & (b.period == "A"))[0]
    te = np.where(m & (b.period == "B"))[0]
    rng = np.random.default_rng(seed)
    tr, va = _group_holdout(b.group[tr_all], b.y[tr_all], val_frac, rng)
    return tr_all[tr], tr_all[va], te, devs


def e3_class_heldout(b: Bundle, openness: float, seed: int,
                     forced_unknown: list[str] | None = None):
    """Return (known_classes, U_val, U_test) with the three sets disjoint.

    `forced_unknown` is where the tail classes go: BlipCareBPMeter (54 flows in
    the whole export) and HelloBarbie (150) can never be trainable classes, so
    they are unknown by construction rather than by sampling luck.
    """
    rng = np.random.default_rng(seed)
    labels = [c for c in b.labels if c not in set(forced_unknown or [])]
    n_unknown = max(2, int(round(openness * len(b.labels))))
    n_unknown = min(n_unknown, len(labels) - 2)
    pool = list(labels)
    rng.shuffle(pool)
    unknown = pool[:n_unknown] + list(forced_unknown or [])
    rng.shuffle(unknown)
    half = max(1, len(unknown) // 2)
    u_val, u_test = unknown[:half], unknown[half:]
    known = [c for c in b.labels if c not in set(u_val) | set(u_test)]
    assert not (set(u_val) & set(u_test)), "U_val and U_test must be disjoint"
    return known, u_val, u_test


def e4_cross_network(b: Bundle, source: str, target: str, val_frac: float = 0.15,
                     seed: int = 0):
    """Train on one capture environment, test on another.

    Returns (train, val, test, shared_labels). Only labels present in BOTH
    environments are returned, because a class the target does not contain
    cannot be scored there and a class the source does not contain cannot be
    learned.

    Two warnings that belong in any result built on this split. First, the two
    DATA2025 networks are distributed in different representations (v1
    bidirectional, v2 upstream-only), so `Config.cf_upstream_only` must stay
    True or the measured drop is partly the exporter. Second, a device type
    shared between the networks is usually a DIFFERENT physical unit in each, so
    a cross-network drop mixes the network effect with the unit effect; use
    `e6_instance_heldout` within one network to estimate the unit effect and
    separate the two.
    """
    if b.network is None:
        raise ValueError("this Bundle carries no network labels")
    src = b.network == source
    tgt = b.network == target
    if not src.any() or not tgt.any():
        raise ValueError(f"network {source!r} or {target!r} absent "
                         f"(have {sorted(set(b.network.tolist()))})")
    labs = np.array(b.labels)
    shared = sorted(set(labs[b.y[src]].tolist()) & set(labs[b.y[tgt]].tolist()))
    keep = np.isin(labs[b.y], shared)
    tr_all = np.where(src & keep)[0]
    te = np.where(tgt & keep)[0]
    rng = np.random.default_rng(seed)
    tr, va = _group_holdout(b.group[tr_all], b.y[tr_all], val_frac, rng)
    return tr_all[tr], tr_all[va], te, shared


def e6_instance_heldout(b: Bundle, seed: int = 0, val_frac: float = 0.15,
                        network: str | None = None):
    """Hold out whole physical units, for the device types that have several.

    This is what separates "the model recognises this box" from "the model
    recognises this product". DATA2025 supports it: Tapo Motion Sensors and
    Meross Plugs come in fours, Tuya Globes in threes, DLink and Tapo plugs and
    vTech baby monitors in pairs. Classes with a single unit stay in training so
    the label space is unchanged, and the returned list names the classes the
    test set can actually speak to.
    """
    if b.instance is None:
        raise ValueError("this Bundle carries no instance labels")
    m = np.ones(len(b.y), bool) if network is None else (b.network == network)
    rng = np.random.default_rng(seed)
    labs = np.array(b.labels)
    held: list[str] = []
    test_classes: list[str] = []
    for cls in sorted(set(labs[b.y[m]].tolist())):
        units = sorted(set(b.instance[m & (labs[b.y] == cls)].tolist()))
        if len(units) < 2:
            continue
        held.append(units[int(rng.integers(len(units)))])
        test_classes.append(cls)
    if not held:
        raise ValueError("no device type in this selection has two units")
    te = np.where(m & np.isin(b.instance, held))[0]
    tr_all = np.where(m & ~np.isin(b.instance, held))[0]
    tr, va = _group_holdout(b.group[tr_all], b.y[tr_all], val_frac,
                            np.random.default_rng(seed + 1))
    return tr_all[tr], tr_all[va], te, test_classes, held
