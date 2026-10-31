"""Window cache and split-aware materialisation.

Two-step on purpose. Splitting needs to know which device, when and which
adjacency group each window belongs to, but it must NOT need the features --
because the features (vocabularies, scalers) are fitted on the training half of
whatever split is chosen. So:

    records  = build_windows(...)          # once, cached
    index    = make_index(records)         # identity/time only, no features
    train,val,test = some_split(index)
    bundle   = materialise(records, cfg, train_idx=train)   # fits on train only

Calling `materialise` with the full index as `train_idx` would leak the test
distribution into the scalers, so the experiment drivers never do that.
"""
from __future__ import annotations

import hashlib
import os
import pickle

import numpy as np

from .config import Config, Paths
from .views import Bundle, ViewSpace
from .windows import WindowRecord, build_windows


def _cache_key(cfg: Config, devices) -> str:
    payload = f"T{cfg.T_minutes}|L{cfg.max_flows_per_window}|min{cfg.min_flows_per_window}|" \
              f"salt{cfg.hash_salt}|dg{cfg.dest_group_vocab}|{sorted(devices or [])}"
    return hashlib.blake2b(payload.encode(), digest_size=8).hexdigest()


def load_records(paths: Paths, cfg: Config, devices=None, use_cache: bool = True,
                 verbose: bool = True) -> list[WindowRecord]:
    os.makedirs(paths.cache, exist_ok=True)
    p = os.path.join(paths.cache, f"windows_{_cache_key(cfg, devices)}.pkl")
    if use_cache and os.path.exists(p):
        with open(p, "rb") as fh:
            return pickle.load(fh)
    if verbose:
        print(f"building windows (T={cfg.T_minutes} min) ...")
    recs = build_windows(paths, cfg, devices, verbose=verbose)
    with open(p, "wb") as fh:
        pickle.dump(recs, fh, protocol=4)
    return recs


def make_index(records: list[WindowRecord], cfg: Config,
               labels: list[str] | None = None) -> Bundle:
    labs = labels or sorted({r.label for r in records})
    N = len(records)
    z = np.zeros((N, 0), dtype=np.float32)
    return Bundle(
        Xseq=np.zeros((N, 0, 0), dtype=np.float32), seq_mask=np.zeros((N, 0), dtype=np.float32),
        Xagg=z, Xh=z, Xc=z,
        mH=np.array([1.0 if (r.hs["tls_present"] or r.hs["dhcp_present"]) else 0.0
                     for r in records], dtype=np.float32),
        mC=np.array([1.0 if r.n_flows >= 2 else 0.0 for r in records], dtype=np.float32),
        h_field_mask=np.zeros((N, 4), dtype=np.float32),
        y=np.array([labs.index(r.label) if r.label in labs else -1 for r in records]),
        labels=labs,
        device=np.array([r.device for r in records]),
        t_start=np.array([r.t_start for r in records], dtype=float),
        period=np.array([r.period for r in records]),
        group=np.array([f"{getattr(r, 'network', '')}|{r.device}|"
                        f"{int(r.t_start // (cfg.group_block_hours * 3600))}"
                        for r in records]),
        n_flows=np.array([r.n_flows for r in records]),
        network=np.array([getattr(r, "network", "") for r in records]),
        instance=np.array([r.device for r in records]))


def materialise(records: list[WindowRecord], cfg: Config, train_idx: np.ndarray,
                labels: list[str] | None = None, allow_v6: bool = False
                ) -> tuple[Bundle, ViewSpace]:
    vs = ViewSpace(cfg).fit(records, np.asarray(train_idx))
    b = vs.transform(records, labels=labels,
                     allow_v6=allow_v6 and cfg.allow_endpoint_upper_bound)
    return b, vs


def filter_records(records: list[WindowRecord], keep_labels) -> tuple[list, np.ndarray]:
    keep = set(keep_labels)
    idx = np.array([i for i, r in enumerate(records) if r.label in keep])
    return [records[i] for i in idx], idx


def class_counts(records: list[WindowRecord]) -> dict:
    from collections import Counter
    return dict(Counter(r.label for r in records).most_common())


def cap_records_per_class(records: list[WindowRecord], cap: int, cfg: Config,
                          seed: int = 0, verbose: bool = True) -> list[WindowRecord]:
    """Subsample to at most `cap` windows per class, by whole adjacency group.

    Why group-wise and not row-wise: windows from one device inside one
    adjacency block are near-duplicates of each other, so a row-wise sample
    would scatter members of the same block across the training and test sides
    of a later split and leak. Dropping whole groups keeps `e1_session_grouped`
    honest.

    Why cap at all: the full DATA2025 span is roughly twenty times the
    development set, and `cf_cli threshold` refits its model twenty times
    (5 seeds x 4 axes). Capping bounds the expensive arm WITHOUT shortening the
    calendar span -- which matters, because the span is the thing the time axis
    measures. Groups are drawn in shuffled order, so the retained windows stay
    spread across the whole period rather than clustering at the start.
    """
    if not cap or cap <= 0:
        return records
    block = max(1, int(cfg.group_block_hours)) * 3600
    by_class: dict[str, dict[str, list[int]]] = {}
    for i, r in enumerate(records):
        g = f"{getattr(r, 'network', '')}|{r.device}|{int(r.t_start // block)}"
        by_class.setdefault(r.label, {}).setdefault(g, []).append(i)

    rng = np.random.default_rng(seed)
    keep: list[int] = []
    capped: list[str] = []
    for cls, groups in sorted(by_class.items()):
        total = sum(len(v) for v in groups.values())
        if total <= cap:
            for v in groups.values():
                keep += v
            continue
        names = sorted(groups)
        rng.shuffle(names)
        n = 0
        for g in names:
            if n >= cap:
                break
            keep += groups[g]
            n += len(groups[g])
        capped.append(f"{cls}({total}->{n})")
    keep.sort()
    if verbose and capped:
        print(f"  capped {len(capped)} class(es) at {cap} windows: "
              + ", ".join(capped[:6]) + ("..." if len(capped) > 6 else ""))
        print(f"  {len(records):,} -> {len(keep):,} windows")
    return [records[i] for i in keep]
