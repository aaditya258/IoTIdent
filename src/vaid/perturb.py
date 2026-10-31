"""Experiment 8: traffic-perturbation simulation.

These operate on window records BEFORE feature construction, so a perturbation
propagates into every feature it should touch (jitter changes interarrival
statistics, burstiness and periodicity; loss changes counts, byte totals and
destination cardinality) instead of being applied to a finished matrix where
half the features would silently disagree with each other.

Sensitivity, not adversarial robustness: none of these is optimised against the
model (§9.7).
"""
from __future__ import annotations

import copy

import numpy as np

from .windows import WindowRecord

KINDS = ("jitter_ms", "loss_frac", "pad_bytes", "missing_handshake_prob")

#: The levels the proposal names in Experiment 8.
LEVELS = {
    "jitter_ms": [5, 10, 25, 50, 100],
    "loss_frac": [0.01, 0.02, 0.05, 0.10],
    "pad_bytes": [0, 16, 64, 256],
    "missing_handshake_prob": [0.10, 0.25, 0.50, 0.75, 1.00],
}


def _copy(r: WindowRecord) -> WindowRecord:
    n = copy.copy(r)
    n.seq = r.seq.copy()
    n.agg = dict(r.agg)
    n.comm = dict(r.comm)
    n.hs = copy.deepcopy(r.hs)
    n.dest_groups = list(r.dest_groups)
    return n


def perturb(records: list[WindowRecord], kind: str, level: float,
            rng: np.random.Generator) -> list[WindowRecord]:
    if kind not in KINDS:
        raise ValueError(f"unknown perturbation {kind}")
    out = []
    for r in records:
        n = _copy(r)
        if kind == "jitter_ms" and level > 0 and len(n.seq):
            # column 9 is log1p(dt_prev); perturb in linear space and re-log
            dt = np.expm1(n.seq[:, 9])
            dt = np.clip(dt + rng.normal(0, level / 1000.0, size=dt.shape), 0, None)
            n.seq[:, 9] = np.log1p(dt)
            iat = np.expm1(n.seq[:, 5])
            iat = np.clip(iat + rng.normal(0, level / 1000.0, size=iat.shape), 0, None)
            n.seq[:, 5] = np.log1p(iat)
            n.agg["mean_log_dt"] = float(np.log1p(dt).mean())
            n.agg["std_log_dt"] = float(np.log1p(dt).std())
            n.agg["burst_frac"] = float(np.mean(dt < 1.0))
            cv = float(dt.std() / (dt.mean() + 1e-9))
            n.comm["periodicity"] = float(1.0 / (1.0 + cv))
        elif kind == "loss_frac" and level > 0 and len(n.seq) > 1:
            keep = rng.random(len(n.seq)) >= level
            if keep.sum() == 0:
                keep[rng.integers(len(keep))] = True
            n.seq = n.seq[keep]
            n.n_flows = max(1, int(round(r.n_flows * (1 - level))))
            n.agg["n_flows"] = float(n.n_flows)
            n.agg["log_n_flows"] = float(np.log1p(n.n_flows))
            n.agg["flows_per_min"] = r.agg["flows_per_min"] * (1 - level)
            n.agg["log_total_src_bytes"] = float(np.log1p(np.expm1(r.agg["log_total_src_bytes"]) * (1 - level)))
            n.agg["log_total_dst_bytes"] = float(np.log1p(np.expm1(r.agg["log_total_dst_bytes"]) * (1 - level)))
        elif kind == "pad_bytes" and level > 0 and len(n.seq):
            pad = rng.integers(0, int(level) + 1, size=len(n.seq))
            for col in (0, 1, 7, 8):
                n.seq[:, col] = np.log1p(np.expm1(n.seq[:, col]) + pad)
            sb = np.expm1(n.agg["log_total_src_bytes"]) + pad.sum()
            n.agg["log_total_src_bytes"] = float(np.log1p(sb))
        elif kind == "missing_handshake_prob" and level > 0:
            if rng.random() < level:
                n.hs = dict(n.hs)
                n.hs.update({"tls_present": 0.0, "n_tls": 0.0, "version": "",
                             "ciphers": [], "ext_types": [], "groups": [], "ecpf": [],
                             "n_ciphers": 0.0, "n_exts": 0.0, "ext_len_total": 0.0,
                             "n_sig_algs": 0.0, "has_alpn": 0.0, "has_ticket": 0.0,
                             "has_keyshare": 0.0, "has_sni": 0.0, "sni_len": 0.0,
                             "n_fingerprints": 0.0})
        out.append(n)
    return out
