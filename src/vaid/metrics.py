"""Evaluation metrics (§6.4) and the statistics §6.6 asks for.

Macro-F1 is primary. Accuracy is never reported alone for open-set results.
Every comparison between two methods on the same test observations goes through
a paired bootstrap; comparisons across folds go through Wilcoxon.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             f1_score, matthews_corrcoef, roc_auc_score)


# --------------------------------------------------------------------------
# Closed set
# --------------------------------------------------------------------------

def closed_set_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                       proba: np.ndarray | None = None,
                       classes: np.ndarray | None = None) -> dict:
    """`classes` maps probability columns to class ids. It matters whenever a
    model was trained on a subset of the label space (E5 restricts to devices
    present in both periods), because column j is then not class j."""
    out = {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if len(set(y_true)) > 1 else 0.0,
        "accuracy": float(np.mean(y_true == y_pred)),
        "n": int(len(y_true)),
    }
    if proba is not None and proba.size:
        out.update(calibration_metrics(y_true, proba, classes))
    return out


def per_class_f1(y_true, y_pred, labels) -> dict:
    f = f1_score(y_true, y_pred, average=None, labels=list(range(len(labels))),
                 zero_division=0)
    return {labels[i]: float(f[i]) for i in range(len(labels))}


def calibration_metrics(y_true: np.ndarray, proba: np.ndarray,
                        classes: np.ndarray | None = None, bins: int = 15) -> dict:
    cls = np.arange(proba.shape[1]) if classes is None else np.asarray(classes)
    conf = proba.max(axis=1)
    pred = cls[proba.argmax(axis=1)]
    correct = (pred == y_true).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for i in range(bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if m.sum():
            ece += (m.sum() / len(conf)) * abs(correct[m].mean() - conf[m].mean())
    onehot = np.zeros_like(proba)
    pos = {int(c): j for j, c in enumerate(cls)}
    for r, yt in enumerate(y_true):
        j = pos.get(int(yt))
        if j is not None:          # a class the model never saw stays all-zero
            onehot[r, j] = 1.0
    return {"ece": float(ece), "brier": float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))}


def reliability_curve(y_true, proba, bins: int = 15):
    conf = proba.max(axis=1)
    correct = (proba.argmax(1) == y_true).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    xs, ys, ns = [], [], []
    for i in range(bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if m.sum():
            xs.append(float(conf[m].mean())); ys.append(float(correct[m].mean()))
            ns.append(int(m.sum()))
    return {"confidence": xs, "accuracy": ys, "count": ns}


# --------------------------------------------------------------------------
# Open set
# --------------------------------------------------------------------------

def open_set_metrics(is_unknown: np.ndarray, unknown_score: np.ndarray,
                     y_true_known: np.ndarray, y_pred_known: np.ndarray,
                     accept: np.ndarray) -> dict:
    """`unknown_score` is higher for more-unknown-looking observations."""
    out = {}
    if len(set(is_unknown.tolist())) > 1:
        out["auroc_unknown"] = float(roc_auc_score(is_unknown, unknown_score))
        out["aupr_unknown"] = float(average_precision_score(is_unknown, unknown_score))
    else:
        out["auroc_unknown"] = float("nan")
        out["aupr_unknown"] = float("nan")

    known = ~is_unknown.astype(bool)
    out["unknown_recall"] = float(np.mean(~accept[is_unknown.astype(bool)])) if is_unknown.any() else float("nan")
    out["known_acceptance"] = float(np.mean(accept[known])) if known.any() else float("nan")
    out["fpr_at_tpr"] = 1.0 - out["unknown_recall"] if np.isfinite(out["unknown_recall"]) else float("nan")

    # Known-class macro-F1 after rejection: rejected knowns count as errors.
    if known.any():
        yk = y_true_known[known]
        pk = np.where(accept[known], y_pred_known[known], -1)
        out["known_macro_f1_after_rejection"] = float(
            f1_score(yk, pk, average="macro", zero_division=0,
                     labels=sorted(set(yk.tolist()))))
    out["oscr"] = oscr(is_unknown, unknown_score, y_true_known, y_pred_known)
    return out


def oscr(is_unknown: np.ndarray, unknown_score: np.ndarray,
         y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Open-Set Classification Rate: area under CCR vs FPR as the rejection
    threshold sweeps. CCR counts a known sample only if it is both accepted and
    correctly classified."""
    known = ~is_unknown.astype(bool)
    if not known.any() or not is_unknown.any():
        return float("nan")
    thr = np.unique(unknown_score)
    ccr, fpr = [], []
    correct = known & (y_true == y_pred)
    for t in thr:
        acc = unknown_score <= t
        ccr.append(float(np.sum(correct & acc) / max(1, known.sum())))
        fpr.append(float(np.sum(acc & is_unknown.astype(bool)) / max(1, is_unknown.sum())))
    order = np.argsort(fpr)
    return float(np.trapezoid(np.array(ccr)[order], np.array(fpr)[order]))


# --------------------------------------------------------------------------
# Statistics (§6.6)
# --------------------------------------------------------------------------

def paired_bootstrap(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray,
                     metric=None, n: int = 2000, seed: int = 0) -> dict:
    """Paired bootstrap over the same test observations."""
    metric = metric or (lambda yt, yp: f1_score(yt, yp, average="macro", zero_division=0))
    rng = np.random.default_rng(seed)
    N = len(y_true)
    diffs = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, N, N)
        diffs[i] = metric(y_true[idx], pred_a[idx]) - metric(y_true[idx], pred_b[idx])
    obs = metric(y_true, pred_a) - metric(y_true, pred_b)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p = 2 * min(np.mean(diffs <= 0), np.mean(diffs >= 0))
    return {"delta": float(obs), "ci_low": float(lo), "ci_high": float(hi),
            "p_bootstrap": float(min(1.0, p)),
            "significant": bool(lo > 0 or hi < 0)}


def wilcoxon_across_folds(a: list[float], b: list[float]) -> dict:
    from scipy.stats import wilcoxon
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or np.allclose(a, b):
        return {"p_wilcoxon": float("nan"), "effect_size_r": float("nan"),
                "note": "too few folds or identical scores"}
    stat, p = wilcoxon(a, b)
    d = a - b
    # matched-pairs rank-biserial correlation
    r = float(np.sum(np.sign(d) * np.abs(d).argsort().argsort()) /
              (len(d) * (len(d) + 1) / 2))
    return {"p_wilcoxon": float(p), "effect_size_r": r,
            "median_delta": float(np.median(d))}


def holm_bonferroni(pvals: dict[str, float], alpha: float = 0.05) -> dict:
    items = sorted([(k, v) for k, v in pvals.items() if np.isfinite(v)], key=lambda x: x[1])
    m = len(items)
    out, prev = {}, 0.0
    for i, (k, p) in enumerate(items):
        adj = max(prev, min(1.0, (m - i) * p))
        prev = adj
        out[k] = {"p_raw": float(p), "p_holm": float(adj), "reject": bool(adj < alpha)}
    return out


def mean_ci(values: list[float]) -> dict:
    v = np.asarray(values, float)
    if len(v) == 0:
        return {"mean": float("nan"), "std": float("nan"), "ci95": float("nan")}
    return {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
            "ci95": float(1.96 * v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0,
            "n": int(len(v))}
