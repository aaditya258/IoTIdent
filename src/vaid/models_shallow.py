"""Conventional baselines (§6.3) and a torch-free fallback for the proposed
fusion, so the whole protocol still runs on a machine without PyTorch.

`FlatClassifier` covers Random Forest, gradient boosting and an MLP on a single
concatenated feature vector -- the "fixed-view concatenation" family the paper
argues against.

`ShallowVisibilityFusion` is the fallback for the proposed method: one expert
per view, then a fusion layer whose weights are conditioned on the availability
mask. It is a mixture of experts rather than a gated neural fusion, but it tests
the same hypothesis (fusion weights should depend on which views were observed)
and it degrades the same way when a view disappears.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from .config import Config
from .views import Bundle

EPS = 1e-9


def seq_summary(b: Bundle) -> np.ndarray:
    """Mean/std/max over the flow-event sequence, so that flat models see the
    same information the TCN sees, only without the ordering."""
    m = b.seq_mask[..., None]
    cnt = np.clip(m.sum(1), 1, None)
    mean = (b.Xseq * m).sum(1) / cnt
    var = ((b.Xseq - mean[:, None, :]) ** 2 * m).sum(1) / cnt
    mx = np.where(m > 0, b.Xseq, -1e9).max(1)
    return np.concatenate([mean, np.sqrt(np.clip(var, 0, None)), mx], axis=1)


def flat_features(b: Bundle, views=("F", "H", "C")) -> np.ndarray:
    parts = []
    if "F" in views:
        parts += [seq_summary(b), b.Xagg]
    if "H" in views:
        parts += [b.Xh * b.mH[:, None], b.mH[:, None], b.h_field_mask]
    if "C" in views:
        parts += [b.Xc * b.mC[:, None], b.mC[:, None]]
    X = np.concatenate(parts, axis=1) if parts else np.zeros((len(b.y), 1))
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


class FlatClassifier:
    def __init__(self, kind: str = "rf", cfg: Config | None = None,
                 views=("F", "H", "C"), seed: int = 0):
        self.kind, self.views, self.seed = kind, tuple(views), seed
        self.cfg = cfg or Config()
        self.model = None
        self.classes_ = None

    def _make(self):
        if self.kind == "rf":
            return RandomForestClassifier(n_estimators=300, n_jobs=-1,
                                          class_weight="balanced_subsample",
                                          random_state=self.seed)
        if self.kind in ("gbm", "hgb", "xgboost"):
            # class_weight matters here: macro-F1 over 27 imbalanced classes
            # collapses without it, and the Random Forest above IS weighted, so
            # leaving it off compares a balanced model against an unbalanced one.
            # Measured on DATA2025: 0.372 +- 0.328 unweighted, swinging 0.07 to
            # 0.99 across folds purely with which classes were rare in the test
            # split.
            try:
                return HistGradientBoostingClassifier(
                    max_iter=200, learning_rate=0.1, random_state=self.seed,
                    class_weight="balanced")
            except TypeError:           # sklearn < 1.4
                return HistGradientBoostingClassifier(
                    max_iter=200, learning_rate=0.1, random_state=self.seed)
        if self.kind == "mlp":
            return MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=400,
                                 random_state=self.seed)
        raise ValueError(self.kind)

    def fit(self, tr: Bundle, va: Bundle | None = None):
        X, y = flat_features(tr, self.views), tr.y
        self.model = self._make().fit(X, y)
        self.classes_ = self.model.classes_
        return self

    def predict(self, b: Bundle) -> dict:
        X = flat_features(b, self.views)
        proba = self.model.predict_proba(X)
        return {"proba": proba, "logits": np.log(proba + EPS),
                "z": np.log(proba + EPS),
                "y_pred": self.classes_[proba.argmax(1)]}


class ShallowVisibilityFusion:
    """Per-view experts + mask-conditioned fusion (torch-free fallback)."""

    def __init__(self, cfg: Config, fusion: str = "gated", views=("F", "H", "C"),
                 seed: int = 0, view_dropout: bool = True):
        self.cfg, self.fusion, self.views, self.seed = cfg, fusion, tuple(views), seed
        self.view_dropout = view_dropout
        self.experts: dict = {}
        self.gate = None
        self.classes_ = None

    @staticmethod
    def _view_X(b: Bundle, v: str) -> np.ndarray:
        if v == "F":
            return np.nan_to_num(np.concatenate([seq_summary(b), b.Xagg], axis=1))
        if v == "H":
            return np.nan_to_num(b.Xh * b.mH[:, None])
        return np.nan_to_num(b.Xc * b.mC[:, None])

    def _mask(self, b: Bundle, v: str) -> np.ndarray:
        return np.ones(len(b.y)) if v == "F" else (b.mH if v == "H" else b.mC)

    def fit(self, tr: Bundle, va: Bundle | None = None):
        rng = np.random.default_rng(self.seed)
        self.classes_ = np.unique(tr.y)
        for v in self.views:
            m = self._mask(tr, v) > 0.5
            idx = np.where(m)[0] if v != "F" else np.arange(len(tr.y))
            if len(idx) < 20 or len(np.unique(tr.y[idx])) < 2:
                self.experts[v] = None
                continue
            self.experts[v] = HistGradientBoostingClassifier(
                max_iter=150, random_state=self.seed).fit(
                    self._view_X(tr, v)[idx], tr.y[idx])

        Z, M = self._stack(tr, rng if self.view_dropout else None)
        self.gate = LogisticRegression(max_iter=1000, multi_class="auto").fit(
            np.concatenate([Z, M], axis=1), tr.y)
        return self

    def _expert_logp(self, b: Bundle, v: str) -> np.ndarray:
        K = len(self.classes_)
        exp = self.experts.get(v)
        if exp is None:
            return np.full((len(b.y), K), np.log(1.0 / K))
        p = np.full((len(b.y), K), 1.0 / K)
        pp = exp.predict_proba(self._view_X(b, v))
        for j, c in enumerate(exp.classes_):
            k = int(np.where(self.classes_ == c)[0][0])
            p[:, k] = pp[:, j]
        p = p / p.sum(1, keepdims=True)
        return np.log(p + EPS)

    def _stack(self, b: Bundle, rng=None) -> tuple[np.ndarray, np.ndarray]:
        Zs, Ms = [], []
        for v in self.views:
            m = self._mask(b, v).copy()
            if rng is not None and v != "F":
                p = self.cfg.p_drop_H if v == "H" else self.cfg.p_drop_C
                m = m * (rng.random(len(m)) >= p)
            lp = self._expert_logp(b, v)
            if self.fusion == "gated":
                lp = lp * m[:, None] + (1 - m[:, None]) * np.log(1.0 / len(self.classes_))
            Zs.append(lp)
            Ms.append(m[:, None])
        return np.concatenate(Zs, axis=1), np.concatenate(Ms, axis=1)

    def predict(self, b: Bundle) -> dict:
        Z, M = self._stack(b, None)
        if self.fusion in ("late", "fixed"):
            K = len(self.classes_)
            acc = np.zeros((len(b.y), K)); w = np.zeros((len(b.y), 1))
            for i, v in enumerate(self.views):
                acc += Z[:, i * K:(i + 1) * K] * M[:, i: i + 1]
                w += M[:, i: i + 1]
            logits = acc / np.clip(w, 1e-6, None)
        else:
            proba = self.gate.predict_proba(np.concatenate([Z, M], axis=1))
            logits = np.log(proba + EPS)
        p = np.exp(logits - logits.max(1, keepdims=True))
        p = p / p.sum(1, keepdims=True)
        return {"proba": p, "logits": logits, "z": Z,
                "y_pred": self.classes_[p.argmax(1)]}


def make_model(cfg: Config, name: str, seed: int = 0, views=("F", "H", "C")):
    """Factory used by the experiment drivers.

    Names: rf, gbm, mlp                       -- conventional flat baselines
           cnn1d, tcn, transformer            -- sequence baselines (flow view)
           gated, concat, late, fixed, attention -- fusion mechanisms
    """
    from .models_torch import TORCH_AVAILABLE, NeuralClassifier
    backend = cfg.backend
    if backend == "auto":
        backend = "torch" if TORCH_AVAILABLE else "shallow"

    if name in ("rf", "gbm", "mlp", "xgboost"):
        return FlatClassifier(name, cfg, views, seed)
    if name in ("cnn1d", "tcn", "transformer"):
        if backend == "torch":
            return NeuralClassifier(cfg, fusion="gated", seq_encoder=name,
                                    use_views=("F",), view_dropout=False, seed=seed)
        return FlatClassifier("gbm", cfg, ("F",), seed)
    if name in ("gated", "concat", "late", "fixed", "attention"):
        if backend == "torch":
            return NeuralClassifier(cfg, fusion=name, seq_encoder="tcn",
                                    use_views=views,
                                    view_dropout=(name == "gated"), seed=seed)
        return ShallowVisibilityFusion(cfg, fusion=name, views=views, seed=seed,
                                       view_dropout=(name == "gated"))
    raise ValueError(f"unknown model {name}")
