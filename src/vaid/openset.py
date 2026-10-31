"""Calibrated open-set decision layer (§5.9).

Prototype distance is the proposed mechanism; energy score is the comparison
method. The rejection threshold is fitted on VALIDATION unknowns only, and the
experiment driver guarantees U_val and U_test are disjoint -- calibrating on the
classes you later report against is the single easiest way to make an open-set
result look better than it is.
"""
from __future__ import annotations

import numpy as np


def _l2(z: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(z, axis=1, keepdims=True)
    return z / np.clip(n, 1e-9, None)


class PrototypeOpenSet:
    def __init__(self, metric: str = "cosine"):
        self.metric = metric
        self.prototypes: np.ndarray | None = None
        self.classes_: np.ndarray | None = None
        self.tau: float | None = None

    def fit(self, z: np.ndarray, y: np.ndarray) -> "PrototypeOpenSet":
        self.classes_ = np.unique(y)
        zz = _l2(z) if self.metric == "cosine" else z
        self.prototypes = np.stack([zz[y == c].mean(0) for c in self.classes_])
        if self.metric == "cosine":
            self.prototypes = _l2(self.prototypes)
        return self

    def distances(self, z: np.ndarray) -> np.ndarray:
        zz = _l2(z) if self.metric == "cosine" else z
        if self.metric == "cosine":
            return 1.0 - zz @ self.prototypes.T
        return np.linalg.norm(zz[:, None, :] - self.prototypes[None], axis=2)

    def score(self, z: np.ndarray) -> np.ndarray:
        """Unknown-ness: distance to the nearest known prototype."""
        return self.distances(z).min(axis=1)

    def nearest_class(self, z: np.ndarray) -> np.ndarray:
        return self.classes_[self.distances(z).argmin(axis=1)]

    def calibrate(self, z_known_val: np.ndarray, target_known_tpr: float = 0.95,
                  z_unknown_val: np.ndarray | None = None) -> float:
        """tau at the quantile of known-validation scores that accepts
        `target_known_tpr` of them. Validation unknowns are used only to report
        where that threshold lands, never to move it beyond this quantile."""
        s = self.score(z_known_val)
        self.tau = float(np.quantile(s, target_known_tpr))
        if z_unknown_val is not None and len(z_unknown_val):
            self.unknown_rejection_at_tau = float(np.mean(self.score(z_unknown_val) > self.tau))
        return self.tau

    def accept(self, z: np.ndarray) -> np.ndarray:
        assert self.tau is not None, "calibrate() before accept()"
        return self.score(z) <= self.tau


class EnergyOpenSet:
    """Comparison method: free energy over the classifier logits."""

    def __init__(self):
        self.tau: float | None = None

    @staticmethod
    def score(logits: np.ndarray) -> np.ndarray:
        m = logits.max(axis=1, keepdims=True)
        return -(m.squeeze(1) + np.log(np.exp(logits - m).sum(axis=1)))

    def calibrate(self, logits_known_val: np.ndarray, target_known_tpr: float = 0.95) -> float:
        self.tau = float(np.quantile(self.score(logits_known_val), target_known_tpr))
        return self.tau

    def accept(self, logits: np.ndarray) -> np.ndarray:
        return self.score(logits) <= self.tau


def openness(n_train_classes: int, n_test_classes: int) -> float:
    """Standard openness definition, reported alongside the % of held-out classes."""
    return 1.0 - np.sqrt(2 * n_train_classes / (n_train_classes + n_test_classes))
