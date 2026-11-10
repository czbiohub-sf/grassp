"""Multiclass SVM probabilities by pairwise coupling, as libsvm's ``probability=True`` did.

scikit-learn deprecated ``SVC(probability=True)`` in 1.9 (removal in 1.11) and points to
``CalibratedClassifierCV(SVC(), ensemble=False)``. For two classes that is the same
model; for more it is not. ``CalibratedClassifierCV`` calibrates ``SVC.decision_function``
one-vs-rest, and those scores are built from one-vs-one *votes*; on the Hein 2024 markers
(16 classes) held-out accuracy fell from 0.904 to 0.865 and log-loss rose from 0.39 to
0.65. libsvm instead:

1. uses the one-vs-one SVMs that ``SVC`` fits anyway, one per class pair;
2. fits a Platt sigmoid per pair on cross-validated decision values, with Platt's target
   smoothing;
3. couples the ``k(k-1)/2`` pairwise probabilities into one distribution per sample
   (Wu, Lin & Weng 2004, "Probability estimates for multi-class classification by pairwise
   coupling", JMLR 5, method 2).

This module rebuilds that from public pieces: ``SVC(decision_function_shape="ovo")`` for
the pairwise margins, ``cross_val_predict`` for the out-of-fold margins, scipy for the
sigmoid fits and one batched linear solve for the coupling. On the Hein markers it matches
``SVC(probability=True)`` to within cross-validation noise (top call agrees for 99% of
proteins, probabilities correlate at 0.9997). Unlike libsvm's version it uses no shared
random state, which is what made ``probability=True`` thread-unsafe.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

from scipy.optimize import minimize
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_predict

if TYPE_CHECKING:
    from sklearn.svm import SVC

# Rows coupled per batch: the coupling builds a (rows, k+1, k+1) system.
_COUPLING_CHUNK = 4096


def _fit_platt(decision: np.ndarray, positive: np.ndarray) -> tuple[float, float]:
    """Platt (1999) sigmoid ``P(positive | f) = 1 / (1 + exp(A f + B))``, as in libsvm.

    The targets are smoothed (``(n+ + 1) / (n+ + 2)`` and ``1 / (n- + 2)``) so that a
    perfectly separated pair does not drive the sigmoid to a step.
    """
    n_pos = int(positive.sum())
    n_neg = positive.size - n_pos
    target = np.where(positive, (n_pos + 1) / (n_pos + 2), 1 / (n_neg + 2))

    def neg_log_likelihood(ab):
        z = ab[0] * decision + ab[1]
        log1p_exp_z = np.logaddexp(0.0, z)  # -log P(positive), computed stably
        return np.sum(target * log1p_exp_z + (1 - target) * (log1p_exp_z - z))

    start = np.array([0.0, np.log((n_neg + 1) / (n_pos + 1))])
    a, b = minimize(neg_log_likelihood, start, method="L-BFGS-B").x
    return float(a), float(b)


def couple_pairwise(r: np.ndarray) -> np.ndarray:
    """Couple pairwise class probabilities into one distribution per sample.

    Wu, Lin & Weng (2004), method 2, which libsvm implements: minimise ``p' Q p`` subject
    to ``sum(p) = 1`` with ``Q_ii = sum_{j != i} r_ji^2`` and ``Q_ij = -r_ji r_ij``.

    Parameters
    ----------
    r
        ``(n, k, k)`` array with ``r[:, i, j] = P(class i | class i or j)``; the diagonal
        is ignored.

    Returns
    -------
    ``(n, k)`` array of probabilities; each row sums to 1.
    """
    n, k, _ = r.shape
    rt = r.transpose(0, 2, 1)  # rt[:, i, j] = r[:, j, i]
    off = ~np.eye(k, dtype=bool)
    Q = np.where(off, -rt * r, 0.0)
    Q[:, np.arange(k), np.arange(k)] = np.where(off, rt**2, 0.0).sum(axis=2)
    system = np.zeros((n, k + 1, k + 1))
    system[:, :k, :k] = Q
    system[:, :k, k] = 1.0
    system[:, k, :k] = 1.0
    rhs = np.zeros((n, k + 1))
    rhs[:, k] = 1.0
    p = np.linalg.solve(system, rhs[..., None])[:, :k, 0]
    # the constrained minimum is non-negative up to round-off
    p = np.clip(p, 0.0, None)
    return p / p.sum(axis=1, keepdims=True)


class PairwiseCoupledSVC:
    """An :class:`~sklearn.svm.SVC` with libsvm-style multiclass probabilities.

    Parameters
    ----------
    svc
        Unfitted :class:`~sklearn.svm.SVC` (``probability`` must be left unset). It is
        cloned; its ``decision_function_shape`` is set to ``"ovo"``.
    cv
        Number of stratified folds for the out-of-fold decision values the sigmoids are
        fitted on. Every class needs at least ``cv`` samples.
    random_state
        Seed for the fold assignment.
    """

    def __init__(self, svc: SVC, cv: int = 5, random_state: int | None = 0):
        self.svc = svc
        self.cv = cv
        self.random_state = random_state

    def fit(self, X, y) -> PairwiseCoupledSVC:
        y = np.asarray(y)
        base = clone(self.svc).set_params(decision_function_shape="ovo")
        self.svc_ = clone(base).fit(X, y)
        self.classes_ = self.svc_.classes_
        k = len(self.classes_)
        # same order as the columns of SVC's one-vs-one decision_function
        self.pairs_ = [(i, j) for i in range(k) for j in range(i + 1, k)]
        folds = StratifiedKFold(self.cv, shuffle=True, random_state=self.random_state)
        decision = self._pairwise(
            cross_val_predict(base, X, y, cv=folds, method="decision_function")
        )
        self.sigmoids_ = []
        for col, (i, j) in enumerate(self.pairs_):
            in_pair = (y == self.classes_[i]) | (y == self.classes_[j])
            # a positive one-vs-one decision value is a vote for the pair's first class
            self.sigmoids_.append(
                _fit_platt(decision[in_pair, col], y[in_pair] == self.classes_[i])
            )
        return self

    def _pairwise(self, decision: np.ndarray) -> np.ndarray:
        """One column per pair, positive = vote for the pair's first class.

        That is SVC's one-vs-one convention for three or more classes. With two, SVC
        returns a single column that is positive for the *second* class, so it is negated.
        """
        if len(self.classes_) == 2:
            return -decision.reshape(-1, 1)
        return decision

    def predict_proba(self, X) -> np.ndarray:
        decision = self._pairwise(self.svc_.decision_function(X))
        k = len(self.classes_)
        out = np.empty((decision.shape[0], k))
        for start in range(0, decision.shape[0], _COUPLING_CHUNK):
            block = decision[start : start + _COUPLING_CHUNK]
            r = np.full((block.shape[0], k, k), 0.5)
            for col, (i, j) in enumerate(self.pairs_):
                a, b = self.sigmoids_[col]
                # 1 / (1 + exp(a f + b)), computed without overflow
                p = np.exp(-np.logaddexp(0.0, a * block[:, col] + b))
                p = np.clip(p, 1e-7, 1 - 1e-7)
                r[:, i, j], r[:, j, i] = p, 1 - p
            out[start : start + block.shape[0]] = couple_pairwise(r)
        return out
