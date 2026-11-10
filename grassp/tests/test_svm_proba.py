"""Tests for the libsvm-style pairwise-coupled SVM probabilities (grassp.tools._svm_proba)."""

import warnings

import numpy as np
import pytest

from sklearn.svm import SVC

from grassp.tools._svm_proba import PairwiseCoupledSVC, couple_pairwise


def _pairwise_from(p):
    """r[:, i, j] = P(i | i or j) implied by a full distribution p."""
    return p[:, :, None] / (p[:, :, None] + p[:, None, :])


def _blobs(n_classes, n_per_class=60, spread=1.0, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, 2.5, (n_classes, 5))
    X = np.vstack([rng.normal(c, spread, (n_per_class, 5)) for c in centers])
    y = np.repeat(np.arange(n_classes), n_per_class)
    return X, y


class TestCoupling:
    def test_recovers_a_consistent_distribution(self):
        """If the pairwise probabilities come from one distribution, coupling returns it."""
        rng = np.random.default_rng(0)
        p = rng.dirichlet(np.ones(6), size=50)
        np.testing.assert_allclose(couple_pairwise(_pairwise_from(p)), p, atol=1e-10)

    def test_rows_are_distributions(self):
        rng = np.random.default_rng(1)
        r = rng.uniform(0.01, 0.99, (200, 5, 5))
        r = np.triu(r, 1) + np.tril(1 - r.transpose(0, 2, 1), -1)  # r_ji = 1 - r_ij
        p = couple_pairwise(r)
        assert (p >= 0).all()
        np.testing.assert_allclose(p.sum(axis=1), 1.0)

    def test_two_classes_is_the_pairwise_probability(self):
        r = np.array([[[0.5, 0.8], [0.2, 0.5]]])
        np.testing.assert_allclose(couple_pairwise(r), [[0.8, 0.2]])


class TestPairwiseCoupledSVC:
    @pytest.mark.parametrize("n_classes", [2, 4])
    def test_probabilities_favour_the_true_class(self, n_classes):
        """Also pins SVC's sign conventions: one-vs-one decision values are positive for
        the pair's first class, except with two classes, where SVC flips it."""
        X, y = _blobs(n_classes, spread=0.5)
        model = PairwiseCoupledSVC(SVC(gamma=0.1), cv=5).fit(X, y)
        P = model.predict_proba(X)
        np.testing.assert_allclose(P.sum(axis=1), 1.0)
        assert (model.classes_[P.argmax(axis=1)] == y).mean() > 0.95

    def test_deterministic(self):
        X, y = _blobs(3, spread=1.5)
        a = PairwiseCoupledSVC(SVC(gamma=0.1), random_state=3).fit(X, y).predict_proba(X)
        b = PairwiseCoupledSVC(SVC(gamma=0.1), random_state=3).fit(X, y).predict_proba(X)
        np.testing.assert_array_equal(a, b)

    def test_matches_libsvm_probability(self):
        """Same method as SVC(probability=True); differences come from the CV splits."""
        X, y = _blobs(5, spread=2.0, seed=4)
        svc_kwargs = dict(C=1.0, gamma=0.1, class_weight="balanced")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            try:
                old = SVC(probability=True, random_state=0, **svc_kwargs).fit(X, y)
            except (TypeError, ValueError):
                pytest.skip("this scikit-learn no longer has SVC(probability=True)")
        new = PairwiseCoupledSVC(SVC(**svc_kwargs), cv=5).fit(X, y)
        P_old, P_new = old.predict_proba(X), new.predict_proba(X)
        assert (P_old.argmax(axis=1) == P_new.argmax(axis=1)).mean() > 0.95
        assert np.corrcoef(P_old.ravel(), P_new.ravel())[0, 1] > 0.99
