"""Tests for the shared neighbour-graph operator in :mod:`grassp.tools._graph`.

Both annotation families diffuse over the same operator. It used to be written out twice
and the copies had drifted on their convergence test, so these pin the properties both
callers rely on.
"""

import anndata as ad
import numpy as np
import pytest
import scanpy as sc
import scipy.sparse as sp

from grassp.tools import _graph


@pytest.fixture
def graph():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(m, 1.0, (30, 6)) for m in (0.0, 3.0, 6.0)])
    data = ad.AnnData(X)
    data.obs_names = [f"P{i}" for i in range(data.n_obs)]
    sc.pp.neighbors(data, n_neighbors=10, use_rep="X")
    return data


class TestSymmetricNormalized:
    def test_is_symmetric_and_bounded(self, graph):
        S = _graph.symmetric_normalized(graph.obsp["connectivities"])
        assert sp.issparse(S)
        assert abs((S - S.T)).max() < 1e-12
        assert S.data.max() <= 1.0 + 1e-12

    def test_isolated_rows_do_not_become_nan(self):
        """A zero-degree protein must contribute nothing, not poison the matrix."""
        W = sp.csr_matrix(np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]))
        S = _graph.symmetric_normalized(W)
        assert np.isfinite(S.toarray()).all()
        assert np.allclose(S.toarray()[2], 0.0)


class TestSpread:
    @staticmethod
    def _seed(n, k, rng):
        labels = rng.integers(0, k, n)
        return np.eye(k)[labels].astype(float)

    def test_alpha_zero_returns_the_seed(self, graph):
        rng = np.random.default_rng(0)
        S = _graph.symmetric_normalized(graph.obsp["connectivities"])
        Y0 = self._seed(graph.n_obs, 3, rng)
        assert np.allclose(_graph.spread(S, Y0, alpha=0.0), Y0)

    def test_alpha_outside_the_unit_interval_is_rejected(self, graph):
        S = _graph.symmetric_normalized(graph.obsp["connectivities"])
        Y0 = np.zeros((graph.n_obs, 2))
        for bad in (-0.1, 1.5):
            with pytest.raises(ValueError, match=r"alpha must be in \[0, 1\]"):
                _graph.spread(S, Y0, alpha=bad)

    @pytest.mark.parametrize("alpha", [0.5, 0.8, 0.95])
    def test_rtol_bounds_the_relative_error_whatever_the_shape(self, graph, alpha):
        """``rtol`` must mean the same thing regardless of how many classes there are,
        how many proteins there are, or how large the seed values are.

        An absolute L1 test fails all three: it tightens as the matrix grows and loosens
        as the numbers shrink. Here the achieved relative error must track ``rtol`` and
        must not drift with the shape or the scale of the input.
        """
        rng = np.random.default_rng(0)
        S = _graph.symmetric_normalized(graph.obsp["connectivities"])
        errors = {}
        for k in (2, 5, 20):
            for scale in (1.0, 1000.0):
                Y0 = self._seed(graph.n_obs, k, rng) * scale
                result = _graph.spread(S, Y0, alpha=alpha, rtol=1e-4)
                exact = _graph.spread(S, Y0, alpha=alpha, rtol=1e-12, max_iter=20000)
                errors[(k, scale)] = np.abs(result - exact).sum() / np.abs(exact).sum()
        assert max(errors.values()) < 1e-4, errors
        # and the same order of magnitude everywhere, rather than drifting with k, n or scale
        assert max(errors.values()) / max(min(errors.values()), 1e-12) < 10, errors

    def test_rtol_scales_the_error_linearly(self, graph):
        """Tightening ``rtol`` by 10x must tighten the achieved error by about 10x --
        that is what makes the number interpretable rather than merely monotone."""
        S = _graph.symmetric_normalized(graph.obsp["connectivities"])
        Y0 = self._seed(graph.n_obs, 5, np.random.default_rng(0))
        exact = _graph.spread(S, Y0, alpha=0.9, rtol=1e-12, max_iter=20000)
        err = {}
        for rtol in (1e-3, 1e-4, 1e-5):
            got = _graph.spread(S, Y0, alpha=0.9, rtol=rtol, max_iter=20000)
            err[rtol] = np.abs(got - exact).sum() / np.abs(exact).sum()
        assert 3 < err[1e-3] / err[1e-4] < 30, err
        assert 3 < err[1e-4] / err[1e-5] < 30, err

    def test_warns_only_when_asked(self, graph):
        S = _graph.symmetric_normalized(graph.obsp["connectivities"])
        Y0 = self._seed(graph.n_obs, 3, np.random.default_rng(0))
        with pytest.warns(UserWarning, match="without convergence"):
            _graph.spread(S, Y0, alpha=0.99, max_iter=1, rtol=1e-12, warn_context="ctx")
        import warnings as w

        with w.catch_warnings():
            w.simplefilter("error")
            _graph.spread(S, Y0, alpha=0.99, max_iter=1, rtol=1e-12)


class TestAffinity:
    def test_connectivities_are_used_as_stored(self, graph):
        T = _graph.affinity(graph, "connectivities")
        assert T is graph.obsp["connectivities"]

    def test_distances_become_an_rbf_kernel_and_are_cached(self, graph):
        T = _graph.affinity(graph, "distances")
        assert "W_spreading" in graph.obsp
        assert T.nnz == graph.obsp["distances"].nnz  # same sparsity pattern
        assert T.data.max() <= 1.0
        assert (T.data > 0).all()

    def test_cache_can_be_declined(self, graph):
        _graph.affinity(graph, "distances", cache=False)
        assert "W_spreading" not in graph.obsp

    def test_missing_key_names_the_remedy(self, graph):
        with pytest.raises(KeyError, match="neighbors"):
            _graph.affinity(graph, "nope")


class TestEffectiveNeighbourhood:
    def test_kish_is_bounded_by_the_degree(self, graph):
        T = graph.obsp["connectivities"]
        n_eff = _graph.neff_kish(T)
        degree = np.asarray((T > 0).sum(axis=1)).ravel()
        assert (n_eff >= 0).all()
        assert (n_eff <= degree + 1e-9).all()

    def test_kish_of_a_uniform_row_is_its_count(self):
        W = sp.csr_matrix(np.array([[0.0, 1.0, 1.0, 1.0]]))
        assert np.isclose(_graph.neff_kish(W)[0], 3.0)

    def test_hutchinson_tracks_kish_at_shallow_depth(self, graph):
        """At small alpha the diffused operator is nearly the graph itself, so the two
        estimators should agree in rank even though they measure different things."""
        S = _graph.symmetric_normalized(graph.obsp["connectivities"])
        diffuse = _graph.make_diffuser(S)
        ones = np.ones(graph.n_obs)
        denominator = diffuse(ones, 0.2)
        estimated = _graph.neff_hutchinson(
            diffuse, 0.2, denominator, n_probe=256, rng=np.random.RandomState(0)
        )
        assert np.isfinite(estimated).all()
        assert (estimated > 0).all()
        kish = _graph.neff_kish(graph.obsp["connectivities"])
        assert np.corrcoef(estimated, kish)[0, 1] > 0.5
