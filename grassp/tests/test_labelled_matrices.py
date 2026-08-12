"""Tests for labelled ``.obsm`` matrices -- the DataFrame contract.

Tools that score every protein against every compartment write an ``(n_obs, n_classes)``
matrix to ``.obsm`` as a :class:`~pandas.DataFrame`, so the class each column stands for is
carried by the matrix itself rather than reapplied positionally from somewhere else. These
tests pin that contract down at three points: what the writers produce, that it survives a
round trip through h5ad, and that readers still accept the bare ndarrays written before the
switch.
"""

import matplotlib

matplotlib.use('Agg')  # Use non-interactive backend for testing

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
import scanpy as sc  # noqa: E402

from anndata import AnnData  # noqa: E402

from grassp.preprocessing import simple  # noqa: E402
from grassp.tools import localization, scoring, tagm  # noqa: E402
from grassp.util import get_matrix, set_matrix  # noqa: E402


def make_annotated_data(n_proteins=60, n_samples=6, n_compartments=4):
    """A small object with compartment structure and markers, ready to propagate."""
    rng = np.random.default_rng(0)
    compartments = [f"Compartment{i}" for i in range(1, n_compartments + 1)]

    X = np.zeros((n_proteins, n_samples))
    blocks = np.array_split(np.arange(n_proteins), n_compartments)
    for i, rows in enumerate(blocks):
        pattern = rng.normal(0, 0.3, n_samples)
        pattern[i % n_samples] += 4.0
        X[rows, :] = pattern + rng.normal(0, 0.2, (len(rows), n_samples))

    obs = pd.DataFrame(index=[f"P{i:05d}" for i in range(n_proteins)])
    markers = np.array([None] * n_proteins, dtype=object)
    for i, rows in enumerate(blocks):
        # label the first third of each block, leave the rest to be predicted
        for row in rows[: max(1, len(rows) // 3)]:
            markers[row] = compartments[i]
    obs["markers"] = pd.Categorical(markers, categories=compartments)

    var = pd.DataFrame(index=[f"Sample{i}" for i in range(n_samples)])
    data = AnnData(X=X, obs=obs, var=var)
    simple.neighbors(data, n_neighbors=min(10, n_proteins - 1))
    return data, compartments


@pytest.fixture
def annotated():
    return make_annotated_data()


# ==============================================================================
# The helpers
# ==============================================================================


class TestHelpers:
    def test_set_matrix_writes_named_dataframe(self, annotated):
        data, compartments = annotated
        values = np.arange(data.n_obs * len(compartments), dtype=float).reshape(
            data.n_obs, len(compartments)
        )
        set_matrix(data, "scores", values, compartments)

        stored = data.obsm["scores"]
        assert isinstance(stored, pd.DataFrame)
        assert list(stored.columns) == compartments
        assert stored.index.equals(data.obs_names)
        np.testing.assert_array_equal(stored.to_numpy(), values)

    def test_set_matrix_coerces_labels_to_str(self, annotated):
        data, _ = annotated
        set_matrix(data, "ints", np.zeros((data.n_obs, 2)), [0, 1])
        assert list(data.obsm["ints"].columns) == ["0", "1"]

    def test_set_matrix_takes_dataframe_positionally(self, annotated):
        """A DataFrame input is data, not a lookup -- its own labels are discarded."""
        data, _ = annotated
        incoming = pd.DataFrame(
            np.ones((data.n_obs, 2)),
            index=[f"other{i}" for i in range(data.n_obs)],
            columns=["x", "y"],
        )
        set_matrix(data, "renamed", incoming, ["a", "b"])
        assert list(data.obsm["renamed"].columns) == ["a", "b"]
        assert data.obsm["renamed"].index.equals(data.obs_names)

    def test_set_matrix_rejects_duplicate_labels(self, annotated):
        data, _ = annotated
        with pytest.raises(ValueError, match="Duplicate column labels"):
            set_matrix(data, "dupes", np.zeros((data.n_obs, 2)), ["a", "a"])

    def test_get_matrix_round_trips(self, annotated):
        data, compartments = annotated
        values = np.zeros((data.n_obs, len(compartments)))
        set_matrix(data, "scores", values, compartments)

        got, columns = get_matrix(data, "scores")
        assert isinstance(got, np.ndarray)
        assert columns == compartments
        np.testing.assert_array_equal(got, values)

    def test_get_matrix_accepts_legacy_ndarray(self, annotated):
        """Objects written before the switch store a bare array and carry no names."""
        data, compartments = annotated
        values = np.zeros((data.n_obs, len(compartments)))
        data.obsm["legacy"] = values

        got, columns = get_matrix(data, "legacy")
        assert columns is None
        np.testing.assert_array_equal(got, values)

    def test_set_matrix_varm(self, annotated):
        data, _ = annotated
        set_matrix(data, "loadings", np.zeros((data.n_vars, 2)), ["a", "b"], axis="var")
        assert isinstance(data.varm["loadings"], pd.DataFrame)
        assert data.varm["loadings"].index.equals(data.var_names)


# ==============================================================================
# The writers
# ==============================================================================


class TestWriters:
    def test_competitive_propagation(self, annotated):
        data, compartments = annotated
        localization.competitive_propagation(data, gt_col="markers", key_added="cp")

        for key in ("cp_probabilities", "cp_one_hot_labels"):
            stored = data.obsm[key]
            assert isinstance(stored, pd.DataFrame), key
            assert list(stored.columns) == compartments, key
            assert stored.index.equals(data.obs_names), key
        # the one-hot block stays boolean rather than widening to float
        assert set(data.obsm["cp_one_hot_labels"].dtypes) == {np.dtype(bool)}

    def test_probabilities_columns_match_predicted_label(self, annotated):
        """The argmax column name must be the label written to .obs."""
        data, _ = annotated
        localization.competitive_propagation(
            data, gt_col="markers", key_added="cp", min_probability=0
        )
        probabilities = data.obsm["cp_probabilities"]
        by_column = probabilities.columns[probabilities.to_numpy().argmax(axis=1)]
        assert list(by_column) == list(data.obs["cp"].astype(str))

    def test_svm_annotation(self, annotated):
        data, compartments = annotated
        localization.svm_annotation(data, gt_col="markers", key_added="svm", C=1.0, gamma=0.1)
        stored = data.obsm["svm_probabilities"]
        assert isinstance(stored, pd.DataFrame)
        assert list(stored.columns) == compartments

    def test_tagm_map(self, annotated):
        data, compartments = annotated
        params = tagm.tagm_map_train(data, gt_col="markers", numIter=2, seed=0)
        tagm.tagm_map_predict(data, params=params)
        stored = data.obsm["tagm.map.probabilities"]
        assert isinstance(stored, pd.DataFrame)
        assert list(stored.columns) == compartments

    def test_replicate_cv(self):
        rng = np.random.default_rng(0)
        obs = pd.DataFrame(index=[f"P{i:03d}" for i in range(20)])
        var = pd.DataFrame({"group": ["a", "a", "b", "b"]}, index=[f"S{i}" for i in range(4)])
        data = AnnData(X=rng.random((20, 4)), obs=obs, var=var)
        simple.calculate_replicate_cv(data, grouping_columns="group", is_log=False)

        stored = data.obsm["replicate_cv"]
        assert isinstance(stored, pd.DataFrame)
        assert list(stored.columns) == ["a", "b"]
        # the uns header stays for readers of older objects
        assert data.uns["obsm_replicate_cv_headers"] == ["a", "b"]

    def test_embeddings_stay_arrays(self, annotated):
        """Embedding columns are unnamed dimensions -- scanpy's contract, left alone."""
        data, _ = annotated
        sc.tl.umap(data)
        assert isinstance(data.obsm["X_umap"], np.ndarray)


# ==============================================================================
# Persistence
# ==============================================================================


class TestRoundTrip:
    def test_h5ad_preserves_labels(self, annotated, tmp_path):
        data, compartments = annotated
        localization.competitive_propagation(data, gt_col="markers", key_added="cp")

        path = tmp_path / "labelled.h5ad"
        data.write_h5ad(path)
        reloaded = sc.read_h5ad(path)

        for key in ("cp_probabilities", "cp_one_hot_labels"):
            stored = reloaded.obsm[key]
            assert isinstance(stored, pd.DataFrame), key
            assert list(stored.columns) == compartments, key
            assert stored.index.equals(reloaded.obs_names), key
        np.testing.assert_allclose(
            reloaded.obsm["cp_probabilities"].to_numpy(),
            data.obsm["cp_probabilities"].to_numpy(),
        )
        assert set(reloaded.obsm["cp_one_hot_labels"].dtypes) == {np.dtype(bool)}

    def test_view_subsetting_keeps_labels(self, annotated):
        data, compartments = annotated
        localization.competitive_propagation(data, gt_col="markers", key_added="cp")
        view = data[data.obs["markers"] == compartments[0]]
        assert list(view.obsm["cp_probabilities"].columns) == compartments
        assert view.obsm["cp_probabilities"].index.equals(view.obs_names)


# ==============================================================================
# Consumers
# ==============================================================================


class TestConsumers:
    def test_confusion_matrix_soft(self, annotated):
        """Regression: DataFrame obsm used to reach ``DataFrame.sum(keepdims=True)``."""
        data, compartments = annotated
        localization.competitive_propagation(
            data, gt_col="markers", key_added="cp", min_probability=0
        )
        cm = scoring.knn_confusion_matrix(
            data, gt_col="markers", pred_col="cp", soft=True, plot=False
        )
        cm = np.asarray(cm)
        assert cm.shape == (len(compartments), len(compartments))
        np.testing.assert_allclose(cm.sum(axis=1), 1.0)

    def test_confusion_matrix_hard(self, annotated):
        data, compartments = annotated
        localization.competitive_propagation(
            data, gt_col="markers", key_added="cp", min_probability=0
        )
        cm = np.asarray(
            scoring.knn_confusion_matrix(
                data, gt_col="markers", pred_col="cp", soft=False, plot=False
            )
        )
        assert cm.shape[0] == cm.shape[1]

    def test_confusion_matrix_accepts_legacy_ndarray(self, annotated):
        """An object saved before the switch still scores, via the .obs fallback."""
        data, compartments = annotated
        localization.competitive_propagation(
            data, gt_col="markers", key_added="cp", min_probability=0
        )
        for key in ("cp_probabilities", "cp_one_hot_labels"):
            data.obsm[key] = data.obsm[key].to_numpy()

        cm = np.asarray(
            scoring.knn_confusion_matrix(
                data, gt_col="markers", pred_col="cp", soft=True, plot=False
            )
        )
        assert cm.shape == (len(compartments), len(compartments))

    def test_soft_seed_reads_columns_without_uns_key(self, annotated):
        """A labelled seed names its own classes; the uns key becomes optional."""
        data, compartments = annotated
        rng = np.random.default_rng(1)
        seed = rng.random((data.n_obs, len(compartments)))
        seed /= seed.sum(axis=1, keepdims=True)
        set_matrix(data, "soft_seed", seed, compartments)

        localization.competitive_propagation(
            data, gt_col=None, key_added="soft", seed_obsm_key="soft_seed"
        )
        assert list(data.obsm["soft_probabilities"].columns) == compartments

    def test_soft_seed_without_labels_still_needs_uns_key(self, annotated):
        data, compartments = annotated
        data.obsm["bare_seed"] = np.full((data.n_obs, len(compartments)), 0.25)
        with pytest.raises(ValueError, match="carries no column names"):
            localization.competitive_propagation(
                data, gt_col=None, key_added="bare", seed_obsm_key="bare_seed"
            )
