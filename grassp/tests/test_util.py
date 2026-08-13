"""Tests for :mod:`grassp.util`.

Only the structural diff lives here for now; the labelled-matrix helpers are covered by
``test_labelled_matrices.py``, next to the writers whose contract they define.
"""

import anndata
import numpy as np
import pandas as pd
import pytest

from grassp.util import diff_anndata


def make_pair(n_obs=6, n_vars=4):
    """Two structurally identical objects, ready to be pulled apart by a test."""
    obs = pd.DataFrame(
        {"markers": pd.Categorical(["ER", "Golgi"] * (n_obs // 2))},
        index=[f"P{i}" for i in range(n_obs)],
    )
    var = pd.DataFrame({"fraction": [f"F{i}" for i in range(n_vars)]}, index=range(n_vars))
    var.index = [f"F{i}" for i in range(n_vars)]
    data = anndata.AnnData(np.zeros((n_obs, n_vars)), obs=obs, var=var)
    return data, data.copy()


class TestNoDifference:
    def test_identical_objects_diff_to_nothing(self):
        before, after = make_pair()
        assert diff_anndata(before, after).empty

    def test_an_object_against_itself(self):
        before, _ = make_pair()
        assert diff_anndata(before, before).empty

    def test_the_columns_are_stable_even_when_empty(self):
        before, after = make_pair()
        assert list(diff_anndata(before, after).columns) == ["change", "slot", "key", "detail"]


class TestKeys:
    @pytest.mark.parametrize(
        ("slot", "key", "value"),
        [
            ("obsm", "X_umap", np.zeros((6, 2))),
            ("varm", "PCs", np.zeros((4, 2))),
            ("layers", "log", np.ones((6, 4))),
            ("obsp", "connectivities", np.zeros((6, 6))),
            ("varp", "corr", np.zeros((4, 4))),
        ],
    )
    def test_added_and_removed_are_symmetric(self, slot, key, value):
        before, after = make_pair()
        getattr(after, slot)[key] = value

        added = diff_anndata(before, after)
        assert added.loc[0, ["change", "slot", "key"]].tolist() == ["added", slot, key]
        # the same pair the other way round is the same difference, reported as a removal
        removed = diff_anndata(after, before)
        assert removed.loc[0, ["change", "slot", "key"]].tolist() == ["removed", slot, key]

    def test_obs_and_var_columns(self):
        before, after = make_pair()
        after.obs["svm"] = pd.Categorical(["ER"] * after.n_obs)
        del after.var["fraction"]

        report = diff_anndata(before, after)
        assert ("added", "obs", "svm") in list(
            report[["change", "slot", "key"]].itertuples(index=False, name=None)
        )
        assert ("removed", "var", "fraction") in list(
            report[["change", "slot", "key"]].itertuples(index=False, name=None)
        )

    def test_a_flat_uns_key(self):
        before, after = make_pair()
        after.uns["title"] = "AC16 cardiomyocytes"
        assert diff_anndata(before, after).loc[0, ["slot", "key"]].tolist() == ["uns", "title"]

    def test_nested_uns_keys_are_addressed_by_path(self):
        before, after = make_pair()
        after.uns["neighbors"] = {"params": {"n_neighbors": 15}}
        assert diff_anndata(before, after).loc[0, "key"] == "neighbors.params.n_neighbors"

    def test_an_empty_uns_mapping_is_a_leaf_of_its_own(self):
        """Recursing into it would find nothing, so the key itself would go unreported."""
        before, after = make_pair()
        after.uns["empty"] = {}
        assert diff_anndata(before, after).loc[0, "key"] == "empty"

    def test_the_main_matrix_is_not_reported_as_a_layer(self):
        """anndata >= 0.13 backs .X with layers[None], which is not a layer anyone added."""
        before, after = make_pair()
        assert diff_anndata(before, after).empty
        assert "None" not in set(diff_anndata(before, after)["key"])


class TestChanges:
    def test_a_dtype_change_in_obs(self):
        before, after = make_pair()
        after.obs["markers"] = after.obs["markers"].astype(object)

        row = diff_anndata(before, after).iloc[0]
        assert row["change"] == "changed"
        assert row["slot"] == "obs"
        assert row["key"] == "markers"
        assert row["detail"] == "category -> object"

    def test_an_obsm_array_that_became_a_dataframe(self):
        """What a round trip through pRoloc does: the matrix comes back naming its own columns."""
        before, after = make_pair()
        before.obsm["scores"] = np.zeros((before.n_obs, 2))
        after.obsm["scores"] = pd.DataFrame(
            np.zeros((after.n_obs, 2)), index=after.obs_names, columns=["ER", "Golgi"]
        )

        row = diff_anndata(before, after).iloc[0]
        assert (row["change"], row["slot"], row["key"]) == ("changed", "obsm", "scores")
        assert row["detail"] == "ndarray -> DataFrame"

    @pytest.mark.parametrize(
        ("before_value", "after_value", "detail"),
        [
            # the classic round-trip drift: a scalar comes back as a one-element array
            (15, np.array([15]), "int -> ndarray[int64][1]"),
            (["a", "b"], np.array(["a", "b"]), "list[2] -> ndarray[<U1][2]"),
            (np.zeros(3), np.zeros(4), "ndarray[float64][3] -> ndarray[float64][4]"),
            (np.zeros(3, dtype=int), np.zeros(3), "ndarray[int64][3] -> ndarray[float64][3]"),
            (1.0, "1.0", "float -> str"),
        ],
    )
    def test_uns_leaves_report_type_dtype_shape_and_length(
        self, before_value, after_value, detail
    ):
        before, after = make_pair()
        before.uns["nested"] = {"value": before_value}
        after.uns["nested"] = {"value": after_value}

        row = diff_anndata(before, after).iloc[0]
        assert (row["change"], row["slot"], row["key"]) == ("changed", "uns", "nested.value")
        assert row["detail"] == detail

    def test_an_unchanged_nested_uns_is_quiet(self):
        before, after = make_pair()
        for data in (before, after):
            data.uns["neighbors"] = {"params": {"n_neighbors": 15, "method": "umap"}}
        assert diff_anndata(before, after).empty

    def test_a_shape_change(self):
        before, after = make_pair()
        subset = after[:3].copy()

        row = diff_anndata(before, subset).iloc[0]
        assert (row["change"], row["slot"]) == ("changed", "shape")
        assert row["detail"] == "(6, 4) -> (3, 4)"


class TestCheckDtypes:
    """``check_dtypes=False`` is the "just tell me what appeared and disappeared" mode."""

    @pytest.fixture
    def moved(self):
        before, after = make_pair()
        after.obs["markers"] = after.obs["markers"].astype(object)  # a dtype change
        before.obsm["scores"] = np.zeros((before.n_obs, 2))  # a container change
        after.obsm["scores"] = pd.DataFrame(
            np.zeros((after.n_obs, 2)), index=after.obs_names, columns=["ER", "Golgi"]
        )
        before.uns["neighbors"] = {"params": {"n_neighbors": 15}}  # an uns leaf change
        after.uns["neighbors"] = {"params": {"n_neighbors": np.array([15])}}
        after.obs["svm"] = pd.Categorical(["ER"] * after.n_obs)  # a real addition
        return before, after

    def test_on_by_default(self, moved):
        report = diff_anndata(*moved)
        assert set(report["change"]) == {"added", "changed"}
        assert sorted(report.loc[report["change"] == "changed", "slot"]) == [
            "obs",
            "obsm",
            "uns",
        ]

    def test_off_leaves_only_additions_and_removals(self, moved):
        report = diff_anndata(*moved, check_dtypes=False)
        assert set(report["change"]) == {"added"}
        assert report["key"].tolist() == ["svm"]

    def test_off_still_reports_a_shape_change(self, moved):
        """A different number of proteins is not a question of how anything is stored."""
        before, after = moved
        report = diff_anndata(before, after[:3].copy(), check_dtypes=False)
        assert report.loc[0, ["change", "slot"]].tolist() == ["changed", "shape"]
        assert "obs" not in set(report.loc[report["change"] == "changed", "slot"])

    def test_off_is_a_subset_of_on(self, moved):
        strict, loose = diff_anndata(*moved), diff_anndata(*moved, check_dtypes=False)
        assert len(loose) < len(strict)
        assert set(map(tuple, loose.to_numpy())) <= set(map(tuple, strict.to_numpy()))


class TestValues:
    def test_values_are_not_compared(self):
        """Deliberate: uns is arbitrarily nested and matrices can be sparse or dask."""
        before, after = make_pair()
        after.X = np.ones_like(after.X)
        after.uns["note"] = "same key, different value"
        del after.uns["note"]
        after.obs["markers"] = pd.Categorical(
            ["Golgi", "ER"] * (after.n_obs // 2), categories=["ER", "Golgi"]
        )

        assert diff_anndata(before, after).empty
