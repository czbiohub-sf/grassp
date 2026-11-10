import warnings

import anndata
import numpy as np
import pandas as pd
import pytest

from anndata import AnnData

from grassp.preprocessing import enrichment


def make_basic_anndata():
    X = np.array(
        [
            [1, 2, 5, 4, 3],
            [0, 1, 2, 3, 6],
            [1, 0, 1, 0, 2],
        ],
        dtype=float,
    )
    obs = pd.DataFrame(index=[f"prot{i}" for i in range(3)])
    var = pd.DataFrame(
        {
            "subcellular_enrichment": ["A", "A", "UNTAGGED", "B", "UNTAGGED"],
            "covariate_1": ["x", "x", "x", "y", "y"],
            "covariate_2": ["c1", "c1", "c1", "c2", "c2"],
            "biological_replicate": ["1", "2", "1", "2", "1"],
        },
    )
    var.index = (
        var["subcellular_enrichment"]
        + "_"
        + var["covariate_1"]
        + "_"
        + var["covariate_2"]
        + "_"
        + var["biological_replicate"]
    )
    ad1 = AnnData(X=X, obs=obs, var=var)
    ad2 = AnnData(X=X, obs=obs, var=var)
    ad1.var["biological_replicate"] = "1"
    ad2.var["biological_replicate"] = "2"
    ad2.X = ad2.X + 2
    ad2.var_names = ad2.var_names + "_2"
    adata = anndata.concat([ad1, ad2], axis=1)
    adata.var_names_make_unique()
    return adata


def test_check_covariates_basic():
    adata = make_basic_anndata()
    covs = enrichment._check_covariates(adata, None)
    assert set(covs) == {"covariate_1", "covariate_2"}
    covs2 = enrichment._check_covariates(adata, ["covariate_1"])
    assert covs2 == ["covariate_1"]
    covs3 = enrichment._check_covariates(adata, "covariate_2")
    assert covs3 == ["covariate_2"]
    with pytest.raises(ValueError):
        enrichment._check_covariates(adata, ["not_a_cov"])


def test_calculate_enrichment_vs_untagged_basic():
    adata = make_basic_anndata()
    # Should not raise
    result = enrichment.calculate_enrichment_vs_untagged(
        adata,
        covariates=["covariate_1"],
        subcellular_enrichment_column="subcellular_enrichment",
        untagged_name="UNTAGGED",
        drop_untagged=True,
        keep_raw=True,
    )
    assert isinstance(result, AnnData)
    assert "pvals" in result.layers
    assert result.X.shape[1] == 2  # Only 'A' and 'B' remain
    # Test with no untagged present
    adata2 = make_basic_anndata()
    adata2.var["subcellular_enrichment"] = ["A", "A", "A", "B", "B"] * 2
    with pytest.raises(ValueError):
        enrichment.calculate_enrichment_vs_untagged(
            adata2,
            covariates=["covariate_1"],
            subcellular_enrichment_column="subcellular_enrichment",
            untagged_name="UNTAGGED",
            drop_untagged=True,
        )


def test_vs_untagged_leaves_no_helper_column():
    """_experimental_condition used to stay on the caller's var (unless it was a view) and in
    .raw."""
    adata = make_basic_anndata()
    var_before = adata.var.copy()
    result = enrichment.calculate_enrichment_vs_untagged(
        adata, covariates=["covariate_1"], keep_raw=True
    )
    pd.testing.assert_frame_equal(adata.var, var_before)
    assert "_experimental_condition" not in result.var.columns
    assert "_experimental_condition" not in result.raw.var.columns


def test_vs_untagged_matches_the_untagged_label_exactly():
    """untagged_name was a regex prefix, so "UNTAGGED_2" samples joined the control."""
    adata = make_basic_anndata()
    labels = adata.var["subcellular_enrichment"].to_numpy().copy()
    labels[1] = "UNTAGGED_2"  # one condition-x "A" sample, relabelled
    adata.var["subcellular_enrichment"] = labels
    result = enrichment.calculate_enrichment_vs_untagged(
        adata, covariates=["covariate_1"], drop_untagged=False
    )

    x = adata.var["covariate_1"] == "x"
    ip = adata.X[:, (x & (adata.var["subcellular_enrichment"] == "UNTAGGED_2")).to_numpy()]
    ctrl = adata.X[:, (x & (adata.var["subcellular_enrichment"] == "UNTAGGED")).to_numpy()]
    expected = np.median(ip, axis=1) - np.median(ctrl, axis=1)
    col = (
        (result.var["subcellular_enrichment"] == "UNTAGGED_2")
        & (result.var["covariate_1"] == "x")
    ).to_numpy()
    np.testing.assert_allclose(result.X[:, col].ravel(), expected)


def test_original_intensities_layer_is_not_overwritten():
    """The layer used to alias X, so writing the fold changes into X overwrote it."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="invalid value encountered in divide", category=RuntimeWarning
        )
        untagged = enrichment.calculate_enrichment_vs_untagged(
            make_basic_anndata(),
            covariates=["covariate_1"],
            subcellular_enrichment_column="subcellular_enrichment",
            untagged_name="UNTAGGED",
            drop_untagged=False,
            original_intensities_key="original_intensities",
        )
        vs_all = enrichment.calculate_enrichment_vs_all(
            make_basic_anndata(),
            covariates=["covariate_1"],
            subcellular_enrichment_column="subcellular_enrichment",
            enrichment_method="lfc",
            correlation_threshold=1.0,
        )
    for result in (untagged, vs_all):
        layer = result.layers["original_intensities"]
        assert not np.shares_memory(layer, result.X)
        assert not np.allclose(layer, result.X)
        # aggregate_samples averages replicates, so every raw value is non-negative --
        # unlike the log fold changes now in X.
        assert (layer >= 0).all()


def test_calculate_enrichment_vs_all_basic():
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="invalid value encountered in divide", category=RuntimeWarning
        )
        adata = make_basic_anndata()
        # Should not raise
        result = enrichment.calculate_enrichment_vs_all(
            adata,
            covariates=["covariate_1"],
            subcellular_enrichment_column="subcellular_enrichment",
            enrichment_method="lfc",
            correlation_threshold=1.0,
            keep_raw=True,
        )
        assert isinstance(result, AnnData)
        assert "pvals" in result.layers
        assert "enriched_vs" in result.var.columns
        # Test with missing covariate column
        adata2 = make_basic_anndata()
        adata2.var = adata2.var.drop(columns=["covariate_1"])
        with pytest.raises(ValueError):
            enrichment.calculate_enrichment_vs_all(
                adata2,
                covariates=["covariate_1"],
                subcellular_enrichment_column="subcellular_enrichment",
            )
