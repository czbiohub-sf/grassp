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
