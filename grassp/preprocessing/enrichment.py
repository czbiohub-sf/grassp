from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Optional

from anndata import AnnData
from .simple import aggregate_samples
import scipy.stats as stats
import numpy as np


def calculate_enrichment_vs_untagged(
    data: AnnData,
    covariates: Optional[list[str]] = None,
    subcellular_enrichment_column: str = "subcellular_enrichment",
    untagged_name: str = "UNTAGGED",
    original_intensities_key: Optional[str] = None,
) -> AnnData:
    """Calculate enrichment scores and p-values with a t-test comparing tagged vs untagged samples.

    Parameters
    ----------
    data
        Annotated data matrix with proteins as observations (rows)
    covariates
        List of column names in data.var to use as covariates for grouping samples.
        If None, uses columns starting with "covariate_"
    subcellular_enrichment_column
        Column in .var containing subcellular enrichment labels
    untagged_name
        Label in subcellular_enrichment_column identifying untagged control samples

    Returns
    -------
    Aggregated AnnData
        Annotated data matrix with enrichment scores and p-values.
        Enrichment scores are stored in .X as log2 fold changes vs untagged.
        P-values from t-tests are stored in .layers["pvals"].
        Raw values are stored in .layers["raw"].
    """

    if covariates is None:
        covariates = data.var.columns[data.var.columns.str.startswith("covariate_")]
    else:
        # Check that all covariates are in the data
        for c in covariates:
            if c not in data.var.columns:
                raise ValueError(f"Covariate {c} not found in data.var.columns")

    # Create a temporary column that contains the experimental conditions
    data.var["_experimental_condition"] = data.var[covariates].apply(
        lambda x: "_".join(x.dropna().astype(str)),
        axis=1,
    )

    # Create aggregated data with the desired output shape
    grouping_columns = [subcellular_enrichment_column] + covariates.tolist()
    data_aggr = aggregate_samples(data, grouping_columns=grouping_columns)
    data_aggr.var_names = data_aggr.var_names.str.replace(r"_\d+", "", regex=True)

    if original_intensities_key is not None:
        data_aggr.layers[original_intensities_key] = data_aggr.X
    data_aggr.layers["pvals"] = np.zeros_like(data_aggr.X)

    for experimental_condition in data_aggr.var["_experimental_condition"].unique():
        data_sub = data[
            :, data.var["_experimental_condition"] == experimental_condition
        ]
        intensities_control = data_sub[
            :, data_sub.var[subcellular_enrichment_column] == untagged_name
        ].X
        if intensities_control.shape[1] == 0:
            raise ValueError(
                f"No {untagged_name} samples found for condition: "
                + experimental_condition
            )
        for subcellular_enrichment in data_sub.var[
            subcellular_enrichment_column
        ].unique():
            intensities_ip = data_sub[
                :, data_sub.var[subcellular_enrichment_column] == subcellular_enrichment
            ].X
            scores, pv = stats.ttest_ind(intensities_ip.T, intensities_control.T)
            lfc = np.median(intensities_ip, axis=1) - np.median(
                intensities_control, axis=1
            )
            aggr_mask = (
                data_aggr.var["_experimental_condition"] == experimental_condition
            ) & (data_aggr.var[subcellular_enrichment_column] == subcellular_enrichment)
            if aggr_mask.sum() > 1:
                raise Warning(
                    "Multiple samples found for condition: " + experimental_condition
                )
            data_aggr.layers["pvals"][:, aggr_mask] = pv[:, None]
            data_aggr[:, aggr_mask].X = lfc[:, None]

    # Now remove the untagged samples
    data_aggr = data_aggr[
        :, data_aggr.var[subcellular_enrichment_column] != untagged_name
    ]
    data_aggr.var.drop(columns=["_experimental_condition"], inplace=True)
    return data_aggr


def calculate_enrichment_vs_all(
    adata: AnnData,
    covariates: Optional[list[str]] = None,
    subcellular_enrichment_column: str = "subcellular_enrichment",
    original_intensities_key: str | None = "original_intensities",
) -> AnnData:
    """Calculate enrichment of each subcellular enrichment vs all other samples as the background.

    Parameters
    ----------
    adata
        AnnData object containing protein intensities
    covariates
        List of column names in adata.var to use as covariates for grouping samples.
        If None, uses columns starting with "covariate_"
    subcellular_enrichment_column
        Column in adata.var containing subcellular enrichment labels
    original_intensities_key
        If provided, store the original intensities in this layer

    Returns
    -------
    AnnData object with enrichment scores and p-values stored in .X as log2 fold changes
    vs all other conditions. P-values from t-tests are stored in .layers["pvals"].
    Raw values are stored in .layers[original_intensities_key] if provided.
    """

    data = adata.copy()

    if covariates is None:
        covariates = data.var.columns[data.var.columns.str.startswith("covariate_")]
    else:
        # Check that all covariates are in the data
        for c in covariates:
            if c not in data.var.columns:
                raise ValueError(f"Covariate {c} not found in data.var.columns")

    # Create aggregated data with the desired output shape
    grouping_columns = [subcellular_enrichment_column] + covariates.tolist()
    # Create a temporary column that contains the experimental conditions
    data.var["_experimental_condition"] = data.var[grouping_columns].apply(
        lambda x: "_".join(x.dropna().astype(str)),
        axis=1,
    )

    data_aggr = aggregate_samples(data, grouping_columns=grouping_columns)
    data_aggr.var_names = data_aggr.var_names.str.replace(r"_\d+", "", regex=True)

    if original_intensities_key is not None:
        data_aggr.layers[original_intensities_key] = data_aggr.X
    data_aggr.layers["pvals"] = np.zeros_like(data_aggr.X)

    for experimental_condition in data_aggr.var["_experimental_condition"].unique():
        intensities_control = data[
            :, data.var["_experimental_condition"] != experimental_condition
        ].X
        intensities_ip = data[
            :, data.var["_experimental_condition"] == experimental_condition
        ].X
        scores, pv = stats.ttest_ind(intensities_ip.T, intensities_control.T)
        lfc = np.median(intensities_ip, axis=1) - np.median(intensities_control, axis=1)
        aggr_mask = data_aggr.var["_experimental_condition"] == experimental_condition
        data_aggr.layers["pvals"][:, aggr_mask] = pv[:, None]
        data_aggr[:, aggr_mask].X = lfc[:, None]

    data_aggr.var.drop(columns=["_experimental_condition"], inplace=True)
    return data_aggr
