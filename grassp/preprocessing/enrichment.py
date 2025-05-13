from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Optional

import warnings

import numpy as np
import scipy.stats as stats

from anndata import AnnData

from .simple import aggregate_samples


def _check_covariates(data: AnnData, covariates: Optional[list[str]] = None):
    if covariates is None:
        covariates = data.var.columns[data.var.columns.str.startswith("covariate_")]
    # Check that all covariates are in the data
    for c in covariates:
        if c not in data.var.columns:
            raise ValueError(f"Covariate {c} not found in data.var.columns")

    if not isinstance(covariates, list):
        covariates = [covariates]
    return covariates


def calculate_enrichment_vs_untagged(
    data: AnnData,
    covariates: Optional[list[str]] = [],
    subcellular_enrichment_column: str = "subcellular_enrichment",
    untagged_name: str = "UNTAGGED",
    original_intensities_key: Optional[str] = None,
    drop_untagged: bool = True,
    keep_raw: bool = True,
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
    original_intensities_key
        Key in data.layers to store the original intensities
    drop_untagged
        Whether to drop the untagged samples from the returned AnnData object
    keep_raw
        Whether to keep the unaggregated data in the .raw attribute of the returned AnnData object

    Returns
    -------
    Aggregated AnnData
        Annotated data matrix with enrichment scores and p-values.
        Enrichment scores are stored in .X as log2 fold changes vs untagged.
        P-values from t-tests are stored in .layers["pvals"].
        Raw values are stored in .layers["raw"].
    """

    # if covariates is None:
    #     covariates = data.var.columns[data.var.columns.str.startswith("covariate_")]
    # else:
    # Check that all covariates are in the data
    if data.is_view:
        data = data.copy()
    for c in covariates:
        if c not in data.var.columns:
            raise ValueError(f"Covariate {c} not found in data.var.columns")

    if not isinstance(covariates, list):
        covariates = [covariates]
    # Create a temporary column that contains the experimental conditions
    data.var["_experimental_condition"] = data.var[covariates].apply(
        lambda x: "_".join(x.dropna().astype(str)),
        axis=1,
    )

    # Create aggregated data with the desired output shape
    grouping_columns = [subcellular_enrichment_column] + covariates
    data_aggr = aggregate_samples(data, grouping_columns=grouping_columns)
    data_aggr.var_names = data_aggr.var_names.str.replace(r"_\d+", "", regex=True)

    if original_intensities_key is not None:
        data_aggr.layers[original_intensities_key] = data_aggr.X
    data_aggr.layers["pvals"] = np.zeros_like(data_aggr.X)

    for experimental_condition in data_aggr.var["_experimental_condition"].unique():
        data_sub = data[:, data.var["_experimental_condition"] == experimental_condition]
        intensities_control = data_sub[
            :,
            data_sub.var[subcellular_enrichment_column].str.match(untagged_name),
        ].X
        if intensities_control.shape[1] == 0:
            raise ValueError(
                f"No {untagged_name} samples found for condition: " + experimental_condition
            )
        for subcellular_enrichment in data_sub.var[subcellular_enrichment_column].unique():
            intensities_ip = data_sub[
                :, data_sub.var[subcellular_enrichment_column] == subcellular_enrichment
            ].X
            scores, pv = stats.ttest_ind(intensities_ip.T, intensities_control.T)
            lfc = np.median(intensities_ip, axis=1) - np.median(intensities_control, axis=1)
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
    if drop_untagged:
        data_aggr = data_aggr[
            :, ~data_aggr.var[subcellular_enrichment_column].str.match(untagged_name)
        ]
    data_aggr.var.drop(columns=["_experimental_condition"], inplace=True)
    if keep_raw:
        data_aggr.raw = data.copy()
    return data_aggr


def calculate_noc_proportions(
    adata: AnnData,
    covariates: Optional[list[str]] = None,
    subcellular_enrichment_column: str = "subcellular_enrichment",
    use_layer: Optional[str] = None,
    original_intensities_key: str | None = None,
    keep_raw: bool = True,
) -> AnnData:
    pass


def calculate_enrichment_vs_all(
    adata: AnnData,
    covariates: Optional[list[str]] = None,
    subcellular_enrichment_column: str = "subcellular_enrichment",
    correlation_threshold: float = 1.0,
    original_intensities_key: str | None = "original_intensities",
    keep_raw: bool = True,
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
    keep_raw
        Whether to keep the unaggregated data in the .raw attribute of the returned AnnData object

    Returns
    -------
    AnnData object with enrichment scores and p-values stored in .X as log2 fold changes
    vs all other conditions. P-values from t-tests are stored in .layers["pvals"].
    Raw values are stored in .layers[original_intensities_key] if provided.
    """

    data = adata.copy()

    if covariates is None:
        covariates = data.var.columns[data.var.columns.str.startswith("covariate_")].tolist()
    # Check that all covariates are in the data
    for c in covariates:
        if c not in data.var.columns:
            raise ValueError(f"Covariate {c} not found in data.var.columns")

    if not isinstance(covariates, list):
        covariates = [covariates]
    # Create aggregated data with the desired output shape
    grouping_columns = [subcellular_enrichment_column] + covariates
    # Create a temporary column that contains the experimental conditions
    data.var["_experimental_condition"] = data.var[grouping_columns].apply(
        lambda x: "_".join(x.dropna().astype(str)),
        axis=1,
    )

    data_aggr = aggregate_samples(data, grouping_columns=grouping_columns, keep_raw=False)
    data_aggr.var_names = data_aggr.var_names.str.replace(r"_\d+", "", regex=True)

    if original_intensities_key is not None:
        data_aggr.layers[original_intensities_key] = data_aggr.X
    data_aggr.layers["pvals"] = np.zeros_like(data_aggr.X)
    data_aggr.var["enriched_vs"] = ""

    intensities = data_aggr.X.copy()
    corr_matrix = np.corrcoef(intensities.T)

    for experimental_condition in data_aggr.var["_experimental_condition"].unique():
        mask = data_aggr.var["_experimental_condition"] == experimental_condition
        corr_mat_sub = corr_matrix[mask, :].mean(axis=0)
        control_mask = ~mask & (corr_mat_sub < correlation_threshold)
        if control_mask.sum() < 10:
            warnings.warn(
                f"Less than 10 ({control_mask.sum()}) control samples found for condition: {experimental_condition}"
            )
        intensities_control = intensities[:, control_mask]
        intensities_ip = intensities[:, mask]
        scores, pv = stats.ttest_ind(intensities_ip.T, intensities_control.T)
        lfc = np.median(intensities_ip, axis=1) - np.median(intensities_control, axis=1)
        aggr_mask = data_aggr.var["_experimental_condition"] == experimental_condition
        data_aggr.layers["pvals"][:, aggr_mask] = pv[:, None]
        data_aggr[:, aggr_mask].X = lfc[:, None]
        data_aggr.var.loc[aggr_mask, "enriched_vs"] = ",".join(
            data_aggr.var_names[control_mask]
        )
    data_aggr.var.drop(columns=["_experimental_condition"], inplace=True)
    if keep_raw:
        data_aggr.raw = data.copy()
    return data_aggr
