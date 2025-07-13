from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Optional, Literal

import warnings

import numpy as np
import scipy.stats as stats

from anndata import AnnData

from .simple import aggregate_samples


def _check_covariates(data: AnnData, covariates: Optional[list[str]] = None) -> list[str]:
    """
    Checks for covariates in ``data.var`` and returns a list of validated covariate names.

    If `covariates` is None, this function identifies columns in ``data.var``
    that start with "covariate_" and uses them as the covariates. It also
    ensures that all specified covariates exist in ``data.var``.

    Parameters
    ----------
    data
        An AnnData object with covariate information in ``.var``.
    covariates
        A list of covariate names to check. If None, covariates are
        inferred from columns starting with "covariate_".

    Returns
    -------
    A list of validated covariate names.

    Raises
    ------
    ValueError
        If a specified covariate is not found in ``data.var.columns``.
    """
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
    """
    Calculates enrichment scores and p-values by comparing tagged samples against untagged controls.

    This function performs a t-test to determine the significance of protein
    enrichment in tagged samples relative to untagged controls. The enrichment
    is calculated as the log2 fold change of median intensities.

    Parameters
    ----------
    data
        An AnnData object with protein intensities in ``.X``.
    covariates
        A list of column names in ``data.var`` to group samples. If None,
        columns starting with ``covariate_`` are used.
    subcellular_enrichment_column
        The column in ``.var`` that contains subcellular enrichment labels.
    untagged_name
        The label in `subcellular_enrichment_column` that identifies untagged
        control samples.
    original_intensities_key
        If specified, the original intensity values are stored in
        ``data.layers[original_intensities_key]``.
    drop_untagged
        If True, untagged samples are removed from the returned AnnData object.
    keep_raw
        If True, the original unaggregated data is stored in ``.raw``.

    Returns
    -------
    AnnData
        Aggregated AnnData object with enrichment scores and p-values, with:

        * ``.X``: log2 fold changes relative to untagged controls.
        * ``.layers["pvals"]``: p-values from the t-tests.
        * ``.layers[original_intensities_key]``: raw intensity values if `original_intensities_key` is set.
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


def calculate_enrichment_vs_all(
    adata: AnnData,
    covariates: Optional[list[str]] = None,
    subcellular_enrichment_column: str = "subcellular_enrichment",
    enrichment_method: Literal["lfc", "proportion"] = "lfc",
    correlation_threshold: float = 1.0,
    original_intensities_key: str | None = "original_intensities",
    keep_raw: bool = True,
    min_comparison_warning: int | None = None,
) -> AnnData:
    """
    Calculates enrichment of each sample against all other samples as the background.

    This function determines enrichment by comparing each sample's protein
    intensities to a background composed of all other samples that are not
    highly correlated with it.

    Parameters
    ----------
    adata
        An AnnData object with protein intensities in ``.X``.
    covariates
        A list of column names in ``adata.var`` for grouping. If None,
        columns starting with ``covariate_`` are used.
    subcellular_enrichment_column
        The column in ``.var`` with subcellular enrichment labels.
    enrichment_method
        The method for calculating enrichment. Either "lfc" (log-fold change)
        or "proportion" (proportion of total intensity).
    correlation_threshold
        The correlation value above which samples are excluded from the background
        to prevent comparing a sample against itself or highly similar ones.
    original_intensities_key
        If provided, the original intensities are stored in this layer.
    keep_raw
        If True, the original unaggregated data is stored in ``.raw``.
    min_comparison_warning
        If the number of control samples for a given comparison is below this
        threshold, a warning is issued.

    Returns
    -------
    AnnData
        An AnnData object with enrichment scores and p-values.

        * ``.X`` contains enrichment scores (log2 fold changes or proportions).
        * ``.layers["pvals"]`` stores p-values from the t-tests.
        * ``.var["enriched_vs"]`` lists the conditions used as the background.
    """

    if enrichment_method not in ["lfc", "proportion"]:
        raise ValueError("enrichment_method must be either 'lfc' or 'proportion'")

    data = adata.copy()

    if covariates is None:
        covariates = data.var.columns[data.var.columns.str.startswith("covariate_")].tolist()
    # Check that all covariates are in the data
    for c in covariates:
        if c not in data.var.columns:
            raise ValueError(f"Covariate {c} not found in data.var.columns")

    grouping_columns = [subcellular_enrichment_column] + covariates

    data.var["_experimental_condition"] = data.var[grouping_columns].apply(
        lambda x: "_".join(x.dropna().astype(str)),
        axis=1,
    )
    data.var["_covariates"] = data.var[covariates].apply(
        lambda x: "_".join(x.dropna().astype(str)), axis=1
    )

    data_aggr = aggregate_samples(data, grouping_columns=grouping_columns, keep_raw=False)
    data_aggr.var_names = data_aggr.var_names.str.replace(r"_\d+", "", regex=True)
    data_aggr = aggregate_samples(data, grouping_columns=grouping_columns, keep_raw=False)

    if original_intensities_key is not None:
        data_aggr.layers[original_intensities_key] = data_aggr.X
    data_aggr.layers["pvals"] = np.zeros_like(data_aggr.X)
    data_aggr.var["enriched_vs"] = ""

    intensities = data_aggr.X.copy()
    corr_matrix = np.corrcoef(intensities.T)

    for experimental_condition in data_aggr.var["_experimental_condition"].unique():
        mask = data_aggr.var["_experimental_condition"] == experimental_condition

        intensities_ip = intensities[:, mask]
        covariate = data.var.loc[
            data.var._experimental_condition == experimental_condition, "_covariates"
        ].values[0]
        covariate_mask = data_aggr.var["_covariates"] == covariate
        control_mask = ~mask & covariate_mask
        corr_mat_sub = corr_matrix[mask, control_mask].mean(axis=0)
        control_mask = control_mask & (corr_mat_sub < correlation_threshold)
        intensities_control = intensities[:, control_mask]
        if min_comparison_warning is not None:
            if control_mask.sum() < min_comparison_warning:
                warnings.warn(
                    f"Less than {min_comparison_warning} ({control_mask.sum()}) control samples found for condition: {experimental_condition}"
                )  # Check for statistical power (if fewer than 10 samples selected )

        scores, pv = stats.ttest_ind(intensities_ip.T, intensities_control.T)

        if enrichment_method == "lfc":
            enrichment_values = np.median(intensities_ip, axis=1) - np.median(
                intensities_control, axis=1
            )
        else:
            enrichment_values = np.nansum(intensities_ip, axis=1) / (
                np.nansum(intensities_ip, axis=1) + np.nansum(intensities_control, axis=1)
            )

        aggr_mask = data_aggr.var["_experimental_condition"] == experimental_condition
        data_aggr.layers["pvals"][:, aggr_mask] = pv[:, None]
        data_aggr.X[:, aggr_mask] = enrichment_values[:, None]
        data_aggr.var.loc[aggr_mask, "enriched_vs"] = ",".join(
            data_aggr.var_names[control_mask]
        )

    data_aggr.var.drop(columns=["_experimental_condition"], inplace=True)
    if keep_raw:
        data_aggr.raw = data.copy()
    return data_aggr
