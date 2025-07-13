from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import scanpy as sc

if TYPE_CHECKING:
    from typing import Callable, Collection, List, Literal, Optional

    from scanpy._compat import DaskArray
    from scipy.sparse import spmatrix

    # Define a type hint for functions that take an ndarray and an optional axis argument
    NDArrayAxisFunction = Callable[[np.ndarray, Optional[int]], np.ndarray]


from anndata import AnnData

from ..util import confirm_proteins_as_obs


def filter_samples(
    data: AnnData | spmatrix | np.ndarray | DaskArray,
    *,
    min_counts: int | None = None,
    min_proteins: int | None = None,
    max_counts: int | None = None,
    max_proteins: int | None = None,
    inplace: bool = True,
    copy: bool = False,
) -> AnnData | tuple[np.ndarray, np.ndarray] | None:
    """Filter samples based on number of counts or proteins.

    Parameters
    ----------
    data
        The annotated data matrix of shape `n_obs` x `n_vars`. Rows correspond to proteins
        and columns to samples.
    min_counts
        Minimum number of counts required for a sample to pass filtering.
    min_proteins
        Minimum number of proteins expressed required for a sample to pass filtering.
    max_counts
        Maximum number of counts required for a sample to pass filtering.
    max_proteins
        Maximum number of proteins expressed required for a sample to pass filtering.
    inplace
        Perform computation inplace or return result.
    copy
        If an AnnData is passed, determines whether a copy is returned.

    Returns
    -------
    AnnData | tuple[np.ndarray, np.ndarray] | None
        * If `inplace` is **True**, the function updates `data` in place and
          returns **None**.
        * If `inplace` is **False** and *data* is an :class:`~anndata.AnnData`
          object, the filtered AnnData is returned.
        * If *data* is an array-like matrix (``np.ndarray``, ``scipy.sparse`` or
          ``dask`` array) the function returns a tuple *(mask_obs,
          mask_vars)* indicating which observations and variables were kept.
    """
    if isinstance(data, AnnData):
        confirm_proteins_as_obs(data)

    return sc.pp.filter_genes(
        data,
        min_counts=min_counts,
        min_cells=min_proteins,
        max_counts=max_counts,
        max_cells=max_proteins,
        inplace=inplace,
        copy=copy,
    )


def filter_proteins(
    data: AnnData | spmatrix | np.ndarray | DaskArray,
    *,
    min_counts: int | None = None,
    min_samples: int | None = None,
    max_counts: int | None = None,
    max_samples: int | None = None,
    inplace: bool = True,
    min_consecutive: int | None = None,
) -> AnnData | tuple[np.ndarray, np.ndarray] | None:
    """Filter proteins based on number of counts or samples.

    Parameters
    ----------
    data
        The annotated data matrix of shape `n_obs` x `n_vars`. Rows correspond to proteins
        and columns to samples.
    min_counts
        Minimum number of counts required for a protein to pass filtering.
    min_samples
        Minimum number of samples expressed required for a protein to pass filtering.
    max_counts
        Maximum number of counts required for a protein to pass filtering.
    max_samples
        Maximum number of samples expressed required for a protein to pass filtering.
    inplace
        Perform computation inplace or return result.
    min_consecutive
        (If using fractionation instead of organellar IP) Minimum number of consecutive fractions that a protein is present in.

    Returns
    -------
    AnnData | tuple[np.ndarray, np.ndarray] | None
        * If `inplace` is **True**, `data` is modified in place and **None** is returned.
        * If `inplace` is **False** and *data* is an :class:`~anndata.AnnData`
          object, the filtered AnnData is returned.
        * If *data* is an array-like matrix, the function returns a tuple
          *(mask_obs, mask_vars)* with the retained observations and variables.
    """
    # if isinstance(data, AnnData):
    #     confirm_proteins_as_obs(data)

    return sc.pp.filter_cells(
        data,
        min_counts=min_counts,
        min_genes=min_samples,
        max_counts=max_counts,
        max_genes=max_samples,
        inplace=inplace,
    )


def filter_min_consecutive_fractions(
    data: AnnData,
    min_consecutive: int = 2,
    replicate_column: str | None = None,
    min_replicates: int | None = None,
    inplace: bool = True,
) -> np.ndarray | None:
    """
    Filters for proteins present in at least `min_consecutive` of specified consecutive fractions.

    Parameters
    ----------
    data
        The annotated data matrix of shape `n_obs` x `n_vars`. Rows correspond to proteins
        and columns to samples.
    min_consecutive : int
        Minimum number of consecutive fractions in which a protein must be detected to pass filtering.
    replicate_column : str, optional
        Column name in data.obs that contains replicate identifiers.
        If provided, consecutive fraction filtering is applied per replicate,
        and proteins must meet the min_consecutive threshold in at least min_replicates number of replicates.
        If None, filtering is applied across all samples as a single dataset.
    min_replicates : int, optional
        Minimum number of replicates that must satisfy the consecutive fraction requirement for a protein to be retained.
    inplace : bool, default True
        If True, modifies the input AnnData object in place and returns None.
        If False, returns a new filtered AnnData object.

    Returns
    -------
    Depending on `inplace` and input type, returns either:

    None
        if `inplace=True`
    np.ndarray
        A boolean mask of proteins that passed the filter, if `inplace=False`.
    """

    if replicate_column is None:
        consecutive_fractions = longest_consecutive_run_per_row(data.X)
        filtered_subset = consecutive_fractions >= min_consecutive
        if inplace:
            data.obs["consecutive_fractions"] = consecutive_fractions
    else:
        groups = data.var.groupby(replicate_column)
        protein_subset = np.repeat(0, repeats=data.n_obs)
        for _, g in groups:
            ad_sub = data[:, g.index].copy()

            gs = filter_min_consecutive_fractions(
                ad_sub, min_consecutive=min_consecutive, inplace=False
            )
            # isn't this returning anndata object and not a tuple ---> need to fix this portion, take a look at how he did it before

            protein_subset = protein_subset + gs
        filtered_subset = protein_subset >= min_replicates
        if inplace:
            data.obs[f"n_replicates_with_minimum_{min_consecutive}_fractions"] = protein_subset
        else:
            return protein_subset

    if not inplace:
        return filtered_subset
    data._inplace_subset_obs(data.obs.index[filtered_subset])


def filter_proteins_per_replicate(
    data: AnnData,
    grouping_columns: str | List[str],
    min_replicates: int = 1,
    min_samples: int = 1,
    inplace: bool = True,
) -> np.ndarray | None:
    """Filter proteins based on detection in replicates.

    Parameters
    ----------
    data
        The annotated data matrix with proteins as observations (rows).
    grouping_columns
        Column name(s) in data.var to group samples into replicates.
    min_replicates
        Minimum number of replicates a protein must be detected in to pass filtering.
    min_samples
        Minimum number of sample groups a protein must be detected in to pass filtering.
    inplace
        Whether to modify data in place or return a copy.

    Returns
    -------
    np.ndarray | None
        * If `inplace` is **False**, returns a boolean mask indicating which
          proteins passed the filter.
        * If `inplace` is **True**, updates *data* in place and returns **None**.

    Notes
    -----
    This function filters proteins based on their detection pattern across replicates.
    For each group of samples (defined by grouping_columns), it requires proteins to be
    detected in at least min_replicates samples. The protein must pass this threshold
    in at least min_samples groups to be kept.
    """
    confirm_proteins_as_obs(data)
    groups = data.var.groupby(grouping_columns)
    protein_subset = np.repeat(0, repeats=data.n_obs)
    for _, g in groups:
        ad_sub = data[:, g.index]
        gs, _ = filter_proteins(ad_sub, min_samples=min_replicates, inplace=False)
        protein_subset = protein_subset + gs
    gene_subset = protein_subset >= min_samples
    if not inplace:
        return gene_subset
    data._inplace_subset_obs(data.obs.index[gene_subset])


def longest_consecutive_run_per_row(a1: np.ndarray) -> np.ndarray:
    """
    Calculates the length of the longest consecutive run of non-zero values in each row of a 2D array.

    Parameters
    ----------
    a1
        A 2D numpy array.

    Returns
    -------
    A 1D numpy array containing the length of the longest consecutive run for each row.
    """
    # Ensure input is binary (0 or 1)
    a1 = (a1 != 0).astype(int)

    # Add zero padding to the start and end of each row to catch runs at the edges
    padded = np.pad(a1, ((0, 0), (1, 1)), constant_values=0)

    # Find the positions where values change (start or end of a run)
    diff = np.diff(padded, axis=1)

    # Start and end indices of runs
    starts = np.where(diff == 1)
    ends = np.where(diff == -1)

    # Calculate lengths of runs
    run_lengths = ends[1] - starts[1]

    # Aggregate max run per row
    max_lengths = np.zeros(a1.shape[0], dtype=int)
    np.maximum.at(max_lengths, starts[0], run_lengths)

    return max_lengths


def remove_contaminants(
    data: AnnData,
    filter_columns: List[str] | None = None,
    filter_value: str | None = None,
    inplace: bool = True,
) -> AnnData | None:
    """Remove contaminant proteins from the data matrix.

    Parameters
    ----------
    data
        The annotated data matrix with proteins as observations (rows).
    filter_columns
        Column names in data.obs to use for filtering contaminants. If None, uses
        columns specified in data.uns['RawInfo']['filter_columns'].
    filter_value
        If provided, first convert filter columns to boolean by comparing to this value.
        If None, assumes filter columns are already boolean.
    inplace
        Whether to modify data in place or return a copy.

    Returns
    -------
    AnnData or None
        If inplace=False, returns filtered data. If inplace=True, returns None.
    """

    confirm_proteins_as_obs(data)
    if filter_columns is None:
        filter_columns = data.uns["RawInfo"]["filter_columns"]
    elif isinstance(filter_columns, str):
        filter_columns = [filter_columns]

    if filter_value is not None:
        data.obs[filter_columns] = data.obs[filter_columns].eq(filter_value)
    is_contaminant = data.obs[filter_columns].any(axis=1)
    if not inplace:
        return data.copy()[~is_contaminant, :]
    data._inplace_subset_obs(data.obs.index[~is_contaminant])


def aggregate_proteins(
    data: AnnData,
    grouping_columns: str | List[str],
    agg_func: NDArrayAxisFunction = np.median,
) -> AnnData:
    """Aggregate proteins based on grouping columns.

    Parameters
    ----------
    data
        The annotated data matrix with proteins as observations (rows).
    grouping_columns
        Column name(s) in data.obs to group proteins by.
    agg_func
        Function to aggregate proteins within each group. Must take an array and axis
        argument. Default is np.median.

    Returns
    -------
    AnnData
        New AnnData object with aggregated proteins.

    Notes
    -----
    This function aggregates proteins based on shared values in the specified grouping
    columns. For each group, the proteins are combined using the provided aggregation
    function. The resulting AnnData object has one observation per unique group.
    """
    groups = data.obs.groupby(grouping_columns)
    X_list = []
    obs_list = []
    # Determine obs columns to keep
    g = groups.get_group(list(groups.groups)[0])
    unique_col_indices = g.nunique() == 1

    for _, ind in groups.indices.items():
        g = data.obs.iloc[ind]
        obs_sub = g.loc[g.index[[0]], unique_col_indices]
        obs_sub["n_merged_proteins"] = ind.size
        X_sub = data.X[ind, :]
        X_sub = agg_func(X_sub, axis=0)
        X_list.append(X_sub)
        obs_list.append(obs_sub)
    obs = pd.concat(obs_list, axis=0)
    X = np.vstack(X_list)
    retdata = AnnData(X=X, obs=obs, var=data.var, uns=data.uns, varp=data.varp, varm=data.varm)
    return retdata


def aggregate_samples(
    data: AnnData,
    grouping_columns: str | List[str],
    agg_func: NDArrayAxisFunction = np.median,
    keep_raw: bool = False,
) -> AnnData:
    """Aggregate samples based on grouping columns.

    Parameters
    ----------
    data
        The annotated data matrix with proteins as observations (rows).
    grouping_columns
        Column name(s) in data.var to group samples by.
    agg_func
        Function to aggregate samples within each group. Must take an array and axis
        argument. Default is np.median.
    keep_raw
        Whether to keep the unaggregated data in the .raw attribute of the returned AnnData object.

    Returns
    -------
    AnnData
        New AnnData object with aggregated samples.

    Notes
    -----
    This function aggregates samples based on shared values in the specified grouping
    columns. For each group, the samples are combined using the provided aggregation
    function. The resulting AnnData object has one variable per unique group.
    """
    groups = data.var.groupby(grouping_columns, observed=True)
    X_list = []
    var_list = []
    layers_dict = {layer: [] for layer in data.layers.keys()}
    # Determine obs columns to keep
    g = groups.get_group(list(groups.groups)[0])
    unique_col_indices = g.nunique() == 1

    for names, ind in groups.indices.items():
        g = data.var.iloc[ind]
        if isinstance(names, str):
            names = [names]
        var_sub = g.loc[:, unique_col_indices].iloc[[0]]
        var_sub["n_merged_samples"] = ind.size
        var_sub.index = ["_".join(names)]
        X_sub = data.X[:, ind]
        X_sub = agg_func(X_sub, axis=1)
        X_list.append(X_sub)
        var_list.append(var_sub)

        # Aggregate layers
        for layer_name, layer_data in data.layers.items():
            layer_sub = layer_data[:, ind]
            layer_sub = agg_func(layer_sub, axis=1)
            layers_dict[layer_name].append(layer_sub)

    var = pd.concat(var_list, axis=0)
    X = np.vstack(X_list).T
    aggregated_layers = {
        layer: np.vstack(layer_list).T for layer, layer_list in layers_dict.items()
    }
    retdata = AnnData(
        X=X,
        obs=data.obs,
        var=var,
        uns=data.uns,
        obsp=data.obsp,
        obsm=data.obsm,
        layers=aggregated_layers,
    )
    if keep_raw:
        retdata.raw = data.copy()

    return retdata


def calculate_qc_metrics(
    data: AnnData,
    qc_vars: Collection[str] | str = (),
    percent_top: Collection[int] | None = (50, 100, 200, 500),
    layer: str | None = None,
    use_raw: bool = False,
    inplace: bool = True,
    log1p: bool = True,
    var_type: str = "proteins",
    expr_type: str = "intensity",
    parallel: bool | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """Calculate quality control metrics.

    Parameters
    ----------
    data
        The annotated data matrix with proteins as observations (rows).
    qc_vars
        Keys for boolean columns in .var that indicate a protein is a quality control
        protein.
    percent_top
        Which proportions of top proteins to compute as QC metrics.
        Set to None to disable.
    layer
        If provided, use `data.layers[layer]` for expression values.
    use_raw
        If True, use `data.raw` for expression values.
    inplace
        Whether to add metrics to input object or return them.
    log1p
        If True, compute log1p of expression values.
    var_type
        Name for variables (e.g. 'proteins', 'genes', etc).
    expr_type
        Name for expression values (e.g. 'intensity', 'counts', etc).
    parallel
        Whether to parallelize computation.

    Returns
    -------
    tuple of (pd.DataFrame, pd.DataFrame) or None
        If `inplace` is False, returns:

        - A DataFrame with protein-based metrics (`.var`)
        - A DataFrame with sample-based metrics (`.obs`)

        If `inplace` is True, returns None and adds metrics to the input object.

    Notes
    -----
    Calculates quality control metrics for both proteins and samples, including:

    - Number of samples expressing each protein
    - Total intensity per sample
    - Number of proteins detected per sample
    - Percentage of intensity from top proteins

    **Added to `.obs`**:
    - `n_samples_by_intensity`: number of samples where each protein has non-zero intensity
    - `mean_intensity`: mean intensity of the protein
    - `log1p_mean_intensity`: log1p of the mean intensity
    - `pct_dropout_by_intensity`: percent of samples with zero intensity
    - `total_intensity`: total intensity per sample
    - `log1p_total_intensity`: log1p of the total intensity

    **Added to `.var`**:
    - `n_proteins_by_intensity`: number of proteins detected in a sample
    - `log1p_n_proteins_by_intensity`: log1p of that number
    - `total_intensity`: total intensity of the protein
    - `log1p_total_intensity`: log1p of the total intensity
    - `pct_intensity_in_top_n_proteins`: percent of intensity from top n proteins
    - `pct_dropout_by_intensity`: percent of samples where the protein is not detected
    """

    confirm_proteins_as_obs(data)
    dfs = sc.pp.calculate_qc_metrics(
        data.copy().T,
        expr_type=expr_type,
        var_type=var_type,
        inplace=False,
        layer=layer,
        use_raw=use_raw,
        log1p=log1p,
        parallel=parallel,
        percent_top=percent_top,
        qc_vars=qc_vars,
    )
    var_df, obs_df = dfs
    var_df["pct_dropout_by_intensity"] = 100 - (
        100 * (var_df[f"n_{var_type}_by_{expr_type}"] / obs_df.shape[0])
    )
    if not inplace:
        return dfs
    obs_df.columns = obs_df.columns.str.replace(
        "cells", "samples"
    )  # This fixes a bug in scanpy
    data.obs[obs_df.columns] = obs_df
    data.var[var_df.columns] = var_df


def highly_variable_proteins(
    data: AnnData,
    inplace: bool = True,
    n_top_proteins: int | None = None,
    flavor: Literal["seurat", "cell_ranger", "seurat_v3", "seurat_v3_paper"] = "seurat",
    subset: bool = False,
    batch_key: str | None = None,
    **kwargs,
) -> pd.DataFrame | None:
    """Identify highly variable proteins.

    Parameters
    ----------
    data
        The annotated data matrix with proteins as observations (rows).
    inplace
        Whether to store results in data.obs or return them.
    n_top_proteins
        Number of highly-variable proteins to keep. If None, use flavor-specific defaults.
    flavor
        Method for identifying highly variable proteins. Options are:
        'seurat' - Seurat's method (default)
        'cell_ranger' - Cell Ranger's method
        'seurat_v3' - Seurat v3 method
        'seurat_v3_paper' - Method from Seurat v3 paper
    subset
        Whether to subset the data to highly variable proteins.
    batch_key
        If specified, highly-variable proteins are selected within each batch separately.
    **kwargs
        Additional arguments to pass to scanpy.pp.highly_variable_genes.

    Returns
    -------
    pandas.DataFrame or None
        If inplace=False, returns DataFrame of highly variable proteins.
        If inplace=True, returns None and stores results in data.obs.

    Notes
    -----
    This function identifies highly variable proteins using methods adapted from
    single-cell RNA sequencing analysis. The results are stored in ``data.obs`` with the
    following fields::

        * highly_variable – boolean indicator
        * means – mean expression
        * dispersions – dispersion of expression
        * dispersions_norm – normalized dispersion
    """
    confirm_proteins_as_obs(data)
    df = sc.pp.highly_variable_genes(
        data.T, inplace=False, n_top_genes=n_top_proteins, **kwargs
    )
    if not inplace:
        if subset:
            df = df.loc[df["highly_variable"]]
        return df

    data.uns["hvg"] = {"flavor": flavor}
    data.obs["highly_variable"] = df["highly_variable"]
    data.obs["means"] = df["means"]
    data.obs["dispersions"] = df["dispersions"]
    data.obs["dispersions_norm"] = df["dispersions_norm"].astype(np.float32, copy=False)

    if batch_key is not None:
        data.obs["highly_variable_nbatches"] = df["highly_variable_nbatches"]
        data.obs["highly_variable_intersection"] = df["highly_variable_intersection"]
    if subset:
        data._inplace_subset_var(df["highly_variable"])


# @TODO: This needs fixing. Does not return the scaled  with inplace=True
def normalize_total(
    data: AnnData,
    inplace: bool = True,
    **kwargs,
) -> AnnData | dict[str, np.ndarray] | None:
    """Normalize expression values for each sample to sum to a constant value.

    Parameters
    ----------
    data
        The annotated data matrix with proteins as observations (rows).
    inplace
        Whether to modify data in place or return normalization factors.
    **kwargs
        Additional arguments to pass to scanpy.pp.normalize_total.

    Returns
    -------
    AnnData or dict or None
        If copy=True, returns a copy of the AnnData object.
        If inplace=False, returns dictionary containing normalization factors.
        If inplace=True, returns None and modifies input data.

    Notes
    -----
    This function normalizes each sample (column) to have the same total intensity.
    This function serves as a convenient wrapper around scanpy.pp.normalize_total,
    automatically handling the transposition required to work with protein data
    (where proteins are rows rather than columns as in typical single-cell data).
    """
    normd = sc.pp.normalize_total(
        data.T,
        inplace=False,
        copy=False,
        **kwargs,
    )
    if inplace:
        data.X = normd["X"].T
    else:
        retdata = data.copy()
        retdata.X = normd["X"].T
        return retdata


def drop_excess_MQ_metadata(
    data: AnnData,
    colname_regex: str = "Peptide|peptide|MS/MS|Evidence IDs|Taxonomy|Oxidation|Intensity|Total Spectral Count|Unique Spectral Count|Spectral Count|Identification type|Sequence coverage|MS/MS count",
    inplace: bool = True,
) -> AnnData | None:
    """Drop excess metadata columns from MaxQuant output.

    Parameters
    ----------
    data
        The annotated data matrix with proteins as observations (rows).
    colname_regex
        Regular expression pattern to match column names that should be dropped.
        Default pattern matches common MaxQuant metadata columns.
    inplace
        Whether to modify data in place or return filtered metadata.

    Returns
    -------
    AnnData or None
        If inplace=False, returns filtered metadata.
        If inplace=True, returns None and modifies input data.

    Notes
    -----
    This function removes metadata columns that match the provided regular expression
    pattern. The default pattern removes common MaxQuant metadata columns that are
    typically not needed for downstream analysis.
    """

    obs = data.obs
    drop_mask = obs.columns.str.contains(colname_regex, regex=True)
    obs = obs.loc[:, ~drop_mask]
    if not inplace:
        return obs
    data.obs = obs


# def pca(data: AnnData, **kwargs) -> None:
#     scanpy.pp.pca(data.T, **kwargs)
