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

# from ..util import confirm_proteins_as_obs


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
    Depending on `inplace` and input type, returns either:

    * ``None`` if ``inplace=True``
    * AnnData if input is AnnData and ``inplace=False``
    * A tuple of arrays (``retained_samples``, ``retained_proteins``) if input is not AnnData
    """
    # if isinstance(data, AnnData):
    #     confirm_proteins_as_obs(data)

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
    Depending on ``inplace`` and input type, returns either:

    * ``None`` if ``inplace=True``
    * AnnData if input is AnnData and ``inplace=False``
    * A tuple of arrays ``(retained_proteins, retained_samples)`` if input is not AnnData
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
        Column name in ``data.obs`` that contains replicate identifiers.
        If provided, consecutive fraction filtering is applied per replicate,
        and proteins must meet the `min_consecutive` threshold in at least `min_replicates` number of replicates.
        If None, filtering is applied across all samples as a single dataset.
    min_replicates : int, optional
        Minimum number of replicates that must satisfy the consecutive fraction requirement for a protein to be retained.
    inplace : bool, default True
        If True, modifies the input AnnData object in place and returns None.
        If False, returns a new filtered AnnData object.

    Returns
    -------
    Depending on ``inplace`` and input type, returns either:

    ``None``
        if ``inplace=True``
    ``np.ndarray``
        A boolean mask of proteins that passed the filter, if ``inplace=False``.
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
            data.obs[f"n_replicates_with_minimum_{min_consecutive}_fractions"] = (
                protein_subset
            )
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
        Column name(s) in ``data.var`` to group samples into replicates.
        Note: Typically the grouping columns will not be the column with the replicate information, but rather the columns with the sample (IP/fraction) information. Samples that are grouped by these columns will be considered replicates.
    min_replicates
        Minimum number of replicates a protein must be detected in to pass filtering.
    min_samples
        Minimum number of sample groups a protein must be detected in to pass filtering.
    inplace
        Whether to modify data in place or return a copy.

    Returns
    -------
    numpy.ndarray or None

    * If ``inplace=False``, returns boolean mask indicating which proteins passed filtering.
    * If ``inplace=True``, returns None and modifies input data.

    Notes
    -----
    This function filters proteins based on their detection pattern across replicates.
    For each group of samples (defined by ``grouping_columns``), it requires proteins to be
    detected in at least ``min_replicates`` samples. The protein must pass this threshold
    in at least ``min_samples`` groups to be kept.
    """
    # confirm_proteins_as_obs(data)
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

    * If `inplace=False`, returns filtered data.
    * If `inplace=True`, returns None.
    """

    # confirm_proteins_as_obs(data)
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
    """Aggregates protein intensities across samples using a given function.

    Parameters
    ----------
    data
        The annotated data matrix with proteins as observations (rows).
    grouping_columns
        Column name(s) in ``data.obs`` to group samples into replicates.
    agg_func
        Function to use for aggregation. Defaults to ``np.median``.

    Returns
    -------
    A new :class:`~anndata.AnnData` object with aggregated expression values. The number of
    variables (samples) remains the same, but the number of observations
    (proteins) will correspond to the number of unique groups defined by
    ``grouping_columns``.

    Notes
    -----
    This function is useful for e.g. combining multiple proteins that belong to the same gene. For each protein, it groups
    the samples based on the provided ``grouping_columns`` and then aggregates
    the intensity values using the specified ``agg_func``.
    """
    groups = data.obs.groupby(grouping_columns, observed=True)
    X_list = []
    obs_list = []
    layers_dict = {layer: [] for layer in data.layers.keys()}

    for _, ind in groups.indices.items():
        g = data.obs.iloc[ind]
        # Determine obs columns to keep
        unique_col_indices = g.nunique() == 1
        obs_sub = g.loc[g.index[[0]], unique_col_indices]
        obs_sub["n_merged_proteins"] = ind.size
        X_sub = data.X[ind, :]
        X_sub = agg_func(X_sub, axis=0)
        X_list.append(X_sub)
        obs_list.append(obs_sub)
        # Aggregate layers
        for layer_name, layer_data in data.layers.items():
            layer_sub = layer_data[ind, :]
            layer_sub = agg_func(layer_sub, axis=0)
            layers_dict[layer_name].append(layer_sub)

        obs_list.append(obs_sub)
        # Aggregate layers
        for layer_name, layer_data in data.layers.items():
            layer_sub = layer_data[ind, :]
            layer_sub = agg_func(layer_sub, axis=0)
            layers_dict[layer_name].append(layer_sub)

    obs = pd.concat(obs_list, axis=0)
    X = np.vstack(X_list)
    aggregated_layers = {
        layer: np.vstack(layer_list) for layer, layer_list in layers_dict.items()
    }
    retdata = AnnData(
        X=X,
        obs=obs,
        var=data.var,
        uns=data.uns,
        varp=data.varp,
        varm=data.varm,
        layers=aggregated_layers,
    )
    return retdata


def aggregate_samples(
    data: AnnData,
    grouping_columns: str | List[str],
    agg_func: NDArrayAxisFunction = np.median,
    keep_raw: bool = False,
) -> AnnData:
    """Aggregates sample expression across samples using a given function.

    Parameters
    ----------
    data
        The annotated data matrix with proteins as observations (rows).
    grouping_columns
        Column name(s) in ``data.obs`` to group proteins.
    agg_func
        Function to use for aggregation. Defaults to ``np.median``.
    keep_raw
        Whether to keep the raw data in the returned AnnData object.

    Returns
    -------
    A new :class:`~anndata.AnnData` object with aggregated expression values. The number of
    observations (proteins) remains the same, but the number of variables
    (samples) will correspond to the number of unique groups defined by
    ``grouping_columns``.

    Notes
    -----
    This function is useful for combining replicates or creating an averaged profile across conditions.
    For each sample, it groups the samples based on the provided ``grouping_columns`` and then aggregates
    the expression values using the specified ``agg_func``.
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
    """\
    Calculate quality control metrics.

    This function is a wrapper around :func:`scanpy:scanpy.pp.calculate_qc_metrics` ``scanpy.pp.calculate_qc_metrics``.
    It calculates quality control metrics for proteins and samples and adds
    them to ``data.obs`` and ``data.var``.

    Parameters
    ----------
    data
        The annotated data matrix.
    qc_vars
        Column names in ``.obs`` to add to the QC metrics.
    percent_top
        Which proportions of top genes to cover.
    layer
        Layer to use for QC metric calculation.
    use_raw
        Whether to use ``.raw`` for calculation.
    inplace
        Whether to add the QC metrics to the AnnData object.
    log1p
        Whether to log1p the expression values before calculating QC metrics.
    var_type
        The type of variables in the data.
    expr_type
        The type of expression values in the data.
    parallel
        Whether to run the calculation in parallel.

    Returns
    -------
    if ``inplace=True``.
        ``None`` and modifies the data ``.obs`` and ``.var`` with the QC metrics.
    if ``inplace=False``, a tuple with protein-wise and sample-wise QC metrics:
        * `protein_qc_metrics`: ``pd.DataFrame`` with protein-wise QC metrics
        * `sample_qc_metrics`: ``pd.DataFrame`` with sample-wise QC metrics
    """

    # confirm_proteins_as_obs(data)
    return sc.pp.calculate_qc_metrics(
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
    If ``inplace=False``, returns :class:`~pandas.DataFrame` of highly variable proteins.
    If ``inplace=True``, returns ``None`` and stores results in ``data.obs``.

    Notes
    -----
    This function identifies highly variable proteins wrapping the scanpy function :func:`~scanpy.pp.highly_variable_genes`.
    The results are stored in ``data.obs`` with the following fields:

    * ``highly_variable``: boolean indicator
    * ``means``: mean expression
    * ``dispersions``: dispersion of expression
    * ``dispersions_norm``: normalized dispersion
    """
    # confirm_proteins_as_obs(data)
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
) -> AnnData | None:
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
    - If ``inplace=True``, modifies the input :class:`~anndata.AnnData` object and returns ``None``.
    - If ``inplace=False``, returns a new :class:`~anndata.AnnData` object with normalized values.

    Notes
    -----
    This function normalizes each sample (column) to have the same total intensity.
    This function serves as a convenient wrapper around :func:`~scanpy.pp.normalize_total`,
    automatically handling the transposition required to work with subcellular protein data
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
) -> None | pd.DataFrame:
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
    - If ``inplace=True``, modifies the input :class:`~anndata.AnnData` object and returns ``None``.
    - If ``inplace=False``, returns a :class:`~pandas.DataFrame` with filtered metadata.

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
