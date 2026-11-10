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

from ..util import layer_names, set_matrix


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

    Examples
    --------
    >>> import grassp as gr
    >>> adata = gr.datasets.hein_2024(enrichment="raw")
    >>> adata.shape
    (8538, 183)
    >>> gr.pp.filter_samples(adata, min_proteins=500)
    >>> adata.shape  # Fewer samples after filtering
    (8538, 183)
    """

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

    Only one criterion can be given per call, as in :func:`scanpy.pp.filter_cells`.

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
        (If using fractionation instead of organellar IP) Minimum number of consecutive
        fractions that a protein is present in; see
        :func:`filter_min_consecutive_fractions`. Requires an AnnData.

    Returns
    -------
    Depending on ``inplace`` and input type, returns either:

    * ``None`` if ``inplace=True``
    * A tuple of arrays ``(retained_proteins, retained_samples)`` if ``inplace=False``
    * A boolean mask of retained proteins for ``min_consecutive`` with ``inplace=False``

    Examples
    --------
    >>> import grassp as gr
    >>> adata = gr.datasets.hein_2024(enrichment="raw")
    >>> adata.shape
    (8538, 183)
    >>> gr.pp.filter_proteins(adata, min_samples=5)
    >>> adata.shape  # Fewer proteins after filtering
    (8358, 183)
    """
    # min_consecutive used to be accepted and silently ignored.
    if min_consecutive is not None:
        if any(v is not None for v in (min_counts, min_samples, max_counts, max_samples)):
            raise ValueError("Only provide one of the filtering criteria per call.")
        if not isinstance(data, AnnData):
            raise TypeError("min_consecutive requires an AnnData object.")
        return filter_min_consecutive_fractions(
            data, min_consecutive=min_consecutive, inplace=inplace
        )

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
        Column name in ``data.var`` that contains replicate identifiers.
        If provided, consecutive fraction filtering is applied per replicate,
        and proteins must meet the `min_consecutive` threshold in at least `min_replicates` number of replicates.
        If None, filtering is applied across all samples as a single dataset.
    min_replicates : int, optional
        Minimum number of replicates that must satisfy the consecutive fraction requirement
        for a protein to be retained. ``None`` (default) means 1.
    inplace : bool, default True
        If True, removes the proteins that fail from the input AnnData and returns None.
        If False, returns the boolean mask of proteins that pass.

    Returns
    -------
    Depending on ``inplace`` and input type, returns either:

    ``None``
        if ``inplace=True``
    ``np.ndarray``
        A boolean mask of proteins that passed the filter, if ``inplace=False``.

    Examples
    --------
    >>> import grassp as gr
    >>> adata = gr.datasets.hek_dc_2025(enrichment="raw")
    >>> adata.shape
    (8599, 42)
    >>> gr.pp.filter_min_consecutive_fractions(adata, min_consecutive=3)
    >>> adata.shape  # Fewer proteins after filtering
    (8599, 42)
    """

    if replicate_column is None:
        consecutive_fractions = _longest_consecutive_run_per_row(data.X)
        filtered_subset = consecutive_fractions >= min_consecutive
        if inplace:
            data.obs["consecutive_fractions"] = consecutive_fractions
    else:
        # A None default used to reach `>= None` and raise TypeError.
        if min_replicates is None:
            min_replicates = 1
        groups = data.var.groupby(replicate_column, observed=True)
        protein_subset = np.repeat(0, repeats=data.n_obs)
        for _, g in groups:
            ad_sub = data[:, g.index].copy()
            # boolean mask of the proteins passing within this replicate
            gs = filter_min_consecutive_fractions(
                ad_sub, min_consecutive=min_consecutive, inplace=False
            )
            protein_subset = protein_subset + gs
        filtered_subset = protein_subset >= min_replicates
        if inplace:
            data.obs[f"n_replicates_with_minimum_{min_consecutive}_fractions"] = protein_subset

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

    Examples
    --------
    >>> import grassp as gr
    >>> adata = gr.datasets.hein_2024(enrichment="raw")
    >>> adata.shape
    (8538, 183)
    >>> gr.pp.filter_proteins_per_replicate(
    ...     adata,
    ...     grouping_columns='subcellular_enrichment',
    ...     min_replicates=2,
    ...     min_samples=3
    ... )
    >>> adata.shape  # Fewer proteins after filtering
    (7869, 183)
    """
    groups = data.var.groupby(grouping_columns, observed=True)
    protein_subset = np.repeat(0, repeats=data.n_obs)
    for _, g in groups:
        ad_sub = data[:, g.index]
        gs, _ = filter_proteins(ad_sub, min_samples=min_replicates, inplace=False)
        protein_subset = protein_subset + gs
    gene_subset = protein_subset >= min_samples
    if not inplace:
        return gene_subset
    data._inplace_subset_obs(data.obs.index[gene_subset])


def _longest_consecutive_run_per_row(a1: np.ndarray) -> np.ndarray:
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

    Examples
    --------
    >>> import grassp as gr
    >>> adata = gr.datasets.hein_2024(enrichment="raw")
    >>> adata.shape
    (14551, 222)
    >>> aggregated = gr.pp.aggregate_proteins(
    ...     adata,
    ...     grouping_columns='Gene names'
    ... )
    >>> aggregated.shape  # Fewer proteins after aggregation
    (12617, 222)

    Proteins with a missing value in ``grouping_columns`` are not merged with anything: each
    is kept as its own row under its original name (here the 2,371 proteins without a gene
    name). ``n_merged_proteins`` counts the input proteins behind each row, and adds up when
    an already aggregated object is aggregated again.
    """
    obs_in = data.obs
    keys = obs_in[grouping_columns]
    missing = keys.isna() if keys.ndim == 1 else keys.isna().any(axis=1)
    # groupby drops rows with a missing key, which used to delete those proteins silently;
    # grouping them (dropna=False) would merge every unnamed protein into a single row.
    groups = obs_in.groupby(grouping_columns, observed=True)
    entries = list(groups.indices.items()) + [
        (obs_in.index[p], np.array([p])) for p in np.flatnonzero(missing.to_numpy())
    ]

    # Start from each group's first row, so every column keeps its dtype (an empty frame
    # filled row by row made them all object, which write_h5ad cannot store), then blank the
    # columns whose values differ within a merged group.
    obs = obs_in.iloc[[pos[0] for _, pos in entries]].copy()
    obs.index = pd.Index([key for key, _ in entries])
    varies = np.zeros(obs.shape, dtype=bool)
    for i, (_, pos) in enumerate(entries):
        if len(pos) > 1:
            varies[i] = (obs_in.iloc[pos].nunique() > 1).to_numpy()
    obs = obs.mask(varies)
    for col in obs.columns:  # bool + NaN would be object; keep it storable
        if obs_in[col].dtype == bool and obs[col].dtype == object:
            obs[col] = obs[col].astype("boolean")
    prior = (
        obs_in["n_merged_proteins"].to_numpy()
        if "n_merged_proteins" in obs_in.columns
        else np.ones(len(obs_in), dtype=int)
    )
    obs["n_merged_proteins"] = [int(prior[pos].sum()) for _, pos in entries]

    X = np.empty((len(entries), data.n_vars))
    layers_dict = {layer: X.copy() for layer in layer_names(data)}
    for i, (_, pos) in enumerate(entries):
        if len(pos) == 1:
            X[i] = data.X[pos[0]]
            for layer_name in layers_dict:
                layers_dict[layer_name][i] = data.layers[layer_name][pos[0]]
        else:
            X[i] = agg_func(data.X[pos, :], axis=0)
            for layer_name in layers_dict:
                layers_dict[layer_name][i] = agg_func(data.layers[layer_name][pos, :], axis=0)

    retdata = AnnData(
        X=X,
        obs=obs,
        var=data.var,
        uns=data.uns,
        varp=data.varp,
        varm=data.varm,
        layers=layers_dict,
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

    Examples
    --------
    >>> import grassp as gr
    >>> adata = gr.datasets.hein_2024(enrichment="raw")
    >>> adata.shape  # Original shape
    (8538, 183)
    >>> aggregated = gr.pp.aggregate_samples(
    ...     adata,
    ...     grouping_columns='subcellular_enrichment'
    ... )
    >>> aggregated.shape  # Fewer samples after aggregation
    (8538, 41)
    >>> int(aggregated.var['n_merged_samples'].max())  # Max samples merged
    12
    """
    groups = data.var.groupby(grouping_columns, observed=True)
    X_list = []
    var_list = []
    layers_dict = {layer: [] for layer in layer_names(data)}
    # Determine obs columns to keep
    # g = groups.get_group((list(groups.groups)[0],))
    g0 = next(iter(groups))[1]
    singleton_col_indices = g0.nunique() == 1

    for names, ind in groups.indices.items():
        g = data.var.iloc[ind]
        if isinstance(names, str):
            names = [names]
        var_sub = g.loc[:, singleton_col_indices].iloc[[0]]
        var_sub["n_merged_samples"] = ind.size
        var_sub.index = ["_".join(names)]
        X_sub = data.X[:, ind]
        X_sub = agg_func(X_sub, axis=1)
        X_list.append(X_sub)
        var_list.append(var_sub)

        # Aggregate layers
        for layer_name in layers_dict:
            layer_sub = data.layers[layer_name][:, ind]
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

    Examples
    --------
    >>> import grassp as gr
    >>> adata = gr.datasets.hein_2024(enrichment="raw")
    >>> gr.pp.calculate_qc_metrics(adata)
    >>> adata.obs[['n_samples_by_intensity', 'total_intensity']].head()  # doctest: +NORMALIZE_WHITESPACE
                n_samples_by_intensity  total_intensity
    index
    A0A023T6R1                      78     1.123382e+09
    Q9Y5S9                         183     1.244060e+10
    A0A0C4DFM1                     167     1.080222e+10
    A0A024QYR6                       8     6.084850e+07
    Q99805                         175     3.038691e+10
    """

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

    Examples
    --------
    >>> import grassp as gr
    >>> import scanpy as sc
    >>> adata = gr.datasets.hein_2024(enrichment="enriched")
    >>> gr.pp.highly_variable_proteins(adata, n_top_proteins=1000)
    >>> int(adata.obs['highly_variable'].sum())  # Number of HVPs
    1000
    >>> highly_variable_subset = adata[adata.obs['highly_variable']]
    """
    # Proteins are observations here, so scanpy runs on the transpose and `batch_key` names a
    # sample (var) column -- which is an obs column of `data.T`, exactly where scanpy looks.
    df = sc.pp.highly_variable_genes(
        data.T,
        inplace=False,
        n_top_genes=n_top_proteins,
        flavor=flavor,
        batch_key=batch_key,
        **kwargs,
    )
    if not inplace:
        if subset:
            df = df.loc[df["highly_variable"]]
        return df

    data.uns["hvg"] = {"flavor": flavor}
    # Copy whatever statistics the flavor produced: the seurat_v3 flavors report
    # variances/variances_norm/highly_variable_rank instead of dispersions.
    for col in df.columns:
        data.obs[col] = df[col].to_numpy()
    if "dispersions_norm" in df.columns:
        data.obs["dispersions_norm"] = data.obs["dispersions_norm"].astype(
            np.float32, copy=False
        )
    if subset:
        data._inplace_subset_obs(df["highly_variable"].to_numpy())


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

    Examples
    --------
    >>> import grassp as gr
    >>> import numpy as np
    >>> adata = gr.datasets.hein_2024(enrichment="raw")
    >>> float(np.median(adata.X.sum(axis=0)))  # Column sums before normalization  # doctest: +ELLIPSIS
    627735...
    >>> gr.pp.normalize_total(adata, target_sum=1e6)
    >>> float(np.median(adata.X.sum(axis=0)))  # All columns sum to target after  # doctest: +ELLIPSIS
    1000000.0...
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

    Examples
    --------
    >>> import grassp as gr
    >>> adata = gr.datasets.hein_2024(enrichment="raw")
    >>> gr.pp.drop_excess_MQ_metadata(adata)
    """

    obs = data.obs
    drop_mask = obs.columns.str.contains(colname_regex, regex=True)
    obs = obs.loc[:, ~drop_mask]
    if not inplace:
        return obs
    data.obs = obs


# def pca(data: AnnData, **kwargs) -> None:
#     scanpy.pp.pca(data.T, **kwargs)


def neighbors(data, layer=None, copy: bool = False, **kwargs) -> AnnData | None:
    """Compute the nearest neighbors distance matrix using scanpy.pp.neighbors.

    This function is a wrapper around :func:`scanpy:scanpy.pp.neighbors` ``scanpy.pp.neighbors`` adding support for using different layers.

    Parameters
    ----------
    data
        The annotated data matrix with proteins as observations (rows).
    layer
        Layer to use for nearest neighbors calculation.
    copy
        Return a copy with the neighbors graph instead of writing it to ``data``.
    **kwargs
        Additional arguments to pass to scanpy.pp.neighbors.

    Returns
    -------
    A copy of ``data`` with the neighbors graph if ``copy=True``, else ``None``.
    """
    # The copy is made here and scanpy runs in place on it. Forwarding copy=True to scanpy
    # made it write the graph to a copy of its own that was discarded, so the returned
    # object had no neighbors graph.
    if copy:
        data = data.copy()
    if layer is None:
        sc.pp.neighbors(data, **kwargs)
    else:
        X = data.X
        data.X = data.layers[layer]
        kwargs["use_rep"] = "X"  # when specifying a layer, we should always use X
        try:
            sc.pp.neighbors(data, **kwargs)
        finally:  # never leave the layer in X, even if scanpy raises
            data.X = X
    return data if copy else None


def _calculate_cv(
    x: np.ndarray, is_log: bool = False, axis: int = 1, ignore_zeros: bool = True
) -> float:
    """Calculate the coefficient of variation for a given data matrix.

    Parameters
    ----------
    data
        The data matrix.
    is_log
        Whether the data is log-transformed.
    """
    x = np.asarray(x)
    if ignore_zeros:
        x[x == 0] = np.nan
    if is_log:
        cv = np.sqrt(np.exp(np.nanvar(x, axis=axis)) - 1)
    else:
        cv = np.std(x, axis=axis) / np.mean(x, axis=axis)
    if ignore_zeros:
        cv[np.sum(np.isnan(x), axis=1) > 0] = np.nan
    return cv


def calculate_replicate_cv(
    data: AnnData,
    grouping_columns: str | List[str],
    is_log: bool,
    layer: str | None = None,
    ignore_zeros: bool = True,
    inplace: bool = True,
) -> pd.DataFrame | None:
    """Calculate the coefficient of variation for each protein across replicates.

    Parameters
    ----------
    data
        The annotated data matrix with proteins as observations (rows).
    grouping_columns
        Column name(s) in ``data.obs`` to group samples into replicates.
    is_log
        Whether the data is log-transformed, this determines the CV formula used.
    layer
        Layer to use for calculation, if None, the .X object is used.
    ignore_zeros
        Whether to ignore zero values.
    inplace
        Whether to modify data in place or return the result. If False, returns a :class:`~pandas.DataFrame` with the CVs.

    Returns
    -------
    - If ``inplace=True``, modifies the input :class:`~anndata.AnnData` object and returns
      ``None``. The following fields are added to the :class:`~anndata.AnnData` object:
      - ``.obs["mean_replicate_cv"]``: Mean CV across replicates.
      - ``.obsm["replicate_cv"]``: Replicate CVs per group.
      - ``.uns["obsm_replicate_cv_headers"]``: Headers for the replicate CVs matrix.
    - If ``inplace=False``, returns a :class:`~pandas.DataFrame` with the CVs per group.
    """

    if layer is not None:
        X = data.layers[layer]
    else:
        X = data.X

    groups = data.var.groupby(grouping_columns, sort=False, observed=True)
    # cvs = {name: calculate_cv(X_sub, is_log=is_log, axis=1, ignore_zeros=ignore_zeros) for name, g in groups}
    cvs = {
        name: _calculate_cv(X[:, g], is_log=is_log, axis=1, ignore_zeros=ignore_zeros)
        for name, g in groups.indices.items()
    }
    cvs_df = pd.DataFrame(cvs)

    if inplace:
        data.obs["mean_replicate_cv"] = cvs_df.mean(axis=1).values
        set_matrix(data, "replicate_cv", cvs_df.values, cvs_df.columns)
        data.uns["obsm_replicate_cv_headers"] = cvs_df.columns.tolist()
        return None
    else:
        return cvs_df
