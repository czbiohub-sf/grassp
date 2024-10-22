from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Callable, Collection, List, Literal, Optional

    import numpy as np

    from scanpy._compat import DaskArray
    from scipy.sparse import spmatrix

    # Define a type hint for functions that take an ndarray and an optional axis argument
    NDArrayAxisFunction = Callable[[np.ndarray, Optional[int]], np.ndarray]

import numpy as np
import pandas as pd
import scanpy

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

    if isinstance(data, AnnData):
        confirm_proteins_as_obs(data)

    return scanpy.pp.filter_genes(
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
    copy: bool = False,
) -> AnnData | tuple[np.ndarray, np.ndarray] | None:

    if isinstance(data, AnnData):
        confirm_proteins_as_obs(data)

    return scanpy.pp.filter_cells(
        data,
        min_counts=min_counts,
        min_genes=min_samples,
        max_counts=max_counts,
        max_genes=max_samples,
        inplace=inplace,
        copy=copy,
    )


def remove_contaminants(
    data: AnnData, filter_columns: List[str] | None = None, filter_value: str | None = None
) -> AnnData:
    confirm_proteins_as_obs(data)
    if filter_columns is None:
        filter_columns = data.uns["RawInfo"]["filter_columns"]

    if filter_value is not None:
        data.obs[filter_columns] = data.obs[filter_columns].eq(filter_value)
    is_contaminant = data.obs[filter_columns].any(axis=1)
    data = data[~is_contaminant, :]
    return data


def filter_proteins_per_replicate(
    data: AnnData,
    grouping_columns: str | List[str],
    min_replicates: int = 1,
    min_samples: int = 1,
    inplace: bool = True,
) -> np.ndarray | None:
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


def aggregate_proteins(
    data: AnnData,
    grouping_columns: str | List[str],
    agg_func: NDArrayAxisFunction = np.median,
):
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
):
    groups = data.var.groupby(grouping_columns)
    X_list = []
    var_list = []
    # Determine obs columns to keep
    g = groups.get_group(list(groups.groups)[0])
    unique_col_indices = g.nunique() == 1

    for _, ind in groups.indices.items():
        g = data.var.iloc[ind]
        var_sub = g.loc[g.index[[0]], unique_col_indices]
        var_sub["n_merged_samples"] = ind.size
        X_sub = data.X[:, ind]
        X_sub = agg_func(X_sub, axis=1)
        X_list.append(X_sub)
        var_list.append(var_sub)
    var = pd.concat(var_list, axis=0)
    X = np.vstack(X_list).T
    retdata = AnnData(X=X, obs=data.obs, var=var, uns=data.uns, obsp=data.obsp, obsm=data.obsm)
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
    subset: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame] | None:

    confirm_proteins_as_obs(data)
    dfs = scanpy.pp.calculate_qc_metrics(
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
    if not inplace:
        return dfs
    var_df, obs_df = dfs
    obs_df.columns = obs_df.columns.str.replace(
        "cells", "sample"
    )  # This fixes a bug in scanpy
    data.obs = pd.concat([data.obs, obs_df], axis=1)
    data.obs = data.obs.loc[:, ~data.obs.columns.duplicated(keep="last")]
    data.var = pd.concat([data.var, var_df], axis=1)
    data.var = data.var.loc[:, ~data.var.columns.duplicated(keep="last")]


def highly_variable_proteins(
    data: AnnData,
    inplace: bool = True,
    n_top_proteins: int | None = None,
    flavor: Literal["seurat", "cell_ranger", "seurat_v3", "seurat_v3_paper"] = "seurat",
    subset: bool = False,
    batch_key: str | None = None,
    **kwargs,
) -> pd.DataFrame | None:
    confirm_proteins_as_obs(data)
    df = scanpy.pp.highly_variable_genes(
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


def normalize_total(
    data: AnnData,
    copy: bool = False,
    inplace: bool = True,
    **kwargs,
) -> AnnData | dict[str, np.ndarray] | None:
    data = data.T.copy()
    dat = scanpy.pp.normalize_total(
        data.T,
        inplace=False,
        **kwargs,
    )
    if copy:
        return data
    elif not inplace:
        return dat


# def pca(data: AnnData, **kwargs) -> None:
#     scanpy.pp.pca(data.T, **kwargs)
