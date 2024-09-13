from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Callable, List, Optional

    import numpy as np

    # Define a type hint for functions that take an ndarray and an optional axis argument
    NDArrayAxisFunction = Callable[[np.ndarray, Optional[int]], np.ndarray]

import numpy as np
import pandas as pd
import scanpy

from anndata import AnnData

filter_samples = scanpy.pp.filter_cells
filter_proteins = scanpy.pp.filter_genes


def remove_contaminants(data: AnnData, filter_columns: List[str] | None = None) -> AnnData:
    if filter_columns is None:
        filter_columns = data.uns["RawInfo"]["filter_columns"]

    is_contaminant = data.var[filter_columns].any(axis=1)
    data = data[:, ~is_contaminant]
    return data


def filter_proteins_per_replicate(
    data: AnnData,
    grouping_columns: str | List[str],
    min_replicates: int = 1,
    min_samples: int = 1,
    inplace: bool = True,
):
    groups = data.obs.groupby(grouping_columns)
    gene_subset = np.repeat(0, repeats=data.n_vars)
    for _, g in groups:
        # print(_)
        ad_sub = data[g.index, :]
        gs, _ = filter_proteins(ad_sub, min_cells=min_replicates, inplace=False)
        gene_subset = gene_subset + gs
    gene_subset = gene_subset >= min_samples
    if inplace:
        data = data[:, gene_subset]
        return data
    return gene_subset


def aggregate_replicates(
    data: AnnData, grouping_columns: str | List[str], agg_func: NDArrayAxisFunction = np.median
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
        obs_sub["n_merged_replicates"] = ind.size
        X_sub = data.X[ind, :]
        X_sub = agg_func(X_sub, axis=0)
        X_list.append(X_sub)
        obs_list.append(obs_sub)
    obs = pd.concat(obs_list, axis=0)
    X = np.vstack(X_list)
    retdata = AnnData(X=X, obs=obs, var=data.var, uns=data.uns, varp=data.varp, varm=data.varm)
    return retdata


def aggregate_proteins(
    data: AnnData, grouping_columns: str | List[str], agg_func: NDArrayAxisFunction = np.median
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
        var_sub["n_merged_proteins"] = ind.size
        X_sub = data.X[:, ind]
        X_sub = agg_func(X_sub, axis=1)
        X_list.append(X_sub)
        var_list.append(var_sub)
    var = pd.concat(var_list, axis=0)
    X = np.vstack(X_list).T
    retdata = AnnData(X=X, obs=data.obs, var=var, uns=data.uns, obsp=data.obsp, obsm=data.obsm)
    return retdata
