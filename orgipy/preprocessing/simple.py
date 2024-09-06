from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import List

import numpy as np
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
    data: AnnData, grouping_columns: List[str], min_samples: int = 1
):
    groups = data.obs.groupby(grouping_columns)
    gene_subset = np.repeat(True, repeats=data.n_vars)
    for _, g in groups:
        ad_sub = data[g.index, :]
        gs, _ = filter_proteins(ad_sub, min_cells=min_samples, inplace=False)
        gene_subset = gene_subset & gs
    data = data[:, gene_subset]
    return data
