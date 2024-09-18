from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from anndata import AnnData
    from typing import List

import numpy as np
import pandas as pd


def knn_annotation(
    data: AnnData,
    orig_ann_col: str,
    key_added: str = "consensus_graph_annotation",
    exclude_category: str | List[str] | None = None,
) -> AnnData:
    nrow = data.obs.shape[0]
    orig_ann = data.obs[orig_ann_col]
    if isinstance(exclude_category, str):
        exclude_category = [exclude_category]
    orig_ann.replace(exclude_category, np.nan, inplace=True)

    df = pd.DataFrame(np.tile(orig_ann, (nrow, 1)))
    conn = data.obsp['distances']
    mask = ~(conn != 0).todense()  # This avoids expensive conn == 0 for sparse matrices
    df[mask] = np.nan

    majority_cluster = df.mode(axis=1, dropna=True).loc[
        :, 0
    ]  # take the first if there are ties
    data.obs[key_added] = majority_cluster.values
    return data
