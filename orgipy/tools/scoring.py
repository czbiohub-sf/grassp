from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

import sklearn.metrics


def silhouette_score(
    data, gt_col, use_rep="X_umap", key_added="silhouette", inplace=True
) -> None | np.ndarray:
    mask = data.obs[gt_col].notna()
    data_sub = data[mask]
    ss = sklearn.metrics.silhouette_samples(data_sub.obsm[use_rep], data_sub.obs[gt_col])
    if inplace:
        data_sub.obs[key_added] = ss
        cluster_mean_ss = data_sub.obs.groupby(gt_col)[key_added].mean()
        data.uns[key_added] = {
            "mean_silhouette_score": ss.mean(),
            "cluster_mean_silhouette": cluster_mean_ss.to_dict(),
            "cluster_balanced_silhouette_score": cluster_mean_ss.mean(),
        }
        data.obs.loc[mask, key_added] = ss
    else:
        return ss


def calinski_habarasz_score(
    data, gt_col, use_rep="X_umap", key_added="ch_score", inplace=True
) -> None | float:
    mask = data.obs[gt_col].notna()
    data_sub = data[mask]
    ch = sklearn.metrics.calinski_harabasz_score(data_sub.obsm[use_rep], data_sub.obs[gt_col])
    if inplace:
        data.uns[key_added] = ch
    else:
        return ch
