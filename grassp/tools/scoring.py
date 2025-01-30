from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

import sklearn.metrics


def silhouette_score(
    data, gt_col, use_rep="X_umap", key_added="silhouette", inplace=True
) -> None | np.ndarray:
    """Calculate silhouette scores for clustered data.

    Parameters
    ----------
    data : AnnData
        Annotated data matrix.
    gt_col : str
        Column name in data.obs containing cluster labels.
    use_rep : str, optional
        Key for representation in data.obsm to use for score calculation.
        Defaults to 'X_umap'.
    key_added : str, optional
        Key under which to add the silhouette scores. Defaults to 'silhouette'.
    inplace : bool, optional
        If True, store results in data, else return scores. Defaults to True.

    Returns
    -------
    None or ndarray
        If inplace=True, returns None and stores results in data.
        If inplace=False, returns array of silhouette scores.
    """
    mask = data.obs[gt_col].notna()
    data_sub = data[mask]
    sub_obs = data_sub.obs.copy()
    ss = sklearn.metrics.silhouette_samples(data_sub.obsm[use_rep], sub_obs[gt_col])
    if inplace:
        sub_obs[key_added] = ss
        cluster_mean_ss = sub_obs.groupby(gt_col)[key_added].mean()
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
    """Calculate Calinski-Harabasz score for clustered data.

    Parameters
    ----------
    data : AnnData
        Annotated data matrix.
    gt_col : str
        Column name in data.obs containing cluster labels.
    use_rep : str, optional
        Key for representation in data.obsm to use for score calculation.
        Defaults to 'X_umap'.
    key_added : str, optional
        Key under which to add the score. Defaults to 'ch_score'.
    inplace : bool, optional
        If True, store results in data, else return score. Defaults to True.

    Returns
    -------
    None or float
        If inplace=True, returns None and stores result in data.
        If inplace=False, returns the Calinski-Harabasz score.
    """
    mask = data.obs[gt_col].notna()
    data_sub = data[mask]
    ch = sklearn.metrics.calinski_harabasz_score(
        data_sub.obsm[use_rep], data_sub.obs[gt_col]
    )
    if inplace:
        data.uns[key_added] = ch
    else:
        return ch
