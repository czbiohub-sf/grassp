from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from anndata import AnnData

import warnings

import numpy as np
import pandas as pd
import sklearn.metrics


def class_balance(
    data: AnnData, label_key: str, min_class_size: int = 10, seed: int = 42
) -> AnnData:
    """Balance classes in the data.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    label_key : str
        Key in adata.obs containing cluster labels.
    min_class_size : int, optional
        Minimum number of samples per class. Defaults to 10.
    """
    # Check if label_key is in adata.obs
    if label_key not in data.obs.columns:
        raise ValueError(f"Label key {label_key} not found in adata.obs")
    # Remove all samples with missing labels
    data_sub = data[data.obs[label_key].notna()]
    # Check if smallest class has at least min_class_size samples
    min_class_s = data_sub.obs[label_key].value_counts().min()
    min_class = data_sub.obs[label_key].value_counts().idxmin()
    if min_class_s < min_class_size:
        raise ValueError(
            f"Smallest class ({min_class}) has less than {min_class_size} samples."
        )
    if min_class_s < 10:
        warnings.warn(
            f"Smallest class ({min_class}) has less than 10 samples, this might not yield a stable score."
        )

    obs_names = []
    for label in data_sub.obs[label_key].unique():
        obs_names.extend(
            data_sub.obs[data_sub.obs[label_key] == label]
            .sample(min_class_s, replace=False, random_state=seed)
            .index.values
        )
    data_sub = data_sub[obs_names, :]
    return data_sub


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
    data,
    gt_col,
    use_rep="X_umap",
    key_added="ch_score",
    class_balance=False,
    inplace=True,
    seed=42,
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
    if class_balance:
        min_class_size = data_sub.obs[gt_col].value_counts().min()
        if min_class_size < 10:
            warnings.warn(
                "Smallest class has less than 10 samples, this might not yield a stable score."
            )
        obs_names = []
        for label in data_sub.obs[gt_col].unique():
            obs_names.extend(
                data_sub.obs[data_sub.obs[gt_col] == label]
                .sample(min_class_size, replace=False, random_state=seed)
                .index.values
            )
        data_sub = data_sub[obs_names, :]
    ch = sklearn.metrics.calinski_harabasz_score(data_sub.obsm[use_rep], data_sub.obs[gt_col])
    if inplace:
        data.uns[key_added] = ch
    else:
        return ch


def qsep_score(
    data: AnnData,
    label_key: str,
    use_rep: str = "X",
    distance_key: str = "full_distances",
    inplace: bool = True,
) -> None | np.ndarray:
    """Calculate QSep scores for spatial proteomics data.

    Parameters
    ----------
    data : AnnData
        Annotated data matrix.
    label_key : str
        Key in data.obs containing cluster labels.
    use_rep : str, optional
        Key for representation to use for distance calculation.
        Either 'X' or a key in data.obsm. Defaults to 'X'.
    distance_key : str, optional
        Key under which to store the full distances in data.obs.
        Defaults to 'full_distances'.
    inplace : bool, optional
        If True, store results in data, else return matrices.
        Defaults to True.

    Returns
    -------
    None or np.ndarray
        If inplace=True, returns None and stores results in data.
        If inplace=False, returns cluster_distances.
    """
    # Get data matrix
    if use_rep == "X":
        X = data.X
    else:
        X = data.obsm[use_rep]

    # Calculate pairwise distances between all points
    full_distances = sklearn.metrics.pairwise_distances(X)

    # Get valid clusters (non-NA)
    mask = data.obs[label_key].notna()
    valid_clusters = data.obs[label_key][mask].unique()

    # Calculate cluster distances
    cluster_distances = np.zeros((len(valid_clusters), len(valid_clusters)))
    cluster_indices = {
        cluster: np.where(data.obs[label_key] == cluster)[0] for cluster in valid_clusters
    }

    for i, cluster1 in enumerate(valid_clusters):
        for j in range(i, len(valid_clusters)):
            # for j, cluster2 in enumerate(valid_clusters[i + 1 :]):
            cluster2 = valid_clusters[j]
            idx1 = cluster_indices[cluster1]
            idx2 = cluster_indices[cluster2]

            # Get submatrix of distances between clusters
            submatrix = full_distances[np.ix_(idx1, idx2)]
            cluster_distances[i, j] = np.mean(submatrix)
            cluster_distances[j, i] = np.mean(submatrix)

    if inplace:
        # Store full distances
        data.obs[distance_key] = pd.Series(
            np.mean(full_distances, axis=1), index=data.obs.index
        )

        # Store cluster distances and metadata
        data.uns["cluster_distances"] = {
            "distances": cluster_distances,
            "clusters": valid_clusters.tolist(),
        }
    else:
        return cluster_distances
