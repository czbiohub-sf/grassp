from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from anndata import AnnData

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy
import seaborn as sns
import sklearn.metrics

from .clustering import knn_annotation


def class_balance(
    data: AnnData, label_key: str, min_class_size: int = 10, seed: int = 42
) -> AnnData:
    """Return a balanced subset with equally sized clusters.

    Samples the same number of observations from each category in
    ``data.obs[label_key]`` (the size is determined by the smallest class).

    Parameters
    ----------
    data
        Input :class:`~anndata.AnnData`.
    label_key
        Observation column with cluster or class labels.
    min_class_size
        Raise an error if the smallest class contains fewer than this
        number of observations (default ``10``).
    seed
        Random seed for reproducible sampling.

    Returns
    -------
    An AnnData *view* containing the balanced subset.
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
    """Per-group silhouette scores.

    Computes the silhouette score for each group in ``data.obs[gt_col]``.

    Parameters
    ----------
    data
        AnnData object containing an embedding in ``.obsm``.
    gt_col
        Column in ``data.obs`` with cluster labels.
    use_rep
        Key of the embedding to evaluate (default ``"X_umap"``).
    key_added
        Base key under which results are stored (default ``"silhouette"``).
    inplace
        If ``True`` (default) store return ``None``, if ``False`` return the silhouette scores.


    Returns
    -------
    If ``inplace`` is ``True``:
        ``data.obs[key_added]``
            Vector of silhouette scores.
        ``data.uns[key_added]['mean_silhouette_score']``
            Global mean.
        ``data.uns[key_added]['cluster_mean_silhouette']``
            Mapping of cluster → mean score.
    If ``inplace`` is ``False``:
        Vector of silhouette scores.
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
    """Calinski–Harabasz score of cluster compactness vs separation.

    Parameters
    ----------
    data
        AnnData with an embedding under ``.obsm[use_rep]``.
    gt_col
        Observation column containing cluster assignments.
    use_rep
        Name of embedding to use (default ``"X_umap"``).
    key_added
        Key under which to store the score when ``inplace`` is ``True``.
    class_balance
        If ``True`` subsample each cluster to equal size before computing the
        score (calls ``class_balance`` internally).
    inplace, seed
        Standard behaviour flags.

    Returns
    -------
    If ``inplace`` is ``True``:
        ``data.uns[key_added]``
            Score.
    If ``inplace`` is ``False``:
        Score.
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
    gt_col: str,
    use_rep: str = "X",
    distance_key: str = "full_distances",
    inplace: bool = True,
) -> None | np.ndarray:
    """QSep cluster-separation metric for spatial proteomics.

    Implements the *QSep* statistic from Gatto *et al.* (2014) which
    measures within- vs between-cluster distances.

    Parameters
    ----------
    data
        AnnData object.
    gt_col
        Observation column with ground-truth cluster labels.
    use_rep
        Representation used for distance computation – ``"X"`` or a key in
        ``data.obsm``.
    distance_key
        Column name to store per-protein mean distances (only when
        ``inplace`` is ``True``).
    inplace
        Control write-back vs return behaviour.

    Returns
    -------
    If ``inplace`` is ``True``:
        ``None``
    If ``inplace`` is ``False``:
        ``cluster_distances``
    """
    # Get data matrix
    if use_rep == "X":
        X = data.X
    else:
        X = data.obsm[use_rep]

    # Calculate pairwise distances between all points
    full_distances = sklearn.metrics.pairwise_distances(X)

    # Get valid clusters (non-NA)
    mask = data.obs[gt_col].notna()
    valid_clusters = data.obs[gt_col][mask].unique()

    # Calculate cluster distances
    cluster_distances = np.zeros((len(valid_clusters), len(valid_clusters)))
    cluster_indices = {
        cluster: np.where(data.obs[gt_col] == cluster)[0] for cluster in valid_clusters
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


def knn_f1_score(data, gt_col, pred_col=None, weights=None, average="macro"):
    """F1 score.

    Parameters
    ----------
    data
        AnnData object.
    gt_col
        Observation column with ground-truth labels.
    pred_col
        Observation column with predicted labels.
    weights
        Weights for the F1 score.
    average
        Average method for the F1 score. If ``None`` the score for each label is returned.

    Returns
    -------
    F1 score.
    """
    if pred_col is None:
        knnres = knn_annotation(data, gt_col, inplace=False, min_probability=0)
        pred = knnres["labels"][knnres["probabilities"].argmax(axis=1)]
    else:
        pred = data.obs[pred_col]

    gt = data.obs[gt_col]
    mask = gt.notna() & pred.notna()
    y_true_raw = gt[mask]
    y_pred_raw = pred[mask]

    if weights is not None:
        labels = list(weights.keys())
        cats = (
            pd.Index(labels)
            .union(pd.Index(y_true_raw.unique()))
            .union(pd.Index(y_pred_raw.unique()))
        )
        y_true = pd.Categorical(y_true_raw, categories=cats).codes
        y_pred = pd.Categorical(y_pred_raw, categories=cats).codes
        label_idx = [cats.get_loc(label) for label in labels]
        f1_arr = sklearn.metrics.f1_score(y_true, y_pred, average=None, labels=label_idx)
        w = np.asarray([weights[label] for label in labels], float)
        w /= w.sum() if w.sum() > 0 else 1.0
        if average is None:
            return f1_arr * w
        return float((f1_arr * w).sum())
    else:
        cats = pd.Index(y_true_raw.unique()).union(pd.Index(y_pred_raw.unique()))
        y_true = pd.Categorical(y_true_raw, categories=cats).codes
        y_pred = pd.Categorical(y_pred_raw, categories=cats).codes
        return sklearn.metrics.f1_score(y_true, y_pred, average=average)


def knn_confusion_matrix(data, gt_col, pred_col=None, soft=False, cluster=False, plot=True):
    """Plot the confusion matrix of KNN-predicted versus ground-truth labels.

    Parameters
    ----------
    data
        AnnData object.
    gt_col
        Observation column with ground-truth labels.
    pred_col
        Observation column with predicted labels. If None, KNN annotation is computed.
    soft
        If True, use probabilistic (soft) confusion matrix instead of hard assignments.
    plot=True
        If True, plot the heatmap of the confusion matrix, otherwise return the confusion matrix.
    cluster
        If True, reorder matrix rows and columns via hierarchical clustering for visualization.

    Returns
    -------
    None. Displays a heatmap of the confusion matrix if plot is True, otherwise returns the confusion matrix.
    """

    # Notation: n observations, g ground truth label classes
    if pred_col is None:
        knnres = knn_annotation(data, gt_col, inplace=False, min_probability=0)
    else:
        knnres = {
            "probabilities": data.obsm[f"{pred_col}_probabilities"],
            "labels": data.obs[pred_col],
            "one_hot_labels": data.obsm[f"{pred_col}_one_hot_labels"],
        }

    if soft:
        M = knnres["one_hot_labels"].T @ knnres["probabilities"]  # shape (g, g)
        # Row-normalize to get fractions (each row sums to 1)
        cm = M / M.sum(axis=1, keepdims=True)
        cm = np.nan_to_num(cm, nan=0.0)
        labels = list(knnres["labels"])  # consistent label order with matrix axes
    else:
        gt = data.obs[gt_col]
        pred = knnres["labels"][knnres["probabilities"].argmax(axis=1)]
        mask = gt.notna() & pred.notna()
        y_true_raw = gt[mask]
        y_pred_raw = pred[mask]
        # Ensure we know the label order used in the confusion matrix
        labels = list(pd.Index(y_true_raw.unique()).union(pd.Index(y_pred_raw.unique())))
        cm = sklearn.metrics.confusion_matrix(y_true_raw, y_pred_raw, labels=labels)
        cm = cm / cm.sum(axis=1, keepdims=True)

    if not plot:
        return cm

    plt.figure(figsize=(10, 10))
    # Derive a single ordering for both axes so labels align and high values
    # lie near the diagonal. Use hierarchical clustering on a symmetrized matrix.

    if cluster:
        sym = (cm + cm.T) / 2.0
        order = scipy.cluster.hierarchy.leaves_list(
            scipy.cluster.hierarchy.linkage(1 - sym, method="ward")
        )

        cm_ordered = cm[np.ix_(order, order)]
        ordered_labels = [labels[i] for i in order]
    else:
        ordered_labels = labels
        cm_ordered = cm

    sns.heatmap(
        cm_ordered,
        annot=True,
        fmt=".2f",
        cmap="rocket_r",
        cbar=True,
        vmax=1,
        vmin=0,
        linewidths=0.5,
        square=True,
        xticklabels=ordered_labels,
        yticklabels=ordered_labels,
    )
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(f"{'Soft' if soft else 'Hard'} confusion matrix from knn annotation")
    plt.tight_layout()
    plt.show()
