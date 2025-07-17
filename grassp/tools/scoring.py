from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from anndata import AnnData

import warnings

import numpy as np
import pandas as pd
import itertools
import seaborn as sns
import anndata as ad
import sklearn.metrics
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict


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
    gt_col: str,
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


def separability_auc(data,
                label_col="consensus_graph_annnotation",
                coord_cols=["umap_1", "umap_2"],
                C_margin=1.0,
                auc_model="svm",
                svm_kernel="rbf",
                C_auc=1.0,
                cv_auc=5,
                heatmap_size=(12,11),
                inplace=True):

    """
    Compute pair-wise SVM AUC between all label pairs.

    Parameters
    ----------
    data : DataFrame or AnnData 
    label_col : str 
        class labels (y in the classifier) 
        if AnnData, then use .obs[label_col]
        if DataFrame, then use column name as label
    coord_cols : list of str
        coordinates (X in the classifier)
        if AnnData, use .obsm[coord_cols] if coord_cols is a *str*, and .var[coord_cols] if *list*
        if DataFrame, then interpret as list of column names
    auc_model : {"lr", "svm"}
        Which estimator to use for AUC calculation.
    svm_kernel : str
        Kernel for SVC when auc_model == "svm" ("linear", "rbf", …).
    C_margin : float
        Soft-margin parameter for the *margin* LinearSVC.
    C_auc : float
        Regularisation strength for the AUC classifier (LR or SVM).
    cluster_heatmap : bool
        Whether to cluster rows/columns in the AUC heatmap.
    display_heatmaps : bool
        Whether to display the heatmaps.
    heatmap_size : tuple
        Size of the heatmaps.

    Returns
    -------
    margin_mat : DataFrame
        Pair-wise margin matrix
    auc_mat : DataFrame
        Pair-wise AUC matrix
    figures: matplotlib.pyplot.Figure
    """
    #------------------------------------------------------------------
    # construct dataframe if input is anndata
    #------------------------------------------------------------------
    if isinstance(data, ad.AnnData):
        assert(label_col in data.obs.columns), f"label_col {label_col} not in data.obs.columns"
        if isinstance(coord_cols, str):
            X_all = data.obsm[coord_cols]
            df = pd.DataFrame(X_all)
            coord_cols = df.columns.tolist() # update coord_cols to the actual column names
            df[label_col] = data.obs[label_col].tolist()
        elif isinstance(coord_cols, list):
            col_idx = data.var_names.get_indexer(coord_cols)
            X_all = data.X[:, col_idx]
            df = pd.DataFrame(X_all)
            coord_cols = df.columns.tolist() # update coord_cols to the actual column names
            df[label_col] = data.obs[label_col].tolist()
        else:
            raise ValueError(f"coord_cols must be a string or list, got {type(coord_cols)}")
    elif isinstance(data, pd.DataFrame):
        df = data
        if not isinstance(coord_cols, list):
            raise ValueError(f"coord_cols must be a list if data is a DataFrame, got {type(coord_cols)}")
        if label_col not in df.columns:
            raise ValueError(f"label_col {label_col} not in data.columns")
    else:
        raise ValueError(f"data must be an AnnData or DataFrame, got {type(data)}")

    # ------------------------------------------------------------------
    #  Basic prep
    # ------------------------------------------------------------------
    # Drop rows with missing coords or labels
    df = df.dropna(subset=[label_col, *coord_cols])

    X_all = df[list(coord_cols)].values
    y_all = df[label_col].values
    labels = pd.unique(y_all)         # handles mixed types safely
    k = len(labels)

    margin_mat = pd.DataFrame(
        np.zeros((k, k)), index=labels, columns=labels)
    auc_mat = pd.DataFrame(
        np.ones((k, k)), index=labels, columns=labels)

    # ------------------------------------------------------------------
    #  Pipelines
    # ------------------------------------------------------------------
    margin_svm = Pipeline([
        ("scale", StandardScaler()),
        ("svm",   LinearSVC(C=C_margin,
                            class_weight="balanced",
                            dual=False,
                            max_iter=20000,
                            random_state=0))
    ])

    if auc_model == "lr":
        auc_clf = Pipeline([
            ("scale", StandardScaler()),
            ("logit", LogisticRegression(
                C=C_auc,
                class_weight="balanced",
                max_iter=5000,
                solver="lbfgs",
                random_state=0))
        ])
    elif auc_model == "svm":
        auc_clf = Pipeline([
            ("scale", StandardScaler()),
            ("svc", SVC(
                kernel=svm_kernel,
                C=C_auc,
                probability=True,          # enables predict_proba
                class_weight="balanced",
                random_state=0))
        ])
    else:
        raise ValueError('auc_model must be "lr" or "svm"')

    # ------------------------------------------------------------------
    #  Pair-wise loop
    # ------------------------------------------------------------------
    for lab_i, lab_j in itertools.combinations(labels, 2):
        mask = (y_all == lab_i) | (y_all == lab_j)
        X_pair = X_all[mask]
        y_pair = (y_all[mask] == lab_j).astype(int)

        # ----- margin -----
        margin_svm.fit(X_pair, y_pair)
        w = margin_svm[-1].coef_.ravel()
        margin = 2.0 / np.linalg.norm(w)
        margin_mat.loc[lab_i, lab_j] = margin_mat.loc[lab_j, lab_i] = margin

        # ----- AUC -----
        prob_oof = cross_val_predict(
            auc_clf, X_pair, y_pair,
            cv=cv_auc, method="predict_proba")[:, 1]
        auc = roc_auc_score(y_pair, prob_oof)
        auc_mat.loc[lab_i, lab_j] = auc_mat.loc[lab_j, lab_i] = auc

    # ------------------------------------------------------------------
    #  Set diagonal to 1 for AUC matrix (self-comparison isn't meaningful)
    # ------------------------------------------------------------------
    np.fill_diagonal(auc_mat.values, 0.5)
    
    # ------------------------------------------------------------------
    #  visual summary
    # ------------------------------------------------------------------
    figures = {}
    # Create clustered AUC heatmap
    auc_clustermap = sns.clustermap(auc_mat, 
                                    square=True, 
                                    annot=True, 
                                    fmt=".2f",
                                    cmap="rocket", 
                                    vmin=0.5, 
                                    vmax=1,
                                    cbar_kws=dict(label=f"ROC-AUC ({auc_model.upper()})"),
                                    figsize=(heatmap_size[0], heatmap_size[1]))
    auc_clustermap.fig.suptitle("Label separability\nPair-wise classifier-AUC")
    auc_clustermap.ax_heatmap.set_xticklabels(
        auc_clustermap.ax_heatmap.get_xticklabels(), rotation=45, ha='right')
    figures['auc_fig'] = auc_clustermap
    
    # Get the clustered order for returning
    auc_mat = auc_mat.iloc[auc_clustermap.dendrogram_row.reordered_ind, 
                            auc_clustermap.dendrogram_col.reordered_ind]

    if inplace:
        data.uns[f"separability ({label_col})"] = {
            "auc_mat": auc_mat,
            "margin_mat": margin_mat,
            "figures": figures
        }
    else:
            return auc_mat, margin_mat, figures