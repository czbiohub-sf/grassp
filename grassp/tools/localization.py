from __future__ import annotations
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from anndata import AnnData
    from typing import List

import warnings

import numpy as np
import pandas as pd
import scipy.sparse as sp

from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.svm import SVC


def _get_knn_annotation_df(
    data: AnnData, obs_ann_col: str, exclude_category: str | List[str] | None = None
) -> pd.DataFrame:
    """
    Get a dataframe with a column of .obs repeated for each protein.
    """
    nrow = data.obs.shape[0]
    obs_ann = data.obs[obs_ann_col]
    if isinstance(exclude_category, str):
        exclude_category = [exclude_category]
    if exclude_category is not None:
        obs_ann.replace(exclude_category, np.nan, inplace=True)

    df = pd.DataFrame(np.tile(obs_ann, (nrow, 1)))
    return df


def _knn_annotation(
    data: AnnData,
    gt_col: str,
    class_balance: bool = True,
    obsp_key="connectivities",
    iterative: bool = False,
    max_iter: int = 30,
    tol: float = 1e-3,
    verbose: bool = True,
    fix_markers: bool = False,
    method: Literal["propagation", "spreading"] = "propagation",
    alpha: float = 0.8,
):
    """Helper function that does label propagation/spreading with fixed min_probability."""
    labels = data.obs[gt_col].astype("category")
    labels_one_hot = pd.get_dummies(labels).values
    if obsp_key == "distances":
        # Build a Gaussian RBF affinity W from the kNN distance graph, restricted
        # to the existing sparsity pattern. sigma defaults to the median nonzero
        # distance, which gives a data-driven kernel width that's narrower than
        # UMAP's fuzzy-union "connectivities" and thus produces more
        # boundary-localized smoothing in label spreading. The result is cached
        # at adata.obsp["W_spreading"].
        D = data.obsp[obsp_key]
        sigma = float(np.median(D.data)) if D.nnz > 0 else 1.0
        W = D.copy()
        W.data = np.exp(-(D.data**2) / (2.0 * sigma**2))
        data.obsp["W_spreading"] = W
        T = W
    else:
        T = data.obsp[obsp_key]
    if method == "spreading":
        # Label spreading (Zhou et al., 2003): build the symmetric normalized
        # operator S = D^(-1/2) W D^(-1/2) once, then iterate
        #     F(t+1) = alpha * S @ F(t) + (1 - alpha) * Y_0
        # alpha controls soft clamping: small alpha keeps seeds close to Y_0,
        # alpha -> 1 lets labeled rows drift freely.
        if not 0 <= alpha <= 1:
            raise ValueError(f"alpha must be in [0, 1), got {alpha}")
        d = np.asarray(T.sum(axis=1)).ravel()
        d_inv_sqrt = np.zeros_like(d, dtype=float)
        nz = d > 0
        d_inv_sqrt[nz] = 1.0 / np.sqrt(d[nz])
        D_inv_sqrt = sp.diags(d_inv_sqrt)
        S = D_inv_sqrt @ T @ D_inv_sqrt
        Y_0 = labels_one_hot.astype(float)
        Y = Y_0.copy()
        for i in range(max_iter):
            Y_new = alpha * np.asarray(S @ Y) + (1 - alpha) * Y_0
            diff = np.abs(Y_new - Y).sum()
            Y = Y_new
            if verbose:
                print(f"Diff: {diff:.3f}, Iteration {i} completed")
            if diff < tol:
                if verbose:
                    print(f"Diff: {diff:.3f}, Converged")
                break
        else:
            warnings.warn(
                f"knn_annotation: max_iter={max_iter} reached without convergence "
                f"(tol={tol})."
            )
    elif iterative:
        # Iterative propagation with hard clamping, mirroring
        # sklearn.semi_supervised.LabelPropagation: at each step propagate along T,
        # row-normalize, then reset labeled rows to their original one-hot. Stops when
        # |Y - Y_prev|.sum() < tol or after max_iter iterations.
        unlabeled = labels_one_hot.sum(axis=1) == 0
        labeled_oh = labels_one_hot.astype(float)
        Y = labeled_oh.copy()
        Y_prev = np.zeros_like(Y)
        for _ in range(max_iter):
            diff = np.abs(Y - Y_prev).sum()
            if diff < tol:
                if verbose:
                    print(f"Diff: {diff:.3f}, Converged")
                break
            Y_prev = Y
            Y = np.asarray(T @ Y)
            row_sums = Y.sum(axis=1)
            row_sums[row_sums == 0] = 1
            Y = Y / row_sums[:, None]
            if fix_markers:
                Y[~unlabeled] = labeled_oh[~unlabeled]
            if verbose:
                print(f"Diff: {diff:.3f}, Iteration {_} completed")
        else:
            warnings.warn(
                f"knn_annotation: max_iter={max_iter} reached without convergence "
                f"(tol={tol})."
            )
    else:
        # Single-step propagation along T
        Y = T @ labels_one_hot
    Y[Y.sum(axis=1) == 0] = 1 / Y.shape[1]

    # Class balance
    if class_balance:
        # gt_compartments with a lot of proteins are more likely to be in the neighborhood of a protein
        # Adjust probability based on the number of proteins in the compartment
        Y = Y / np.nansum(Y, axis=0) * labels_one_hot.sum(axis=0)
        #
    # Normalize the propagated labels to get probabilities
    if any(Y.sum(axis=1) == 0):
        print(Y[Y.sum(axis=1) == 0])
    Y = Y / np.nansum(Y, axis=1)[:, None]

    return Y, labels, labels_one_hot


def knn_annotation(
    data: AnnData,
    gt_col: str,
    fix_markers: bool = False,
    class_balance: bool = True,
    min_probability: float | None = None,
    plot_optimization: bool = True,
    inplace: bool = True,
    obsp_key="connectivities",
    key_added: str = "knn_annotation",
    iterative: bool = False,
    max_iter: int = 1000,
    tol: float = 1e-3,
    verbose: bool = True,
    method: Literal["propagation", "spreading"] = "propagation",
    alpha: float = 0.8,
):
    """Propagate categorical annotations along the *k*-NN graph.

    For each observation the function inspects its neighbourhood in
    ``adata.obsp[obsp_key]`` (generated by :func:`scanpy.pp.neighbors`) and
    calculates the a weighted probability for each label category.

    Parameters
    ----------
    data
        :class:`anndata.AnnData` with a populated neighbour graph (*distances*
        or *connectivities*).
    gt_col
        Observation column containing the *source* annotations to be
        propagated.
    fix_markers
        If ``True`` marker probabilities do not get overwritten by the propagated labels.
    class_balance
        If ``True`` ground truth compartments with a lot of proteins are downweighted proportional to their size to prevent them from dominating the propagated labels.
    min_probability
        If the probability of the most probable label is below this threshold, the label is set to ``np.nan``. If ``None`` (default), the threshold is automatically
        determine by the data. Specifically the threshold is chosen to maximize the F1 score for the given ground truth labels.
    plot_optimization
        If ``True`` a plot is shown showing the F1 score for different minimum probability thresholds.
    obsp_key
        Name of the neighbour connectivity graph to use (default ``"connectivities"``).
        If ``obsp_key="distances"`` is passed, a Gaussian RBF affinity
        ``W = exp(-d² / (2σ²))`` is built on the kNN distance graph (with σ
        the median nonzero distance) and used as the spreading/propagation
        operator. The resulting matrix is cached at ``adata.obsp["W_spreading"]``
        for inspection. This typically gives a narrower effective kernel than
        UMAP's fuzzy-union ``connectivities``, which is useful when you want
        boundary-localized uncertainty in :func:`label spreading <knn_annotation>`.
    key_added
        Name of the new column that will hold the propagated annotation
        (default ``"knn_annotation"``).
    iterative
        If ``True`` perform multi-step label propagation with hard clamping (in the
        style of :class:`sklearn.semi_supervised.LabelPropagation`). At every step
        the label distribution is propagated along ``T``, row-normalized, then
        labeled rows are reset to their initial one-hot encoding. Iteration stops
        when ``|Y - Y_prev|.sum() < tol`` or when ``max_iter`` is reached.
        If ``False`` (default) only a single propagation step is performed.
        Ignored when ``method="spreading"`` (spreading is always iterative).
    max_iter
        Maximum number of propagation iterations when ``iterative=True`` or
        ``method="spreading"`` (default 30).
    tol
        Convergence tolerance on the L1 change of the label distribution between
        consecutive iterations (default ``1e-3``).
    verbose
        If ``True`` print progress to the console.
    method
        Either ``"propagation"`` (default, Zhu & Ghahramani, 2002) or
        ``"spreading"`` (Zhou et al., 2003). Spreading uses the symmetric
        normalized operator ``S = D^{-1/2} W D^{-1/2}`` and a soft clamp
        controlled by ``alpha``, which makes it more robust to noisy seeds.
    alpha
        Soft-clamping parameter for label spreading, in ``[0, 1)``. The update
        rule is ``F(t+1) = alpha * S @ F(t) + (1 - alpha) * Y_0``: small
        ``alpha`` keeps predictions close to the initial seeds, ``alpha`` close
        to 1 lets labeled rows drift. Ignored when ``method="propagation"``.
        Default ``0.8`` (matches :class:`sklearn.semi_supervised.LabelSpreading`).


    Returns
    -------
    Modified anndata object with the following new entries:
    - .obsm[f"{key_added}_probabilities"] containing the propagated probabilities
    - .obs[f"{key_added}"] containing the propagated labels (most probable label)
    - .uns[f"{key_added}_colors"] to make sure plotting uses the same colors as the ground truth labels
    - .obs[f"{key_added}_probability"] containing the probability of the most probable label
    """

    if min_probability is None:
        min_probability = 0.0
    #     min_probabilities = np.linspace(0.5, 1, 100)
    #     f1 = []
    #     for prob in min_probabilities:
    #         Y, labels, labels_one_hot = _knn_annotation(
    #             data,
    #             gt_col=gt_col,
    #             class_balance=class_balance,
    #             obsp_key=obsp_key,
    #             iterative=iterative,
    #             max_iter=max_iter,
    #             tol=tol,
    #             verbose=verbose,
    #         )

    #         gt = data.obs[gt_col]
    #         pred = pd.Categorical(
    #             labels.cat.categories[Y.argmax(axis=1)],
    #             categories=labels.cat.categories,
    #             ordered=labels.cat.ordered,
    #         )
    #         pred[Y.max(axis=1) < prob] = np.nan
    #         mask = gt.notna() #& pred.notna()
    #         y_true_raw = gt[mask]
    #         y_pred_raw = pred[mask]
    #         cats = pd.Index(y_true_raw.unique()).union(pd.Index(y_pred_raw.unique())).dropna()
    #         y_true = pd.Categorical(y_true_raw, categories=cats).codes
    #         y_pred = pd.Categorical(y_pred_raw, categories=cats).codes
    #         f1.append(f1_score(y_true, y_pred, average="macro", labels=np.arange(len(cats))))
    #     min_probability = min_probabilities[np.argmax(f1)]

    # if plot_optimization:
    #     plt.plot(min_probabilities, f1, label="F1 score")
    #     plt.axvline(
    #         x=min_probability,
    #         color="red",
    #         linestyle="--",
    #         label=f"Optimal cutoff = {min_probability:.3f}",
    #     )
    #     plt.xlabel("Minimum probability cutoff")
    #     plt.ylabel("F1 score")
    #     plt.title("F1 score optimization")
    #     plt.legend()
    #     plt.show()

    Y, labels, labels_one_hot = _knn_annotation(
        data,
        gt_col=gt_col,
        class_balance=class_balance,
        obsp_key=obsp_key,
        iterative=iterative,
        max_iter=max_iter,
        tol=tol,
        verbose=verbose,
        fix_markers=fix_markers,
        method=method,
        alpha=alpha,
    )

    if fix_markers:
        # Pin marker rows to their original one-hot encoding after the
        # propagation/spreading + class_balance + row-normalize pipeline.
        # This guarantees marker probabilities are 1.0 for their seed class
        # regardless of method ("propagation"/"spreading") or iterative mode.
        marker_mask = labels_one_hot.sum(axis=1) == 1
        Y[marker_mask] = labels_one_hot[marker_mask].astype(float)

    if inplace:
        data.obsm[f"{key_added}_probabilities"] = Y
        data.obsm[f"{key_added}_one_hot_labels"] = labels_one_hot
        predicted = pd.Categorical(
            labels.cat.categories[Y.argmax(axis=1)],
            categories=labels.cat.categories,
            ordered=labels.cat.ordered,
        )
        data.obs[f"{key_added}"] = predicted
        data.obs[f"{key_added}_probability"] = np.max(Y, axis=1)
        data.obs.loc[
            data.obs[f"{key_added}_probability"] < min_probability, f"{key_added}"
        ] = np.nan
        if f"{gt_col}_colors" in data.uns:
            data.uns[f"{key_added}_colors"] = data.uns[f"{gt_col}_colors"]

    else:
        return {
            "probabilities": Y,
            "labels": labels.cat.categories,
            "one_hot_labels": labels_one_hot,
        }


def knn_annotation_old(
    data: AnnData,
    obs_ann_col: str,
    key_added: str = "consensus_graph_annotation",
    exclude_category: str | List[str] | None = None,
    inplace: bool = True,
) -> AnnData | None:
    """Propagate categorical annotations along the *k*-NN graph.

    For each observation the function inspects its neighbourhood in
    ``adata.obsp['distances']`` (generated by :func:`scanpy.pp.neighbors`) and
    assigns the majority category found in ``obs_ann_col``.  Ties are broken
    arbitrarily using :func:`pandas.DataFrame.mode`.

    Parameters
    ----------
    data
        :class:`anndata.AnnData` with a populated neighbour graph (*distances*
        or *connectivities*).
    obs_ann_col
        Observation column containing the *source* annotations to be
        propagated.
    key_added
        Name of the new column that will hold the *consensus* annotation
        (default ``"consensus_graph_annotation"``).
    exclude_category
        One or multiple category labels that should be ignored when computing
        the neighbourhood majority (useful for *unknown* / *NA* categories).
    inplace
        If ``True`` (default) modify *data* in place.  Otherwise return a
        copy with the additional column.

    Returns
    -------
    Modified object when ``inplace`` is ``False`` with a new column in .obs[key_added].
    """
    df = _get_knn_annotation_df(data, obs_ann_col, exclude_category)

    conn = data.obsp["distances"]
    mask = ~(conn != 0).todense()  # This avoids expensive conn == 0 for sparse matrices
    df[mask] = np.nan

    majority_cluster = df.mode(axis=1, dropna=True).loc[
        :, 0
    ]  # take the first if there are ties
    data.obs[key_added] = majority_cluster.values
    return data if not inplace else None


def svm_train(
    data: AnnData,
    gt_col: str,
    C_range: np.ndarray = np.array([0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16]),
    gamma_range: np.ndarray = np.array([0.001, 0.01, 0.1, 1, 10, 100]),
    class_weight: None | dict | Literal["balanced"] = "balanced",
    cv_splits: int = 5,
    cv_repeats: int = 20,
    n_jobs: int = -1,
    random_state: int | None = None,
    key_added: str = "svm",
    inplace: bool = True,
) -> dict | None:
    """Train SVM classifier with hyperparameter tuning using marker proteins.

    Performs grid search over C and gamma parameters using repeated stratified
    cross-validation. Best hyperparameters are stored in ``.uns`` for later use
    with :func:`svm_annotation`.

    Parameters
    ----------
    data
        :class:`anndata.AnnData` object with proteins as observations.
    gt_col
        Observation column containing marker annotations. Proteins with NaN
        values are considered unknown and excluded from training.
    C_range
        Array of C (regularization) values to search. Default: 2^-4 to 2^4.
    gamma_range
        Array of gamma (kernel coefficient) values. Default: 10^-3 to 10^2.
    cv_splits
        Number of cross-validation folds (default 5).
    cv_repeats
        Number of CV repetitions (default 20). Total fits per parameter
        combination: ``cv_splits × cv_repeats``.
    n_jobs
        Number of parallel jobs. -1 uses all available cores.
    random_state
        Random seed for reproducibility.
    key_added
        Key prefix for storing results in ``.uns`` (default ``"svm"``).
    inplace
        If ``True`` store results in ``.uns``; if ``False`` return grid search
        object and dictionary with best parameters and CV results. This can be
        useful if you want to inspect the grid search object or use the best
        parameters for other tasks.

    Returns
    -------
    None or dict
        If ``inplace=False``, returns dictionary with best parameters and CV
        results. Otherwise modifies ``data.uns[f"{key_added}.params"]`` in place.

    Examples
    --------
    >>> import grassp as gr
    >>> adata = gr.ds.hein_2024(enrichment="enriched")

    # When actually training, increase cv_repeats and cv_splits
    # We recommend >20 repeats with 5 splits
    >>> gr.tl.svm_train(adata, gt_col="hein2024_gt_component", cv_repeats=2, cv_splits=2, random_state=42)
    Fitting 4 folds for each of 54 candidates, totalling 216 fits
    >>> adata.uns["svm.params"]["best_params"]
    {'C': 2.0, 'gamma': 0.01}
    """
    # Validate gt_col exists
    if gt_col not in data.obs.columns:
        raise KeyError(f"Column '{gt_col}' not found in data.obs")

    # Extract markers (non-NaN proteins)
    marker_mask = data.obs[gt_col].notna()
    if not marker_mask.any():
        raise ValueError(f"No marker proteins found in '{gt_col}' (all values are NaN)")

    X_train = data.X[marker_mask]
    y_train = data.obs.loc[marker_mask, gt_col]

    # Check for sufficient classes
    if y_train.nunique() < 2:
        raise ValueError("Need at least 2 classes for SVM training")

    # Warn if too few samples per class for CV
    min_class_count = y_train.value_counts().min()
    if min_class_count < cv_splits:
        warnings.warn(
            f"Smallest class has {min_class_count} samples but cv_splits={cv_splits}. "
            "Consider reducing cv_splits or ensuring balanced marker representation."
        )

    # Configure cross-validation
    # 5-fold CV repeated 20 times = 100 fits per hyperparameter combo
    cv = RepeatedStratifiedKFold(
        n_splits=cv_splits, n_repeats=cv_repeats, random_state=random_state
    )

    # Create parameter grid
    param_grid = {
        'C': C_range,
        'gamma': gamma_range,
    }

    # Configure SVM
    svm = SVC(
        kernel='rbf',
        class_weight=class_weight,
        probability=True,  # Needed for svm_annotation
    )

    # Run grid search
    grid_search = GridSearchCV(
        estimator=svm,
        param_grid=param_grid,
        cv=cv,
        scoring='f1_macro',  # Balanced metric for multiclass
        n_jobs=n_jobs,
        verbose=1,
    )

    grid_search.fit(X_train, y_train.cat.codes)

    # Store results
    params = {
        "method": "SVM-RBF",
        "gt_col": gt_col,
        "best_params": {
            "C": float(grid_search.best_params_["C"]),
            "gamma": float(grid_search.best_params_["gamma"]),
        },
        "cv_results": {
            "best_score": float(grid_search.best_score_),
            "cv_splits": cv_splits,
            "cv_repeats": cv_repeats,
            "total_fits": cv_splits * cv_repeats,
            "scoring": "f1_macro",
        },
        "search_space": {
            "C_range": C_range.tolist(),
            "gamma_range": gamma_range.tolist(),
        },
        "class_weight": class_weight,
        "class_labels": y_train.cat.categories.tolist(),
        "n_markers": int(X_train.shape[0]),
        "random_state": random_state,
        "n_jobs": n_jobs,
        "kernel": "rbf",
        "cv_splits": cv_splits,
        "cv_repeats": cv_repeats,
    }

    if inplace:
        data.uns[f"{key_added}.params"] = params
        return None
    else:
        return grid_search, params


def svm_annotation(
    data: AnnData,
    gt_col: str = "markers",
    C: float | None = None,
    gamma: float | str | None = None,
    fix_markers: bool = False,
    min_probability: float = 0.5,
    inplace: bool = True,
    key_added: str = "svm_annotation",
    params_key: str | None = None,
    class_weight: None | dict | Literal["balanced"] = "balanced",
) -> dict | None:
    """Classify proteins using SVM with marker-based training.

    Trains an SVM classifier on marker proteins (non-NaN values in ``gt_col``)
    and predicts localization for all proteins. Hyperparameters can be provided
    manually or loaded from prior :func:`svm_train` call.

    Similar to :func:`knn_annotation` but uses SVM instead of graph propagation.

    Parameters
    ----------
    data
        :class:`anndata.AnnData` with feature matrix in ``.X``.
    gt_col
        Observation column with marker labels (NaN for unknowns).
    C
        SVM regularization parameter. If ``None``, loads from ``.uns``.
    gamma
        RBF kernel coefficient. If ``None``, loads from ``.uns``.
    fix_markers
        If ``True`` marker proteins retain their original labels with
        probability 1.0.
    min_probability
        Confidence threshold; predictions below this are set to NaN.
    inplace
        If ``True`` modify data in place; else return dict.
    key_added
        Base name for results (default ``"svm_annotation"``).
    params_key
        Key to load hyperparameters from ``.uns`` (default ``"svm.params"``).

    Returns
    -------
    None or dict
        If ``inplace=True``, modifies data with:

        - ``.obs[f"{key_added}"]``: Predicted labels
        - ``.obs[f"{key_added}_probability"]``: Max probability per protein
        - ``.obsm[f"{key_added}_probabilities"]``: Full probability matrix
        - ``.uns[f"{key_added}_colors"]``: Color scheme (copied from ``gt_col``)

        If ``inplace=False``, returns dict with predictions and probabilities.

    Raises
    ------
    ValueError
        If no hyperparameters found and none provided manually.
    KeyError
        If ``gt_col`` not found in ``.obs``.

    Examples
    --------
    >>> import grassp as gr
    >>> import scanpy as sc
    >>> adata = gr.ds.hein_2024(enrichment="enriched")

    ##### Option 1: Annotate directly, with fixed hyperparameters #####
    >>> gr.tl.svm_annotation(
    ...     adata,
    ...     gt_col="hein2024_gt_component",
    ...     min_probability=0.5,
    ...     C=10,
    ...     gamma=0.01,
    ... )
    >>> sc.pl.umap(adata, color="svm_annotation") # doctest: +SKIP

    ##### Option 2: Train SVM hyperparameters, then annotate #####
    # When actually training, increase cv_repeats and cv_splits
    # We recommend >20 repeats with 5 splits
    >>> gr.tl.svm_train(adata, gt_col="hein2024_gt_component", cv_repeats=2, cv_splits=2, random_state=42)
    Fitting 4 folds for each of 54 candidates, totalling 216 fits
    >>> adata.uns["svm.params"]["best_params"]
    {'C': 2.0, 'gamma': 0.01}
    >>> gr.tl.svm_annotation(adata, gt_col="hein2024_gt_component", min_probability=0.5)
    >>> sc.pl.umap(adata, color="svm_annotation") # doctest: +SKIP
    """
    # If hyperparameters not provided, try loading from .uns
    if C is None or gamma is None:
        if params_key is None:
            params_key = "svm.params"

        if params_key not in data.uns:
            raise ValueError(
                f"No hyperparameters found in data.uns['{params_key}']. "
                "Either:\n"
                "  1) Run svm_train() first to tune hyperparameters, or\n"
                "  2) Provide C and gamma explicitly (e.g., C=1.0, gamma=0.1)"
            )

        stored_params = data.uns[params_key]
        if C is None:
            C = stored_params["best_params"]["C"]
        if gamma is None:
            gamma = stored_params["best_params"]["gamma"]
        if class_weight is None:
            class_weight = stored_params["class_weight"]

    # Validate gt_col
    if gt_col not in data.obs.columns:
        raise KeyError(f"Column '{gt_col}' not found in data.obs")

    # Extract markers
    marker_mask = data.obs[gt_col].notna()
    if not marker_mask.any():
        raise ValueError(f"No marker proteins found in '{gt_col}'")

    X_train = data.X[marker_mask]
    y_train = data.obs.loc[marker_mask, gt_col]
    X_all = data.X

    categories = y_train.cat.categories
    y_train_codes = y_train.cat.codes

    # Train SVM
    svm = SVC(
        C=C,
        gamma=gamma,
        kernel='rbf',
        class_weight=class_weight,
        probability=True,
        random_state=42,
    )
    svm.fit(X_train, y_train_codes)

    # Get probability matrix (n_proteins, n_classes)
    probabilities = svm.predict_proba(X_all)

    # Get predictions (argmax)
    pred_codes = np.argmax(probabilities, axis=1)
    pred_labels = categories[pred_codes].to_numpy()

    # Get max probability
    max_prob = np.max(probabilities, axis=1)

    # Handle fix_markers
    if fix_markers:
        # Set marker probabilities to 1.0 for their true class
        marker_indices = np.where(marker_mask)[0]
        for idx, label in zip(marker_indices, y_train):
            label_idx = categories.get_loc(label)
            probabilities[idx, :] = 0.0
            probabilities[idx, label_idx] = 1.0
            pred_labels[idx] = label
            max_prob[idx] = 1.0

    # Apply probability threshold
    low_conf_mask = max_prob < min_probability
    pred_labels[low_conf_mask] = np.nan

    # Store results
    if inplace:
        # Probabilities matrix
        data.obsm[f"{key_added}_probabilities"] = probabilities

        # Predicted labels (categorical)
        data.obs[f"{key_added}"] = pd.Categorical(pred_labels, categories=categories)

        # Max probability
        data.obs[f"{key_added}_probability"] = max_prob

        # Copy colors if available
        if f"{gt_col}_colors" in data.uns:
            data.uns[f"{key_added}_colors"] = data.uns[f"{gt_col}_colors"]

        # Store metadata
        data.uns[f"{key_added}_params"] = {
            "method": "SVM-RBF",
            "C": C,
            "gamma": gamma,
            "gt_col": gt_col,
            "fix_markers": fix_markers,
            "min_probability": min_probability,
        }

        return None
    else:
        return {
            "probabilities": probabilities,
            "labels": pred_labels,
            "max_probability": max_prob,
            "categories": categories,
        }


def prune_markers_knn(
    adata: AnnData, gt_col: str, key_added: str | None = None, min_probability: float = 0.9
) -> AnnData:
    """Remove "outliers" from marker proteins whose compartment label is not supported by their k-NN neighbourhood.

    Runs :func:`knn_annotation` on the existing markers and retains only those
    whose neighbours confidently predict the same compartment label. Markers
    whose predicted label disagrees with their annotated label, or whose
    neighbourhood confidence falls below ``min_probability``, are set to NaN in
    the output column. Non-marker proteins (NaN in ``gt_col``) are always set
    to NaN.

    This is useful for cleaning noisy or misannotated training sets before
    semi-supervised classification.

    Parameters
    ----------
    adata
        :class:`anndata.AnnData` with a populated neighbour graph
        (``adata.obsp["connectivities"]``).
    gt_col
        ``.obs`` column containing the ground-truth compartment labels.
        Proteins with NaN values are treated as unannotated.
    key_added
        ``.obs`` column to write the pruned labels to. Defaults to
        ``"{gt_col}_pruned"``.
    min_probability
        Minimum k-NN probability for a marker to be considered
        neighbourhood-consistent. Markers below this threshold are removed.
        Increasing this value will remove more markers, leading to more consistent but sparser annotations.
        Default is ``0.9``.

    Returns
    -------
    Modifies ``adata.obs[key_added]`` in place: retained markers keep their
    original label; removed markers are set to NaN.
    """
    key_added = key_added or f"{gt_col}_pruned"
    knnres = knn_annotation(
        adata, gt_col, min_probability=min_probability, inplace=False, fix_markers=False
    )
    labels, Y = knnres["labels"], knnres["probabilities"]

    predicted = pd.Categorical(
        labels[Y.argmax(axis=1)],
        categories=adata.obs[gt_col].cat.categories,
        ordered=adata.obs[gt_col].cat.ordered,
    )
    adata.obs[key_added] = predicted
    adata.obs.loc[adata.obs[gt_col].isna(), key_added] = np.nan  # Remove non-markers
    adata.obs.loc[adata.obs[gt_col] != adata.obs[key_added], key_added] = (
        np.nan
    )  # Remove non-predictable markers
