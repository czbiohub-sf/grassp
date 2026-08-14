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

from ..util import get_matrix, set_matrix


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


def _propagate_soft(
    T,
    seed: np.ndarray,
    class_balance: bool = True,
    method: Literal["propagation", "spreading"] = "propagation",
    iterative: bool = False,
    alpha: float = 0.8,
    max_iter: int = 30,
    tol: float = 1e-3,
    verbose: bool = False,
    fix_markers: bool = False,
):
    """Propagate a (n_obs, n_categories) seed matrix over the affinity operator ``T``.

    This is the shared propagation core used by :func:`_competitive_propagation` and by the
    permutation null in :func:`resolve_soft_labels`. It is agnostic to whether ``seed``
    is a one-hot encoding or an arbitrary non-negative soft-label matrix, so the null
    re-propagation is byte-identical to production.

    ``T`` is the affinity operator (e.g. ``adata.obsp["connectivities"]``, or the RBF
    matrix built from distances by the caller). Returns the row-normalized propagated
    probability matrix ``Y``.
    """
    seed = np.asarray(seed, dtype=float)
    if method == "spreading":
        # Label spreading (Zhou et al., 2003): S = D^(-1/2) W D^(-1/2), then iterate
        # F(t+1) = alpha * S @ F(t) + (1 - alpha) * Y_0.
        if not 0 <= alpha <= 1:
            raise ValueError(f"alpha must be in [0, 1), got {alpha}")
        d = np.asarray(T.sum(axis=1)).ravel()
        d_inv_sqrt = np.zeros_like(d, dtype=float)
        nz = d > 0
        d_inv_sqrt[nz] = 1.0 / np.sqrt(d[nz])
        D_inv_sqrt = sp.diags(d_inv_sqrt)
        S = D_inv_sqrt @ T @ D_inv_sqrt
        Y_0 = seed
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
                f"competitive_propagation: max_iter={max_iter} reached without convergence "
                f"(tol={tol})."
            )
    elif iterative:
        # Iterative propagation with hard clamping, mirroring
        # sklearn.semi_supervised.LabelPropagation.
        unlabeled = seed.sum(axis=1) == 0
        labeled_oh = seed
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
                f"competitive_propagation: max_iter={max_iter} reached without convergence "
                f"(tol={tol})."
            )
    else:
        # Single-step propagation along T
        Y = np.asarray(T @ seed)
    Y[Y.sum(axis=1) == 0] = 1 / Y.shape[1]

    # Class balance: down-weight large seed classes so they don't dominate simply by
    # being numerous in neighbourhoods. Guard against categories with zero total
    # propagated mass (e.g. an all-zero `unknown` column when every cluster is
    # confidently annotated) — those columns must stay zero, not become NaN.
    if class_balance:
        col_mass = np.nansum(Y, axis=0)
        scale = np.divide(
            seed.sum(axis=0),
            col_mass,
            out=np.zeros_like(col_mass, dtype=float),
            where=col_mass > 0,
        )
        Y = Y * scale
    # Normalize the propagated labels to get probabilities
    row = np.nansum(Y, axis=1)
    Y = np.divide(Y, row[:, None], out=np.zeros_like(Y), where=row[:, None] > 0)
    return Y


def _competitive_propagation(
    data: AnnData,
    gt_col: str | None,
    class_balance: bool = True,
    obsp_key="connectivities",
    iterative: bool = False,
    max_iter: int = 30,
    tol: float = 1e-3,
    verbose: bool = True,
    fix_markers: bool = False,
    method: Literal["propagation", "spreading"] = "propagation",
    alpha: float = 0.8,
    seed_matrix: np.ndarray | None = None,
    seed_categories: list | None = None,
):
    """Helper function that does label propagation/spreading with fixed min_probability."""
    if seed_matrix is not None:
        # Soft-seed mode: use a caller-supplied (n_obs, n_categories) row-stochastic
        # (or non-negative) seed matrix instead of a one-hot encoding of gt_col.
        # The propagation math below is agnostic to whether the seed is one-hot,
        # so only the seed construction differs.
        if seed_categories is None:
            raise ValueError("seed_categories must be provided when seed_matrix is given.")
        labels_one_hot = np.asarray(seed_matrix, dtype=float)
        if labels_one_hot.shape[1] != len(seed_categories):
            raise ValueError(
                f"seed_matrix has {labels_one_hot.shape[1]} columns but "
                f"{len(seed_categories)} seed_categories were provided."
            )
        # Build a categorical whose categories match the seed columns so the
        # downstream argmax-to-label / colour-mapping code works unchanged.
        seed_categories = list(seed_categories)
        labels = pd.Series(
            pd.Categorical(
                np.take(seed_categories, labels_one_hot.argmax(axis=1)),
                categories=seed_categories,
            ),
            index=data.obs_names,
        )
    else:
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

    Y = _propagate_soft(
        T,
        labels_one_hot,
        class_balance=class_balance,
        method=method,
        iterative=iterative,
        alpha=alpha,
        max_iter=max_iter,
        tol=tol,
        verbose=verbose,
        fix_markers=fix_markers,
    )

    return Y, labels, labels_one_hot


def competitive_propagation(
    data: AnnData,
    gt_col: str | None = None,
    fix_markers: bool = False,
    class_balance: bool = True,
    min_probability: float | None = None,
    plot_optimization: bool = True,
    inplace: bool = True,
    obsp_key="connectivities",
    key_added: str = "competitive_propagation",
    iterative: bool = False,
    max_iter: int = 1000,
    tol: float = 1e-3,
    verbose: bool = True,
    method: Literal["propagation", "spreading"] = "propagation",
    alpha: float = 0.8,
    seed_obsm_key: str | None = None,
    seed_categories_uns_key: str | None = None,
    unknown_label: str | None = "unknown",
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
        boundary-localized uncertainty in :func:`label spreading <competitive_propagation>`.
    key_added
        Name of the new column that will hold the propagated annotation
        (default ``"competitive_propagation"``).
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
    seed_obsm_key
        If given, seed the propagation with a *soft* per-observation label
        distribution stored in ``data.obsm[seed_obsm_key]`` (shape
        ``(n_obs, n_categories)``) instead of a one-hot encoding of ``gt_col``.
        Use this to propagate enrichment uncertainty produced by
        :func:`~grassp.tl.enrichment_to_cluster_distribution` /
        :func:`~grassp.tl.soft_cluster_annotation`. ``gt_col`` becomes optional
        when this is set, and ``fix_markers`` is disabled (its one-hot marker
        test does not apply to soft seeds).
    seed_categories_uns_key
        Name of the ``data.uns`` entry holding the ordered list of category
        names matching the columns of the soft seed matrix. Optional when the
        seed is a :class:`~pandas.DataFrame`, whose own column names are used
        instead; required when it is a bare ndarray, as written by grassp
        before labelled ``obsm`` matrices were introduced.
    unknown_label
        Name of the background/unknown category in the soft seed. Observations
        whose most probable label is this category are reported as unassigned
        (``NaN``) in ``data.obs[key_added]`` while the full probability matrix
        (including the unknown column) is kept in ``obsm``. Set to ``None`` to
        keep the unknown label as a regular category. Only used with soft seeds.


    Returns
    -------
    Modified anndata object with the following new entries:
    - .obsm[f"{key_added}_probabilities"] containing the propagated probabilities
    - .obs[f"{key_added}"] containing the propagated labels (most probable label)
    - .uns[f"{key_added}_colors"] to make sure plotting uses the same colors as the ground truth labels
    - .obs[f"{key_added}_probability"] containing the probability of the most probable label
    """

    # Resolve an optional soft seed. When provided, the propagation is seeded
    # with a per-observation probability distribution over `seed_categories`
    # instead of a one-hot encoding of `gt_col`.
    seed_matrix = None
    seed_categories = None
    if seed_obsm_key is not None:
        if seed_obsm_key not in data.obsm:
            raise KeyError(f"seed_obsm_key '{seed_obsm_key}' not found in data.obsm.")
        seed_matrix, seed_categories = get_matrix(data, seed_obsm_key)
        seed_matrix = seed_matrix.astype(float)
        # An explicit uns key still wins, for callers that pass one and for seeds written
        # as bare ndarrays by older versions. Otherwise the seed's own column names say
        # what its classes are -- which is the point of storing it as a DataFrame.
        if seed_categories_uns_key is not None:
            if seed_categories_uns_key not in data.uns:
                raise KeyError(
                    f"seed_categories_uns_key '{seed_categories_uns_key}' not found in data.uns."
                )
            seed_categories = list(data.uns[seed_categories_uns_key])
        elif seed_categories is None:
            raise ValueError(
                "seed_categories_uns_key must be provided when seed_obsm_key is set and "
                f"data.obsm['{seed_obsm_key}'] carries no column names."
            )
        if fix_markers:
            warnings.warn(
                "fix_markers is ignored when seeding competitive_propagation with a soft "
                "seed_matrix (its one-hot marker test does not apply to soft seeds)."
            )
            fix_markers = False
    elif gt_col is None:
        raise ValueError("Either gt_col or seed_obsm_key must be provided.")

    if min_probability is None:
        min_probability = 0.0
    #     min_probabilities = np.linspace(0.5, 1, 100)
    #     f1 = []
    #     for prob in min_probabilities:
    #         Y, labels, labels_one_hot = _competitive_propagation(
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

    Y, labels, labels_one_hot = _competitive_propagation(
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
        seed_matrix=seed_matrix,
        seed_categories=seed_categories,
    )

    if fix_markers:
        # Pin marker rows to their original one-hot encoding after the
        # propagation/spreading + class_balance + row-normalize pipeline.
        # This guarantees marker probabilities are 1.0 for their seed class
        # regardless of method ("propagation"/"spreading") or iterative mode.
        marker_mask = labels_one_hot.sum(axis=1) == 1
        Y[marker_mask] = labels_one_hot[marker_mask].astype(float)

    if inplace:
        set_matrix(data, f"{key_added}_probabilities", Y, labels.cat.categories)
        set_matrix(data, f"{key_added}_one_hot_labels", labels_one_hot, labels.cat.categories)
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
        # When seeding with a soft distribution that carries an explicit
        # background/unknown class, treat proteins whose most probable label is
        # that class as unassigned (NaN) while keeping the full probability
        # matrix (including the unknown column) in obsm.
        if seed_matrix is not None and unknown_label is not None:
            data.obs.loc[data.obs[f"{key_added}"] == unknown_label, f"{key_added}"] = np.nan
            data.obs[f"{key_added}"] = data.obs[f"{key_added}"].astype("category")
        if gt_col is not None and f"{gt_col}_colors" in data.uns:
            data.uns[f"{key_added}_colors"] = data.uns[f"{gt_col}_colors"]

    else:
        return {
            "probabilities": Y,
            "labels": labels.cat.categories,
            "one_hot_labels": labels_one_hot,
        }


def knn_annotation(*args, **kwargs):
    """Deprecated alias for :func:`competitive_propagation`.

    Renamed to pair with :func:`~grassp.tools.independent_diffusion`: the two graph
    annotation families are *competitive* propagation (mutually-exclusive labels, simplex
    output) and *independent* diffusion (overlapping/ontology labels, per-term output).
    """
    warnings.warn(
        "knn_annotation is deprecated and will be removed in a future release; "
        "use competitive_propagation instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return competitive_propagation(*args, **kwargs)


def soft_cluster_annotation(
    data: AnnData,
    enr_res: pd.DataFrame | None = None,
    cluster_key: str = "leiden",
    key_added: str = "soft_annotation",
    cluster_distribution: tuple[pd.DataFrame, list] | None = None,
    ranking_metric: Literal[
        "Adjusted P-value",
        "Adjusted P-value Bonferroni",
        "P-value",
    ] = "Adjusted P-value Bonferroni",
    threshold: float = 0.05,
    temperature: float = 1.0,
    s0: float = 0.0,
    s_max: float = 300.0,
    unknown_label: str | None = "unknown",
    weight_by: Literal["evidence", "odds_ratio"] = "evidence",
    class_balance: bool = True,
    min_probability: float | None = None,
    obsp_key: str = "connectivities",
    method: Literal["propagation", "spreading"] = "propagation",
    iterative: bool = False,
    alpha: float = 0.8,
    seed_obsm_key: str | None = None,
    seed_categories_uns_key: str | None = None,
    resolve: Literal["threshold", "entropy_null"] = "threshold",
    unknown_gate: float = 0.5,
    null: Literal["permutation", "analytic"] | None = "permutation",
    n_permutations: int = 1000,
    alpha_fdr: float = 0.05,
    multi_label_cum: float = 0.8,
    single_eff_k: float = 1.5,
    max_labels: int = 3,
    min_secondary_mass: float = 0.2,
    canonical_order: bool = False,
    random_state: int = 0,
    set_colors: bool = True,
    verbose: bool = True,
) -> None:
    """Soft, uncertainty-aware version of the cluster-annotation pipeline.

    Ties together the three steps needed to propagate enrichment *uncertainty*
    rather than a single hard top term per cluster:

    1. Convert the per-(cluster, term) enrichment table ``enr_res`` into a
       per-cluster probability distribution over a shared compartment vocabulary
       (plus an explicit ``unknown`` class) via
       :func:`~grassp.tl.enrichment_to_cluster_distribution`.
    2. Broadcast each cluster's distribution to its member proteins, producing a
       soft seed matrix stored in ``data.obsm[f"{key_added}_seed"]`` with the
       category order in ``data.uns[f"{key_added}_categories"]``.
    3. Propagate the soft seed over the neighbour graph with
       :func:`~grassp.tl.competitive_propagation`, writing the propagated distribution to
       ``data.obsm[f"{key_added}_probabilities"]`` and the argmax label (with
       ``unknown`` mapped to ``NaN``) to ``data.obs[key_added]``.

    Parameters
    ----------
    data
        AnnData with a populated neighbour graph and ``cluster_key`` in
        ``data.obs``.
    enr_res
        Enrichment table from
        :func:`~grassp.tl.calculate_cluster_enrichment` (``return_enrichment_res=True``),
        computed on the *same* ``cluster_key``. Required unless
        ``cluster_distribution`` is given.
    cluster_key
        Column in ``data.obs`` (and ``enr_res``) with the cluster labels the
        enrichment was computed on.
    key_added
        Base name for the outputs described above.
    cluster_distribution
        Optional precomputed ``(Q, categories)`` where ``Q`` is a row-stochastic
        (cluster x category) DataFrame and ``categories`` its column order — e.g.
        from :func:`~grassp.tl.mgsa_to_cluster_distribution`. When given, it is
        used as the seed directly and ``enr_res``/the enrichment knobs are ignored,
        letting any per-cluster distribution (MGSA, enrichment, custom) drive the
        propagation + entropy-null resolver.
    ranking_metric, threshold, temperature, s0, s_max, unknown_label
        Forwarded to :func:`~grassp.tl.enrichment_to_cluster_distribution` (unused
        when ``cluster_distribution`` is supplied).
    class_balance, min_probability, obsp_key, method, iterative, alpha
        Forwarded to :func:`~grassp.tl.competitive_propagation`.
    verbose
        Passed through to :func:`~grassp.tl.competitive_propagation`.

    Returns
    -------
    None. ``data`` is modified in place.
    """
    # The per-cluster distribution over compartments (+ optional unknown) can come
    # from the p-value/odds-ratio enrichment (default) or be supplied directly
    # (e.g. an MGSA posterior via `mgsa_to_cluster_distribution`). Either way it is
    # a (cluster x category) row-stochastic DataFrame + the category order.
    if cluster_distribution is not None:
        Q, categories = cluster_distribution
        categories = list(categories)
    else:
        if enr_res is None:
            raise ValueError(
                "Provide either `enr_res` or a precomputed `cluster_distribution`."
            )
        from .enrichment import enrichment_to_cluster_distribution

        Q, categories = enrichment_to_cluster_distribution(
            enr_res,
            cluster_key=cluster_key,
            ranking_metric=ranking_metric,
            threshold=threshold,
            temperature=temperature,
            s0=s0,
            s_max=s_max,
            unknown_label=unknown_label,
            weight_by=weight_by,
        )

    # Broadcast the per-cluster distribution to a per-protein (n_obs, C) seed.
    cluster_labels = data.obs[cluster_key].astype(str)
    seed = Q.reindex(cluster_labels.to_numpy()).to_numpy()
    # Clusters absent from Q (shouldn't happen, but be safe) get full mass on
    # the unknown class (or uniform if there is no unknown class).
    missing = np.isnan(seed).all(axis=1)
    if missing.any():
        seed[missing] = 0.0
        if unknown_label is not None and unknown_label in categories:
            seed[missing, categories.index(unknown_label)] = 1.0
        else:
            seed[missing] = 1.0 / seed.shape[1]

    set_matrix(data, f"{key_added}_seed", seed, categories)
    data.uns[f"{key_added}_categories"] = list(categories)

    competitive_propagation(
        data,
        gt_col=None,
        key_added=key_added,
        class_balance=class_balance,
        min_probability=min_probability,
        obsp_key=obsp_key,
        method=method,
        iterative=iterative,
        alpha=alpha,
        verbose=verbose,
        seed_obsm_key=f"{key_added}_seed",
        seed_categories_uns_key=f"{key_added}_categories",
        unknown_label=unknown_label,
    )

    if set_colors:
        # Consistent compartment colours for the base propagated label (covers the
        # threshold path; the entropy_null path additionally colours its resolved
        # columns below).
        try:
            from ..preprocessing import set_sensible_compartment_colors

            set_sensible_compartment_colors(
                data, columns=[key_added], cutoff=0.0, verbose=False, plot_mapping=False
            )
        except Exception as exc:  # pragma: no cover
            warnings.warn(f"soft_cluster_annotation: could not set colours ({exc}).")

    if resolve == "entropy_null":
        # Replace the scalar min_probability decision with the per-protein entropy
        # null test, emitting single / multi / unresolved calls into
        # `{key_added}_resolved*` columns.
        resolve_soft_labels(
            data,
            prob_key=f"{key_added}_probabilities",
            categories_key=f"{key_added}_categories",
            seed_key=f"{key_added}_seed",
            obsp_key=obsp_key,
            unknown_label=unknown_label,
            unknown_gate=unknown_gate,
            null=null,
            n_permutations=n_permutations,
            alpha_fdr=alpha_fdr,
            class_balance=class_balance,
            multi_label_cum=multi_label_cum,
            single_eff_k=single_eff_k,
            max_labels=max_labels,
            min_secondary_mass=min_secondary_mass,
            canonical_order=canonical_order,
            key_added=f"{key_added}_resolved",
            random_state=random_state,
            set_colors=set_colors,
        )


def _entropy_rows(P: np.ndarray) -> np.ndarray:
    """Row-wise Shannon entropy (nats) of a row-stochastic matrix."""
    P = np.clip(P, 1e-12, 1.0)
    return -(P * np.log(P)).sum(axis=1)


def _real_renorm(P: np.ndarray, real_idx: list[int]) -> np.ndarray:
    """Restrict a probability matrix to real (non-unknown) columns and renormalize.

    Rows with no real mass (all on unknown) fall back to uniform so their entropy is
    maximal; such rows are gated as unresolved by the unknown-mass test anyway.
    """
    R = P[:, real_idx]
    rs = R.sum(axis=1, keepdims=True)
    return np.divide(R, rs, out=np.full_like(R, 1.0 / R.shape[1]), where=rs > 0)


def _bh_fdr(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR-adjusted q-values."""
    p = np.asarray(p, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(1, n + 1))
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q = np.empty(n)
    q[order] = np.clip(ranked, 0, 1)
    return q


def resolve_soft_labels(
    data: AnnData,
    prob_key: str,
    categories_key: str,
    seed_key: str | None = None,
    obsp_key: str = "connectivities",
    unknown_label: str | None = "unknown",
    unknown_gate: float = 0.5,
    null: Literal["permutation", "analytic"] | None = "permutation",
    n_permutations: int = 1000,
    alpha_fdr: float = 0.05,
    class_balance: bool = True,
    multi_label_cum: float = 0.8,
    single_eff_k: float = 1.5,
    eff_k_max: float = 3.0,
    max_labels: int = 3,
    min_secondary_mass: float = 0.2,
    canonical_order: bool = False,
    key_added: str | None = None,
    random_state: int = 0,
    set_colors: bool = True,
    inplace: bool = True,
):
    """Resolve soft propagated probabilities into single / multi / unresolved labels.

    A single ``min_probability`` cutoff on the top propagated probability cannot tell
    apart two opposite cases: a protein in an *unresolved* region (mass smeared over many
    compartments) and a *genuine intermediate* (mass concentrated on 2-3 clean
    compartments). Both have a low top probability, but the *shape* of the distribution
    differs. This function decides using the entropy of the propagated distribution,
    compared to a per-protein null.

    For each protein it computes the Shannon entropy ``H`` of the propagated distribution
    over *real* compartments (``unknown`` excluded and handled by a separate gate); the
    effective number of compartments is ``exp(H)``. It then tests ``H`` against a null in
    which the protein carries no local structure:

    - ``null="permutation"`` (default): randomly permute which seed vector sits on which
      protein, keep the graph fixed, re-propagate with the same math, recompute ``H``.
      Repeated ``n_permutations`` times this gives a per-protein null entropy distribution
      (preserving graph geometry, each protein's effective neighbourhood size, the
      class-balance normalization and the global class prior). A left-tail p-value is
      BH-FDR adjusted.
    - ``null="analytic"``: a fast approximation using the population mean seed and each
      protein's effective neighbour count ``n_eff`` (``E[H|H0] ≈ H(π̄) - (k-1)/(2 n_eff)``
      with a delta-method variance), avoiding permutations.
    - ``null=None``: no test; a protein is resolved iff ``exp(H) <= eff_k_max``.

    Decision per protein: ``unknown_mass >= unknown_gate`` -> *unresolved*; else a
    significantly-low entropy (``q < alpha_fdr``) -> *single* (``exp(H) < single_eff_k``)
    or *multi*; otherwise *unresolved*. For resolved proteins the emitted compartment set
    is the smallest reaching cumulative mass ``multi_label_cum``.

    Parameters
    ----------
    data
        AnnData carrying the propagated probabilities in ``data.obsm[prob_key]`` and the
        matching category order in ``data.uns[categories_key]``.
    prob_key
        ``obsm`` key with the ``(n_obs, n_categories)`` propagated probability matrix
        (e.g. ``"ann_soft_probabilities"``).
    categories_key
        ``uns`` key with the ordered category names for the columns of ``prob_key``.
    seed_key
        ``obsm`` key with the soft seed matrix; required when ``null="permutation"``.
    obsp_key
        Affinity graph used for (re-)propagation (default ``"connectivities"``).
    unknown_label
        Name of the background/unknown category in the vocabulary, or ``None`` if there
        is no such class. Used both to exclude it from the entropy and as a gate.
    unknown_gate
        Proteins with ``unknown`` mass at or above this are called unresolved outright.
    null, n_permutations, alpha_fdr
        Null model, number of permutations, and FDR level (see above).
    class_balance
        Must match the setting used to produce ``prob_key`` so the null re-propagation is
        faithful.
    multi_label_cum
        Cumulative probability mass used to select how many compartments to emit for the
        *detailed* ``multiloc_label``.
    single_eff_k
        ``exp(H)`` below this is labelled *single*, otherwise *multi*.
    eff_k_max
        Only used when ``null=None``: resolved iff ``exp(H) <= eff_k_max``.
    max_labels
        Hard cap on the number of compartments in the *compact* ``label_compact``.
    min_secondary_mass
        A secondary compartment is only added to ``label_compact`` if it holds at least
        this probability mass. This is what suppresses gene-set-overlap tails (e.g. a
        0.16 Lipid-droplet share on ER proteins) and keeps the number of distinct
        compact labels small.
    canonical_order
        If ``True``, sort the compartments in ``label_compact`` alphabetically so that
        ``"A / B"`` and ``"B / A"`` collapse into one plot category (drops primary
        ordering). Default ``False`` keeps the primary compartment first.
    key_added
        Output prefix in ``.obs``. Defaults to ``f"{prob_key}_resolved"``.
    random_state
        Seed for the permutation null.
    set_colors
        If ``True`` (default), assign consistent compartment colours to the emitted
        label columns via :func:`~grassp.pp.set_sensible_compartment_colors`, so the
        same compartment renders identically here and in other annotation columns.
        Ignored when ``inplace=False``.
    inplace
        If ``True`` (default) write results to ``data``; otherwise return a dict.

    Returns
    -------
    None or dict
        Writes ``obs[key]`` (primary label / NaN), ``obs[key+"_multiloc"]`` (bool),
        ``obs[key+"_multiloc_label"]`` (detailed ``"A / B"`` string from the
        cumulative-mass rule), ``obs[key+"_label_compact"]`` (plot-friendly label:
        primary + secondaries above ``min_secondary_mass``, capped at ``max_labels``)
        and ``obs[key+"_multiloc_compact"]`` (bool), ``obs[key+"_secondary"]``,
        ``obs[key+"_type"]`` and diagnostics ``_entropy``, ``_eff_k``, ``_zscore``,
        ``_qvalue``, ``_unknown_mass``; a null summary in ``uns[key+"_null"]``.
    """
    key = key_added or f"{prob_key}_resolved"
    P = np.asarray(data.obsm[prob_key], dtype=float)
    cats = list(data.uns[categories_key])
    N = P.shape[0]

    uk = (
        cats.index(unknown_label)
        if (unknown_label is not None and unknown_label in cats)
        else None
    )
    real_idx = [i for i in range(len(cats)) if i != uk]
    real_cats = [cats[i] for i in real_idx]
    unknown_mass = P[:, uk] if uk is not None else np.zeros(N)

    R = _real_renorm(P, real_idx)
    H = _entropy_rows(R)
    eff_k = np.exp(H)

    T = data.obsp[obsp_key]
    w = np.asarray(T.sum(axis=1)).ravel()
    w2 = np.asarray(T.multiply(T).sum(axis=1)).ravel()
    n_eff = np.where(w2 > 0, w**2 / w2, 0.0)

    Hmean = Hsd = None
    if null == "permutation":
        if seed_key is None:
            raise ValueError("seed_key is required when null='permutation'.")
        S = np.asarray(data.obsm[seed_key], dtype=float)
        rng = np.random.default_rng(random_state)
        Hnull = np.empty((N, n_permutations))
        for b in range(n_permutations):
            Yb = _propagate_soft(T, S[rng.permutation(N)], class_balance=class_balance)
            Hnull[:, b] = _entropy_rows(_real_renorm(Yb, real_idx))
        Hmean = Hnull.mean(axis=1)
        Hsd = Hnull.std(axis=1) + 1e-9
        z = (H - Hmean) / Hsd
        pleft = ((Hnull <= H[:, None]).sum(axis=1) + 1) / (n_permutations + 1)
        q = _bh_fdr(pleft)
    elif null == "analytic":
        if seed_key is None:
            raise ValueError("seed_key is required when null='analytic'.")
        S = np.asarray(data.obsm[seed_key], dtype=float)
        pi = _real_renorm(S.mean(axis=0, keepdims=True), real_idx).ravel()
        pic = np.clip(pi, 1e-12, 1.0)
        H_pi = -(pic * np.log(pic)).sum()
        k_eff = int((pi > 0).sum())
        # delta-method variance of the plug-in entropy estimate under n_eff samples
        var_term = (pic * np.log(pic) ** 2).sum() - (pic * np.log(pic)).sum() ** 2
        with np.errstate(divide="ignore", invalid="ignore"):
            Hmean = H_pi - (k_eff - 1) / (2.0 * np.where(n_eff > 0, n_eff, np.nan))
            Hsd = np.sqrt(np.abs(var_term) / np.where(n_eff > 0, n_eff, np.nan)) + 1e-9
        Hmean = np.nan_to_num(Hmean, nan=H_pi)
        z = (H - Hmean) / Hsd
        from scipy.stats import norm

        pleft = norm.cdf(z)
        q = _bh_fdr(pleft)
    else:
        z = np.full(N, np.nan)
        q = np.full(N, np.nan)

    # Decision
    if null is None:
        resolved = eff_k <= eff_k_max
    else:
        resolved = q < alpha_fdr
    if uk is not None:
        resolved = resolved & (unknown_mass < unknown_gate)

    ptype = np.array(["unresolved"] * N, dtype=object)
    ptype[resolved & (eff_k < single_eff_k)] = "single"
    ptype[resolved & (eff_k >= single_eff_k)] = "multi"

    # Label emission from the real-class distribution.
    # Two label variants per resolved protein:
    #  - detailed (`multiloc_label`): smallest set reaching cumulative mass
    #    `multi_label_cum` (faithful, but grows long tails => many distinct combos).
    #  - compact (`label_compact`): a plot-friendly label. It keeps the primary and
    #    adds further compartments only if they each hold >= `min_secondary_mass`,
    #    up to `min(round(eff_k), max_labels)` compartments. `round(eff_k)` alone is
    #    NOT compact (diffuse-but-resolved proteins have large eff_k); the mass floor
    #    and the hard `max_labels` cap are what collapse the number of combinations.
    #    With `canonical_order=True` the compartments are sorted alphabetically so
    #    "A / B" and "B / A" merge into one plot category (loses primary ordering).
    order = np.argsort(-R, axis=1)
    primary = np.full(N, np.nan, dtype=object)
    secondary = np.full(N, np.nan, dtype=object)
    combined = np.full(N, np.nan, dtype=object)
    multiloc = np.zeros(N, dtype=bool)
    combined_compact = np.full(N, np.nan, dtype=object)
    multiloc_compact = np.zeros(N, dtype=bool)
    for i in np.where(resolved)[0]:
        oi = order[i]
        cum = np.cumsum(R[i, oi])
        k = min(int(np.searchsorted(cum, multi_label_cum)) + 1, len(oi))
        labs = [real_cats[j] for j in oi[:k]]
        primary[i] = labs[0]
        combined[i] = " / ".join(labs)
        if k > 1:
            secondary[i] = labs[1]
            multiloc[i] = True
        # compact label: primary + secondaries above the mass floor, capped
        kc = min(max(1, int(np.floor(eff_k[i] + 0.5))), max_labels, len(oi))
        labs_c = [real_cats[oi[0]]]
        for j in range(1, kc):
            if R[i, oi[j]] >= min_secondary_mass:
                labs_c.append(real_cats[oi[j]])
            else:
                break
        if canonical_order and len(labs_c) > 1:
            labs_c = sorted(labs_c)
        combined_compact[i] = " / ".join(labs_c)
        multiloc_compact[i] = len(labs_c) > 1

    out = {
        key: pd.Categorical(primary),
        # confident single-label view: the primary label, but NaN wherever the call is
        # multi-localised (multiloc=True) or unresolved (primary already NaN).
        f"{key}_single": pd.Categorical(np.where(multiloc, np.nan, primary)),
        f"{key}_secondary": pd.Categorical(secondary),
        f"{key}_multiloc_label": pd.Categorical(combined),
        f"{key}_multiloc": multiloc,
        f"{key}_label_compact": pd.Categorical(combined_compact),
        f"{key}_multiloc_compact": multiloc_compact,
        f"{key}_type": pd.Categorical(ptype, categories=["single", "multi", "unresolved"]),
        f"{key}_entropy": H,
        f"{key}_eff_k": eff_k,
        f"{key}_zscore": z,
        f"{key}_qvalue": q,
        f"{key}_unknown_mass": unknown_mass,
    }
    null_summary = {
        "method": str(null),
        "n_permutations": int(n_permutations) if null == "permutation" else 0,
        "null_mean_entropy": float(np.nanmean(Hmean)) if Hmean is not None else None,
        "observed_mean_entropy": float(H.mean()),
        "corr_null_n_eff": (
            float(np.corrcoef(n_eff, Hmean)[0, 1]) if Hmean is not None else None
        ),
        "n_single": int((ptype == "single").sum()),
        "n_multi": int((ptype == "multi").sum()),
        "n_unresolved": int((ptype == "unresolved").sum()),
    }

    if not inplace:
        return {"obs": out, "null_summary": null_summary}

    for col, val in out.items():
        data.obs[col] = val
    data.uns[f"{key}_null"] = null_summary

    if set_colors:
        # Give the label columns consistent compartment colours from the shared
        # MARKER_COLORS palette, so the same compartment renders the same colour
        # here and across other annotation columns (ann_hard, ann_soft, ...). The
        # composite compact/multiloc columns are included with a low cutoff so their
        # single-compartment categories still get canonical colours (composite
        # categories fall back to distinct palette colours).
        try:
            from ..preprocessing import set_sensible_compartment_colors

            color_cols = [
                c
                for c in (
                    key,
                    f"{key}_secondary",
                    f"{key}_label_compact",
                    f"{key}_multiloc_label",
                )
                if c in data.obs
            ]
            set_sensible_compartment_colors(
                data, columns=color_cols, cutoff=0.0, verbose=False, plot_mapping=False
            )
        except Exception as exc:  # pragma: no cover - colouring must never break the run
            warnings.warn(f"resolve_soft_labels: could not set compartment colours ({exc}).")

    return None


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

    # Configure SVM. No `probability=True` here: the grid search scores with f1_macro, which goes
    # through predict(), and only the winning *parameters* are kept -- svm_annotation fits its own
    # estimator. Asking for probabilities would run libsvm's internal 5-fold Platt scaling on every
    # fit in the grid for nothing.
    svm = SVC(
        kernel='rbf',
        class_weight=class_weight,
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

    Similar to :func:`competitive_propagation` but uses SVM instead of graph propagation.

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
        set_matrix(data, f"{key_added}_probabilities", probabilities, categories)

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

    Runs :func:`competitive_propagation` on the existing markers and retains only those
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
    knnres = competitive_propagation(
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
