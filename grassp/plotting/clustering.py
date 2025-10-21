from typing import List, Literal, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from anndata import AnnData
from scanpy.plotting._tools.scatterplots import _components_to_dimensions
from scanpy.tl import Ingest
from scipy.stats import multivariate_normal


def sample_tagm_map(adata: AnnData, size: int = 100) -> list[np.ndarray]:
    """Return synthetic samples from the TAGM posterior distribution.

    This helper draws *size* samples for each TAGM component using the
    multivariate normal parameters stored in
    ``adata.uns["tagm.map.params"]["posteriors"]``.

    Parameters
    ----------
    adata
        Annotated data matrix produced by :func`grassp.tl.tagm_map_train`.
        The function expects that posterior means (``mu``) and covariances
        (``sigma``) are stored under
        ``adata.uns["tagm.map.params"]["posteriors"]``.
    size
        Number of samples to draw *per component* (cluster).

    Returns
    -------
    A list containing one array per component. Each array has shape
    ``(size, n_features)`` where ``n_features`` equals the dimensionality of
    the data on which TAGM was fitted (typically the number of fractions
    or enriched samples).
    """

    params = adata.uns["tagm.map.params"]
    mu = params["posteriors"]["mu"]
    sigma = params["posteriors"]["sigma"]
    mv = [multivariate_normal.rvs(mu[i], sigma[i], size=size) for i in range(mu.shape[0])]
    return mv


def tagm_map_contours(
    adata: AnnData,
    embedding: Literal["umap", "pca"] = "umap",
    size: int = 100,
    components: str | Sequence[str] | None = None,
    dimensions: tuple[int, int] | Sequence[tuple[int, int]] | None = None,
    levels: int = 4,
    ax: plt.Axes | None = None,
    **kwargs,
) -> plt.Axes:
    """Plot posterior density contours of TAGM components in embedding space.

    The function draws synthetic observations from the TAGM posterior via
    :func:`sample_tagm_map` and projects them into either UMAP or PCA space
    (depending on ``embedding``).  A kernel–density estimate is then computed
    for each component and visualized as contour lines.

    Parameters
    ----------
    adata
        Annotated data matrix that already contains TAGM posterior parameters
        in ``adata.uns["tagm.map.params"]`` **and** an embedding (UMAP or PCA)
        fitted by :func:`~scanpy.tl.umap` or :func:`scanpy.pp.pca`.
    embedding
        Target space for the contours, either ``"umap"`` or ``"pca"``.
    size
        Number of posterior samples to draw *per component*.
    components, dimensions
        Specify which dimensions of the embedding to use.  Only one pair of
        dimensions is allowed.  Use ``components`` (Scanpy style string) or a
        ``dimensions`` tuple, not both.
    levels
        Number of contour levels passed to :func:`seaborn.kdeplot`.
    ax
        Matplotlib axes to plot on.  If ``None``, the current axes returned by
        :func:`matplotlib.pyplot.gca` are used.
    **kwargs
        Additional keyword arguments forwarded to :func:`seaborn.kdeplot`.

    Returns
    -------
    The axes object containing the contour plot (returned for
    convenience).
    """
    if ax is None:
        ax = plt.gca()

    dimensions = _components_to_dimensions(
        components=components, dimensions=dimensions, total_dims=2
    )
    if len(dimensions) > 1:
        raise ValueError(
            "Only one dimension pair at a time is supported for the TAGM map contours."
        )
    else:
        dimensions = dimensions[0]
    lmv = sample_tagm_map(adata, size=size)
    mv = np.concatenate(lmv, axis=0)
    ing = Ingest(adata)
    if embedding == "umap":
        emb = ing._umap.transform(mv)
    elif embedding == "pca":
        if ing._pca_centered:
            mv -= mv.mean(axis=0)
        emb = np.dot(mv, ing._pca_basis)
    else:
        raise ValueError(f"Invalid embedding: {embedding}")
    idx = 0
    # Check if adata.uns has color key
    if "tagm.map.allocation_colors" not in adata.uns:
        adata.uns["tagm.map.allocation_colors"] = sns.color_palette(
            "husl", adata.obs[adata.uns["tagm.map.params"]["gt_col"]].nunique()
        )

    for v, c in zip(lmv, adata.uns["tagm.map.allocation_colors"]):
        x = emb[idx : idx + v.shape[0], dimensions[0]]
        y = emb[idx : idx + v.shape[0], dimensions[1]]
        sns.kdeplot(x=x, y=y, levels=levels, ax=ax, color=c, **kwargs)
        idx += v.shape[0]
    return ax


def _plot_covariance_ellipse(cov, pos, nstd=2, ax=None, **kwargs):
    """
    Plots an ellipse representing the covariance matrix.

    Parameters
    ----------
    cov : array-like, shape (2, 2)
        The 2x2 covariance matrix
    pos : array-like, shape (2,)
        The location of the center of the ellipse
    nstd : float
        The number of standard deviations to determine the ellipse's radius
    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into
    **kwargs
        Additional arguments passed to the ellipse patch

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    import matplotlib.pyplot as plt
    import numpy as np

    if ax is None:
        ax = plt.gca()

    # Calculate eigenvalues and eigenvectors
    vals, vecs = np.linalg.eigh(cov)

    # Calculate angle of rotation
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

    # Calculate width and height
    width, height = 2 * nstd * np.sqrt(vals)

    # Create the ellipse
    ellip = plt.matplotlib.patches.Ellipse(
        xy=pos, width=width, height=height, angle=theta, **kwargs
    )

    ax.add_patch(ellip)
    return ellip


def tagm_map_pca_ellipses(
    adata: AnnData,
    stds: List[int] = [1, 2, 3],
    dimensions: tuple[int, int] | None = None,
    components: str | Sequence[str] | None = None,
    ax: plt.Axes | None = None,
    scatter_kwargs: dict | None = {},
    **kwargs,
) -> plt.Axes:
    """Visualize TAGM component covariance as ellipses in PCA space.

    For each TAGM component the posterior covariance matrix is projected into
    PCA space and visualised as an ellipse representing *n* standard-
    deviation contours (defined by the *stds* list).

    Parameters
    ----------
    adata
        Annotated data matrix that contains
        ``adata.varm["PCs"]`` (loadings) and TAGM posterior parameters under
        ``adata.uns["tagm.map.params"]``.
    stds
        Radii of the ellipses in standard deviations.  Typical choices are
        ``[1, 2, 3]`` which correspond to the 68 %, 95 % and 99.7 %
        confidence regions of a multivariate normal distribution.
    dimensions, components
        Specify which principal components to plot—either via an explicit
        *dimensions* tuple (0-based indices) or the Scanpy-style *components*
        string (e.g. ``"1,2"``).
    ax
        Matplotlib axes to plot on.  If *None*, :func:`matplotlib.pyplot.gca`
        is used.
    scatter_kwargs
        Additional keyword arguments forwarded to
        :func:`~matplotlib.pyplot.scatter`.
    **kwargs
        Additional keyword arguments forwarded to
        :class:`~matplotlib.patches.Ellipse` (e.g. ``linewidth``, ``alpha``).

    Returns
    -------
    The axes with the ellipse overlay (returned if *show* is False).
    """
    dimensions = _components_to_dimensions(
        components=components, dimensions=dimensions, total_dims=2
    )
    if len(dimensions) > 1:
        raise ValueError(
            "Only one dimension pair at a time is supported for the TAGM map PCA ellipses."
        )
    else:
        dimensions = dimensions[0]

    if "PCs" not in adata.varm:
        raise ValueError("PCA must be computed before plotting the TAGM map PCA ellipses.")
    if "tagm.map.params" not in adata.uns:
        raise ValueError(
            "TAGM map must be computed before plotting the TAGM map PCA ellipses."
        )
    if ax is None:
        ax = plt.gca()

    pcs = adata.varm["PCs"][:, dimensions]
    mu_pca = (adata.uns["tagm.map.params"]["posteriors"]["mu"] - adata.X.mean(axis=0)) @ pcs

    temp = np.matmul(adata.uns["tagm.map.params"]["posteriors"]["sigma"], pcs)
    Sigma_pca = np.matmul(pcs.T[None, :, :], temp)

    for i in range(Sigma_pca.shape[0]):
        color = adata.uns["tagm.map.allocation_colors"][i]

        ax.scatter(
            mu_pca[i][0],
            mu_pca[i][1],
            marker="X",
            s=40,
            color=color,
            **scatter_kwargs,
        )
        for nstd in stds:
            _plot_covariance_ellipse(
                Sigma_pca[i][:2, :2],
                mu_pca[i][:2],
                ax=ax,
                alpha=1,
                color=color,
                fill=False,
                nstd=nstd,
                **kwargs,
            )
    return ax


def knn_marker_df(data: AnnData, gt_col: str, pred_col: str) -> pd.DataFrame:
    labels = data.obs[gt_col].astype("category")
    labels_one_hot = pd.get_dummies(labels).values
    probabilities = data.obsm[f"{pred_col}_probabilities"]
    true_prob = np.sum(probabilities * labels_one_hot, axis=1)
    marker_df = pd.DataFrame({"gt_col": labels, "pred_prob": true_prob}).dropna()
    return marker_df


def knn_violin(
    data: AnnData, gt_col: str, pred_col: str, ax: plt.Axes | None = None, **kwargs
) -> plt.Axes:
    """Violin plot of KNN annotation.

    Parameters
    ----------
    data
        AnnData object.
    gt_col
        Observation column with ground-truth labels.
    pred_col
        Observation column with predicted labels.
    """
    if ax is None:
        ax = plt.gca()
    plot_df = knn_marker_df(data, gt_col, pred_col)
    sns.violinplot(
        data=plot_df,
        x="gt_col",
        y="pred_prob",
        ax=ax,
        **kwargs,
        cut=0,
        inner=None,  # no box, no points inside
        alpha=0.5,
    )
    sns.stripplot(
        data=plot_df,
        x="gt_col",
        y="pred_prob",
        ax=ax,
        color="k",
        size=2,
        alpha=0.5,
        jitter=True,
        dodge=False,
    )
    ax.set_ylabel("Predicted Probability")
    ax.set_xlabel("Ground Truth")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right")
    return ax
