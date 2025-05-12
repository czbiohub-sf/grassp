from typing import List, Literal, Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from anndata import AnnData
from scanpy.plotting._tools.scatterplots import _components_to_dimensions
from scanpy.tl import Ingest
from scipy.stats import multivariate_normal


def sample_tagm_map(adata, size=100):
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
):
    """Plot the TAGM map of the adata object.

    Args:
        adata: The adata object.
        size: The number of samples to draw from the TAGM map.
        ax: The axis to plot the TAGM map on.
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
    **kwargs,
):
    """Plot the PCA ellipses of the TAGM map of the adata object.

    Args:
        adata: The adata object.
        ax: The axis to plot the TAGM map on.
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

        ax.scatter(mu_pca[i][0], mu_pca[i][1], marker="X", s=40, color=color)
        for nstd in stds:
            _plot_covariance_ellipse(
                Sigma_pca[i][:2, :2],
                mu_pca[i][:2],
                ax=ax,
                alpha=1,
                color=color,
                fill=False,
                nstd=nstd,
            )
