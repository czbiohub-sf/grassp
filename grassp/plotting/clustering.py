import numpy as np
from scipy.stats import multivariate_normal
import seaborn as sns
from scanpy.tl import Ingest
import matplotlib.pyplot as plt


def sample_tagm_map(adata, size=100):
    params = adata.uns["tagm.map.params"]
    mu = params["posteriors"]["mu"]
    sigma = params["posteriors"]["sigma"]
    mv = [
        multivariate_normal.rvs(mu[i], sigma[i], size=size) for i in range(mu.shape[0])
    ]
    return mv


def tagm_map_contours(adata, size=100, ax=None):
    """Plot the TAGM map of the adata object.

    Args:
        adata: The adata object.
        size: The number of samples to draw from the TAGM map.
        ax: The axis to plot the TAGM map on.
    """
    if ax is None:
        ax = plt.gca()

    lmv = sample_tagm_map(adata, size=size)
    mv = np.concatenate(lmv, axis=0)
    ing = Ingest(adata)
    umap = ing._umap.transform(mv)
    idx = 0
    # Check if adata.uns has color key
    if "tagm.map.allocation_colors" not in adata.uns:
        adata.uns["tagm.map.allocation_colors"] = sns.color_palette(
            "husl", adata.obs[adata.uns["tagm.map.params"]["gt_col"]].nunique()
        )

    for v, c in zip(lmv, adata.uns["tagm.map.allocation_colors"]):
        x = umap[idx : idx + v.shape[0], 0]
        y = umap[idx : idx + v.shape[0], 1]
        sns.kdeplot(x=x, y=y, levels=4, ax=ax, color=c)
        idx += v.shape[0]
