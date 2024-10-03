from __future__ import annotations
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from anndata import AnnData
    from typing import List, Literal

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pysankey
import scanpy
import seaborn as sns

from matplotlib import gridspec


def remodeling_score(
    remodeling_score: np.ndarray,
    show: bool | None = None,
    save: bool | str | None = None,
) -> List[plt.Axes] | None:
    # Create grid layout
    gs = gridspec.GridSpec(2, 1, height_ratios=[5, 1])
    # Histogram
    ax0 = plt.subplot(gs[0])
    sns.histplot(remodeling_score, ax=ax0, kde=False)
    ax0.set(xlabel="")
    # turn off x tick labels
    ax0.set_xticklabels([])

    # Boxplot
    ax1 = plt.subplot(gs[1])
    sns.boxplot(
        x=remodeling_score,
        ax=ax1,
        flierprops=dict(
            marker="o", markeredgecolor="orange", markerfacecolor="none", markersize=6
        ),
    )
    ax1.set(xlabel="Remodeling score")
    axs = [ax0, ax1]

    show = scanpy.settings.autoshow if show is None else show
    scanpy.plotting._utils.savefig_or_show("remodeling_score", show=show, save=save)
    if show:
        return None
    return axs


remodeling_legend = [
    mlines.Line2D(
        [], [], color="black", marker="*", linestyle="None", label="remodeled_proteins"
    ),
    mlines.Line2D(
        [], [], color="grey", linestyle="-", linewidth=2, label="remodeling trajectory"
    ),
]


def _get_cluster_colors(data: AnnData, color_key: str = "leiden") -> np.ndarray[str, Any]:
    if f"{color_key}_colors" not in data.uns.keys():
        scanpy.pl._utils._set_default_colors_for_categorical_obs(data, color_key)
    return np.array(
        [data.uns[f"{color_key}_colors"][x] for x in data.obs[color_key].cat.codes]
    )


def aligned_umap(
    data: AnnData,
    data2: AnnData,
    highlight_hits: List[str] | np.ndarray[bool, Any] | None = None,
    aligned_umap_key: str = "X_aligned_umap",
    data1_label: str = "data1",
    data2_label: str = "data2",
    color_by: Literal["perturbation", "cluster"] = "perturbation",
    data1_color: str = "#C7E8F9",
    data2_color: str = "#FFCCC2",
    figsize: tuple[float, float] = (8.25, 6),
    size: int = 80,
    alpha: float = 0.4,
    ax: plt.Axes | None = None,
    show: bool | None = None,
    save: bool | str | None = None,
) -> plt.Axes | None:
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    if color_by == "cluster":
        data1_colors = _get_cluster_colors(data, data1_color)
        data2_colors = _get_cluster_colors(data2, data2_color)
    else:
        data1_colors = data1_color
        data2_colors = data2_color

    embedding1 = data.obsm[aligned_umap_key]
    embedding2 = data2.obsm[aligned_umap_key]

    # Plot the two embeddings as scatter plots
    ax.scatter(
        embedding1[:, 0],
        embedding1[:, 1],
        c=data1_colors,
        s=size,
        alpha=alpha,
        label=data1_label,
        marker=".",
        linewidths=0,
        edgecolor=None,
    )
    ax.scatter(
        embedding2[:, 0],
        embedding2[:, 1],
        c=data2_colors,
        s=size,
        alpha=alpha,
        label=data2_label,
        marker="+",
        linewidths=1,
        edgecolor=None,
    )

    if highlight_hits is not None:
        embedding1_hits = embedding1[highlight_hits]
        embedding2_hits = embedding2[highlight_hits]
        # Plot trajectory lines
        for start, end in zip(embedding1_hits, embedding2_hits):
            # Draw line
            ax.plot(
                [start[0], end[0]],
                [start[1], end[1]],
                color="grey",
                linewidth=0.7,
                alpha=0.5,
            )
            # Draw marker at the end point
        for start, end in zip(embedding1_hits, embedding2_hits):
            ax.scatter(start[0], start[1], color="black", s=30, marker="*", edgecolor=None)

    # Combine scatter plot legend with remodeling legend
    handles, labels = ax.get_legend_handles_labels()
    combined_handles = handles + remodeling_legend

    # Add combined legend to the plot
    ax.legend(handles=combined_handles)

    ax.set_xlabel(f"{aligned_umap_key.replace('X_', '')}1")
    ax.set_ylabel(f"{aligned_umap_key.replace('X_', '')}2")
    ax.set_xticks([])
    ax.set_yticks([])

    show = scanpy.settings.autoshow if show is None else show
    scanpy.plotting._utils.savefig_or_show("aligned_umap", show=show, save=save)
    if show:
        return None
    return ax


def remodeling_sankey(
    data: AnnData,
    data2: AnnData,
    cluster_key: str = "leiden",
    ax: plt.Axes | None = None,
    aspect: float = 20,
    fontsize: int = 12,
    figsize: tuple[float, float] = (10, 11),
    show: bool | None = None,
    save: bool | str | None = None,
) -> plt.Axes | None:
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Check that the anndata objects are aligned
    assert (data.obs_names == data2.obs_names).all()

    pysankey.sankey(
        left=data.obs[cluster_key],
        right=data2.obs[cluster_key],
        aspect=aspect,
        # colorDict=colorDict,
        fontsize=fontsize,
        color_gradient=False,
        # leftLabels=[
        #     "nucleus",
        #     "cytosol",
        #     "mitochondrion",
        #     "ER",
        #     "plasma memb. & actin",
        #     "endo-lysosome & trans-Golgi",
        #     "ERGIC/Golgi",
        #     "translation/RNA granules",
        #     "peroxisome",
        # ],
        # rightLabels=[
        #     "nucleus",
        #     "cytosol",
        #     "mitochondrion",
        #     "ER",
        #     "plasma memb. & actin",
        #     "endo-lysosome & trans-Golgi",
        #     "translation/RNA granules",
        #     "peroxisome",
        #     "COPI vesicle",
        # ],
        ax=ax,
    )

    show = scanpy.settings.autoshow if show is None else show
    scanpy.plotting._utils.savefig_or_show("aligned_umap", show=show, save=save)
    if show:
        return None
    return ax
