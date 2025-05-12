from __future__ import annotations
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from anndata import AnnData
    from typing import List, Literal
    from matplotlib.axes import Axes


import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pysankey
import scanpy
import seaborn as sns

from matplotlib import gridspec
from matplotlib.markers import MarkerStyle


def remodeling_score(
    remodeling_score: np.ndarray,
    show: bool | None = None,
    save: bool | str | None = None,
) -> List[plt.Axes] | None:
    """Plot remodeling score distribution.

    Parameters
    ----------
    remodeling_score
        Array of remodeling scores to plot
    show
        Show the plot. If None, use scanpy's plotting settings
    save
        If True or a str, save the figure. A string is appended to the default
        filename. Infer the filetype if ending on {'.pdf', '.png', '.svg'}

    Returns
    -------
    list or None
        If show=False, returns list of matplotlib axes objects for histogram and boxplot.
        If show=True, returns None and displays plot.
    """

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
    scanpy.pl._utils.savefig_or_show("remodeling_score", show=show, save=save)
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
    """Get cluster colors for a given color key.

    Parameters
    ----------
    data
        Annotated data matrix
    color_key
        Key in data.obs to use for coloring

    Returns
    -------
    np.ndarray
        Array of colors for the given color key
    """
    if f"{color_key}_colors" not in data.uns.keys():
        scanpy.pl._utils._set_default_colors_for_categorical_obs(data, color_key)
    return np.array(
        [data.uns[f"{color_key}_colors"][x] for x in data.obs[color_key].cat.codes]
    )


def aligned_umap(
    data: AnnData,
    data2: AnnData,
    highlight_hits: List[str] | np.ndarray[bool, Any] | None = None,
    highlight_annotation_col: str | None = None,
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
    """Plot aligned UMAP embeddings for two datasets.

    Parameters
    ----------
    data
        First annotated data matrix
    data2
        Second annotated data matrix
    highlight_hits
        List of protein names or boolean mask to highlight trajectories between datasets
    highlight_annotation_col
        Column in data.obs to use for highlighting specific proteins
    aligned_umap_key
        Key in data.obsm containing aligned UMAP coordinates
    data1_label
        Label for first dataset in legend
    data2_label
        Label for second dataset in legend
    color_by
        Whether to color by 'perturbation' or 'cluster'
    data1_color
        Color for first dataset points if coloring by perturbation
    data2_color
        Color for second dataset points if coloring by perturbation
    figsize
        Size of the figure in inches
    size
        Size of scatter points
    alpha
        Transparency of scatter points
    ax
        Matplotlib axes to plot on
    show
        Whether to display the plot
    save
        Whether to save the plot and optional filename

    Returns
    -------
    matplotlib.pyplot.Axes or None
        If show=False returns the axes object, otherwise returns None
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    if color_by == "cluster":
        data1_colors = _get_cluster_colors(data, data1_color)
        data2_colors = _get_cluster_colors(data2, data2_color)

        # Create cluster legend handles
        cluster_handles = []
        unique_clusters = data.obs[data1_color].cat.categories
        cluster_colors = data.uns[f"{data1_color}_colors"]
        for cluster, color in zip(unique_clusters, cluster_colors):
            cluster_handles.append(
                mlines.Line2D(
                    [],
                    [],
                    color=color,
                    marker="o",
                    linestyle="None",
                    markersize=8,
                    label=cluster,
                )
            )
    else:
        data1_colors = data1_color
        data2_colors = data2_color

    embedding1 = data.obsm[aligned_umap_key]
    embedding2 = data2.obsm[aligned_umap_key]

    # Plot the two embeddings as scatter plots
    ax.scatter(
        np.asarray(embedding1)[:, 0],
        np.asarray(embedding1)[:, 1],
        c=data1_colors,
        s=size,
        alpha=alpha,
        label=data1_label,
        marker=MarkerStyle("."),
        linewidths=0,
        edgecolor=None,
    )
    ax.scatter(
        np.asarray(embedding2)[:, 0],
        np.asarray(embedding2)[:, 1],
        c=data2_colors,
        s=size,
        alpha=alpha,
        label=data2_label,
        marker=MarkerStyle("+"),
        linewidths=1,
        edgecolor=None,
    )

    if highlight_hits is not None:
        embedding1_hits = np.asarray(embedding1)[highlight_hits]
        embedding2_hits = np.asarray(embedding2)[highlight_hits]
        # Plot trajectory lines
        for i, (start, end) in enumerate(zip(embedding1_hits, embedding2_hits)):
            # Draw line
            ax.plot(
                [start[0], end[0]],
                [start[1], end[1]],
                color="grey",
                linewidth=0.7,
                alpha=0.5,
            )
            # Draw marker at the end point
            ax.scatter(
                start[0],
                start[1],
                color="black",
                s=30,
                marker=MarkerStyle("*"),
                edgecolor=None,
            )
            if highlight_annotation_col is not None:
                # Add annotation
                ax.annotate(
                    str(data.obs.loc[highlight_hits, highlight_annotation_col].iloc[i]),
                    (start[0], start[1]),
                    color="black",
                    fontsize=5,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=0.5),
                    ha="right",
                    va="bottom",
                    xytext=(5, 5),
                    textcoords="offset points",
                )

    # Combine scatter plot legend with remodeling legend
    handles, labels = ax.get_legend_handles_labels()
    combined_handles = handles + remodeling_legend

    # Add legends to the right of the plot
    if color_by == "cluster":
        # First legend for dataset and remodeling markers
        ax.legend(handles=combined_handles, bbox_to_anchor=(1.15, 1), loc="upper left")
        # Second legend for clusters
        ax.legend(
            handles=cluster_handles,
            bbox_to_anchor=(1.15, 0.5),
            loc="center left",
            title="Clusters",
        )
    else:
        # Single legend if not coloring by cluster
        ax.legend(handles=combined_handles, bbox_to_anchor=(1.15, 1), loc="upper left")

    # Adjust layout to prevent legend cutoff
    plt.tight_layout()

    ax.set_xlabel(f"{aligned_umap_key.replace('X_', '')}1")
    ax.set_ylabel(f"{aligned_umap_key.replace('X_', '')}2")
    ax.set_xticks([])
    ax.set_yticks([])

    show = scanpy.settings.autoshow if show is None else show
    scanpy.pl._utils.savefig_or_show("aligned_umap", show=show, save=save)
    if show:
        return None
    return ax


def remodeling_sankey(
    data: AnnData,
    data2: AnnData,
    cluster_key: str = "leiden",
    ax: plt.Axes | None = None,
    aspect: int = 20,
    fontsize: int = 12,
    figsize: tuple[float, float] = (10, 11),
    show: bool | None = None,
    save: bool | str | None = None,
) -> plt.Axes | None:
    """Plot a Sankey diagram showing protein transitions between clusters.

    Parameters
    ----------
    data
        First annotated data matrix
    data2
        Second annotated data matrix
    cluster_key
        Key in data.obs and data2.obs containing cluster assignments
    ax
        Matplotlib axes object to plot on. If None, creates new figure
    aspect
        Aspect ratio of the plot
    fontsize
        Font size for labels
    figsize
        Size of the figure in inches
    show
        Show the plot. If None, use scanpy's plotting settings
    save
        If True or a str, save the figure. A string is appended to the default
        filename. Infer the filetype if ending on {'.pdf', '.png', '.svg'}

    Returns
    -------
    ax or None
        If show=False, returns matplotlib axes object. Otherwise returns None
    """
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
    scanpy.pl._utils.savefig_or_show("remodeling_sankey", show=show, save=save)
    if show:
        return None
    return ax


def mr_plot(
    data: AnnData,
    mr_key: str = "mr_scores",
    ax: plt.Axes = None,
    m_cutoffs: list[float] = [2, 3, 4],
    r_cutoffs: list[float] = [0.68, 0.81, 0.93],
    highlight_hits: bool = True,
    show: bool | None = None,
    save: bool | str | None = None,
    **kwargs,
) -> Axes | None:
    """Create MR plot for protein translocation analysis with broken x-axis.

    Parameters
    ----------
    data : AnnData
        Annotated data matrix containing MR scores
    mr_key : str, optional
        Key in data.uns containing MR scores. Defaults to "mr_scores"
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, current axes will be used
    m_cutoffs : list[float], optional
        M score cutoffs to show (lenient, stringent, very stringent)
    r_cutoffs : list[float], optional
        R score cutoffs to show (lenient, stringent, very stringent)
    highlight_hits : bool, optional
        Whether to highlight points passing lenient cutoffs
    **kwargs
        Additional arguments passed to plt.scatter

    Returns
    -------
    matplotlib.axes.Axes
        The axes object with the plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    try:
        m_scores = data.obs[f"{mr_key}_M"]
        r_scores = data.obs[f"{mr_key}_R"]
    except KeyError:
        raise ValueError(
            f"MR scores not found in data.obs['{mr_key}_M/R'], run gr.tl.mr_score first"
        )

    # Plot data
    ax.scatter(m_scores, r_scores, alpha=0.5, s=10, color="black", marker=".", **kwargs)

    if highlight_hits:
        hits = (m_scores >= m_cutoffs[0]) & (r_scores >= r_cutoffs[0])
        ax.scatter(m_scores[hits], r_scores[hits], color="red", s=20, marker=".")

    # Add cutoff lines
    colors = ["gray", "darkgray", "lightgray"]
    for m_cut, color in zip(m_cutoffs, colors):
        ax.axvline(m_cut, color=color, linestyle="--", alpha=0.5)
    for r_cut, color in zip(r_cutoffs, colors):
        ax.axhline(r_cut, color=color, linestyle="--", alpha=0.5)

    # Set x-axis limits
    ax.set_xlim(0, np.max(m_scores) + 5)

    # Set labels
    ax.set_xlabel("M score (-log10 Q-value)")
    ax.set_ylabel("R score (minimum correlation)")
    ax.set_title("MR Plot")

    show = scanpy.settings.autoshow if show is None else show
    scanpy.pl._utils.savefig_or_show("remodeling_score", show=show, save=save)
    if show:
        return None
    return ax
