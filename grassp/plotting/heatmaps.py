from __future__ import annotations
from typing import TYPE_CHECKING

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
import scipy.spatial as sp
import seaborn as sns

if TYPE_CHECKING:
    from typing import Callable, Literal, Optional

    from anndata import AnnData

    # Define a type hint for functions that take an ndarray and an optional axis argument
    NDArrayAxisFunction = Callable[[np.ndarray, Optional[int]], np.ndarray]


def protein_clustermap(
    data: AnnData,
    annotation_key: str,
    distance_metric: Literal[
        "euclidean", "cosine", "correlation", "cityblock", "jaccard", "hamming"
    ] = "correlation",
    linkage_method: Literal[
        "single", "complete", "average", "weighted", "centroid", "median", "ward"
    ] = "average",
    linkage_metric: Literal[
        "euclidean", "cosine", "correlation", "cityblock", "jaccard", "hamming"
    ] = "cosine",
    palette="Blues_r",
    show: bool = True,
) -> None:
    """Create a clustered heatmap of protein distances with annotations.

    Parameters
    ----------
    data
        Annotated data matrix with proteins as observations (rows)
    annotation_key
        Key in data.obs for annotating proteins
    distance_metric
        Distance metric to use for calculating pairwise distances between proteins.
        One of 'euclidean', 'cosine', 'correlation', 'cityblock', 'jaccard', 'hamming'
    linkage_method
        Method for hierarchical clustering.
        One of 'single', 'complete', 'average', 'weighted', 'centroid', 'median', 'ward'
    linkage_metric
        Distance metric to use for hierarchical clustering.
        One of 'euclidean', 'cosine', 'correlation', 'cityblock', 'jaccard', 'hamming'
    palette
        Color palette for the heatmap. Default is 'Blues_r'
    show
        If True, display the heatmap. If False, return the Axes object.

    Returns
    -------
    None
        Displays the clustered heatmap
    """

    distance_matrix = sp.distance.pdist(data.X, metric=distance_metric)
    linkage = sch.linkage(distance_matrix, method=linkage_method, metric=linkage_metric)
    row_order = np.array(sch.dendrogram(linkage, no_plot=True, orientation="bottom")["leaves"])
    distance_matrix = sp.distance.squareform(distance_matrix)
    distance_matrix = distance_matrix[row_order, :][:, row_order]

    gt = data.obs[annotation_key].values[row_order]
    unique_categories = np.unique(gt)
    lut = dict(zip(unique_categories, sns.color_palette("tab20", len(unique_categories))))
    row_colors = pd.Series(gt).astype(str).map(lut).to_numpy()

    g = sns.clustermap(
        data=distance_matrix,
        cmap=sns.color_palette(palette, as_cmap=True),
        row_colors=row_colors,
        col_colors=row_colors,
        row_cluster=False,
        col_cluster=False,
        colors_ratio=(0.02, 0.02),  # width of color bar
        cbar_pos=(
            0.38,
            0.9,
            0.5,
            0.03,
        ),  # color bar location coordinates in this format (left, bottom, width, height),
        # cbar_pos=None,
        cbar_kws={
            "orientation": "horizontal",
            "label": "distance",
            "extend": "neither",
        },
        robust=True,
        figsize=(24, 22),
        xticklabels=False,
        yticklabels=False,
        vmin=0.4,
        vmax=0.9,
    )

    g.fig.suptitle(
        "Pairwise distance between proteins in the enrichment space",
        fontsize=25,
        y=1.00,
    )
    g.ax_heatmap.set_xlabel("")
    g.ax_heatmap.set_ylabel("")

    handles = [
        mpatches.Patch(color=lut[category], label=category) for category in unique_categories
    ]

    # Add the legend
    col_legend1 = g.fig.legend(
        handles=handles,
        title="Annotation",
        bbox_to_anchor=(1.0, 0.5),
        bbox_transform=plt.gcf().transFigure,
        loc="upper left",
        fontsize=20,
    )

    plt.gca().add_artist(col_legend1)
    if show:
        return None
    return g


def sample_heatmap(
    data: AnnData,
    distance_metric: Literal[
        "euclidean", "cosine", "correlation", "cityblock", "jaccard", "hamming"
    ] = "correlation",
    show: bool = True,
) -> sns.matrix.ClusterGrid | None:
    """
    Plot a clustermap showing the correlation between samples.

    Parameters
    ----------
    data
        2D numpy array where rows are samples and columns are features.
    distance_metric
        Distance metric to use for calculating pairwise distances between proteins.
        One of 'euclidean', 'cosine', 'correlation', 'cityblock', 'jaccard', 'hamming'
    show
        Whether to display the plot.

    Returns
    -------
    sns.matrix.ClusterGrid or None
        If show=True, returns None. Otherwise returns the seaborn ClusterGrid object.
    """
    # Compute the correlation matrix
    corr = np.corrcoef(data.X, rowvar=False)
    print(corr.shape)
    # Create a clustermap with var_names as annotations
    g = sns.clustermap(
        corr,
        cmap="viridis",
        row_cluster=True,
        col_cluster=True,
        # dendrogram_ratio=(0.0, 0.0),
        xticklabels=data.var_names,
        yticklabels=data.var_names,
    )
    if show:
        plt.show()
        return None
    return g


# def grouped_heatmap(
#     data: AnnData,
#     protein_grouping_key: str,
#     agg_func: NDArrayAxisFunction = np.median,
# ) -> plt.Axis | None:
#     # data = pp.aggregate_proteins(
#     #     data, grouping_columns=protein_grouping_key, agg_func=agg_func
#     # )
#     datat = data.T
#     datat.obs[protein_grouping_key] = adatat.obs["curated_ground_truth_v9.0"].astype(
#         "category"
#     )
#     # adatat.obs.set_index("curated_ground_truth_v9.0", drop=False, inplace=True)
#     adatat
#     sc.pl.matrixplot(
#         adatat,
#         var_names=samples_to_keep,
#         groupby="curated_ground_truth_v9.0",
#         categories_order=orgs,
#         colorbar_title="",
#         cmap="YlGnBu",
#         vmin=0,
#         vmax=5,
#         swap_axes=True,
#         show=False,
#     )


def qsep_heatmap(
    data: AnnData,
    normalize: bool = True,
    ax: plt.Axes = None,
    cmap: str = "RdBu_r",
    vmin: float = None,
    vmax: float = None,
    **kwargs,
) -> plt.Axes:
    """Plot QSep cluster distance heatmap.

    Parameters
    ----------
    data : AnnData
        Annotated data matrix containing QSep results.
    normalize : bool, optional
        If True, normalize distances by diagonal values.
        Defaults to True.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, current axes will be used.
    cmap : str, optional
        Colormap to use. Defaults to "RdBu_r".
    **kwargs
        Additional arguments passed to sns.heatmap.

    Returns
    -------
    matplotlib.axes.Axes
        The axes object with the plot.
    """
    if ax is None:
        ax = plt.gca()

    try:
        distances = data.uns["cluster_distances"]["distances"]
        clusters = data.uns["cluster_distances"]["clusters"]
    except KeyError:
        raise ValueError(
            "Cluster distances not found in data.uns['cluster_distances'], run gr.tl.qsep_score first"
        )

    if normalize:
        # Normalize by diagonal values
        norm_distances = distances / np.diag(distances)[:, np.newaxis]
        plot_data = norm_distances[::-1]
        tvmin = 1.0
        tvmax = np.max(norm_distances)
    else:
        plot_data = distances[::-1]
        tvmin = None
        tvmax = None

    if vmin is None:
        vmin = tvmin
    if vmax is None:
        vmax = tvmax

    # Create heatmap
    sns.heatmap(
        plot_data,
        xticklabels=clusters,
        yticklabels=clusters[::-1],  # Reverse the y-axis labels
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        ax=ax,
        **kwargs,
    )

    ax.set_title("QSep Cluster Distances" + (" (Normalized)" if normalize else ""))

    return ax


def qsep_boxplot(
    data: AnnData,
    normalize: bool = True,
    ax: plt.Axes = None,
    palette: str = "Set2",
    **kwargs,
) -> plt.Axes:
    """Plot QSep cluster distances as boxplots.

    Parameters
    ----------
    data : AnnData
        Annotated data matrix containing QSep results.
    normalize : bool, optional
        If True, normalize distances by diagonal values.
        Defaults to True.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, current axes will be used.
    palette : str, optional
        Color palette for the boxplots. Defaults to "Set2".
    **kwargs
        Additional arguments passed to sns.boxplot.

    Returns
    -------
    matplotlib.axes.Axes
        The axes object with the plot.
    """
    if ax is None:
        ax = plt.gca()

    try:
        distances = data.uns["cluster_distances"]["distances"]
        clusters = data.uns["cluster_distances"]["clusters"]
    except KeyError:
        raise ValueError(
            "Cluster distances not found in data.uns['cluster_distances'], run gr.tl.qsep_score first"
        )

    if normalize:
        # Normalize by diagonal values
        distances = distances / np.diag(distances)[:, np.newaxis]

    # Create DataFrame for plotting
    plot_data = []
    for i, ref_cluster in enumerate(clusters):
        for j, target_cluster in enumerate(clusters):
            plot_data.append(
                {
                    "Reference Cluster": ref_cluster,
                    "Target Cluster": target_cluster,
                    "Distance": distances[i, j],
                    "color": "grey" if i == j else "red",
                }
            )
    plot_df = pd.DataFrame(plot_data)

    # Create boxplot
    sns.boxplot(
        data=plot_df,
        x="Distance",
        y="Reference Cluster",
        color="grey",
        orient="h",
        ax=ax,
        legend=False,
        showfliers=False,
        **kwargs,
    )

    # Add individual points
    sns.stripplot(
        data=plot_df,
        x="Distance",
        y="Reference Cluster",
        hue="color",
        # hue="Target Cluster",
        orient="h",
        size=4,
        # color=".3",
        alpha=0.6,
        ax=ax,
        legend=False,
    )

    # Customize plot
    if normalize:
        ax.axvline(x=1 if normalize else 0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("QSep Cluster Distances" + (" (Normalized)" if normalize else ""))

    # Move legend outside
    # ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)

    return ax
