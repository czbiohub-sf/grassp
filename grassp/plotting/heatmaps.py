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
) -> sns.matrix.ClusterGrid | None:
    """Clustered heat-map of pair-wise protein distances.

    The function computes pairwise distances between proteins (rows of
    ``data.X``) using :func:`scipy.spatial.distance.pdist` and performs
    hierarchical clustering on the resulting distance matrix.  The heat-map
    is rendered with :func:`seaborn.clustermap`; protein annotations provided
    via ``annotation_key`` are visualised as coloured side bars.

    Parameters
    ----------
    data
        Annotated matrix with proteins as **observations** (rows) and samples
        or features as variables (columns).
    annotation_key
        Column in ``data.obs`` containing categorical annotations (e.g.
        curated sub-cellular compartments) to colour the rows/columns.
    distance_metric
        Distance metric used for the pairwise distances passed to
        :func:`scipy.spatial.distance.pdist`.
    linkage_method
        Linkage strategy for :func:`scipy.cluster.hierarchy.linkage`.
    linkage_metric
        Metric used within the linkage algorithm.  Usually identical to
        ``distance_metric`` but can differ.
    palette
        Matplotlib/Seaborn palette used for the heat-map color scale.
    show
        If ``True`` (default) the plot is shown and the function returns
        ``None``.  If ``False`` the underlying
        :class:`seaborn.matrix.ClusterGrid` object is returned for further
        customisation.

    Returns
    -------
    ClusterGrid object if ``show`` is ``False``, otherwise ``None``.
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
    """Correlation heat-map between samples (columns of ``data``).

    Parameters
    ----------
    data
        AnnData object where the **variables** represent individual samples
        (e.g. pull-downs or fractions).  Correlations are computed
        across the observation axis.
    distance_metric
        Distance metric for :func:`scipy.spatial.distance.pdist`.  Not used
        directly at the moment (the function plots correlations), but kept
        for API symmetry with :func:`protein_clustermap`.
    show
        If ``True`` show the figure and return ``None``.  Otherwise return the
        :class:`seaborn.matrix.ClusterGrid` instance.

    Returns
    -------
    ClusterGrid object if ``show`` is ``False``.
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
    show: bool = True,
    **kwargs,
) -> plt.Axes:
    """Plot QSep cluster distance heatmap."""
    """
    Parameters
    ----------
    data
        AnnData object with cluster-wise QSep distances stored under
        ``data.uns['cluster_distances']`` (see
        :func:`~grassp.tl.qsep_score`).
    normalize
        If ``True`` (default) each distance is divided by the intra-cluster
        distance (diagonal of the matrix).
    ax
        Existing matplotlib :class:`~matplotlib.axes.Axes` to plot on.  If
        ``None`` (default) the current axes are used.
    cmap, vmin, vmax
        Passed to :func:`seaborn.heatmap`.
    **kwargs
        Additional keyword arguments forwarded to
        :func:`seaborn.heatmap`.

    Returns
    -------
    Axes object containing the heat-map if ``show`` is ``False``.
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
    if show:
        plt.show()
        return None
    return ax


def qsep_boxplot(
    data: AnnData,
    normalize: bool = True,
    ax: plt.Axes = None,
    show: bool = True,
    **kwargs,
) -> plt.Axes:
    """Plot QSep cluster distances as boxplots."""
    """
    Parameters
    ----------
    data
        AnnData object with QSep distances (see
        :func:`grassp.tl.qsep_score`).
    normalize
        Whether to divide all distances by the respective intra-cluster
        distance (i.e. make the diagonal equal to 1).
    ax
        Matplotlib axes to plot on.  If ``None``, the current axes are used.
    show
        If ``True`` (default) the plot is shown and the function returns
        ``None``.  If ``False`` the underlying
        :class:`seaborn.matrix.ClusterGrid` object is returned for further
        customisation.
    **kwargs
        Additional keyword arguments passed to :func:`seaborn.boxplot` and
        :func:`seaborn.stripplot`.

    Returns
    -------
    Axes object containing the box plots if ``show`` is ``False``.
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

    if show:
        plt.show()
        return None
    return ax
