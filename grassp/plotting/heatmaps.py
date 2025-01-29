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
    row_order = np.array(
        sch.dendrogram(linkage, no_plot=True, orientation="bottom")["leaves"]
    )
    distance_matrix = sp.distance.squareform(distance_matrix)
    distance_matrix = distance_matrix[row_order, :][:, row_order]

    gt = data.obs[annotation_key].values[row_order]
    unique_categories = np.unique(gt)
    lut = dict(
        zip(unique_categories, sns.color_palette("tab20", len(unique_categories)))
    )
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
        mpatches.Patch(color=lut[category], label=category)
        for category in unique_categories
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
