from __future__ import annotations
from typing import TYPE_CHECKING, Literal, Optional, Union

if TYPE_CHECKING:
    from anndata import AnnData

import numpy as np
import pandas as pd
import scanpy

from scipy import cluster, spatial

rank_proteins_groups = scanpy.tl.rank_genes_groups


def calculate_cluster_enrichment(
    data: AnnData,
    cluster_key: str = "leiden",
    gene_name_key: str = "Gene_name_canonical",
    gene_sets: str = "custom_goterms_genes_reviewed.gmt",
    obs_key_added: str = "Cell_compartment",
    enrichment_ranking_metric: Literal["P-value", "Odds Ratio", "Combined Score"] = "P-value",
    return_enrichment_res: bool = True,
    inplace: bool = True,
) -> Optional[Union[AnnData, pd.DataFrame]]:
    """Gene-set enrichment for each *cluster*.

    For every category in ``data.obs[cluster_key]`` the function performs an
    *Enrichr* analysis via ``gseapy`` using the list of proteins (genes)
    present in that cluster.  The most significant term (according to
    ``enrichment_ranking_metric``) is written back to ``data.obs`` under
    ``obs_key_added``.

    Parameters
    ----------
    data
        Input :class:`~anndata.AnnData` with proteins as observations.
    cluster_key
        Categorical column in ``data.obs`` containing cluster labels.
    gene_name_key
        Column in ``data.obs`` that holds gene symbols – required by
        *gseapy*.
    gene_sets
        Gene set database to use for enrichment analysis
    obs_key_added
        Name of the column that will store the top enriched term per
        cluster.
    enrichment_ranking_metric
        Column used to rank results within each cluster.  Valid options are
        ``"P-value"``, ``"Odds Ratio"`` and ``"Combined Score"``.
    return_enrichment_res
        If ``True`` return the full :class:`pandas.DataFrame` of Enrichr
        results.
    inplace
        If ``True`` (default) annotate *data* in place.  Otherwise a modified
        copy is returned.

    Returns
    -------
    Behaviour depends on ``inplace`` and ``return_enrichment_res``:

    - ``inplace=True``  → annotate *data*; return the results
        DataFrame if ``return_enrichment_res`` else ``None``.
    - ``inplace=False`` → return either a new :class:`~anndata.AnnData`
        *or* a ``(adata, results)`` tuple.
    """
    try:
        import gseapy
    except ImportError:
        raise Exception(
            "To calculate cluster enrichment, please install the `gseapy` python package (pip install gseapy)."
        )

    obs_df = data.obs
    groups = obs_df.groupby(cluster_key)

    enrichr_results = []
    enrichr_top_terms = dict()

    for n, group in groups:
        gene_list = group[gene_name_key].tolist()
        er = gseapy.enrich(
            gene_list=gene_list,
            gene_sets=gene_sets,
            background=obs_df[gene_name_key].tolist(),
            outdir=None,
        ).results

        er = pd.DataFrame(er)
        top_term = er.sort_values(enrichment_ranking_metric, ascending=True).iloc[0]["Term"]
        enrichr_top_terms[n] = top_term
        er[cluster_key] = n
        enrichr_results.append(er)

    enrichr_results = pd.concat(enrichr_results)

    if inplace:
        # Add top term annotation to data.obs
        obs_df[obs_key_added] = groups[cluster_key].transform(
            lambda x: enrichr_top_terms[x.name]
        )
        if return_enrichment_res:
            return enrichr_results
        return None
    else:
        if return_enrichment_res:
            return data, enrichr_results
        return data


# Calculate pairwise distance matrix between samples
def calculate_distance_matrix(
    data: AnnData,
    distance_metric: str = "correlation",
    linkage_method: str = "average",
    linkage_metric: str = "cosine",
) -> pd.DataFrame:
    """Pairwise sample-to-sample distance matrix.

    Parameters
    ----------
    data
        AnnData object (proteins × samples).
    distance_metric
        Metric passed to :func:`scipy.spatial.distance.pdist`.
    linkage_method, linkage_metric
        Parameters forwarded to :func:`scipy.cluster.hierarchy.linkage` – used
        here solely to obtain an ordering of samples for the returned matrix.

    Returns
    -------
    pandas.DataFrame
        Square distance matrix with samples in dendrogram order.
    """

    distance_matrix = spatial.distance.pdist(data.X, metric=distance_metric)
    linkage = cluster.hierarchy.linkage(
        distance_matrix, method=linkage_method, metric=linkage_metric
    )  # Hierarchical clustering
    row_order = np.array(
        cluster.hierarchy.dendrogram(linkage, no_plot=True, orientation="bottom")["leaves"]
    )

    distance_matrix = spatial.distance.squareform(distance_matrix)
    distance_matrix = distance_matrix[row_order, :][:, row_order]
    distance_matrix.shape
