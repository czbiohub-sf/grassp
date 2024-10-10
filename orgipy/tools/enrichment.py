from __future__ import annotations
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from anndata import AnnData

import gseapy
import numpy as np
import pandas as pd
import scanpy

from scipy import cluster, spatial

rank_proteins_groups = scanpy.tl.rank_genes_groups


def calculate_cluster_enrichment(
    data: AnnData,
    cluster_key="leiden",
    gene_name_key="Gene_name_canonical",
    gene_sets="custom_goterms_genes_reviewed.gmt",
    obs_key_added="Cell_compartment",
    enrichment_ranking_metric: Literal["P-value", "Odds Ratio", "Combined Score"] = "P-value",
    return_enrichment_res=True,
    inplace=True,
):
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

    distance_matrix = spatial.distance.pdist(data.X, metric=distance_metric)
    linkage = cluster.hierarchy.linkage(
        distance_matrix, method=linkage_method, metric=linkage_metric
    )  # Hierarchical clustering
    row_order = np.array(
        cluster.hierarchy.dendrogram(linkage, no_plot=True, orientation='bottom')['leaves']
    )

    distance_matrix = spatial.distance.squareform(distance_matrix)
    distance_matrix = distance_matrix[row_order, :][:, row_order]
    distance_matrix.shape
