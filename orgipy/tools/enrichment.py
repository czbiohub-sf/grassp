from __future__ import annotations
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from anndata import AnnData

import gseapy
import pandas as pd
import scanpy

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
