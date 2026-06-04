from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Optional, Union

if TYPE_CHECKING:
    from anndata import AnnData

import numpy as np
import pandas as pd
import scanpy

from scipy import cluster, spatial

rank_proteins_groups = scanpy.tl.rank_genes_groups


# Map from species code → filename of the bundled consolidated GMT.
# Shared between `calculate_cluster_enrichment` and `merge_clusters_go`.
_SPECIES_TO_GMT_FILENAME: dict[str, str] = {
    "hsap": "consolidated_goterms_human.gmt",
    "mmus": "consolidated_goterms_mouse.gmt",
    "scer": "consolidated_goterms_yeast.gmt",
}


def _load_gmt(
    path: str | dict[str, list[str]] | None = None,
    species: Literal["hsap", "mmus", "scer"] = "hsap",
) -> dict[str, list[str]]:
    """Resolve a gene-set source into a ``{term: [gene, ...]}`` dict.

    Parameters
    ----------
    path
        One of:
        - ``dict`` — returned as-is.
        - existing file path — parsed as GMT.
        - non-existing string — treated as a ``gseapy`` library name and
          fetched via :func:`gseapy.get_library`.
        - ``None`` — uses the consolidated UniProt subcellular-location gene
          sets bundled with grassp, picked according to ``species``.
    species
        Used only when ``path is None``. One of ``"hsap"``, ``"mmus"``,
        ``"scer"``; selects the matching ``consolidated_goterms_*.gmt`` file
        in ``grassp/datasets/external/``.

    Returns
    -------
    dict[str, list[str]]
        Mapping of term name → list of gene symbols.
    """
    if isinstance(path, dict):
        return path
    if path is None:
        if species not in _SPECIES_TO_GMT_FILENAME:
            raise ValueError(
                f"species must be one of {sorted(_SPECIES_TO_GMT_FILENAME)}, "
                f"got {species!r}"
            )
        path = str(
            Path(__file__).parent.parent
            / "datasets"
            / "external"
            / _SPECIES_TO_GMT_FILENAME[species]
        )

    import os

    if not os.path.exists(path):
        import gseapy as gp

        return gp.get_library(name=path)
    gene_sets: dict[str, list[str]] = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            term = parts[0]
            genes = [g for g in parts[2:] if g]
            gene_sets[term] = genes
    return gene_sets


def calculate_cluster_enrichment(
    data: AnnData,
    cluster_key: str = "leiden",
    gene_name_key: str = "Gene_name_canonical",
    gene_sets: str | None = None,
    obs_key_added: str = "Cell_compartment",
    enrichment_ranking_metric: Literal[
        "Adjusted P-value",
        "Adjusted P-value Bonferroni",
        "P-value",
        "Odds Ratio",
        "Combined Score",
    ] = "Adjusted P-value Bonferroni",
    enrichment_threshold: float = 0.05,
    species: Literal["hsap", "mmus", "scer"] = "hsap",
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
        Path to a Gene set database to use for enrichment analysis in .gmt format
        If None, enrichment is performed against the uniprot subcellular compartment annotations.
        We have found that this is a good default and tends to be less noisy than GO CC.
    obs_key_added
        Name of the column that will store the top enriched term per
        cluster.
    enrichment_ranking_metric
        Column used to rank results within each cluster.  Valid options are
        ``"Adjusted P-value"``, ``"P-value"``, ``"Odds Ratio"`` and ``"Combined Score"``.
    enrichment_threshold
        Threshold for the enrichment ranking metric. Only terms with a ranking metric value less than or equal to this threshold are considered.
    species
        Species code used to pick the default gene-set file when ``gene_sets``
        is ``None``. One of ``"hsap"`` (human, ``consolidated_goterms_human.gmt``),
        ``"mmus"`` (mouse, ``consolidated_goterms_mouse.gmt``), or
        ``"scer"`` (yeast, ``consolidated_goterms_yeast.gmt``). Default
        ``"hsap"``. Ignored when an explicit ``gene_sets`` path is provided.
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
    groups = obs_df.groupby(cluster_key, observed=True)

    enrichr_results = []
    enrichr_top_terms = dict()
    sort_ascending = (
        enrichment_ranking_metric == "P-value"
        or enrichment_ranking_metric == "Adjusted P-value"
        or enrichment_ranking_metric == "Adjusted P-value Bonferroni"
    )
    # print(f"Sorting {enrichment_ranking_metric} in {'ascending' if sort_ascending else 'descending'} order")

    # Resolve the gene-set source to a {term: [gene, ...]} dict. Handles dict
    # passthrough, file paths, gseapy library names, and the species-specific
    # default when `gene_sets is None`.
    gene_sets = _load_gmt(gene_sets, species=species)

    for n, group in groups:
        gene_list = group[gene_name_key].astype(str).tolist()
        er = gseapy.enrich(
            gene_list=gene_list,
            gene_sets=gene_sets,
            background=obs_df[gene_name_key].astype(str).tolist(),
            outdir=None,
        ).results
        if len(er) > 0:
            er = pd.DataFrame(er)
            # Bonferroni adjustment for testing multiple clusters (it's already adjusted for multiple testing against terms by gseapy, but we need to adjust for the number of clusters)
            er["Adjusted P-value Bonferroni"] = np.minimum(
                er["Adjusted P-value"] * len(groups), 1.0
            )
            top_term = er.sort_values(
                enrichment_ranking_metric, ascending=sort_ascending
            ).iloc[0]["Term"]
            enrichr_top_terms[n] = top_term
            er[cluster_key] = n
            enrichr_results.append(er)
        else:
            # Create a single-row DataFrame (all NaN except correct cluster_key)
            er = pd.DataFrame(
                [{"Term": np.nan, "P-value": 1, "Odds Ratio": 0, "Combined Score": 0}],
                columns=["Term", "P-value", "Odds Ratio", "Combined Score"],
            )
            enrichr_top_terms[n] = "NaN"
            er[cluster_key] = n
            enrichr_results.append(er)

    enrichr_results = pd.concat(enrichr_results)

    if inplace:
        # Add top term annotation to data.obs
        obs_df[obs_key_added] = groups[cluster_key].transform(
            lambda x: enrichr_top_terms[x.name]
        )
        obs_df[f"{obs_key_added}_{enrichment_ranking_metric}"] = groups[cluster_key].transform(
            lambda x: enrichr_results[enrichr_results[cluster_key] == x.name]
            .sort_values(enrichment_ranking_metric, ascending=sort_ascending)
            .iloc[0][enrichment_ranking_metric]
        )
        if sort_ascending:
            obs_df.loc[
                obs_df[f"{obs_key_added}_{enrichment_ranking_metric}"] >= enrichment_threshold,
                f"{obs_key_added}",
            ] = np.nan
        else:
            obs_df.loc[
                obs_df[f"{obs_key_added}_{enrichment_ranking_metric}"] <= enrichment_threshold,
                f"{obs_key_added}",
            ] = np.nan
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
