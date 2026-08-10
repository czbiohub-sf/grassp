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


def _deduplicate_gene_sets(
    gene_sets: dict[str, list[str]],
) -> dict[str, list[str]]:
    """Collapse terms with identical gene membership, keeping the first-seen name.

    Fine-grained ontologies (e.g. Enrichr's ``COMPARTMENTS_Curated_2025``) list the
    same compartment under several synonymous names with byte-identical gene sets
    (``PEROXISOME`` ≡ ``MICROBODY``, ``PEROXISOMAL MEMBRANE`` ≡ ``MICROBODY
    MEMBRANE``, …). Under a joint model such as MGSA these interchangeable sets
    split the posterior mass among themselves, so no single one clears a posterior
    threshold even when the compartment is unambiguous. Removing the exact
    duplicates before analysis restores the undiluted signal.

    Terms are grouped by their (order-independent) gene membership; for each group
    the term appearing first in ``gene_sets`` is retained and the rest are dropped.
    """
    seen: dict[frozenset, str] = {}
    deduped: dict[str, list[str]] = {}
    for term, genes in gene_sets.items():
        sig = frozenset(genes)
        if sig in seen:
            continue
        seen[sig] = term
        deduped[term] = list(genes)
    return deduped


def _load_gmt(
    path: str | dict[str, list[str]] | None = None,
    species: Literal["hsap", "mmus", "scer"] = "hsap",
    deduplicate_terms: bool = True,
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
    deduplicate_terms
        If ``True`` (default), collapse terms with identical gene membership via
        :func:`_deduplicate_gene_sets`, keeping the first-seen name. A no-op for
        libraries without duplicate sets (e.g. the bundled consolidated sets).

    Returns
    -------
    dict[str, list[str]]
        Mapping of term name → list of gene symbols.
    """
    if isinstance(path, dict):
        return _deduplicate_gene_sets(path) if deduplicate_terms else dict(path)
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

        gene_sets = gp.get_library(name=path)
        return _deduplicate_gene_sets(gene_sets) if deduplicate_terms else gene_sets
    gene_sets: dict[str, list[str]] = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            term = parts[0]
            genes = [g for g in parts[2:] if g]
            gene_sets[term] = genes
    return _deduplicate_gene_sets(gene_sets) if deduplicate_terms else gene_sets


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
    deduplicate_terms: bool = True,
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
    deduplicate_terms
        If ``True`` (default), collapse gene sets with identical membership to a
        single term (keeping the first-seen name) before enrichment.
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
    gene_sets = _load_gmt(gene_sets, species=species, deduplicate_terms=deduplicate_terms)

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


def enrichment_to_cluster_distribution(
    enr_res: pd.DataFrame,
    cluster_key: str = "leiden",
    ranking_metric: Literal[
        "Adjusted P-value",
        "Adjusted P-value Bonferroni",
        "P-value",
    ] = "Adjusted P-value Bonferroni",
    threshold: float = 0.05,
    temperature: float = 1.0,
    s0: float = 0.0,
    s_max: float = 300.0,
    unknown_label: str | None = "unknown",
    weight_by: Literal["evidence", "odds_ratio"] = "evidence",
) -> tuple[pd.DataFrame, list[str]]:
    """Turn a per-cluster enrichment table into soft label distributions.

    The default annotation pipeline assigns each cluster its single most
    significant term (a hard, one-hot label).  This discards the fact that a
    cluster may be only marginally enriched, or enriched for several terms of
    near-equal significance.  When such a hard label is subsequently propagated
    over the neighbour graph (:func:`~grassp.tl.knn_annotation`) it produces
    over-confident, and often wrong, annotations in poorly resolved regions.

    This function instead converts the *full* per-(cluster, term) enrichment
    table into a probability distribution over a shared compartment vocabulary,
    with an explicit ``unknown`` class that absorbs uncertainty.  The resulting
    distribution can be broadcast to proteins and used as a *soft seed* for
    :func:`~grassp.tl.knn_annotation` (see
    :func:`~grassp.tl.soft_cluster_annotation`).

    For every cluster ``c`` and term ``t`` an evidence score is computed as

    .. math::

        s(c, t) = \\mathrm{clip}\\bigl(\\log_{10}(\\text{threshold} /
                  p_{\\text{adj}}(c, t)),\\; 0,\\; s_{\\max}\\bigr)

    so that a term sitting exactly at ``threshold`` contributes zero evidence,
    stronger enrichment contributes more (up to ``s_max``), and only terms with
    ``p_adj <= threshold`` count.  Probabilities follow a tempered softmax with
    an explicit unknown logit ``s0``::

        w(c, t)       = exp(s(c, t) / temperature)   for significant terms
        w(c, unknown) = exp(s0 / temperature)
        Q[c, :]       = w / w.sum()

    With the defaults (``s0 = 0``) the unknown class ties a term that is only
    marginally significant, so weakly/ambiguously enriched clusters keep most of
    their mass on ``unknown`` instead of committing to a wrong label.  As
    ``temperature -> 0`` the distribution collapses onto the single best term,
    recovering the behaviour of the hard pipeline.

    Parameters
    ----------
    enr_res
        Per-(cluster, term) enrichment table as returned by
        :func:`calculate_cluster_enrichment` (``return_enrichment_res=True``).
        Must contain the columns ``"Term"``, ``ranking_metric`` and
        ``cluster_key``.
    cluster_key
        Name of the column in ``enr_res`` holding the cluster labels.
    ranking_metric
        p-value column used as the evidence metric.  Defaults to
        ``"Adjusted P-value Bonferroni"`` (corrected for both term- and
        cluster-multiplicity), matching the pipeline default.
    threshold
        Significance threshold.  Terms with ``ranking_metric > threshold`` (or
        missing) are treated as non-significant and contribute no evidence.
    temperature
        Softmax temperature.  Larger values give more diffuse distributions;
        ``temperature -> 0`` approaches winner-take-all.
    s0
        Evidence logit assigned to the ``unknown`` class.  ``0`` (default)
        corresponds to a hypothetical term sitting exactly at ``threshold``.
    s_max
        Upper cap on the per-term evidence score ``log10(threshold/padj)``. Its
        only role is to bound a literal ``padj == 0`` (which would give ``inf``);
        it must stay well above realistic ``-log10(padj)`` values. The default
        ``300`` is effectively "off". Do **not** set this small: a small cap
        collapses the evidence of any two strongly-enriched terms to equal,
        producing spurious ties between a compartment and an overlapping one
        (e.g. Lysosome/Endosome, 40S ribosome/Nucleolus).
    weight_by
        How admissible (significant) terms are weighted against each other:

        - ``"evidence"`` (default): logit is the clipped ``log10(threshold/padj)``
          evidence score — confidence-weighted, size-dependent.
        - ``"odds_ratio"``: logit is ``ln`` of the Haldane-Anscombe odds ratio
          (gseapy's already-HA-corrected ``"Odds Ratio"`` column). Effect-size
          weighted and size-independent, so ``P(A)/P(B) = (OR_A/OR_B)^(1/T)``.
          The p-value still gates which terms are admissible; only significant
          terms contribute regardless of this setting.
    unknown_label
        Name of the appended background/unknown category. If ``None``, no unknown
        class is added: the distribution is purely relative over significant terms
        and clusters with no significant term receive a uniform fallback (which
        propagates to a high-entropy, unresolved distribution). Provided mainly to
        ablate the value of the unknown class against the entropy-null resolver.

    Returns
    -------
    Q : pandas.DataFrame
        Row-stochastic matrix indexed by cluster label, with one column per
        compartment in the vocabulary (plus ``unknown_label`` as the final column
        unless ``unknown_label is None``).  Each row sums to 1.
    categories : list of str
        The ordered column vocabulary (significant terms, plus ``unknown_label``
        last unless it is ``None``), suitable as ``seed_categories`` for
        :func:`~grassp.tl.knn_annotation`.
    """
    if ranking_metric not in enr_res.columns:
        raise KeyError(
            f"ranking_metric '{ranking_metric}' not found in enr_res columns: "
            f"{list(enr_res.columns)}"
        )
    if cluster_key not in enr_res.columns:
        raise KeyError(f"cluster_key '{cluster_key}' not found in enr_res columns.")

    # Pivot to a (cluster x term) matrix of the ranking metric. Fallback rows
    # (empty clusters) carry Term = NaN and lack the adjusted-p columns, so they
    # (and any missing cluster/term combination) become NaN here and are treated
    # as non-significant below.
    df = enr_res.dropna(subset=["Term"]).copy()
    df[ranking_metric] = pd.to_numeric(df[ranking_metric], errors="coerce")
    pv = df.pivot_table(
        index=cluster_key, columns="Term", values=ranking_metric, aggfunc="min"
    )
    # Guarantee every cluster is represented, even if all its terms were NaN.
    all_clusters = pd.Index(enr_res[cluster_key].unique(), name=cluster_key)
    pv = pv.reindex(all_clusters)

    # Evidence score s = clip(log10(threshold / padj), 0, s_max). The upper cap
    # only bounds a literal padj == 0 (which gives s = inf); it must stay far above
    # realistic -log10(padj) values. A small cap (the old default of 8) is a bug:
    # in dense data the true compartment is often astronomically significant (e.g.
    # padj ~ 1e-150, s ~ 150) while an overlapping compartment is merely very
    # significant (padj ~ 1e-18, s ~ 18); capping both at 8 collapses a 130-decade
    # difference into an artificial 50/50 tie, which propagation then breaks toward
    # the globally larger compartment (e.g. Lysosome -> Endosome, 40S -> Nucleolus).
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.log10(threshold / pv.to_numpy(dtype=float))
    s = np.clip(s, 0.0, s_max)
    s[~np.isfinite(s)] = 0.0
    s[pv.to_numpy(dtype=float) > threshold] = 0.0
    s = np.nan_to_num(s, nan=0.0)

    # A term contributes only where it is actually significant (padj <= threshold).
    significant = s > 0

    # Keep only terms significant in >=1 cluster, tightening the vocabulary.
    keep_mask = significant.any(axis=0)
    keep_cols = list(pv.columns[keep_mask])
    s = s[:, keep_mask]
    significant = significant[:, keep_mask]

    # Per-term softmax logit. The significance gate (above) is always the p-value;
    # `weight_by` only chooses how admissible terms are *weighted* relative to each
    # other.
    if weight_by == "evidence":
        # logit = clipped -log10(padj) evidence score.
        base_logits = s
    elif weight_by == "odds_ratio":
        # logit = ln(Haldane-Anscombe odds ratio). gseapy's "Odds Ratio" column is
        # already HA-corrected (adds 0.5 to each 2x2 cell), so it is finite and > 0.
        # Weighting by ln(OR) makes P(A)/P(B) = (OR_A / OR_B)^(1/T): relative
        # compartment probability equals relative enrichment odds. Effect size,
        # not confidence, sets the split among significant terms.
        if "Odds Ratio" not in enr_res.columns:
            raise KeyError("weight_by='odds_ratio' requires an 'Odds Ratio' column in enr_res.")
        odf = df.copy()
        odf["Odds Ratio"] = pd.to_numeric(odf["Odds Ratio"], errors="coerce")
        orat = (
            odf.pivot_table(index=cluster_key, columns="Term", values="Odds Ratio", aggfunc="max")
            .reindex(all_clusters)
            .reindex(columns=pv.columns)
            .to_numpy(dtype=float)[:, keep_mask]
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            base_logits = np.log(orat)
        base_logits[~np.isfinite(base_logits)] = 0.0
    else:
        raise ValueError(f"weight_by must be 'evidence' or 'odds_ratio', got {weight_by!r}")

    # Numerically-stable tempered softmax over the significant real terms (+ an
    # optional unknown class at logit s0). Non-significant terms get logit -inf so
    # their weight is exactly 0. Subtracting the per-row max before exp keeps the
    # exponent bounded regardless of how large the evidence (or how small T) is.
    # For weight_by='odds_ratio', s0=0 is a natural baseline (ln(OR)=0 <=> OR=1,
    # i.e. no enrichment), so the unknown class competes as an un-enriched term.
    logits = np.where(significant, base_logits, -np.inf)
    if unknown_label is not None:
        logits = np.concatenate(
            [logits, np.full((logits.shape[0], 1), float(s0))], axis=1
        )
    row_max = np.max(logits, axis=1, keepdims=True)
    row_max = np.where(np.isfinite(row_max), row_max, 0.0)  # rows with all -inf
    with np.errstate(over="ignore"):
        w = np.exp((logits - row_max) / temperature)
    w[~np.isfinite(w)] = 0.0
    den = w.sum(axis=1, keepdims=True)

    if unknown_label is not None:
        # den > 0 always (the unknown logit is finite): weakly/non-enriched clusters
        # (Sig(c)=∅) put all mass on unknown.
        Q_values = w / den
        categories = keep_cols + [unknown_label]
    else:
        # No unknown class: pure relative distribution over significant terms.
        # Clusters with no significant term (den == 0) get a uniform fallback over
        # the vocabulary, which propagates to a high-entropy (unresolved)
        # distribution rather than a spurious confident label.
        empty = den[:, 0] == 0
        Q_values = np.divide(w, den, out=np.zeros_like(w), where=den > 0)
        if empty.any() and w.shape[1] > 0:
            Q_values[empty] = 1.0 / w.shape[1]
        categories = keep_cols

    Q = pd.DataFrame(Q_values, index=pv.index, columns=categories)
    return Q, categories


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
