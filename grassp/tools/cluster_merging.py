from __future__ import annotations
from itertools import combinations
from typing import TYPE_CHECKING, Literal, Optional

import numpy as np
import scanpy as sc
import scipy.cluster.hierarchy as sch

from pandas.api.types import CategoricalDtype
from scipy.sparse import issparse
from scipy.spatial.distance import squareform
from scipy.stats import fisher_exact

from .enrichment import _load_gmt, calculate_cluster_enrichment
from .mgsa import calculate_mgsa, mgsa

__all__ = [  # re-export private helper for callers/tests that imported it here
    "_load_gmt",
    "calculate_cluster_enrichment",
    "dendrogram_cherry_pairs",
    "merge_clusters_go",
    "merge_small_clusters",
    "paga_dendrogram",
]

if TYPE_CHECKING:
    from anndata import AnnData


# ── Pair-testing helpers ──────────────────────────────────────────────────────


def _term_split_pvalue_two_sided(setA: set, setB: set, setTerm: set) -> float:
    """Fisher's exact test: are clusters A and B differentially enriched for *setTerm*?

    A non-significant p-value means the term is not better explained by keeping
    the clusters separate — i.e. it is safe to merge them.

    Parameters
    ----------
    setA, setB
        Gene sets for the two clusters being compared.
    setTerm
        Gene set for the compartment term being tested.

    Returns
    -------
    float
        Two-sided Fisher's exact test p-value.
    """
    a = len(setA & setTerm)
    b = len(setB & setTerm)
    table = [[a, len(setA) - a], [b, len(setB) - b]]
    _, p = fisher_exact(table, alternative='two-sided')
    return p


def _best_term(
    adata: AnnData,
    cluster_col: str,
    cluster_id: str,
    compartment_col: str = 'Cell_compartment',
) -> str:
    """Return the modal compartment term for *cluster_id*.

    Parameters
    ----------
    adata
        AnnData with proteins as observations.
    cluster_col
        ``adata.obs`` column with cluster labels.
    cluster_id
        Cluster to inspect.
    compartment_col
        ``adata.obs`` column with per-protein compartment annotation.

    Returns
    -------
    str
        Most frequent compartment term in the cluster.
    """
    return (
        adata.obs.loc[adata.obs[cluster_col] == cluster_id, compartment_col]
        .value_counts()
        .index[0]
    )


def _gene_set(
    adata: AnnData,
    cluster_col: str,
    cluster_id: str,
    gene_name_key: str = 'Gene_name_canonical',
) -> set[str]:
    """Return unique gene symbols for all proteins in *cluster_id*.

    Parameters
    ----------
    adata
        AnnData with proteins as observations.
    cluster_col
        ``adata.obs`` column with cluster labels.
    cluster_id
        Cluster to inspect.
    gene_name_key
        ``adata.obs`` column holding gene symbols.

    Returns
    -------
    set[str]
        Unique gene symbols in the cluster.
    """
    return set(adata.obs.loc[adata.obs[cluster_col] == cluster_id, gene_name_key].unique())


def _test_pair(
    adata: AnnData,
    cluster_col: str,
    pair: tuple[str, str],
    gene_sets: dict[str, list[str]],
    gene_name_key: str,
    compartment_col: str,
) -> tuple[float, float, float, bool]:
    """Test one cluster pair for mergeability via Fisher's exact test.

    Parameters
    ----------
    adata
        AnnData with proteins as observations.
    cluster_col
        ``adata.obs`` column with cluster labels.
    pair
        Two cluster IDs to compare.
    gene_sets
        Gene-set dict mapping term name → gene list.
    gene_name_key
        ``adata.obs`` column with gene symbols.
    compartment_col
        ``adata.obs`` column with compartment annotation.

    Returns
    -------
    tuple of (min_p, p1, p2, terms_agree)
        min_p       – ``min(p1, p2)``, used for greedy sort ordering.
        p1          – Fisher p-value for the best term of cluster 1.
        p2          – Fisher p-value for the best term of cluster 2.
        terms_agree – True when both clusters share the same best term
                      (unconditional merge candidate, p-values set to 1.0).
    """
    c1, c2 = pair
    t1 = _best_term(adata, cluster_col, c1, compartment_col)
    t2 = _best_term(adata, cluster_col, c2, compartment_col)

    if t1 == t2:
        return (0.0, 1.0, 1.0, True)

    genes1 = _gene_set(adata, cluster_col, c1, gene_name_key)
    genes2 = _gene_set(adata, cluster_col, c2, gene_name_key)

    term_set1 = set(gene_sets.get(t1, []))
    term_set2 = set(gene_sets.get(t2, []))

    p1 = _term_split_pvalue_two_sided(genes1, genes2, term_set1)
    p2 = _term_split_pvalue_two_sided(genes1, genes2, term_set2)

    return (min(p1, p2), p1, p2, False)


def _should_merge(p1: float, p2: float, terms_agree: bool, cutoff: float) -> bool:
    """Return True if a pair should be merged based on Fisher test results.

    Parameters
    ----------
    p1, p2
        Fisher p-values for the two compartment terms.
    terms_agree
        If True, both clusters share the same best compartment term.
    cutoff
        Fisher-test p-value threshold; a pair merges when neither term's test is
        significant (both ``p > cutoff``).

    Returns
    -------
    bool
        True if the pair should be merged.
    """
    return terms_agree or (p1 > cutoff and p2 > cutoff)


# ── MGSA-based merging ─────────────────────────────────────────────────────────


def _annotate_clusters(
    adata: AnnData,
    cluster_col: str,
    gene_sets: dict[str, list[str]],
    gene_name_key: str,
    compartment_col: str,
    merge_method: str,
    n_steps: int,
    n_restarts: int,
    seed: Optional[int],
    max_active: int = 4,
) -> None:
    """Populate ``compartment_col`` with the per-cluster top compartment.

    Uses ORA (:func:`calculate_cluster_enrichment`) or MGSA
    (:func:`~grassp.tl.calculate_mgsa`) depending on ``merge_method``. The MGSA
    path additionally writes per-cluster (log) evidence to
    ``adata.uns[f'{compartment_col}_evidence']`` (used by the merge test).
    """
    if merge_method == 'mgsa_evidence':
        calculate_mgsa(
            adata,
            cluster_key=cluster_col,
            gene_name_key=gene_name_key,
            gene_sets=gene_sets,
            obs_key_added=compartment_col,
            max_active=max_active,
            min_posterior=0.0,  # always assign a top compartment (like ORA threshold=1.0)
            n_steps=n_steps,
            n_restarts=n_restarts,
            seed=seed,
            return_result=False,
            verbose=False,
        )
    else:
        calculate_cluster_enrichment(
            adata,
            cluster_key=cluster_col,
            gene_name_key=gene_name_key,
            gene_sets=gene_sets,
            obs_key_added=compartment_col,
            enrichment_ranking_metric='Adjusted P-value',
            enrichment_threshold=1.0,  # Always assign a top term
        )


def _bf_from_uns(adata: AnnData, compartment_col: str) -> dict[str, float]:
    """Per-cluster log-Bayes-factor (log_evidence - log_null) from the round's MGSA."""
    ev = adata.uns[f'{compartment_col}_evidence']
    return {
        str(cl): float(ev.loc[cl, 'log_evidence'] - ev.loc[cl, 'log_null']) for cl in ev.index
    }


def _merge_score_mgsa_evidence(
    adata: AnnData,
    cluster_col: str,
    pair: tuple[str, str],
    gene_sets: dict[str, list[str]],
    gene_name_key: str,
    bf_cache: dict[str, float],
    population: list[str],
    max_active: int,
) -> float:
    """Model-comparison merge score for one cluster pair.

    ``score = BF(c1 ∪ c2) - BF(c1) - BF(c2)``, where ``BF(S) = logE(S) - logE_null(S)``
    is the MGSA log-Bayes-factor of "some compartment active" vs. "nothing active".
    Subtracting each set's own null cancels the population-background term, so the
    score compares the *enrichment structure* of the merged cluster against the two
    parts. It is positive when the parts share a compartment (splitting double-pays
    that compartment's false-negative cost), ~0 when there is no signal either way,
    and negative when the parts are distinct compartments (evidence against merging).
    """
    c1, c2 = pair
    bf1 = bf_cache.get(str(c1), 0.0)
    bf2 = bf_cache.get(str(c2), 0.0)
    genes = list(
        _gene_set(adata, cluster_col, c1, gene_name_key)
        | _gene_set(adata, cluster_col, c2, gene_name_key)
    )
    d = mgsa(
        genes, gene_sets, population=population, method='exact', max_active=max_active
    ).diagnostics
    bf_union = float(d['log_evidence'] - d['log_null'])
    return bf_union - bf1 - bf2


# ── PAGA dendrogram ───────────────────────────────────────────────────────────


def paga_dendrogram(
    adata: AnnData,
    groupby: str,
    *,
    neighbors_key: Optional[str] = None,
    paga_model: str = 'v1.2',
    linkage_method: str = 'average',
    optimal_ordering: bool = False,
    key_added: Optional[str] = None,
    inplace: bool = True,
) -> Optional[dict]:
    """Build a dendrogram from PAGA connectivity and store it in ``adata.uns``.

    Drop-in replacement for :func:`scanpy.tl.dendrogram` that uses
    ``adata.uns['paga']['connectivities']`` as the similarity matrix instead of
    gene-expression correlations.  The result dict has the same schema as
    :func:`scanpy.tl.dendrogram`, so it can be passed to any scanpy plotting
    function that accepts a ``dendrogram_key`` argument.

    Parameters
    ----------
    adata
        AnnData object.
    groupby
        Categorical ``adata.obs`` column to build the dendrogram for.
    neighbors_key
        Passed to :func:`scanpy.tl.paga`.
    paga_model
        PAGA model version, passed to :func:`scanpy.tl.paga`.
    linkage_method
        Linkage algorithm passed to :func:`scipy.cluster.hierarchy.linkage`.
    optimal_ordering
        Passed to :func:`scipy.cluster.hierarchy.linkage`.
    key_added
        Key written to ``adata.uns``.  Defaults to ``f"dendrogram_{groupby}"``.
    inplace
        If True, write to ``adata.uns`` and return None.  Otherwise return the dict.

    Returns
    -------
    dict or None
        Dendrogram dict when ``inplace=False``, otherwise None.
    """
    if groupby not in adata.obs:
        raise KeyError(f'{groupby!r} not found in adata.obs')
    if not isinstance(adata.obs[groupby].dtype, CategoricalDtype):
        raise ValueError(f'adata.obs[{groupby!r}] must be categorical')

    sc.tl.paga(adata, groups=groupby, neighbors_key=neighbors_key, model=paga_model)

    paga_uns = adata.uns['paga']
    if paga_uns.get('groups') != groupby:
        raise ValueError(
            f"PAGA was computed for groups={paga_uns.get('groups')!r}, "
            f'expected {groupby!r}'
        )

    categories = list(adata.obs[groupby].cat.categories)
    conn = paga_uns['connectivities']
    if issparse(conn):
        conn = conn.toarray()
    conn = np.asarray(conn, dtype=float)

    dist = 1.0 - conn
    np.fill_diagonal(dist, 0.0)

    Z = sch.linkage(
        squareform(dist, checks=False),
        method=linkage_method,
        optimal_ordering=optimal_ordering,
    )
    dendro_info = sch.dendrogram(Z, labels=categories, no_plot=True)

    dat = {
        'linkage': Z,
        'groupby': [groupby],
        'use_rep': None,
        'cor_method': 'paga_connectivity',
        'linkage_method': linkage_method,
        'categories_ordered': dendro_info['ivl'],
        'categories_idx_ordered': dendro_info['leaves'],
        'dendrogram_info': dendro_info,
        'correlation_matrix': conn,
    }

    if key_added is None:
        key_added = f'dendrogram_{groupby}'
    if inplace:
        adata.uns[key_added] = dat
        return None
    return dat


# ── Dendrogram-based candidate discovery ─────────────────────────────────────


def _dendrogram_flat_groups(linkage: np.ndarray, n: int) -> list[frozenset]:
    """Return groups of leaf indices connected by a same-height chain.

    A flat group is built by a chain of merges all at the same height *h*
    where every merge in the chain has at least one leaf child.  This captures
    triplets/quadruplets (a star of leaves equidistant at height *h*) while
    deliberately NOT combining two separate cherry pairs that happen to fuse
    again at the same height.

    Parameters
    ----------
    linkage
        Scipy linkage matrix of shape ``(n-1, 4)``.
    n
        Number of leaves.

    Returns
    -------
    list[frozenset]
        Each frozenset contains the original leaf indices of one flat group.
        Single-leaf groups are excluded; only groups with ≥ 2 leaves are returned.
    """
    node_height: dict[int, float] = {i: 0.0 for i in range(n)}
    for j, (_, __, h, ___) in enumerate(linkage):
        node_height[n + j] = float(h)

    leaf_group: dict[int, Optional[frozenset]] = {i: frozenset([i]) for i in range(n)}
    parent_node: dict[int, int] = {}
    parent_h: dict[int, float] = {}

    for j, (left, right, h, _) in enumerate(linkage):
        node_id = n + j
        li, ri = int(left), int(right)
        parent_node[li] = node_id
        parent_node[ri] = node_id
        parent_h[li] = float(h)
        parent_h[ri] = float(h)

        li_leaf = li < n
        ri_leaf = ri < n
        li_same_h = li_leaf or abs(node_height[li] - float(h)) < 1e-10
        ri_same_h = ri_leaf or abs(node_height[ri] - float(h)) < 1e-10

        if li_same_h and ri_same_h and (li_leaf or ri_leaf):
            lg, rg = leaf_group.get(li), leaf_group.get(ri)
            leaf_group[node_id] = (lg | rg) if (lg is not None and rg is not None) else None
        else:
            leaf_group[node_id] = None

    groups: list[frozenset] = []
    for j, (_, __, h, ___) in enumerate(linkage):
        node_id = n + j
        group = leaf_group[node_id]
        if group is None or len(group) < 2:
            continue
        pid = parent_node.get(node_id)
        chain_continues = (
            pid is not None
            and parent_h.get(node_id, float('inf')) <= float(h) + 1e-10
            and leaf_group.get(pid) is not None
        )
        if not chain_continues:
            groups.append(group)

    return groups


def dendrogram_cherry_pairs(dendrogram_data: dict) -> list[tuple[str, str]]:
    """Return all candidate merge pairs from the dendrogram, expanding ties.

    For each flat region (group of leaves all merging at the same height),
    all pairwise combinations are returned — not just the binary cherry that
    scipy happened to construct first.  This ensures that triplets or
    quadruplets of equidistant clusters are tested exhaustively rather than
    in an arbitrary linkage order.

    Parameters
    ----------
    dendrogram_data
        The dict stored in ``adata.uns[f"dendrogram_{groupby}"]``, as produced
        by :func:`paga_dendrogram`.

    Returns
    -------
    list[tuple[str, str]]
        ``(category_name_a, category_name_b)`` tuples for each candidate pair.
    """
    Z = dendrogram_data['linkage']
    categories_ordered: list[str] = dendrogram_data['categories_ordered']
    categories_idx_ordered: list[int] = dendrogram_data['categories_idx_ordered']
    n = len(categories_ordered)

    original_categories: list[Optional[str]] = [None] * n
    for pos, orig_idx in enumerate(categories_idx_ordered):
        original_categories[orig_idx] = categories_ordered[pos]

    pairs: list[tuple[str, str]] = []
    for group in _dendrogram_flat_groups(Z, n):
        for a, b in combinations(sorted(group), 2):
            pairs.append((original_categories[a], original_categories[b]))
    return pairs


# ── One merge round ───────────────────────────────────────────────────────────


def _one_merge_round_dendrogram(
    adata: AnnData,
    cluster_col: str,
    gene_sets: dict[str, list[str]],
    pv_cutoff: float,
    connectivity_lower: float,
    gene_name_key: str,
    compartment_col: str,
    verbose: bool,
    merge_log: Optional[list] = None,
    merge_method: str = 'ora',
    n_steps: int = 200_000,
    n_restarts: int = 3,
    seed: Optional[int] = 0,
    merge_threshold: float = 0.0,
    max_active: int = 4,
) -> tuple[dict[str, str], int]:
    """One round of dendrogram flat-group pair testing and greedy merging.

    Candidate pairs are drawn from all pairwise combinations within each flat
    group of the PAGA-connectivity dendrogram (see :func:`dendrogram_cherry_pairs`).
    Pairs are sorted by (same compartment term first, then min p-value descending)
    and applied greedily — each cluster participates in at most one merge per round.

    Parameters
    ----------
    adata
        AnnData with proteins as observations.
    cluster_col
        ``adata.obs`` column with current cluster labels.
    gene_sets
        Gene-set dict mapping term name → gene list.
    pv_cutoff
        Fisher-test significance threshold for the merge decision.
    connectivity_lower
        Minimum PAGA connectivity; pairs below this are skipped.
    gene_name_key
        ``adata.obs`` column with gene symbols.
    compartment_col
        ``adata.obs`` column with compartment annotation.
    verbose
        Print per-pair decisions.
    merge_log
        If provided, a dict entry is appended for every merged pair, recording
        ``c1``, ``c2``, ``min_p``, ``p1``, ``p2``, ``terms_agree``, and
        ``merged_score``.

    Returns
    -------
    tuple[dict[str, str], int]
        ``mapping`` – ``{source_cluster: target_cluster}`` for all merges
        this round, and the number of merges performed.
    """
    dendro_key = f'dendrogram_{cluster_col}'
    cherry_pairs = dendrogram_cherry_pairs(adata.uns[dendro_key])

    conn = adata.uns['paga']['connectivities']
    if issparse(conn):
        conn = conn.toarray()

    candidates: list[dict] = []

    # MGSA-evidence merge decisions reuse the round's `calculate_mgsa` per-cluster
    # (log) evidence and run one union MGSA per candidate pair.
    if merge_method == 'mgsa_evidence':
        population = adata.obs[gene_name_key].astype(str).tolist()
        bf_cache = _bf_from_uns(adata, compartment_col)

    for c1, c2 in cherry_pairs:
        idx1 = adata.obs[cluster_col].cat.categories.get_loc(c1)
        idx2 = adata.obs[cluster_col].cat.categories.get_loc(c2)
        conn_val = conn[idx1, idx2]
        if conn_val < connectivity_lower:
            if verbose:
                print(f'  {c1} vs {c2}: conn={conn_val:.3f} < lower threshold, skip')
            continue

        if merge_method == 'mgsa_evidence':
            # Merge unless there is distinct evidence against it: score >= threshold
            # (default 0) merges, so no-signal / same-compartment pairs merge and
            # only clearly-negative (distinct-compartment) pairs are kept apart.
            score = _merge_score_mgsa_evidence(
                adata,
                cluster_col,
                (c1, c2),
                gene_sets,
                gene_name_key,
                bf_cache,
                population,
                max_active,
            )
            if verbose:
                print(
                    f'  {c1} vs {c2}: conn={conn_val:.3f} merge_score={score:.2f}'
                    f' → {"MERGE candidate" if score >= merge_threshold else "keep separate"}'
                )
            if score >= merge_threshold:
                candidates.append(
                    dict(
                        c1=c1,
                        c2=c2,
                        conn_val=conn_val,
                        min_p=np.nan,
                        p1=np.nan,
                        p2=np.nan,
                        terms_agree=False,
                        merged_score=float(score),
                        # apply strongest-evidence merges first
                        _sortkey=(-score,),
                    )
                )
            continue

        min_p, p1, p2, terms_agree = _test_pair(
            adata, cluster_col, (c1, c2), gene_sets, gene_name_key, compartment_col
        )
        if verbose:
            t1 = _best_term(adata, cluster_col, c1, compartment_col)
            t2 = _best_term(adata, cluster_col, c2, compartment_col)
            print(
                f'  {c1}({t1}) vs {c2}({t2}): conn={conn_val:.3f}'
                f'  p1={p1:.3g} p2={p2:.3g}',
                end='',
            )

        if _should_merge(p1, p2, terms_agree, pv_cutoff):
            if verbose:
                reason = 'terms agree' if terms_agree else 'both non-significant'
                print(f' → {reason}, merge candidate')
            candidates.append(
                dict(
                    c1=c1,
                    c2=c2,
                    conn_val=conn_val,
                    min_p=min_p,
                    p1=p1,
                    p2=p2,
                    terms_agree=terms_agree,
                    merged_score=None,
                    # same term first, then highest min_p (least significant = safest)
                    _sortkey=(not terms_agree, -min_p),
                )
            )
        elif verbose:
            print(' → significant split, skip')

    # Apply merges greedily in order of the per-method sort key.
    candidates.sort(key=lambda x: x['_sortkey'])

    used: set[str] = set()
    mapping: dict[str, str] = {}

    for cand in candidates:
        c1, c2 = cand['c1'], cand['c2']
        if c1 in used or c2 in used:
            continue
        mapping[c1] = c2
        used.add(c1)
        used.add(c2)
        if merge_log is not None:
            merge_log.append(
                {
                    'c1': c1,
                    'c2': c2,
                    'min_p': cand['min_p'],
                    'p1': cand['p1'],
                    'p2': cand['p2'],
                    'terms_agree': cand['terms_agree'],
                    'merged_score': cand.get('merged_score'),
                }
            )
        if verbose:
            t1 = _best_term(adata, cluster_col, c1, compartment_col)
            t2 = _best_term(adata, cluster_col, c2, compartment_col)
            reason = 'terms agree' if cand['terms_agree'] else f"min_p={cand['min_p']:.3g}"
            extra = ''
            if cand.get('merged_score') is not None:
                extra = f"  merged_score={cand['merged_score']:.2f}"
            print(
                f'  {c1}({t1}) vs {c2}({t2}): conn={cand["conn_val"]:.3f}'
                f'  p1={cand["p1"]:.3g} p2={cand["p2"]:.3g}{extra}'
                f' → MERGE ({reason})'
            )

    return mapping, len(mapping)


# ── Dendrogram plot ───────────────────────────────────────────────────────────


def _plot_merge_dendrogram(
    adata: AnnData,
    initial_dendro_data: dict,
    initial_cluster_terms: dict[str, str],
    merge_log: list[dict],
    pv_cutoff: float,
    compartment_col: str,
    ax=None,
) -> None:
    """Draw the initial PAGA dendrogram colored by compartment and merge p-value.

    Vertical legs belonging to a fully-merged subtree are colored by the modal
    compartment term of that subtree.  Horizontal merge bars are colored by
    min p-value (magma scale) when the subtree was fully merged; a circle marker
    is added at each merge node.  Unmerged lines are gray.  Leaf nodes are drawn
    starting at y = -0.1 so that zero-distance merges (connectivity = 1) remain
    visible.

    Parameters
    ----------
    adata
        Used to look up compartment colors from ``adata.uns``.
    initial_dendro_data
        The dendrogram dict captured before any merges.
    initial_cluster_terms
        Mapping of cluster ID → best compartment term at the start of the run.
    merge_log
        List of dicts as populated by :func:`_one_merge_round_dendrogram`.
    pv_cutoff
        Bonferroni-corrected p-value cutoff (lower bound of the color scale).
    compartment_col
        Name of the compartment ``adata.obs`` column, used for color lookup.
    ax
        Axes to draw on; a new figure is created when None.
    """
    from collections import Counter

    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    dendro_info = initial_dendro_data['dendrogram_info']
    icoord = dendro_info['icoord']
    dcoord = dendro_info['dcoord']
    ivl: list[str] = dendro_info['ivl']
    n = len(ivl)

    # 1. Leaf x-position → cluster name
    leaf_x: dict[float, str] = {5.0 + 10.0 * i: ivl[i] for i in range(n)}

    # 2. Subtree membership: node x-center → frozenset of cluster names
    subtree: dict[float, frozenset] = {x: frozenset([name]) for x, name in leaf_x.items()}
    for i in range(len(icoord)):
        xs = icoord[i]
        lx, rx = xs[0], xs[3]
        x_center = (lx + rx) / 2.0
        subtree[x_center] = subtree.get(lx, frozenset()) | subtree.get(rx, frozenset())

    # 3. Union-find for final groupings after all merges
    _parent: dict[str, str] = {c: c for c in initial_cluster_terms}

    def _find(x: str) -> str:
        while _parent.get(x, x) != x:
            _parent[x] = _parent.get(_parent[x], _parent[x])
            x = _parent[x]
        return x

    for m in merge_log:
        px, py = _find(m['c1']), _find(m['c2'])
        if px != py:
            _parent[px] = py

    final_group: dict[str, str] = {c: _find(c) for c in _parent}

    def _is_merged(cluster_set: frozenset) -> bool:
        groups = {final_group.get(c) for c in cluster_set if c in final_group}
        return len(groups) == 1

    def _modal_term(cluster_set: frozenset) -> Optional[str]:
        terms = [initial_cluster_terms[c] for c in cluster_set if c in initial_cluster_terms]
        return Counter(terms).most_common(1)[0][0] if terms else None

    def _find_merge_entry(left_cls: frozenset, right_cls: frozenset) -> Optional[dict]:
        for m in merge_log:
            c1, c2 = m['c1'], m['c2']
            if (c1 in left_cls and c2 in right_cls) or (c2 in left_cls and c1 in right_cls):
                return m
        return None

    # 4. Colors
    _gray = '#cccccc'
    terms_agree_color = '#2ca02c'

    cats = adata.obs[compartment_col].cat.categories
    color_key = f'{compartment_col}_colors'
    if color_key in adata.uns and len(adata.uns[color_key]) >= len(cats):
        term_color: dict[str, str] = dict(zip(cats, adata.uns[color_key]))
    else:
        cmap20 = plt.get_cmap('tab20')
        term_color = {
            t: mcolors.to_hex(cmap20(i / max(len(cats), 1))) for i, t in enumerate(cats)
        }

    pv_norm = mcolors.PowerNorm(gamma=0.5, vmin=0, vmax=1)
    pv_cmap = plt.get_cmap('magma_r')

    # 5. Figure layout constants
    _ax_l, _ax_b, _ax_w, _ax_h = 0.04, 0.18, 0.67, 0.78
    _right_x = 0.74
    _cbar_x = 0.77

    own_fig = ax is None
    if own_fig:
        fig = plt.figure(figsize=(max(10, n * 0.35) + 3, 6))
        ax = fig.add_axes([_ax_l, _ax_b, _ax_w, _ax_h])

    lw = 1.5

    # 6. Draw segments
    for i in range(len(icoord)):
        xs, ys = icoord[i], dcoord[i]
        lx, ly = xs[0], ys[0]
        rx, ry = xs[3], ys[3]
        x_center = (lx + rx) / 2.0
        merge_height = ys[1]

        left_cls = subtree.get(lx, frozenset())
        right_cls = subtree.get(rx, frozenset())

        left_color = (
            term_color.get(_modal_term(left_cls) or '', _gray)
            if _is_merged(left_cls)
            else _gray
        )
        right_color = (
            term_color.get(_modal_term(right_cls) or '', _gray)
            if _is_merged(right_cls)
            else _gray
        )

        merge_entry = (
            _find_merge_entry(left_cls, right_cls)
            if _is_merged(left_cls | right_cls)
            else None
        )
        if merge_entry is not None:
            if merge_entry['terms_agree']:
                bar_color = terms_agree_color
            else:
                bar_color = mcolors.to_hex(pv_cmap(pv_norm(merge_entry['min_p'])))
            ax.scatter(
                [x_center],
                [merge_height],
                color='black',
                s=40,
                zorder=5,
                edgecolors='white',
                linewidths=0.5,
            )
        else:
            bar_color = _gray

        # Only true leaf nodes (x in leaf_x AND y==0) get the -0.1 offset;
        # internal nodes with height 0 (connectivity=1) stay at 0.
        actual_ly = -0.1 if (ly == 0.0 and lx in leaf_x) else ly
        actual_ry = -0.1 if (ry == 0.0 and rx in leaf_x) else ry

        ax.plot([xs[0], xs[1]], [actual_ly, merge_height], color=left_color, lw=lw)
        ax.plot([xs[1], xs[2]], [merge_height, merge_height], color=bar_color, lw=lw)
        ax.plot([xs[2], xs[3]], [merge_height, actual_ry], color=right_color, lw=lw)

    # 7. Axes formatting
    ax.set_xticks([5.0 + 10.0 * i for i in range(n)])
    ax.set_xticklabels(ivl, rotation=90, fontsize=6)
    ax.set_ylim(-0.15, 1)
    ax.axhline(0, color='black', lw=0.5, linestyle=':', zorder=0)
    ax.set_ylabel('PAGA distance  (1 − connectivity)')
    ax.set_xlabel('Cluster')
    ax.set_title('Merge dendrogram')

    if not own_fig:
        return

    # 8. Legend
    merged_clusters: set[str] = {c for m in merge_log for c in (m['c1'], m['c2'])}
    merged_terms = sorted(
        {initial_cluster_terms[c] for c in merged_clusters if c in initial_cluster_terms}
    )
    legend_handles = [
        plt.Line2D([0], [0], color=term_color.get(t, _gray), lw=3, label=t)
        for t in merged_terms
    ]
    legend_handles.append(
        plt.Line2D(
            [0],
            [0],
            marker='o',
            linestyle='none',
            color='black',
            markersize=6,
            label='terms agree',
        )
    )
    ax.legend(
        handles=legend_handles,
        loc='upper left',
        bbox_to_anchor=(_right_x, _ax_b + _ax_h),
        bbox_transform=fig.transFigure,
        borderaxespad=0,
        fontsize=7,
        title=compartment_col,
        framealpha=0.9,
    )

    # 9. Colorbar
    cbar_ax = fig.add_axes([_cbar_x, _ax_b, 0.025, 0.25])
    sm = cm.ScalarMappable(cmap=pv_cmap, norm=pv_norm)
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax, label='min p-value')
    plt.show()


# ── Public API ────────────────────────────────────────────────────────────────


def merge_small_clusters(
    adata: AnnData,
    cluster_key: str = 'leiden',
    min_n: int = 3,
    key_added: str | None = None,
    verbose: bool = True,
) -> None:
    """Iteratively merge small clusters into their most-connected neighbor.

    Each round, clusters with fewer than ``min_n`` members are merged into
    the neighbor with the highest PAGA connectivity.  Ties are broken by
    choosing the smallest candidate.  Rounds repeat until no small clusters
    remain.

    Parameters
    ----------
    adata
        AnnData object with a precomputed KNN graph in ``adata.obsp``.
    cluster_key
        ``adata.obs`` column with cluster labels.
    min_n
        Minimum cluster size.  Clusters smaller than this are merged.
    key_added
        ``adata.obs`` column to write merged labels to.
        Defaults to ``f"{cluster_key}_merged"``.
    verbose
        Print per-round merge decisions.
    """
    if key_added is None:
        key_added = f'{cluster_key}_merged'

    adata.obs[key_added] = adata.obs[cluster_key].astype(str).astype('category')

    round_num = 0
    while True:
        round_num += 1
        categories = list(adata.obs[key_added].cat.categories)
        sizes = adata.obs[key_added].value_counts()
        small = [c for c in categories if sizes[c] < min_n]

        if not small:
            if verbose:
                print(f'Round {round_num}: no small clusters. Done.')
            break
        if len(categories) <= 1:
            if verbose:
                print('Only one cluster remains. Done.')
            break

        if verbose:
            print(
                f'\nRound {round_num}: {len(categories)} clusters, '
                f'{len(small)} below min_n={min_n}'
            )

        # Compute PAGA connectivity
        sc.tl.paga(adata, groups=key_added)
        conn = adata.uns['paga']['connectivities']
        if issparse(conn):
            conn = conn.toarray()
        conn = np.asarray(conn, dtype=float)

        cat_idx = {c: i for i, c in enumerate(categories)}

        # For each small cluster, find the highest-connectivity neighbor
        mapping: dict[str, str] = {}
        for cl in small:
            idx = cat_idx[cl]
            conn_row = conn[idx].copy()
            conn_row[idx] = -np.inf  # exclude self

            max_conn = conn_row.max()
            candidates_idx = np.where(np.isclose(conn_row, max_conn))[0]
            candidate_names = [categories[c] for c in candidates_idx]
            target = min(candidate_names, key=lambda c: sizes[c])

            mapping[cl] = target
            if verbose:
                print(
                    f'  {cl} (n={sizes[cl]}) → {target} '
                    f'(n={sizes[target]}, conn={max_conn:.3f})'
                )

        # Resolve chains and cycles via union-find; the larger cluster is
        # kept as the representative label.
        parent: dict[str, str] = {c: c for c in categories}
        uf_size: dict[str, int] = {c: int(sizes[c]) for c in categories}

        def _find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for src, tgt in mapping.items():
            rs, rt = _find(src), _find(tgt)
            if rs == rt:
                continue
            if uf_size[rs] > uf_size[rt]:
                parent[rt] = rs
                uf_size[rs] += uf_size[rt]
            else:
                parent[rs] = rt
                uf_size[rt] += uf_size[rs]

        new_labels = adata.obs[key_added].astype(str).map(_find)
        adata.obs[key_added] = new_labels.astype('category')

        n_after = adata.obs[key_added].nunique()
        if verbose:
            print(f'  → {len(categories) - n_after} merges, now {n_after} clusters')


def merge_clusters_go(
    adata: AnnData,
    pv_cutoff: float = 0.05,
    connectivity_lower: float = 0.5,
    cluster_col: str = 'leiden',
    gene_sets_path: Optional[str | dict] = None,
    species: Literal['hsap', 'mmus', 'scer'] = 'hsap',
    deduplicate_terms: bool = True,
    gene_name_key: str = 'Gene_name_canonical',
    compartment_col: str = 'Cell_compartment',
    key_added: str = 'leiden_merged',
    linkage_method: str = 'average',
    verbose: bool = True,
    plot_iterations: bool = False,
    plot_dendrogram: bool = False,
    merge_method: Literal['ora', 'mgsa_evidence'] = 'ora',
    merge_threshold: float = 0.0,
    max_active: int = 4,
    n_steps: int = 200_000,
    n_restarts: int = 3,
    seed: Optional[int] = 0,
) -> None:
    """Iteratively merge overclustered Leiden clusters using PAGA and GO enrichment.

    Candidate pairs are determined from cherry pairs of a dendrogram built from
    PAGA connectivity (:func:`paga_dendrogram`).  Each round:

    1. Build the PAGA-connectivity dendrogram and identify flat cherry pairs.
    2. For each pair with ``connectivity ≥ connectivity_lower``, run a two-sided
       Fisher's exact test on the best GO term of each cluster.
    3. Pairs where both tests are non-significant at ``pv_cutoff`` (or both
       clusters share the same top term) become merge candidates.
    4. Greedily merge non-overlapping candidates ordered by decreasing min
       p-value (least significant first).
    5. Recompute enrichment annotation and rebuild the dendrogram; repeat until
       no merges occur.

    Results are written to ``adata.obs[key_added]``.

    Parameters
    ----------
    adata
        AnnData object with proteins as observations.  Must already have
        ``cluster_col`` populated (e.g. after :func:`scanpy.tl.leiden`) and a
        precomputed KNN graph in ``adata.obsp`` (e.g. after
        :func:`grassp.pp.neighbors`).
    pv_cutoff
        Significance threshold before Bonferroni correction.
    connectivity_lower
        Minimum PAGA connectivity to consider a cherry pair for merging.
        Pairs below this are ignored regardless of enrichment.
    cluster_col
        ``adata.obs`` column with initial cluster labels.
    gene_sets_path
        Path to a GMT file, a gseapy library name, a pre-loaded
        ``dict[str, list[str]]``, or ``None`` (uses the consolidated UniProt
        subcellular compartment gene sets for the chosen ``species``).
    species
        Species code used to pick the default gene-set file when
        ``gene_sets_path`` is ``None``. One of ``"hsap"`` (human,
        ``consolidated_goterms_human.gmt``), ``"mmus"`` (mouse,
        ``consolidated_goterms_mouse.gmt``), or ``"scer"`` (yeast,
        ``consolidated_goterms_yeast.gmt``). Default ``"hsap"``. Ignored when
        an explicit ``gene_sets_path`` is provided.
    deduplicate_terms
        If ``True`` (default), collapse gene sets with identical membership to a
        single term (keeping the first-seen name) before merging, so synonymous /
        duplicate compartments in fine ontologies do not distort the tests.
    gene_name_key
        ``adata.obs`` column with gene/protein names used for enrichment.
    compartment_col
        ``adata.obs`` column that will hold per-protein compartment annotations
        (written by :func:`~grassp.tl.calculate_cluster_enrichment`).
    key_added
        ``adata.obs`` column to write final merged cluster labels to.
    linkage_method
        Linkage algorithm for :func:`paga_dendrogram`.
    verbose
        Print round-by-round merging decisions.
    plot_iterations
        If True, plot a UMAP after each round showing the current clustering
        and compartment annotation side by side (requires a precomputed UMAP).
    plot_dendrogram
        If True, plot the initial PAGA dendrogram after convergence with leaf
        lines colored by compartment term and merge nodes colored by p-value.
    merge_method
        How to decide merges.

        - ``'ora'`` (default): pairwise Fisher differential-enrichment test on each
          cluster's top term.
        - ``'mgsa_evidence'``: Bayesian model comparison — merge iff
          ``BF(c1∪c2) - BF(c1) - BF(c2) >= merge_threshold`` where
          ``BF(S) = logE(S) - logE_null(S)`` is the MGSA log-Bayes-factor of
          "some compartment active" vs. "nothing active". This *merges by default*
          (no-signal pairs score ~0) and only keeps clusters apart when there is
          distinct evidence against merging (negative score, i.e. the parts are
          different compartments). Handles the small-cluster early-merge regime.
          Self-consistent with the MGSA final annotation.
    merge_threshold
        Only for ``merge_method='mgsa_evidence'``: the minimum merge score to merge.
        ``0.0`` (default) merges unless there is evidence against; a negative value
        merges even in the face of weak evidence against; a positive value requires
        positive evidence for merging.
    max_active
        For ``merge_method='mgsa_evidence'``: cap on simultaneously active sets in
        the exact enumeration (forwarded to :func:`~grassp.tl.calculate_mgsa` /
        :func:`mgsa`).
    n_steps, n_restarts, seed
        MGSA MCMC settings, used only if a run falls back to ``method='mcmc'`` (the
        exact method is used for the small compartment vocabulary). Ignored for
        ``merge_method='ora'``.
    """
    gene_sets = _load_gmt(gene_sets_path, species=species, deduplicate_terms=deduplicate_terms)

    # Initialise working column from the original clustering
    adata.obs[key_added] = adata.obs[cluster_col].astype(str).astype('category')
    n_initial = adata.obs[key_added].nunique()

    # Run enrichment on the initial clusters to populate compartment_col
    _annotate_clusters(
        adata,
        key_added,
        gene_sets,
        gene_name_key,
        compartment_col,
        merge_method,
        n_steps,
        n_restarts,
        seed,
        max_active,
    )

    # Build initial dendrogram and capture state for the optional plot
    paga_dendrogram(adata, key_added, linkage_method=linkage_method)
    initial_dendro_data = adata.uns[f'dendrogram_{key_added}'].copy()
    initial_cluster_terms: dict[str, str] = {
        c: _best_term(adata, key_added, c, compartment_col)
        for c in adata.obs[key_added].cat.categories
    }
    merge_log: list[dict] = []

    if plot_iterations:
        sc.pl.umap(
            adata,
            color=[key_added, compartment_col],
            legend_loc='on data',
            legend_fontsize=5,
            s=10,
            legend_fontoutline=True,
            title=[
                f'Round 0 (initial): {n_initial} clusters',
                f'Round 0 (initial): {compartment_col}',
            ],
        )

    round_num = 0
    while True:
        round_num += 1
        n_clusters = adata.obs[key_added].nunique()
        if verbose:
            print(f'\n=== Round {round_num}: {n_clusters} clusters ===')

        mapping, n_merges = _one_merge_round_dendrogram(
            adata,
            cluster_col=key_added,
            gene_sets=gene_sets,
            pv_cutoff=pv_cutoff,
            connectivity_lower=connectivity_lower,
            gene_name_key=gene_name_key,
            compartment_col=compartment_col,
            verbose=verbose,
            merge_log=merge_log if plot_dendrogram else None,
            merge_method=merge_method,
            n_steps=n_steps,
            n_restarts=n_restarts,
            seed=seed,
            merge_threshold=merge_threshold,
            max_active=max_active,
        )

        if n_merges == 0:
            if verbose:
                print('No merges this round. Converged.')
            if plot_dendrogram:
                _plot_merge_dendrogram(
                    adata=adata,
                    initial_dendro_data=initial_dendro_data,
                    initial_cluster_terms=initial_cluster_terms,
                    merge_log=merge_log,
                    pv_cutoff=pv_cutoff,
                    compartment_col=compartment_col,
                )
            break

        # Apply mapping: relabel source clusters to target cluster
        new_labels = adata.obs[key_added].astype(str).copy()
        for src, tgt in mapping.items():
            new_labels[new_labels == src] = tgt
        adata.obs[key_added] = new_labels.astype('category')

        n_after = adata.obs[key_added].nunique()
        if verbose:
            print(f'  → {n_merges} merges, now {n_after} clusters')

        if n_after <= 1:
            if verbose:
                print('Only one cluster remains. Done.')
            break

        # Recompute annotation for merged clusters and rebuild dendrogram
        _annotate_clusters(
            adata,
            key_added,
            gene_sets,
            gene_name_key,
            compartment_col,
            merge_method,
            n_steps,
            n_restarts,
            seed,
            max_active,
        )
        paga_dendrogram(adata, key_added, linkage_method=linkage_method)

        if plot_iterations:
            sc.pl.umap(
                adata,
                color=[key_added, compartment_col],
                legend_loc='on data',
                legend_fontsize=5,
                s=10,
                legend_fontoutline=True,
                title=[
                    f'Round {round_num}: {n_after} clusters',
                    f'Round {round_num}: {compartment_col}',
                ],
            )
