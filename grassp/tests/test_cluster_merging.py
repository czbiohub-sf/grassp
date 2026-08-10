"""Tests for grassp.tools.cluster_merging."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.cluster.hierarchy as sch

from anndata import AnnData

from grassp.tools.cluster_merging import (
    _best_term,
    _dendrogram_flat_groups,
    _gene_set,
    _load_gmt,
    _should_merge,
    _term_split_pvalue_two_sided,
    _test_pair,
    dendrogram_cherry_pairs,
    merge_clusters_go,
    paga_dendrogram,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_cluster_adata(
    n_proteins: int = 60,
    n_features: int = 15,
    n_clusters: int = 4,
    seed: int = 0,
) -> AnnData:
    """Build a minimal AnnData with cluster and compartment annotations.

    Proteins are stored as observations (grassp convention).  Clusters are
    slightly separated so PAGA finds real connectivity structure, but with
    enough overlap that inter-cluster KNN edges exist.
    """
    import scanpy as sc

    rng = np.random.default_rng(seed)
    cluster_size = n_proteins // n_clusters
    compartments = ['Mitochondrion', 'Nucleus', 'ER', 'Cytoplasm']

    X_blocks = []
    cluster_labels = []
    compartment_labels = []
    gene_names = []
    for k in range(n_clusters):
        # Small, fixed offset so adjacent clusters share KNN edges
        mean = np.zeros(n_features)
        mean[k % n_features] = 1.5
        block = rng.standard_normal((cluster_size, n_features)) + mean
        X_blocks.append(block)
        cluster_labels.extend([str(k)] * cluster_size)
        compartment_labels.extend([compartments[k % len(compartments)]] * cluster_size)
        gene_names.extend([f'GENE{k}_{i}' for i in range(cluster_size)])

    X = np.vstack(X_blocks).astype(np.float32)
    obs = pd.DataFrame(
        {
            'cluster': pd.Categorical(cluster_labels),
            'Cell_compartment': pd.Categorical(compartment_labels),
            'Gene_name_canonical': gene_names,
        },
        index=[f'prot_{i}' for i in range(n_proteins)],
    )
    var = pd.DataFrame(index=[f'feat_{i}' for i in range(n_features)])
    adata = AnnData(X=X, obs=obs, var=var)

    sc.pp.neighbors(adata, n_neighbors=10, use_rep='X')
    return adata


# ── _load_gmt ─────────────────────────────────────────────────────────────────


def test_load_gmt_dict_passthrough():
    d = {'TermA': ['G1', 'G2'], 'TermB': ['G3']}
    # Distinct sets -> dedup is a no-op; content is preserved (as a copy).
    assert _load_gmt(d) == d
    assert _load_gmt(d, deduplicate_terms=False) == d


def test_load_gmt_deduplicates_identical_terms():
    # TermB is a byte-identical synonym of TermA -> collapsed, first-seen kept.
    d = {'TermA': ['G1', 'G2'], 'TermB': ['G2', 'G1'], 'TermC': ['G3']}
    assert _load_gmt(d) == {'TermA': ['G1', 'G2'], 'TermC': ['G3']}
    assert set(_load_gmt(d, deduplicate_terms=False)) == {'TermA', 'TermB', 'TermC'}


def test_load_gmt_reads_gmt_file(tmp_path):
    gmt_file = tmp_path / 'test.gmt'
    gmt_file.write_text('TermA\tDescription\tGENE1\tGENE2\nTermB\tDesc\tGENE3\n')
    result = _load_gmt(str(gmt_file))
    assert result == {'TermA': ['GENE1', 'GENE2'], 'TermB': ['GENE3']}


def test_load_gmt_skips_short_lines(tmp_path):
    gmt_file = tmp_path / 'test.gmt'
    gmt_file.write_text('TermA\tDescription\tGENE1\n\nTermB\n')
    result = _load_gmt(str(gmt_file))
    assert 'TermA' in result
    assert 'TermB' not in result  # only 1 part, skipped


def test_load_gmt_none_returns_dict():
    # None should load the default bundled GMT; just check it returns a non-empty dict
    result = _load_gmt(None)
    assert isinstance(result, dict)
    assert len(result) > 0


# ── _term_split_pvalue_two_sided ──────────────────────────────────────────────


def test_term_split_pvalue_identical_sets():
    # Identical overlap → no differential enrichment → high p-value
    setA = {'G1', 'G2', 'G3', 'G4'}
    setB = {'G5', 'G6', 'G7', 'G8'}
    setTerm = {'G1', 'G5'}  # one from each → balanced
    p = _term_split_pvalue_two_sided(setA, setB, setTerm)
    assert 0.0 <= p <= 1.0
    assert p > 0.05  # balanced → not significant


def test_term_split_pvalue_extreme():
    # All term genes in setA, none in setB → should be significant
    setA = {'G1', 'G2', 'G3', 'G4', 'G5'}
    setB = {'G6', 'G7', 'G8', 'G9', 'G10'}
    setTerm = {'G1', 'G2', 'G3', 'G4', 'G5'}  # all in setA
    p = _term_split_pvalue_two_sided(setA, setB, setTerm)
    assert p < 0.05


def test_term_split_pvalue_empty_term():
    setA = {'G1', 'G2'}
    setB = {'G3', 'G4'}
    setTerm: set = set()
    p = _term_split_pvalue_two_sided(setA, setB, setTerm)
    assert p == 1.0  # no information → p=1


# ── _best_term ────────────────────────────────────────────────────────────────


def test_best_term_returns_modal():
    obs = pd.DataFrame(
        {
            'cluster': ['A', 'A', 'A', 'B'],
            'Cell_compartment': ['Mito', 'Mito', 'ER', 'ER'],
        },
        index=['p0', 'p1', 'p2', 'p3'],
    )
    adata = AnnData(X=np.zeros((4, 2)), obs=obs)
    assert _best_term(adata, 'cluster', 'A') == 'Mito'
    assert _best_term(adata, 'cluster', 'B') == 'ER'


# ── _gene_set ─────────────────────────────────────────────────────────────────


def test_gene_set_returns_unique():
    obs = pd.DataFrame(
        {
            'cluster': ['A', 'A', 'B'],
            'Gene_name_canonical': ['GENE1', 'GENE1', 'GENE2'],
        },
        index=['p0', 'p1', 'p2'],
    )
    adata = AnnData(X=np.zeros((3, 2)), obs=obs)
    assert _gene_set(adata, 'cluster', 'A') == {'GENE1'}
    assert _gene_set(adata, 'cluster', 'B') == {'GENE2'}


# ── _test_pair ────────────────────────────────────────────────────────────────


def test_test_pair_terms_agree():
    obs = pd.DataFrame(
        {
            'cluster': ['A', 'A', 'B', 'B'],
            'Cell_compartment': ['Mito', 'Mito', 'Mito', 'Mito'],
            'Gene_name_canonical': ['G1', 'G2', 'G3', 'G4'],
        },
        index=[f'p{i}' for i in range(4)],
    )
    adata = AnnData(X=np.zeros((4, 2)), obs=obs)
    gene_sets = {'Mito': ['G1', 'G2', 'G3']}
    min_p, p1, p2, terms_agree = _test_pair(
        adata, 'cluster', ('A', 'B'), gene_sets, 'Gene_name_canonical', 'Cell_compartment'
    )
    assert terms_agree is True
    assert p1 == 1.0
    assert p2 == 1.0
    assert min_p == 0.0


def test_test_pair_terms_differ():
    obs = pd.DataFrame(
        {
            'cluster': ['A', 'A', 'B', 'B'],
            'Cell_compartment': ['Mito', 'Mito', 'ER', 'ER'],
            'Gene_name_canonical': ['G1', 'G2', 'G3', 'G4'],
        },
        index=[f'p{i}' for i in range(4)],
    )
    adata = AnnData(X=np.zeros((4, 2)), obs=obs)
    gene_sets = {'Mito': ['G1', 'G2'], 'ER': ['G3', 'G4']}
    min_p, p1, p2, terms_agree = _test_pair(
        adata, 'cluster', ('A', 'B'), gene_sets, 'Gene_name_canonical', 'Cell_compartment'
    )
    assert terms_agree is False
    assert 0.0 <= p1 <= 1.0
    assert 0.0 <= p2 <= 1.0


# ── _should_merge ─────────────────────────────────────────────────────────────


def test_should_merge_terms_agree():
    assert _should_merge(0.001, 0.001, True, 0.05) is True


def test_should_merge_both_nonsig():
    assert _should_merge(0.9, 0.8, False, 0.05) is True


def test_should_merge_one_sig():
    assert _should_merge(0.001, 0.9, False, 0.05) is False


def test_should_merge_both_sig():
    assert _should_merge(0.001, 0.001, False, 0.05) is False


# ── _dendrogram_flat_groups ───────────────────────────────────────────────────


def test_flat_groups_simple_cherries():
    """Two independent cherry pairs should give two flat groups of size 2."""
    # 4 leaves: A(0), B(1), C(2), D(3)
    # A-B merge at 0.2, C-D merge at 0.3, AB-CD merge at 0.6
    Z = np.array(
        [
            [0, 1, 0.2, 2],  # node 4: {0,1}
            [2, 3, 0.3, 2],  # node 5: {2,3}
            [4, 5, 0.6, 4],  # node 6: {0,1,2,3}
        ]
    )
    groups = _dendrogram_flat_groups(Z, 4)
    assert len(groups) == 2
    assert frozenset([0, 1]) in groups
    assert frozenset([2, 3]) in groups


def test_flat_groups_triplet():
    """A triplet where two leaves merge at h=0.1 and then the third joins at the
    same height should yield one group of 3."""
    # 3 leaves: A(0), B(1), C(2)
    # A-B at h=0.1, then (AB)-C at h=0.1 (same height)
    Z = np.array(
        [
            [0, 1, 0.1, 2],  # node 3: {0,1}
            [3, 2, 0.1, 3],  # node 4: {0,1,2} — same height, one leaf child (C=2)
        ]
    )
    groups = _dendrogram_flat_groups(Z, 3)
    assert len(groups) == 1
    assert frozenset([0, 1, 2]) in groups


def test_flat_groups_no_cherries():
    """When all merges happen at different heights, every pair is its own cherry."""
    Z = np.array(
        [
            [0, 1, 0.1, 2],
            [2, 3, 0.5, 2],
            [4, 5, 0.9, 4],
        ]
    )
    groups = _dendrogram_flat_groups(Z, 4)
    # Each merge is at a unique height → 3 binary cherries
    assert all(len(g) == 2 for g in groups)


# ── dendrogram_cherry_pairs ───────────────────────────────────────────────────


def test_dendrogram_cherry_pairs_returns_names():
    """Cherry pairs should be returned as category name tuples, not indices."""
    # 3 categories: "a", "b", "c" — all merge at height 0
    categories = ['a', 'b', 'c']
    Z = np.array([[0, 1, 0.0, 2], [3, 2, 0.0, 3]])
    # Build a minimal dendrogram_data dict
    dendro_info = sch.dendrogram(Z, labels=categories, no_plot=True)
    dendro_data = {
        'linkage': Z,
        'categories_ordered': dendro_info['ivl'],
        'categories_idx_ordered': dendro_info['leaves'],
        'dendrogram_info': dendro_info,
    }
    pairs = dendrogram_cherry_pairs(dendro_data)
    # All returned items must be tuples of strings
    for a, b in pairs:
        assert isinstance(a, str)
        assert isinstance(b, str)
        assert a in categories
        assert b in categories


# ── paga_dendrogram ───────────────────────────────────────────────────────────


def test_paga_dendrogram_inplace():
    adata = _make_cluster_adata()
    paga_dendrogram(adata, 'cluster')
    assert 'dendrogram_cluster' in adata.uns
    dat = adata.uns['dendrogram_cluster']
    for key in ('linkage', 'categories_ordered', 'categories_idx_ordered', 'dendrogram_info'):
        assert key in dat


def test_paga_dendrogram_not_inplace():
    adata = _make_cluster_adata()
    dat = paga_dendrogram(adata, 'cluster', inplace=False)
    assert dat is not None
    assert 'linkage' in dat


def test_paga_dendrogram_missing_column():
    adata = _make_cluster_adata()
    with pytest.raises(KeyError):
        paga_dendrogram(adata, 'nonexistent_col')


def test_paga_dendrogram_non_categorical():
    adata = _make_cluster_adata()
    adata.obs['str_col'] = 'x'  # plain string, not categorical
    with pytest.raises(ValueError, match='must be categorical'):
        paga_dendrogram(adata, 'str_col')


def test_paga_dendrogram_key_added():
    adata = _make_cluster_adata()
    paga_dendrogram(adata, 'cluster', key_added='my_dendro')
    assert 'my_dendro' in adata.uns


# ── merge_clusters_go ─────────────────────────────────────────────────────────


def _mock_enrichment(adata, cluster_key, gene_name_key, gene_sets, obs_key_added, **kwargs):
    """Simulate calculate_cluster_enrichment by assigning each cluster a fixed term."""
    cluster_to_term = {c: f'Term_{c}' for c in adata.obs[cluster_key].cat.categories}
    adata.obs[obs_key_added] = adata.obs[cluster_key].map(cluster_to_term).astype('category')
    return None


def _mock_enrichment_all_same(
    adata, cluster_key, gene_name_key, gene_sets, obs_key_added, **kwargs
):
    """Assign the same compartment term to all clusters → all pairs should merge."""
    adata.obs[obs_key_added] = pd.Categorical(
        ['Mitochondrion'] * len(adata), categories=['Mitochondrion']
    )
    return None


def test_merge_clusters_go_runs(monkeypatch):
    """merge_clusters_go should complete without errors and write key_added."""
    import grassp.tools.cluster_merging as cm

    adata = _make_cluster_adata(n_clusters=4)
    gene_sets = {
        'Term_0': ['GENE0_0'],
        'Term_1': ['GENE1_0'],
        'Term_2': ['GENE2_0'],
        'Term_3': ['GENE3_0'],
    }

    monkeypatch.setattr(cm, 'calculate_cluster_enrichment', _mock_enrichment)

    merge_clusters_go(
        adata,
        cluster_col='cluster',
        gene_sets_path=gene_sets,
        key_added='merged',
        verbose=False,
    )
    assert 'merged' in adata.obs.columns
    assert hasattr(adata.obs['merged'], 'cat')


def test_merge_clusters_go_reduces_clusters(monkeypatch):
    """When all clusters share the same compartment term, all should merge."""
    import grassp.tools.cluster_merging as cm

    adata = _make_cluster_adata(n_clusters=4)
    gene_sets = {'Mitochondrion': ['GENE0_0']}

    monkeypatch.setattr(cm, 'calculate_cluster_enrichment', _mock_enrichment_all_same)

    n_before = adata.obs['cluster'].nunique()
    merge_clusters_go(
        adata,
        cluster_col='cluster',
        gene_sets_path=gene_sets,
        connectivity_lower=0.0,  # accept all pairs
        key_added='merged',
        verbose=False,
    )
    n_after = adata.obs['merged'].nunique()
    assert n_after < n_before


def test_merge_clusters_go_preserves_all_proteins(monkeypatch):
    """Every protein must be assigned to exactly one cluster after merging."""
    import grassp.tools.cluster_merging as cm

    adata = _make_cluster_adata(n_clusters=4)
    gene_sets = {'Mitochondrion': ['GENE0_0']}

    monkeypatch.setattr(cm, 'calculate_cluster_enrichment', _mock_enrichment_all_same)

    merge_clusters_go(
        adata,
        cluster_col='cluster',
        gene_sets_path=gene_sets,
        connectivity_lower=0.0,
        key_added='merged',
        verbose=False,
    )
    assert adata.obs['merged'].isna().sum() == 0
    assert len(adata.obs['merged']) == len(adata)


def test_merge_clusters_go_converges_no_merge(monkeypatch):
    """When every cluster has a distinct, well-separated compartment term,
    the algorithm should converge with no merges."""
    import grassp.tools.cluster_merging as cm

    adata = _make_cluster_adata(n_clusters=4)
    cluster_size = len(adata) // 4
    # Completely non-overlapping gene sets: each cluster owns its own genes only.
    # Fisher test on the pair (cluster_k, cluster_j) for Term_k will find all
    # Term_k genes in cluster_k and none in cluster_j → highly significant → no merge.
    gene_sets = {f'Term_{k}': [f'GENE{k}_{i}' for i in range(cluster_size)] for k in range(4)}

    monkeypatch.setattr(cm, 'calculate_cluster_enrichment', _mock_enrichment)

    n_before = adata.obs['cluster'].nunique()
    merge_clusters_go(
        adata,
        cluster_col='cluster',
        gene_sets_path=gene_sets,
        connectivity_lower=0.0,
        # Use default pv_cutoff=0.05; Fisher p≈1e-5 for perfectly separated sets,
        # which is below the Bonferroni-adjusted threshold → no merge.
        key_added='merged',
        verbose=False,
    )
    n_after = adata.obs['merged'].nunique()
    assert n_after == n_before


# ── MGSA-based merging ─────────────────────────────────────────────────────────


def test_merge_clusters_go_mgsa_evidence_smoke():
    """merge_method='mgsa_evidence' runs end-to-end and collapses same-compartment clusters."""
    adata = _make_cluster_adata(n_clusters=4)
    genes = list(adata.obs['Gene_name_canonical'])
    gene_sets = {
        'Shared01': [g for g in genes if g.startswith(('GENE0_', 'GENE1_'))],
        'Comp2': [g for g in genes if g.startswith('GENE2_')],
        'Comp3': [g for g in genes if g.startswith('GENE3_')],
    }
    n_before = adata.obs['cluster'].nunique()
    merge_clusters_go(
        adata,
        cluster_col='cluster',
        gene_sets_path=gene_sets,
        key_added='merged',
        merge_method='mgsa_evidence',
        merge_threshold=0.0,
        connectivity_lower=0.0,
        verbose=False,
    )
    assert 'merged' in adata.obs.columns
    assert adata.obs['merged'].nunique() < n_before
    # per-cluster evidence was cached for the model-comparison score
    assert 'Cell_compartment_evidence' in adata.uns
