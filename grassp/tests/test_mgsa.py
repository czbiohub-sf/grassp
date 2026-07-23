"""Tests for the MGSA module integrated into grassp (``grassp.tools.mgsa``).

Covers the core sampler (synthetic recovery, determinism, default grids, the
numba-free fallback, gene-set loading) and the grassp integration layer
(``calculate_mgsa``, ``mgsa_to_cluster_distribution`` and feeding an MGSA
posterior into ``soft_cluster_annotation`` as a seed distribution).

The R-agreement cross-check lives in the standalone ``mgsapy`` repo and is not
part of the grassp suite.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import importlib

import anndata as ad

import grassp as gp

# `grassp.tools.mgsa` is re-exported as the *function* in grassp/tools/__init__.py,
# which shadows the submodule attribute. Use importlib to get the actual module so
# module-level helpers (`_chain_core`, `load_gmt`, ...) are reachable.
mgsa_mod = importlib.import_module("grassp.tools.mgsa")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def make_synthetic(seed=0, n_pop=2000, n_sets=30, set_size=40,
                   active=("set3", "set11", "set20"), alpha=0.05, beta=0.2):
    """Synthetic dataset with a known set of active gene sets."""
    rng = np.random.default_rng(seed)
    pop = [f"g{i}" for i in range(n_pop)]
    sets = {
        f"set{k}": list(rng.choice(pop, size=set_size, replace=False))
        for k in range(n_sets)
    }
    hidden = set()
    for a in active:
        hidden.update(sets[a])
    o = []
    for g in pop:
        if g in hidden:
            if rng.random() > beta:
                o.append(g)
        elif rng.random() < alpha:
            o.append(g)
    return o, sets, pop, list(active), alpha, beta


def _post_mean(df):
    return float((df["value"] * df["estimate"]).sum())


def _block_anndata(sizes=(10, 10, 10)):
    """AnnData with block-diagonal connectivity (one clique per cluster)."""
    n = sum(sizes)
    A = np.zeros((n, n))
    start = 0
    labels = []
    for b, s in enumerate(sizes):
        A[start:start + s, start:start + s] = 1.0
        labels += [f"c{b}"] * s
        start += s
    np.fill_diagonal(A, 0.0)
    adata = ad.AnnData(np.zeros((n, 2)))
    adata.obsp["connectivities"] = sp.csr_matrix(A)
    adata.obs["cluster"] = pd.Categorical(labels)
    adata.obs["gene_symbol"] = [f"g{i}" for i in range(n)]
    return adata


# --------------------------------------------------------------------------- #
# Core sampler
# --------------------------------------------------------------------------- #
def test_tiny_hand_case():
    res = mgsa_mod.mgsa(
        ["A", "B"],
        {"set1": ["A", "B", "C"], "set2": ["B", "C", "D"]},
        n_steps=200_000, n_restarts=2, thin=50, seed=1,
    )
    assert res.sets_results.loc["set1", "estimate"] > res.sets_results.loc["set2", "estimate"]
    assert res.diagnostics["population_size"] == 4
    assert res.diagnostics["study_set_size_in_population"] == 2


def test_synthetic_recovery():
    o, sets, pop, active, alpha, beta = make_synthetic(seed=0)
    res = mgsa_mod.mgsa(o, sets, population=pop, n_steps=300_000,
                        n_restarts=4, thin=50, seed=42)
    sr = res.sets_results
    for a in active:
        assert sr.loc[a, "estimate"] > 0.9, f"{a} not recovered: {sr.loc[a, 'estimate']}"
    assert sr.drop(index=active)["estimate"].max() < 0.5
    top3 = set(sr.sort_values("estimate", ascending=False).head(3).index)
    assert top3 == set(active)
    assert abs(_post_mean(res.alpha_post) - alpha) < 0.03
    assert abs(_post_mean(res.beta_post) - beta) < 0.08


def test_determinism():
    o, sets, pop, *_ = make_synthetic(seed=0)
    kw = dict(population=pop, n_steps=100_000, n_restarts=2, thin=50, seed=7)
    r1 = mgsa_mod.mgsa(o, sets, **kw)
    r2 = mgsa_mod.mgsa(o, sets, **kw)
    pd.testing.assert_frame_equal(r1.sets_results, r2.sets_results)
    pd.testing.assert_frame_equal(r1.alpha_post, r2.alpha_post)


def test_default_grids():
    # deduplicate_terms=False so the 21 identical sets are kept (they drive the
    # n_sets-dependent default p grid).
    res = mgsa_mod.mgsa(["A"], {f"s{i}": ["A", "B"] for i in range(21)},
                        deduplicate_terms=False, n_steps=1000, thin=10, seed=0)
    np.testing.assert_allclose(res.alpha_post["value"], np.linspace(0.01, 0.3, 10))
    np.testing.assert_allclose(res.beta_post["value"], np.linspace(0.1, 0.95, 10))
    np.testing.assert_allclose(res.p_post["value"], np.linspace(1, 7, 10) / 21)


def test_study_genes_outside_population_dropped():
    res = mgsa_mod.mgsa(["A", "B", "ZZZ"], {"s1": ["A", "B", "C"]},
                        population=["A", "B", "C"], n_steps=5000, thin=10, seed=0)
    assert res.diagnostics["study_genes_dropped"] == 1
    assert res.diagnostics["study_set_size_in_population"] == 2


def test_numpy_fallback_path(monkeypatch):
    """The un-jitted (numba-free) MCMC core produces a valid result."""
    if not getattr(mgsa_mod._chain_core, "py_func", None):
        pytest.skip("numba not installed; fallback path is already the default")
    monkeypatch.setattr(mgsa_mod, "_chain_core", mgsa_mod._chain_core.py_func)
    monkeypatch.setattr(mgsa_mod, "_toggle", mgsa_mod._toggle.py_func)
    monkeypatch.setattr(mgsa_mod, "_score", mgsa_mod._score.py_func)
    res = mgsa_mod.mgsa(["A", "B"], {"set1": ["A", "B", "C"], "set2": ["B", "C", "D"]},
                        method="mcmc", n_steps=20_000, n_restarts=1, thin=20, seed=1)
    assert res.diagnostics["method"] == "mcmc"
    assert res.sets_results.loc["set1", "estimate"] > res.sets_results.loc["set2", "estimate"]


def test_default_method_is_exact():
    """Small vocabularies default to the exact (variance-free) method."""
    res = mgsa_mod.mgsa(["A", "B"], {"set1": ["A", "B", "C"], "set2": ["B", "C", "D"]})
    assert res.diagnostics["method"] == "exact"
    # exact posterior -> zero Monte-Carlo std error
    assert float(res.sets_results.loc["set1", "std_error"]) == 0.0


def test_exact_matches_mcmc():
    """Exact and MCMC give consistent per-set posteriors on a mid-size case."""
    o, sets, pop, active, *_ = make_synthetic(
        seed=0, n_sets=18, set_size=40, active=("set3", "set8", "set14")
    )
    ex = mgsa_mod.mgsa(o, sets, population=pop, method="exact", max_active=4)
    mc = mgsa_mod.mgsa(o, sets, population=pop, method="mcmc",
                       n_steps=300_000, n_restarts=4, thin=50, seed=1)
    assert ex.diagnostics["method"] == "exact"
    a = ex.sets_results["estimate"]
    b = mc.sets_results["estimate"].reindex(a.index)
    assert np.corrcoef(a.to_numpy(), b.to_numpy())[0, 1] > 0.99
    assert np.abs(a.to_numpy() - b.to_numpy()).max() < 0.05
    # both recover the planted active sets
    for s in active:
        assert a[s] > 0.9 and b[s] > 0.9


def test_auto_falls_back_to_mcmc():
    """`auto` uses MCMC when exact enumeration would be too large."""
    sets = {f"s{i}": ["A", "B"] for i in range(40)}
    res = mgsa_mod.mgsa(["A", "B"], sets, method="auto", deduplicate_terms=False,
                        exact_max_configs=100, n_steps=20_000, n_restarts=1, seed=0)
    assert res.diagnostics["method"] == "mcmc"


def test_load_gmt_dict_and_file(tmp_path):
    d = {"s1": ["A", "B"], "s2": ["C"]}
    assert mgsa_mod.load_gmt(d) == d
    gmt = tmp_path / "test.gmt"
    gmt.write_text("TermA\tsource\tG1\tG2\tG3\nTermB\tsource\tG4\n")
    loaded = mgsa_mod.load_gmt(str(gmt))
    assert loaded == {"TermA": ["G1", "G2", "G3"], "TermB": ["G4"]}


# --------------------------------------------------------------------------- #
# grassp integration layer
# --------------------------------------------------------------------------- #
def test_calculate_mgsa_wrapper():
    """Per-cluster MGSA writes the right top compartment and a posterior matrix."""
    adata = _block_anndata((20, 20, 20))
    genes = list(adata.obs["gene_symbol"])
    sets = {"ER": genes[:20] + genes[40:50], "Nucleus": genes[20:40], "LD": genes[45:60]}
    post = gp.tl.calculate_mgsa(
        adata, cluster_key="cluster", gene_name_key="gene_symbol",
        gene_sets=sets, obs_key_added="mgsa_top",
        n_steps=50_000, n_restarts=2, seed=0, verbose=False,
    )
    # posterior matrix: clusters x sets
    assert list(post.index) == ["c0", "c1", "c2"]
    assert set(post.columns) == {"ER", "Nucleus", "LD"}
    # top compartment per cluster
    top = adata.obs.groupby("cluster", observed=True)["mgsa_top"].first().to_dict()
    assert top["c0"] == "ER" and top["c1"] == "Nucleus"
    # full matrix + MAP indicator persisted to uns
    assert "mgsa_top_posterior" in adata.uns
    assert adata.uns["mgsa_top_posterior"].shape == (3, 3)
    assert "mgsa_top_map" in adata.uns
    assert adata.uns["mgsa_top_map"].shape == (3, 3)
    assert set(np.unique(adata.uns["mgsa_top_map"].to_numpy())) <= {0, 1}


def test_mgsa_to_cluster_distribution():
    post = pd.DataFrame(
        {"ER": [0.99, 0.1, 0.9], "LD": [0.02, 0.1, 0.85], "Nucleus": [0.0, 0.1, 0.01]},
        index=["single", "diffuse", "dual"],
    )
    # MAP: single->ER only; diffuse->none (empty); dual->ER+LD
    mapm = pd.DataFrame(
        {"ER": [1, 0, 1], "LD": [0, 0, 1], "Nucleus": [0, 0, 0]},
        index=["single", "diffuse", "dual"],
    )
    # use_map (recommended): MAP filters which sets get mass
    Q, cats = gp.tl.mgsa_to_cluster_distribution(post, map_matrix=mapm, unknown_label="unknown")
    assert cats[-1] == "unknown"
    np.testing.assert_allclose(Q.sum(axis=1).to_numpy(), 1.0, atol=1e-9)
    assert Q.loc["single", "ER"] > 0.9            # confident single
    assert Q.loc["diffuse", "unknown"] == 1.0     # empty MAP -> all unknown
    assert Q.loc["dual", "ER"] > 0.3 and Q.loc["dual", "LD"] > 0.3  # MAP pair shared

    # use_map=True requires a map_matrix
    with pytest.raises(ValueError):
        gp.tl.mgsa_to_cluster_distribution(post, unknown_label="unknown")

    # use_map=False: all marginals used, rows still sum to 1
    Q2, cats2 = gp.tl.mgsa_to_cluster_distribution(post, use_map=False, unknown_label=None)
    assert "unknown" not in cats2
    np.testing.assert_allclose(Q2.sum(axis=1).to_numpy(), 1.0, atol=1e-9)


def test_mgsa_seed_feeds_soft_cluster_annotation():
    """An MGSA posterior can drive soft_cluster_annotation via cluster_distribution."""
    adata = _block_anndata((15, 15, 15))
    genes = list(adata.obs["gene_symbol"])
    sets = {"ER": genes[:15] + genes[30:38], "Nucleus": genes[15:30], "LD": genes[33:45]}
    post = gp.tl.calculate_mgsa(
        adata, cluster_key="cluster", gene_name_key="gene_symbol",
        gene_sets=sets, obs_key_added="mgsa_top",
        n_steps=50_000, n_restarts=2, seed=0, verbose=False,
    )
    Q, cats = gp.tl.mgsa_to_cluster_distribution(
        post, map_matrix=adata.uns["mgsa_top_map"], unknown_label="unknown"
    )
    gp.tl.soft_cluster_annotation(
        adata, cluster_key="cluster", key_added="ann_mgsa",
        cluster_distribution=(Q, cats), resolve="entropy_null",
        null=None, verbose=False,   # null=None -> fast, deterministic eff_k gate
    )
    # every protein resolved to a single compartment; c0->ER, c1->Nucleus
    types = adata.obs["ann_mgsa_resolved_type"].value_counts().to_dict()
    assert types.get("single", 0) == adata.n_obs
    prim = adata.obs.groupby("cluster", observed=True)["ann_mgsa_resolved"].agg(
        lambda s: s.mode().iloc[0]
    ).to_dict()
    assert prim["c0"] == "ER" and prim["c1"] == "Nucleus"
