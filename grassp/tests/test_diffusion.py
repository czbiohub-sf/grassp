"""Tests for independent (one-vs-rest) label diffusion and the competitive_propagation
rename / knn_annotation deprecation."""
import warnings

import anndata as ad
import numpy as np
import pytest
import scanpy as sc

import grassp as gr


@pytest.fixture
def blob_adata():
    """300 proteins in 3 latent blobs with a kNN graph and gene symbols."""
    rng = np.random.RandomState(0)
    n, d = 300, 20
    centers = rng.randn(3, d) * 3
    lab = rng.randint(0, 3, n)
    X = centers[lab] + rng.randn(n, d)
    genes = np.array([f"G{i}" for i in range(n)])
    a = ad.AnnData(X)
    a.obs["gene_symbol"] = genes
    a.obs["_blob"] = lab
    sc.pp.neighbors(a, n_neighbors=15)
    # overlapping + nested vocabulary: 3 specific blobs + one broad union
    gs = {
        "blobA": list(genes[lab == 0]),
        "blobB": list(genes[lab == 1]),
        "blobC": list(genes[lab == 2]),
        "AorB": list(genes[(lab == 0) | (lab == 1)]),
    }
    return a, gs


def test_independent_diffusion_outputs_and_nonsimplex(blob_adata):
    a, gs = blob_adata
    gr.tl.independent_diffusion(a, gs, gene_key="gene_symbol", resolve="likelihood")
    P = a.obsm["ann_diffusion_probabilities"]
    assert P.shape == (a.n_obs, len(gs))
    assert list(a.uns["ann_diffusion_categories"]) == list(gs)
    assert a.uns["ann_diffusion_alpha"].shape == (len(gs),)
    assert 0.0 <= P.min() and P.max() <= 1.0
    # per-term (one-vs-rest), NOT a simplex: some rows carry mass on >1 term (blob + AorB)
    assert (P.sum(axis=1) > 1.5).any()
    assert "ann_diffusion_maxp" in a.obs
    assert "ann_diffusion_resolved" in a.obs


def test_likelihood_resolver_prefers_specific_over_broad(blob_adata):
    # the broad AorB should be explained away by the specific blobA/blobB
    a, gs = blob_adata
    gr.tl.independent_diffusion(a, gs, gene_key="gene_symbol", resolve="likelihood")
    resolved = a.obs["ann_diffusion_resolved"].astype(object)
    assert (resolved == "AorB").sum() == 0
    # each blob's proteins resolve mostly to their own specific term
    for b, name in [(0, "blobA"), (1, "blobB"), (2, "blobC")]:
        sub = resolved[a.obs["_blob"] == b]
        assert (sub == name).mean() > 0.7


@pytest.mark.parametrize("calibration", ["size_aware", "shrunk", "pooled", "per_term", "none"])
def test_calibration_modes_run(blob_adata, calibration):
    a, gs = blob_adata
    gr.tl.independent_diffusion(a, gs, gene_key="gene_symbol", calibration=calibration,
                                resolve=None)
    P = a.obsm["ann_diffusion_probabilities"]
    assert np.isfinite(P).all()


@pytest.mark.parametrize("resolve", ["likelihood", "specific", "argmax"])
def test_resolve_modes_produce_labels(blob_adata, resolve):
    a, gs = blob_adata
    gr.tl.independent_diffusion(a, gs, gene_key="gene_symbol", resolve=resolve)
    assert a.obs["ann_diffusion_resolved"].notna().any()


def test_copy_does_not_mutate(blob_adata):
    a, gs = blob_adata
    out = gr.tl.independent_diffusion(a, gs, gene_key="gene_symbol", copy=True)
    assert out is not None
    assert "ann_diffusion_probabilities" in out.obsm
    assert "ann_diffusion_probabilities" not in a.obsm


def test_knn_annotation_deprecated_alias():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            gr.tl.knn_annotation(ad.AnnData(np.zeros((2, 2))))
        except Exception:
            pass
    assert any(issubclass(x.category, DeprecationWarning) for x in w)


def test_competitive_propagation_exported():
    assert callable(gr.tl.competitive_propagation)


def test_resolve_diffusion_standalone(blob_adata):
    a, gs = blob_adata
    gr.tl.independent_diffusion(a, gs, gene_key="gene_symbol", resolve="likelihood")
    gr.tl.resolve_diffusion(a, gs, mode="specific", out_key="spec")
    assert a.obs["spec"].notna().any()


def test_min_term_size_floor(blob_adata):
    a, gs = blob_adata
    # a floor larger than every term leaves nothing eligible -> all NA
    gr.tl.independent_diffusion(a, gs, gene_key="gene_symbol", resolve="likelihood",
                                min_term_size=10_000)
    assert a.obs["ann_diffusion_resolved"].notna().sum() == 0
    # a modest floor still annotates most proteins (fallback to larger ancestor terms)
    gr.tl.independent_diffusion(a, gs, gene_key="gene_symbol", resolve="likelihood",
                                min_term_size=5)
    assert a.obs["ann_diffusion_resolved"].notna().mean() > 0.8
