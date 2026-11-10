"""Tests for independent (one-vs-rest) label diffusion and the competitive_diffusion
rename / knn_annotation deprecation."""

import warnings

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scanpy as sc

import grassp as gr

from grassp.util import get_matrix


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
    P, terms = get_matrix(a, "ann_diffusion_probabilities")
    assert P.shape == (a.n_obs, len(gs))
    assert terms == [str(t) for t in gs]
    assert terms == list(gs)  # the matrix names itself; no uns side-channel
    assert a.uns["ann_diffusion_alpha"].shape == (len(gs),)
    assert 0.0 <= P.min() and P.max() <= 1.0
    # per-term (one-vs-rest), NOT a simplex: some rows carry mass on >1 term (blob + AorB)
    assert (P.sum(axis=1) > 1.5).any()
    assert "ann_diffusion_probability" in a.obs
    assert "ann_diffusion" in a.obs


def test_likelihood_resolver_prefers_specific_over_broad(blob_adata):
    # the broad AorB should be explained away by the specific blobA/blobB
    a, gs = blob_adata
    gr.tl.independent_diffusion(a, gs, gene_key="gene_symbol", resolve="likelihood")
    resolved = a.obs["ann_diffusion"].astype(object)
    assert (resolved == "AorB").sum() == 0
    # each blob's proteins resolve mostly to their own specific term
    for b, name in [(0, "blobA"), (1, "blobB"), (2, "blobC")]:
        sub = resolved[a.obs["_blob"] == b]
        assert (sub == name).mean() > 0.7


@pytest.mark.parametrize("calibration", ["size_aware", "shrunk", "pooled", "per_term", "none"])
def test_calibration_modes_run(blob_adata, calibration):
    a, gs = blob_adata
    gr.tl.independent_diffusion(
        a, gs, gene_key="gene_symbol", calibration=calibration, resolve=None
    )
    P, _ = get_matrix(a, "ann_diffusion_probabilities")
    assert np.isfinite(P).all()


@pytest.mark.parametrize("resolve", ["likelihood", "specific", "argmax"])
def test_resolve_modes_produce_labels(blob_adata, resolve):
    a, gs = blob_adata
    gr.tl.independent_diffusion(a, gs, gene_key="gene_symbol", resolve=resolve)
    assert a.obs["ann_diffusion"].notna().any()


def test_copy_does_not_mutate(blob_adata):
    a, gs = blob_adata
    out = gr.tl.independent_diffusion(a, gs, gene_key="gene_symbol", inplace=False)
    assert out is not None
    assert "ann_diffusion_probabilities" in out.obsm
    assert "ann_diffusion_probabilities" not in a.obsm


@pytest.mark.parametrize("alias", ["knn_annotation", "competitive_propagation"])
def test_deprecated_aliases_warn(alias):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            getattr(gr.tl, alias)(ad.AnnData(np.zeros((2, 2))))
        except Exception:
            pass
    assert any(issubclass(x.category, DeprecationWarning) for x in w)


@pytest.mark.parametrize("alias", ["knn_annotation", "competitive_propagation"])
def test_deprecated_aliases_keep_old_key_added(blob_adata, alias):
    """The aliases must not silently move a caller's output columns.

    ``competitive_diffusion`` defaults to ``key_added="competitive_diffusion"``, but code
    written against the old name reads ``obs["competitive_propagation"]``, so both aliases
    pin the historical default.
    """
    a, _ = blob_adata
    a.obs["markers"] = [f"m{v}" if i % 3 else None for i, v in enumerate(a.obs["_blob"])]
    a.obs["markers"] = a.obs["markers"].astype("category")

    b = a.copy()
    gr.tl.competitive_diffusion(a, gt_col="markers", verbose=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        getattr(gr.tl, alias)(b, gt_col="markers", verbose=False)

    assert "competitive_diffusion" in a.obs and "competitive_diffusion_probabilities" in a.obsm
    assert (
        "competitive_propagation" in b.obs
        and "competitive_propagation_probabilities" in b.obsm
    )
    # only the key differs -- the computation is the same one
    assert (
        a.obs["competitive_diffusion"].astype(str)
        == b.obs["competitive_propagation"].astype(str)
    ).all()

    # an explicit key_added still wins over the pinned default
    c = a.copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        getattr(gr.tl, alias)(c, gt_col="markers", key_added="explicit", verbose=False)
    assert "explicit" in c.obs


def test_competitive_diffusion_exported():
    assert callable(gr.tl.competitive_diffusion)
    assert callable(gr.tl.competitive_propagation)


def test_resolve_diffusion_standalone(blob_adata):
    a, gs = blob_adata
    gr.tl.independent_diffusion(a, gs, gene_key="gene_symbol", resolve="likelihood")
    gr.tl.resolve_diffusion(a, gs, mode="specific", out_key="spec")
    assert a.obs["spec"].notna().any()


def test_unannotated_proteins_do_not_dilute_the_score(blob_adata):
    """The raw score is the fraction of the *annotated* neighbourhood carrying a term.

    Drop every other protein from the vocabulary entirely: those stay in the graph but
    carry no term. Inside blob A every annotated neighbour is a blobA member, so the score
    must stay near 1 rather than falling to the ~0.5 an all-protein reference would give.
    """
    a, gs = blob_adata
    genes = a.obs["gene_symbol"].to_numpy()
    kept = set(genes[::2])
    sparse_gs = {t: [g for g in m if g in kept] for t, m in gs.items()}
    gr.tl.independent_diffusion(
        a, sparse_gs, gene_key="gene_symbol", calibration="none", resolve=None
    )
    P, terms = get_matrix(a, "ann_diffusion_probabilities")
    in_a = (a.obs["_blob"] == 0).to_numpy()
    assert np.median(P[in_a, terms.index("blobA")]) > 0.85


def test_duplicate_gene_entries_are_one_unit_of_evidence(blob_adata):
    """Map entries sharing a gene symbol (isoforms, duplicate accessions) count once.

    Three blob-A proteins get a twin entry with the same symbol and a near-identical
    profile, so a term of those three genes has six entries but three genes. It must not
    be scored (the member floor counts genes), where seeding every entry would make it a
    valid term whose twins score it for each other. The twins are recorded in the params.
    """
    a, gs = blob_adata
    rng = np.random.RandomState(1)
    dup = np.where(a.obs["_blob"].to_numpy() == 0)[0][:3]
    twin = ad.AnnData(a.X[dup] + 0.01 * rng.randn(len(dup), a.n_vars))
    twin.obs["gene_symbol"] = a.obs["gene_symbol"].to_numpy()[dup]
    twin.obs["_blob"] = a.obs["_blob"].to_numpy()[dup]
    twin.obs_names = [f"twin{i}" for i in range(len(dup))]
    b = ad.concat([a, twin])
    sc.pp.neighbors(b, n_neighbors=15)
    gs = dict(gs, twins=list(a.obs["gene_symbol"].to_numpy()[dup]))
    gr.tl.independent_diffusion(b, gs, gene_key="gene_symbol", calibration="none")
    P, terms = get_matrix(b, "ann_diffusion_probabilities")
    assert P[:, terms.index("twins")].max() == 0
    assert b.uns["ann_diffusion_params"]["duplicate_gene_entries"] == 3


def test_multilabel_lists_leaves_only(blob_adata):
    """The multi-location call lists the most specific called terms; parents are implied.

    AorB contains blobA, so a blob-A protein's compact label must say blobA, not
    ``blobA; AorB``.
    """
    a, gs = blob_adata
    gr.tl.independent_diffusion(a, gs, gene_key="gene_symbol")
    in_a = (a.obs["_blob"] == 0).to_numpy()
    compact = a.obs["ann_diffusion_label_compact"][in_a].dropna().astype(str)
    assert len(compact) > 0.8 * in_a.sum()
    assert not compact.str.contains("AorB").any()
    assert (compact == "blobA").mean() > 0.8


def test_only_nested_terms_explain_each_other():
    """Containment counts in the likelihood resolver only at ``>= nest_threshold``.

    Two terms sharing half their genes (containment 0.5) are co-localised, not nested: a
    protein probable for both keeps both as active leaves -- a dual location. Soft
    containment would let either explain half of the other away and keep only one, which
    is what lowering the threshold to 0.5 reproduces.
    """
    from grassp.tools.diffusion import _resolve

    gs = {"T1": [f"g{i}" for i in range(20)], "T2": [f"g{i}" for i in range(10, 30)]}
    P = np.array([[0.9, 0.9]])
    sizes = np.array([20, 20])
    args = (P, ["T1", "T2"], gs, sizes, "likelihood", 0.5, 1.0, 0.4, 3, 12, 0)
    labels, leaves = _resolve(*args, 0.8)
    assert set(leaves[0]) == {"T1", "T2"} and labels[0] in {"T1", "T2"}
    labels, leaves = _resolve(*args, 0.5)
    assert len(leaves[0]) == 1 and labels[0] == leaves[0][0]


def test_re_resolving_with_the_same_parameters_is_a_no_op(blob_adata):
    a, gs = blob_adata
    gr.tl.independent_diffusion(a, gs, gene_key="gene_symbol")
    before = a.obs["ann_diffusion"].astype(object).copy()
    compact = list(a.obs["ann_diffusion_label_compact"])
    gr.tl.resolve_diffusion(a, gs, gene_key="gene_symbol")
    pd.testing.assert_series_equal(a.obs["ann_diffusion"].astype(object), before)
    assert list(a.obs["ann_diffusion_label_compact"]) == compact


def test_min_term_size_floor(blob_adata):
    a, gs = blob_adata
    # a floor larger than every term leaves nothing eligible -> all NA
    gr.tl.independent_diffusion(
        a, gs, gene_key="gene_symbol", resolve="likelihood", min_term_size=10_000
    )
    assert a.obs["ann_diffusion"].notna().sum() == 0
    # a modest floor still annotates most proteins (fallback to larger ancestor terms)
    gr.tl.independent_diffusion(
        a, gs, gene_key="gene_symbol", resolve="likelihood", min_term_size=5
    )
    assert a.obs["ann_diffusion"].notna().mean() > 0.8
