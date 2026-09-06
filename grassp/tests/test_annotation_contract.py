"""Tests for the shared annotation output contract in :mod:`grassp.tools._annotation`.

The point of the module is that a consumer can read an annotation without knowing which
annotator produced it, so these test the contract itself rather than any one predictor.
"""

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scanpy as sc

import grassp as gr

from grassp.tools import _annotation


@pytest.fixture
def annotated():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(m, 0.5, (30, 6)) for m in (0.0, 2.5, 5.0)])
    data = ad.AnnData(X)
    data.obs_names = [f"P{i}" for i in range(data.n_obs)]
    markers = np.repeat(["ER", "MITO", "NUC"], 30).astype(object)
    markers[::3] = None
    data.obs["markers"] = pd.Categorical(markers)
    sc.pp.neighbors(data, n_neighbors=10, use_rep="X")
    return data


class TestMarkerLabels:
    def test_one_hot_spans_the_declared_vocabulary(self, annotated):
        annotated.obs["markers"] = pd.Categorical(
            annotated.obs["markers"].astype(object), categories=["ER", "GOLGI", "MITO", "NUC"]
        )
        markers = _annotation.marker_labels(annotated, "markers")
        assert list(markers.categories) == ["ER", "GOLGI", "MITO", "NUC"]
        assert markers.one_hot.shape == (annotated.n_obs, 4)
        # the unobserved class gets a column of zeros rather than being dropped
        assert markers.one_hot[:, 1].sum() == 0
        assert np.allclose(markers.one_hot[markers.mask].sum(axis=1), 1.0)

    def test_object_dtype_is_accepted(self, annotated):
        annotated.obs["markers"] = annotated.obs["markers"].astype(object)
        markers = _annotation.marker_labels(annotated, "markers")
        assert markers.mask.sum() > 0

    def test_missing_and_empty_columns_are_named(self, annotated):
        with pytest.raises(KeyError, match="nope"):
            _annotation.marker_labels(annotated, "nope")
        annotated.obs["blank"] = pd.Categorical([None] * annotated.n_obs, categories=["ER"])
        with pytest.raises(ValueError, match="No annotated proteins"):
            _annotation.marker_labels(annotated, "blank")


class TestResolveSimplex:
    P = np.array([[0.7, 0.2, 0.1], [0.4, 0.35, 0.25], [0.1, 0.1, 0.8]])
    CATS = ["ER", "MITO", "unknown"]

    def test_argmax_and_confidence(self):
        labels, probability = _annotation.resolve_simplex(self.P, self.CATS)
        assert list(labels.astype(object)) == ["ER", "ER", "unknown"]
        assert np.allclose(probability, [0.7, 0.4, 0.8])

    def test_min_probability_abstains(self):
        labels, _ = _annotation.resolve_simplex(self.P, self.CATS, min_probability=0.5)
        assert list(labels.astype(object))[:2] == ["ER", None] or pd.isna(
            labels.astype(object)[1]
        )

    def test_unknown_label_abstains_but_keeps_the_column(self):
        labels, probability = _annotation.resolve_simplex(
            self.P, self.CATS, unknown_label="unknown"
        )
        assert pd.isna(labels.astype(object)[2])
        assert "unknown" not in list(labels.categories)
        # the confidence still reports the mass that won
        assert np.isclose(probability[2], 0.8)

    def test_declared_categories_widen_the_label_vocabulary(self):
        labels, _ = _annotation.resolve_simplex(
            self.P[:, :2], ["ER", "MITO"], declared_categories=["ER", "GOLGI", "MITO"]
        )
        assert list(labels.categories) == ["ER", "GOLGI", "MITO"]

    def test_mismatched_width_is_rejected(self):
        with pytest.raises(ValueError, match="must describe the same matrix"):
            _annotation.resolve_simplex(self.P, ["ER", "MITO"])


class TestWrittenSlots:
    """Every annotator must land in the same slots."""

    EXPECTED_OBS = ("{k}", "{k}_probability")

    @pytest.mark.parametrize(
        "run,key",
        [
            (
                lambda d: gr.tl.competitive_diffusion(d, gt_col="markers", verbose=False),
                "competitive_diffusion",
            ),
            (
                lambda d: gr.tl.svm_annotation(d, gt_col="markers", C=1.0, gamma=0.1),
                "svm_annotation",
            ),
        ],
        ids=["competitive_diffusion", "svm_annotation"],
    )
    def test_slots_and_params(self, annotated, run, key):
        run(annotated)
        for template in self.EXPECTED_OBS:
            assert template.format(k=key) in annotated.obs
        assert f"{key}_probabilities" in annotated.obsm
        assert f"{key}_colors" in annotated.uns

        params = annotated.uns[f"{key}_params"]
        assert params["kind"] == "simplex"
        assert params["gt_col"] == "markers"
        assert params["method"]
        assert params["label_separator"] == "; "

        # and the object describes itself well enough to be read back blind
        P, categories, read_params = _annotation.read_annotation(annotated, key)
        assert P.shape == (annotated.n_obs, len(categories))
        assert read_params["kind"] == "simplex"
        assert np.allclose(P.sum(axis=1), 1.0)

    def test_colours_are_one_per_category(self, annotated):
        gr.tl.competitive_diffusion(annotated, gt_col="markers", verbose=False)
        n_categories = len(annotated.obs["competitive_diffusion"].cat.categories)
        assert len(annotated.uns["competitive_diffusion_colors"]) == n_categories


class TestKindGuard:
    def test_entropy_resolver_refuses_per_term_input(self, annotated):
        annotated.obs["gene_symbol"] = [f"G{i}" for i in range(annotated.n_obs)]
        gene_sets = {
            "ER": [f"G{i}" for i in range(30)],
            "MITO": [f"G{i}" for i in range(30, 60)],
            "NUC": [f"G{i}" for i in range(60, 90)],
        }
        gr.tl.independent_diffusion(annotated, gene_sets, gene_key="gene_symbol", resolve=None)
        with pytest.raises(ValueError, match="per_term"):
            gr.tl.resolve_soft_labels(
                annotated, prob_key="ann_diffusion_probabilities", null=None
            )

    def test_diffusion_resolver_refuses_simplex_input(self, annotated):
        gr.tl.competitive_diffusion(annotated, gt_col="markers", verbose=False)
        with pytest.raises(ValueError, match="simplex"):
            gr.tl.resolve_diffusion(
                annotated, {"ER": ["x"]}, key_added="competitive_diffusion"
            )

    def test_objects_without_params_are_allowed_through(self, annotated):
        """Files written before the contract existed say nothing and must still work."""
        gr.tl.competitive_diffusion(annotated, gt_col="markers", verbose=False)
        del annotated.uns["competitive_diffusion_params"]
        gr.tl.resolve_soft_labels(
            annotated,
            prob_key="competitive_diffusion_probabilities",
            null=None,
            unknown_label=None,
        )
        assert "competitive_diffusion_probabilities_resolved" in annotated.obs


class TestReadAnnotation:
    def test_legacy_bare_array_uses_the_uns_fallback(self, annotated):
        gr.tl.competitive_diffusion(annotated, gt_col="markers", verbose=False)
        values = np.asarray(annotated.obsm["competitive_diffusion_probabilities"])
        annotated.obsm["legacy_probabilities"] = values
        annotated.uns["legacy_categories"] = ["ER", "MITO", "NUC"]
        P, categories, params = _annotation.read_annotation(annotated, "legacy")
        assert categories == ["ER", "MITO", "NUC"]
        assert params == {}

    def test_nameless_matrix_says_what_is_missing(self, annotated):
        annotated.obsm["nameless_probabilities"] = np.zeros((annotated.n_obs, 2))
        with pytest.raises(ValueError, match="cannot be recovered"):
            _annotation.read_annotation(annotated, "nameless")
