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


class TestTagmJoinsTheContract:
    """TAGM was the one predictor no generic consumer could address."""

    @staticmethod
    def _predicted(data):
        gr.tl.tagm_map_train(data, gt_col="markers", numIter=8, random_state=0)
        gr.tl.tagm_map_predict(data, probJoint=True)
        return data

    def test_keys_follow_the_key_added_convention(self, annotated):
        self._predicted(annotated)
        assert "tagm_map" in annotated.obs
        assert "tagm_map_probability" in annotated.obs
        assert "tagm_map_outlier" in annotated.obs
        assert "tagm_map_probabilities" in annotated.obsm
        assert "tagm_map_joint" in annotated.obsm
        assert "tagm_map_colors" in annotated.uns
        assert not [k for k in annotated.obs if k.startswith("tagm.map")]
        assert not [k for k in annotated.obsm if k.startswith("tagm.map")]

    def test_the_model_does_not_collide_with_the_provenance(self, annotated):
        """Both wanted uns[f"{key}_params"]; the prediction would overwrite the model."""
        self._predicted(annotated)
        assert annotated.uns["tagm_map_params"]["kind"] == "simplex"
        assert "posteriors" in annotated.uns["tagm_map_model"]

    def test_generic_consumers_now_accept_it(self, annotated):
        from grassp.tools import scoring

        self._predicted(annotated)
        cm = scoring.annotation_confusion_matrix(
            annotated, "markers", pred_col="tagm_map", plot=False
        )
        assert cm.shape[0] == cm.shape[1]
        f1 = scoring.annotation_f1_score(annotated, "markers", pred_col="tagm_map")
        assert 0.0 <= f1 <= 1.0

    def test_the_legacy_dotted_model_key_is_still_read(self, annotated):
        gr.tl.tagm_map_train(annotated, gt_col="markers", numIter=8, random_state=0)
        annotated.uns["tagm.map.params"] = annotated.uns.pop("tagm_map_model")
        gr.tl.tagm_map_predict(annotated)
        assert annotated.obs["tagm_map"].notna().any()

    def test_key_added_is_honoured(self, annotated):
        gr.tl.tagm_map_train(
            annotated, gt_col="markers", numIter=8, random_state=0, key_added="mine"
        )
        gr.tl.tagm_map_predict(annotated, key_added="mine")
        assert "mine" in annotated.obs and "mine_probabilities" in annotated.obsm
        assert "mine_model" in annotated.uns


class TestParameterNamesAreConsistent:
    """One name per concept across the annotation surface."""

    ANNOTATORS = [
        "competitive_diffusion",
        "independent_diffusion",
        "svm_annotation",
        "tagm_map_train",
        "tagm_map_predict",
        "soft_cluster_annotation",
        "resolve_soft_labels",
    ]

    def _params(self, name):
        import inspect

        return inspect.signature(getattr(gr.tl, name)).parameters

    @pytest.mark.parametrize("name", ANNOTATORS)
    def test_no_legacy_parameter_names_survive(self, name):
        params = self._params(name)
        for retired, replacement in [
            ("seed", "random_state"),
            ("copy", "inplace"),
            ("marker_key", "gt_col"),
            ("obs_key_added", "key_added"),
            ("class_balance", "renormalize_class_mass"),
        ]:
            assert (
                retired not in params
            ), f"{name} still takes {retired!r}, not {replacement!r}"

    @pytest.mark.parametrize("name", ANNOTATORS)
    def test_verbose_defaults_to_quiet(self, name):
        params = self._params(name)
        if "verbose" in params:
            assert params["verbose"].default is False

    @pytest.mark.parametrize("name", ANNOTATORS)
    def test_every_annotator_offers_inplace(self, name):
        assert "inplace" in self._params(name)

    @pytest.mark.parametrize("name", ["svm_annotation", "tagm_map_train", "ccompass"])
    def test_the_markers_default_is_gone(self, name):
        """`gt_col="markers"` was the pRolocdata convention and wrong for any object
        prepared with pp.add_markers, which writes author-named columns. The label column
        is now required, so a missing one fails at the call rather than as a KeyError."""
        import inspect

        assert self._params(name)["gt_col"].default is inspect.Parameter.empty

    def test_diffusion_uses_term_threshold_not_min_probability(self):
        """min_probability means "abstain below this" everywhere; the per-term membership
        cutoff is a different question and now has its own name."""
        assert "term_threshold" in self._params("independent_diffusion")
        assert "min_probability" not in self._params("independent_diffusion")
        assert "term_threshold" in self._params("resolve_diffusion")
        # while the abstention cutoff keeps the shared name
        assert "min_probability" in self._params("competitive_diffusion")
        assert "min_probability" in self._params("svm_annotation")


class TestSvmHyperparameterSearch:
    def test_the_search_does_not_fit_a_throwaway_estimator(self, annotated):
        """GridSearchCV defaults to refit=True, so a full SVC was fitted on every marker
        and discarded, because only the winning parameters are kept."""
        gr.tl.svm_tune_hyperparameters(
            annotated,
            gt_col="markers",
            cv_splits=2,
            cv_repeats=1,
            C_range=np.array([1.0]),
            gamma_range=np.array([0.1]),
            inplace=False,
        )
        search, params = gr.tl.svm_tune_hyperparameters(
            annotated,
            gt_col="markers",
            cv_splits=2,
            cv_repeats=1,
            C_range=np.array([1.0]),
            gamma_range=np.array([0.1]),
            inplace=False,
        )
        assert not hasattr(search, "best_estimator_")
        assert params["best_params"]["C"] == 1.0

    def test_params_key_follows_the_shared_convention(self, annotated):
        gr.tl.svm_tune_hyperparameters(
            annotated,
            gt_col="markers",
            cv_splits=2,
            cv_repeats=1,
            C_range=np.array([1.0]),
            gamma_range=np.array([0.1]),
        )
        assert "svm_params" in annotated.uns
        assert "svm.params" not in annotated.uns
        # and svm_annotation picks it up without being told where
        gr.tl.svm_annotation(annotated, gt_col="markers")
        assert "svm_annotation" in annotated.obs

    def test_the_old_name_is_gone(self):
        assert not hasattr(gr.tl, "svm_train")
        assert callable(gr.tl.svm_tune_hyperparameters)


class TestNullMatchesTheProduction:
    """The entropy null must be propagated the way the probabilities were.

    resolve_soft_labels re-propagates permuted seeds to build a per-protein null entropy,
    but it hard-coded _propagate_soft's defaults -- single-step propagation over
    obsp["connectivities"] -- so a run made with method="spreading" was tested against a
    null from different math. The annotator records its settings in uns[f"{key}_params"],
    so the null can now read them.
    """

    @staticmethod
    def _spread(data):
        gr.tl.competitive_diffusion(data, gt_col="markers", method="spreading", alpha=0.8)
        return data

    def test_the_recorded_propagation_is_used(self, annotated):
        self._spread(annotated)
        assert annotated.uns["competitive_diffusion_params"]["propagation"] == "spreading"
        gr.tl.resolve_soft_labels(
            annotated,
            prob_key="competitive_diffusion_probabilities",
            seed_key="competitive_diffusion_one_hot_labels",
            unknown_label=None,
            n_permutations=40,
            random_state=0,
        )
        key = "competitive_diffusion_probabilities_resolved"
        assert annotated.uns[f"{key}_null"]["method"] == "permutation"
        assert annotated.obs[f"{key}_type"].notna().all()

    def test_a_mismatched_null_gives_a_different_answer(self, annotated):
        """Pinning the difference: this is what the old behaviour computed."""
        faithful = self._spread(annotated.copy())
        mismatched = self._spread(annotated.copy())
        common = dict(
            prob_key="competitive_diffusion_probabilities",
            seed_key="competitive_diffusion_one_hot_labels",
            unknown_label=None,
            n_permutations=40,
            random_state=0,
        )
        gr.tl.resolve_soft_labels(faithful, **common)
        gr.tl.resolve_soft_labels(mismatched, method="propagation", iterative=False, **common)

        key = "competitive_diffusion_probabilities_resolved"
        assert not np.allclose(
            faithful.obs[f"{key}_zscore"].to_numpy(dtype=float),
            mismatched.obs[f"{key}_zscore"].to_numpy(dtype=float),
        )

    def test_the_rbf_graph_is_reproduced(self, annotated):
        """obsp_key="distances" means an RBF kernel, not the raw distance matrix; the
        null read data.obsp[obsp_key] directly and so skipped the kernel entirely."""
        gr.tl.competitive_diffusion(
            annotated, gt_col="markers", obsp_key="distances", method="spreading"
        )
        assert annotated.uns["competitive_diffusion_params"]["obsp_key"] == "distances"
        gr.tl.resolve_soft_labels(
            annotated,
            prob_key="competitive_diffusion_probabilities",
            seed_key="competitive_diffusion_one_hot_labels",
            unknown_label=None,
            n_permutations=20,
            random_state=0,
        )
        assert "W_spreading" in annotated.obsp
