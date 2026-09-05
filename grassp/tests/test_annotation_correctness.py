"""Regression tests for correctness defects in the protein-annotation tools.

Each test here pins a bug that shipped silently, so the assertions are deliberately
about the *specific* wrong behaviour rather than about general happy-path output.
"""

import tempfile

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scanpy as sc

import grassp as gr

from grassp.tools import localization


def _blobs(separation: float = 2.5, spread: float = 0.5, seed: int = 0) -> ad.AnnData:
    """Three compartments of 40 proteins, every third one unlabelled.

    ``separation``/``spread`` control how confidently the graph can call a protein, which
    is what the ``min_probability`` tests need to vary.
    """
    rng = np.random.default_rng(seed)
    X = np.vstack([rng.normal(m, spread, (40, 6)) for m in (0.0, separation, 2 * separation)])
    data = ad.AnnData(X)
    data.obs_names = [f"P{i}" for i in range(data.n_obs)]
    data.var_names = [f"F{i}" for i in range(data.n_vars)]
    truth = np.array(["ER"] * 40 + ["MITO"] * 40 + ["NUC"] * 40, dtype=object)
    markers = truth.copy()
    markers[::3] = None
    data.obs["truth"] = truth
    data.obs["markers"] = pd.Categorical(markers)
    sc.pp.neighbors(data, n_neighbors=10, use_rep="X")
    return data


class TestSvmUnobservedCategory:
    """`svm_annotation` indexed predict_proba columns into the *declared* vocabulary.

    ``svm.classes_`` only holds the codes present among the markers, but a Categorical
    keeps its unused categories (after subsetting an object, or when the marker
    vocabulary is wider than the map). The mismatch shifted every label past the first
    missing class, and later crashed in ``util.set_matrix`` on the width mismatch.
    """

    #: declared vocabulary with the unobserved class at code 1, i.e. *before* two classes
    #: that are observed. Order matters: an unobserved class appended last (what
    #: ``cat.add_categories`` does) still crashes on the width mismatch but cannot expose
    #: the label shift, because no observed code sits above it.
    DECLARED = ["ER", "GOLGI", "MITO", "NUC"]

    @classmethod
    def _with_unobserved(cls) -> ad.AnnData:
        data = _blobs()
        data.obs["markers"] = pd.Categorical(
            data.obs["markers"].astype(object), categories=cls.DECLARED
        )
        assert "GOLGI" not in set(data.obs["markers"].dropna())
        return data

    def test_inplace_matrix_is_labelled_with_the_predicted_classes(self):
        data = self._with_unobserved()
        gr.tl.svm_annotation(data, gt_col="markers", C=1.0, gamma=0.1, min_probability=0.0)

        probs = data.obsm["svm_annotation_probabilities"]
        assert list(probs.columns) == ["ER", "MITO", "NUC"]
        assert probs.shape == (data.n_obs, 3)
        # the obs column keeps the full declared vocabulary, in its declared order, so
        # colours and comparisons against gt_col still line up
        assert list(data.obs["svm_annotation"].cat.categories) == self.DECLARED

    def test_labels_are_not_shifted(self):
        data = self._with_unobserved()
        gr.tl.svm_annotation(data, gt_col="markers", C=1.0, gamma=0.1, min_probability=0.0)
        # separable blobs: a positional shift would send the NUC block to MITO and so on
        agreement = (
            data.obs["svm_annotation"].astype(str) == data.obs["truth"].astype(str)
        ).mean()
        assert agreement > 0.95

    def test_not_inplace_categories_describe_the_probability_columns(self):
        data = self._with_unobserved()
        res = gr.tl.svm_annotation(
            data, gt_col="markers", C=1.0, gamma=0.1, inplace=False, min_probability=0.0
        )
        assert res["probabilities"].shape[1] == len(res["categories"])
        assert list(res["categories"]) == ["ER", "MITO", "NUC"]

    def test_fix_markers_pins_the_annotated_class(self):
        data = self._with_unobserved()
        gr.tl.svm_annotation(
            data,
            gt_col="markers",
            C=1.0,
            gamma=0.1,
            fix_markers=True,
            min_probability=0.0,
        )
        is_marker = data.obs["markers"].notna()
        assert np.allclose(data.obs.loc[is_marker, "svm_annotation_probability"], 1.0)
        assert (
            data.obs.loc[is_marker, "svm_annotation"].astype(str)
            == data.obs.loc[is_marker, "markers"].astype(str)
        ).all()


class TestPruneMarkersThreshold:
    """`prune_markers_knn`'s min_probability had no effect at any setting.

    It forwarded the cutoff to ``competitive_diffusion(inplace=False)``, but that applies
    its threshold only on the inplace path, so nothing was ever removed for low
    confidence.
    """

    def test_raising_the_threshold_shrinks_the_retained_set(self):
        # overlapping compartments, so propagated confidence actually spans (0, 1)
        kept = []
        for min_probability in (0.0, 0.6, 0.8, 0.9, 0.99):
            data = _blobs(separation=1.6, spread=1.4, seed=1)
            localization.prune_markers_knn(
                data, gt_col="markers", min_probability=min_probability
            )
            kept.append(int(data.obs["markers_pruned"].notna().sum()))

        assert kept == sorted(kept, reverse=True), kept
        assert kept[0] > kept[-1], f"threshold still inert: {kept}"

    def test_retained_markers_keep_their_original_label(self):
        data = _blobs(separation=1.6, spread=1.4, seed=1)
        localization.prune_markers_knn(data, gt_col="markers", min_probability=0.8)
        retained = data.obs["markers_pruned"].notna()
        assert retained.any()
        assert (
            data.obs.loc[retained, "markers_pruned"].astype(str)
            == data.obs.loc[retained, "markers"].astype(str)
        ).all()
        # non-markers are never retained
        assert not retained[data.obs["markers"].isna()].any()

    def test_returns_none(self):
        data = _blobs()
        assert localization.prune_markers_knn(data, gt_col="markers") is None


class TestInterfacialnessDoesNotMutateInput:
    """`_get_knn_annotation_df` NaN'd the excluded labels in the *caller's* object.

    It took ``data.obs[col]`` (a view) and called ``.replace(..., inplace=True)``, so
    every annotator run afterwards saw those proteins as unlabelled.
    """

    @staticmethod
    def _with_unknowns() -> ad.AnnData:
        data = _blobs(separation=1.6, spread=1.4, seed=1)
        data.obs["compartment"] = pd.Categorical(
            np.where(data.obs["markers"].isna(), "unknown", data.obs["markers"].astype(str))
        )
        return data

    def test_excluded_labels_survive_in_the_caller(self):
        data = self._with_unknowns()
        before = data.obs["compartment"].copy()
        gr.tl.calculate_interfacialness_score(data, "compartment", exclude_category="unknown")
        pd.testing.assert_series_equal(data.obs["compartment"], before)
        assert (data.obs["compartment"] == "unknown").sum() > 0

    def test_exclusion_still_takes_effect(self):
        data = self._with_unknowns()
        gr.tl.calculate_interfacialness_score(data, "compartment", exclude_category="unknown")
        # the excluded label must never win a neighbourhood
        for col in ("jaccard_k1", "jaccard_k2"):
            assert "unknown" not in set(data.obs[col].dropna().astype(str))

    def test_score_is_unaffected_by_a_second_call(self):
        """The old mutation made the function non-idempotent on its own input."""
        data = self._with_unknowns()
        gr.tl.calculate_interfacialness_score(data, "compartment", exclude_category="unknown")
        first = data.obs["jaccard_score"].to_numpy(dtype=float).copy()
        data.obs = data.obs.drop(
            columns=[c for c in data.obs.columns if c.startswith("jaccard_")]
        )
        gr.tl.calculate_interfacialness_score(data, "compartment", exclude_category="unknown")
        second = data.obs["jaccard_score"].to_numpy(dtype=float)
        assert np.allclose(first, second)


class TestTagmJointAndPersistence:
    """Two TAGM defects that made the model unusable end to end.

    ``probJoint=True`` stacked a marker-only block underneath the full posterior matrix
    and assigned the result to an ``.obs`` column, so it raised for every non-empty
    marker set. Separately, ``uns["tagm.map.params"]`` held a raw ``.shape`` tuple, which
    anndata has no h5ad writer for -- so a trained object could not be saved at all.
    """

    @staticmethod
    def _trained() -> ad.AnnData:
        data = _blobs()
        gr.tl.tagm_map_train(data, gt_col="markers", numIter=8, seed=0)
        return data

    def test_prob_joint_writes_an_obsm_matrix(self):
        data = self._trained()
        gr.tl.tagm_map_predict(data, probJoint=True)

        assert "tagm.map.joint" in data.obsm
        joint = data.obsm["tagm.map.joint"]
        assert joint.shape == (data.n_obs, 3)
        assert list(joint.columns) == ["ER", "MITO", "NUC"]

    def test_marker_rows_are_one_hot_at_their_annotated_class(self):
        data = self._trained()
        gr.tl.tagm_map_predict(data, probJoint=True)
        joint = data.obsm["tagm.map.joint"]
        values = np.asarray(joint)

        is_marker = data.obs["markers"].notna().to_numpy()
        marker_rows = values[is_marker]
        assert np.allclose(marker_rows.sum(axis=1), 1.0)
        assert np.allclose(marker_rows.max(axis=1), 1.0)
        winner = np.asarray(joint.columns)[marker_rows.argmax(axis=1)]
        assert (winner == data.obs.loc[is_marker, "markers"].astype(str).to_numpy()).all()

        # unlabelled rows carry the posterior, not a one-hot
        assert (values[~is_marker].max(axis=1) < 1.0).any()
        assert np.allclose(values.sum(axis=1), 1.0)

    def test_prob_joint_false_writes_nothing(self):
        data = self._trained()
        gr.tl.tagm_map_predict(data, probJoint=False)
        assert "tagm.map.joint" not in data.obsm

    def test_trained_object_round_trips_through_h5ad(self):
        data = self._trained()
        gr.tl.tagm_map_predict(data, probJoint=True)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trained.h5ad"
            data.write_h5ad(path)
            reloaded = ad.read_h5ad(path)

        assert np.allclose(
            np.asarray(reloaded.obsm["tagm.map.joint"]),
            np.asarray(data.obsm["tagm.map.joint"]),
        )
        assert list(reloaded.uns["tagm.map.params"]["datasize"]["data"]) == list(data.shape)

    def test_reloaded_params_still_predict(self):
        """The saved parameters are the only reusable fitted model in the package."""
        data = self._trained()
        gr.tl.tagm_map_predict(data)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trained.h5ad"
            data.write_h5ad(path)
            params = ad.read_h5ad(path).uns["tagm.map.params"]

        target = _blobs()
        target.obs["markers"] = data.obs["markers"].to_numpy()
        gr.tl.tagm_map_predict(target, params=params)
        assert (
            target.obs["tagm.map.allocation"].astype(str)
            == data.obs["tagm.map.allocation"].astype(str)
        ).all()


@pytest.mark.parametrize("gt_col", ["markers"])
def test_competitive_diffusion_tolerates_unobserved_categories(gt_col):
    """Guard the asymmetry that made the svm bug hard to spot.

    `competitive_diffusion` builds its one-hot with ``pd.get_dummies``, which emits a
    column per *declared* category, so it always agreed with its own label list. Only the
    svm path mixed the two vocabularies.
    """
    declared = ["ER", "GOLGI", "MITO", "NUC"]
    data = _blobs()
    data.obs[gt_col] = pd.Categorical(data.obs[gt_col].astype(object), categories=declared)
    gr.tl.competitive_diffusion(data, gt_col=gt_col, verbose=False)
    probs = data.obsm["competitive_diffusion_probabilities"]
    assert list(probs.columns) == declared


def test_tagm_predict_rejects_a_mismatched_variable_count():
    """`D` was read off the target object, so a wrong-width map ran on garbage."""
    data = _blobs()
    gr.tl.tagm_map_train(data, gt_col="markers", numIter=8, seed=0)

    narrower = data[:, :4].copy()
    with pytest.raises(ValueError, match="fractions"):
        gr.tl.tagm_map_predict(narrower, params=data.uns["tagm.map.params"])


class TestTagmColoursAndProvenance:
    """Three more TAGM defects: rotated colours, an object-dtype label column, and a
    provenance record that could name an algorithm which never ran."""

    #: declared in a non-alphabetical order, as a biologically-ordered or pRoloc-imported
    #: marker column would be. The model's vocabulary is np.sort(observed), so a
    #: positional colour copy hands every compartment its neighbour's colour.
    DECLARED = ["NUC", "ER", "MITO"]
    COLORS = ["#ff0000", "#00ff00", "#0000ff"]

    @classmethod
    def _trained(cls) -> ad.AnnData:
        data = _blobs()
        data.obs["markers"] = pd.Categorical(
            data.obs["markers"].astype(object), categories=cls.DECLARED
        )
        data.uns["markers_colors"] = list(cls.COLORS)
        gr.tl.tagm_map_train(data, gt_col="markers", numIter=8, seed=0)
        return data

    def test_colours_follow_the_compartment_not_the_position(self):
        data = self._trained()
        gr.tl.tagm_map_predict(data)

        intended = dict(zip(self.DECLARED, self.COLORS))
        vocabulary = [str(m) for m in data.uns["tagm.map.params"]["markers"]]
        assigned = dict(zip(vocabulary, data.uns["tagm.map.allocation_colors"]))
        assert assigned == {k: intended[k] for k in vocabulary}

    def test_allocation_is_categorical_in_the_model_order(self):
        data = self._trained()
        gr.tl.tagm_map_predict(data)

        allocation = data.obs["tagm.map.allocation"]
        assert isinstance(allocation.dtype, pd.CategoricalDtype)
        vocabulary = [str(m) for m in data.uns["tagm.map.params"]["markers"]]
        # same order as the probability columns and the colour list
        assert list(allocation.cat.categories) == vocabulary
        assert list(data.obsm["tagm.map.probabilities"].columns) == vocabulary

    def test_unimplemented_method_is_rejected(self):
        data = _blobs()
        with pytest.raises(NotImplementedError, match="MAP"):
            gr.tl.tagm_map_train(data, gt_col="markers", method="mcmc", numIter=3)
        assert "tagm.map.params" not in data.uns

    def test_predicts_on_an_object_without_the_marker_column(self):
        """The fitted model exists to be applied elsewhere; the marker column is only
        needed to mark the labelled rows of the optional joint matrix."""
        trained = self._trained()
        target = _blobs()
        del target.obs["markers"]

        gr.tl.tagm_map_predict(target, params=trained.uns["tagm.map.params"])
        assert (
            target.obs["tagm.map.allocation"].astype(str) == target.obs["truth"].astype(str)
        ).mean() > 0.95

        with pytest.raises(KeyError, match="probJoint"):
            gr.tl.tagm_map_predict(
                target, params=trained.uns["tagm.map.params"], probJoint=True
            )

    def test_untrained_marker_class_is_named(self):
        data = self._trained()
        data.obs["markers"] = data.obs["markers"].cat.add_categories(["GOLGI"])
        data.obs.iloc[1, data.obs.columns.get_loc("markers")] = "GOLGI"
        with pytest.raises(ValueError, match="GOLGI"):
            gr.tl.tagm_map_predict(data, probJoint=True)


class TestSvmParameterHandling:
    def test_class_weight_may_be_keyed_by_label(self):
        """Both entry points fit on `.cat.codes`, so a label-keyed dict -- the only one a
        user would write -- was rejected by sklearn outright."""
        data = _blobs()
        weights = {"ER": 2.0, "MITO": 1.0, "NUC": 1.0}
        gr.tl.svm_annotation(data, gt_col="markers", C=1.0, gamma=0.1, class_weight=weights)
        assert "svm_annotation" in data.obs

        gr.tl.svm_train(
            data,
            gt_col="markers",
            cv_splits=2,
            cv_repeats=1,
            C_range=np.array([1.0]),
            gamma_range=np.array([0.1]),
            class_weight=weights,
        )
        assert data.uns["svm.params"]["best_params"]["C"] == 1.0

    def test_class_weight_keyed_by_code_still_works(self):
        data = _blobs()
        gr.tl.svm_annotation(
            data, gt_col="markers", C=1.0, gamma=0.1, class_weight={0: 2.0, 1: 1.0, 2: 1.0}
        )
        assert "svm_annotation" in data.obs

    def test_unknown_class_weight_key_is_named(self):
        data = _blobs()
        with pytest.raises(KeyError, match="NOPE"):
            gr.tl.svm_annotation(
                data, gt_col="markers", C=1.0, gamma=0.1, class_weight={"NOPE": 2.0}
            )

    def test_stored_class_labels_match_the_fitted_vocabulary(self):
        data = _blobs()
        data.obs["markers"] = pd.Categorical(
            data.obs["markers"].astype(object), categories=["ER", "GOLGI", "MITO", "NUC"]
        )
        gr.tl.svm_train(
            data,
            gt_col="markers",
            cv_splits=2,
            cv_repeats=1,
            C_range=np.array([1.0]),
            gamma_range=np.array([0.1]),
        )
        assert data.uns["svm.params"]["class_labels"] == ["ER", "MITO", "NUC"]

    @pytest.mark.parametrize(
        "call",
        [
            lambda d: gr.tl.competitive_diffusion(d, gt_col="markers", verbose=False),
            lambda d: gr.tl.svm_annotation(d, gt_col="markers", C=1.0, gamma=0.1),
            lambda d: localization.prune_markers_knn(d, gt_col="markers"),
        ],
        ids=["competitive_diffusion", "svm_annotation", "prune_markers_knn"],
    )
    def test_object_dtype_label_column_is_accepted(self, call):
        """competitive_diffusion always accepted one, so requiring a Categorical made the
        predictors disagree about what a valid gt_col is."""
        data = _blobs()
        data.obs["markers"] = data.obs["markers"].astype(object)
        call(data)

    def test_numeric_categories_survive_the_abstention_cutoff(self):
        """`pred_labels` was a numeric array that cannot hold the np.nan written by the
        min_probability cutoff."""
        data = _blobs()
        codes = pd.Series(np.repeat([0, 1, 2], 40), index=data.obs_names)
        data.obs["clusters"] = pd.Categorical(
            codes.where(pd.Series(np.arange(data.n_obs) % 3 != 0, index=data.obs_names))
        )
        gr.tl.svm_annotation(
            data, gt_col="clusters", C=1.0, gamma=0.1, min_probability=0.99, key_added="num"
        )
        assert "num" in data.obs


class TestEvaluationWorksAcrossPredictors:
    """`annotation_confusion_matrix` and `annotation_marker_df` were specific to one predictor."""

    @staticmethod
    def _annotated() -> ad.AnnData:
        data = _blobs()
        gr.tl.competitive_diffusion(data, gt_col="markers", verbose=False)
        gr.tl.svm_annotation(data, gt_col="markers", C=1.0, gamma=0.1)
        return data

    @pytest.mark.parametrize("pred_col", ["competitive_diffusion", "svm_annotation"])
    @pytest.mark.parametrize("soft", [False, True])
    def test_confusion_matrix_accepts_any_probability_matrix(self, pred_col, soft):
        from grassp.tools import scoring

        data = self._annotated()
        cm = scoring.annotation_confusion_matrix(
            data, "markers", pred_col=pred_col, soft=soft, plot=False
        )
        assert cm.shape == (3, 3)
        # separable blobs, so the mass belongs on the diagonal
        assert np.mean(np.diag(cm)) > 0.8

    def test_marker_df_aligns_by_name_not_position(self):
        from grassp.plotting import clustering as plotting_clustering

        data = _blobs()
        # svm stores only the classes it can predict, so a positional product would pair
        # each protein's probability with the wrong compartment
        data.obs["markers"] = pd.Categorical(
            data.obs["markers"].astype(object), categories=["ER", "GOLGI", "MITO", "NUC"]
        )
        gr.tl.svm_annotation(data, gt_col="markers", C=1.0, gamma=0.1)
        assert list(data.obsm["svm_annotation_probabilities"].columns) == ["ER", "MITO", "NUC"]

        frame = plotting_clustering.annotation_marker_df(data, "markers", "svm_annotation")
        assert len(frame) == int(data.obs["markers"].notna().sum())
        # each marker's own compartment should carry most of the mass
        assert frame["pred_prob"].mean() > 0.8


class TestClusterEnrichmentWrites:
    @staticmethod
    def _data() -> ad.AnnData:
        rng = np.random.default_rng(0)
        data = ad.AnnData(rng.normal(size=(60, 4)))
        data.obs_names = [f"P{i}" for i in range(60)]
        data.obs["Gene_name_canonical"] = [f"G{i}" for i in range(60)]
        data.obs["leiden"] = pd.Categorical(["0"] * 30 + ["1"] * 30)
        return data

    GENE_SETS = {
        "ER": [f"G{i}" for i in range(30)],
        "MITO": [f"G{i}" for i in range(30, 60)],
    }

    def test_not_inplace_returns_an_annotated_copy(self):
        """The writes sat inside `if inplace:`, so inplace=False returned an object with
        none of the documented columns."""
        data = self._data()
        out = gr.tl.calculate_cluster_enrichment(
            data,
            cluster_key="leiden",
            gene_sets=self.GENE_SETS,
            inplace=False,
            return_enrichment_res=False,
        )
        assert "Cell_compartment" in out.obs
        assert "Cell_compartment" not in data.obs  # the original is untouched
        assert out.obs.groupby("leiden", observed=True)[
            "Cell_compartment"
        ].first().to_dict() == {
            "0": "ER",
            "1": "MITO",
        }

    def test_accepts_a_view(self):
        """`obs_df = data.obs` went stale the moment anndata materialised the view."""
        data = self._data()
        gr.tl.calculate_cluster_enrichment(
            data[data.obs_names[:40]],
            cluster_key="leiden",
            gene_sets=self.GENE_SETS,
            return_enrichment_res=False,
        )

    def test_empty_vocabulary_raises_a_named_error(self):
        """Previously "zero-size array to reduction operation maximum", from numpy."""
        enrichment = pd.DataFrame(
            {
                "leiden": ["0", "0", "1", "1"],
                "Term": ["ER", "MITO", "ER", "MITO"],
                "Adjusted P-value Bonferroni": [0.9, 0.8, 0.95, 0.7],
                "Odds Ratio": [1.0, 1.0, 1.0, 1.0],
                "Genes": ["A;B", "C", "D", "E"],
            }
        )
        with pytest.raises(ValueError, match="vocabulary would be empty"):
            gr.tl.enrichment_to_cluster_distribution(
                enrichment, cluster_key="leiden", unknown_label=None
            )
        # both documented escapes work
        gr.tl.enrichment_to_cluster_distribution(enrichment, cluster_key="leiden")
        gr.tl.enrichment_to_cluster_distribution(
            enrichment, cluster_key="leiden", unknown_label=None, threshold=1.0
        )
