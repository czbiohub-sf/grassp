"""Tests for the grassp <-> pRoloc exchange contract in :mod:`grassp.io._msnset`.

The module is small on purpose: nothing needs translating between the two frameworks that h5ad
cannot carry on its own, so nothing is. What remains is the class names belonging to an
``.obsm`` matrix, the contract version, and two sanity checks.

These are pure functions; no filesystem, no h5py, no R.
"""

import numpy as np
import pandas as pd
import pytest

from grassp.io import _msnset as m

# ==============================================================================
# The "unknown" sentinel -- only needed where no R is involved
# ==============================================================================


class TestNanToUnknown:
    """pRoloc encodes unlabelled features as ``"unknown"`` and genuinely needs it:
    ``markerMSnSet`` and ``unknownMSnSet`` fail outright on ``NA``.

    Applied to *every* text column rather than to a nominated marker column, because pRoloc's
    ``fcol`` is a per-call argument -- one ``MSnSet`` can carry ``markers``, ``markers.orig``,
    ``pd.markers`` and more at the same time, exactly as AnnData can.
    """

    def test_object_dtype(self):
        out = m.nan_to_unknown(pd.Series(["Golgi", None, "ER"], dtype=object))
        assert out.tolist() == ["Golgi", "unknown", "ER"]

    def test_string_dtype(self):
        out = m.nan_to_unknown(pd.Series(pd.array(["Golgi", None], dtype="string")))
        assert out.tolist() == ["Golgi", "unknown"]

    def test_categorical_gains_the_category(self):
        """Assigning to a Categorical without adding the category first would raise."""
        out = m.nan_to_unknown(pd.Series(pd.Categorical(["Golgi", None, "ER"])))
        assert isinstance(out.dtype, pd.CategoricalDtype)
        assert "unknown" in out.cat.categories
        assert out.tolist() == ["Golgi", "unknown", "ER"]

    def test_categorical_that_already_has_the_category(self):
        out = m.nan_to_unknown(pd.Series(pd.Categorical(["Golgi", None, "unknown"])))
        assert out.tolist() == ["Golgi", "unknown", "unknown"]
        assert list(out.cat.categories).count("unknown") == 1

    @pytest.mark.parametrize("values", [[1.0, np.nan, 3.0], [1, 2, 3], [True, False, True]])
    def test_numeric_and_boolean_are_untouched(self, values):
        """Putting a string sentinel in a float column would corrupt it."""
        series = pd.Series(values)
        out = m.nan_to_unknown(series)
        assert out.dtype == series.dtype
        assert list(pd.isna(out)) == list(pd.isna(series))

    def test_is_idempotent(self):
        once = m.nan_to_unknown(pd.Series(["Golgi", None], dtype=object))
        assert m.nan_to_unknown(once).tolist() == once.tolist()


class TestUnknownToNan:
    """The inverse, needed only by :func:`grassp.io.read_prolocdata`, which parses ``.rda``
    files with no R at all."""

    def test_sentinel_becomes_nan(self):
        out = m.unknown_to_nan(pd.Series(["Golgi", "unknown", "ER"]))
        assert out.tolist()[0] == "Golgi"
        assert pd.isna(out.tolist()[1])

    def test_other_values_are_untouched(self):
        values = pd.Series(["Golgi", "ER", "PM"])
        assert m.unknown_to_nan(values).tolist() == values.tolist()

    def test_round_trip(self):
        original = pd.Series(["Golgi", None, "ER"], dtype=object)
        back = m.unknown_to_nan(m.nan_to_unknown(original))
        assert list(pd.isna(back)) == list(pd.isna(original))

    def test_custom_sentinel(self):
        out = m.unknown_to_nan(pd.Series(["Golgi", "n/a"]), unknown_label="n/a")
        assert pd.isna(out.tolist()[1])

    def test_categorical_drops_the_category_not_just_the_values(self):
        """`rdata` maps an R factor to a Categorical, so this is the common .rda case.

        Leaving `"unknown"` as an unused category is not cosmetic: anything iterating
        `.cat.categories` -- `set_sensible_compartment_colors`, scanpy legends -- would show a
        phantom compartment with no members.
        """
        out = m.unknown_to_nan(pd.Series(pd.Categorical(["Golgi", "unknown", "ER"])))
        assert isinstance(out.dtype, pd.CategoricalDtype)
        assert list(out.cat.categories) == ["ER", "Golgi"]
        assert pd.isna(out.tolist()[1])

    def test_categorical_without_the_sentinel_is_untouched(self):
        original = pd.Series(pd.Categorical(["Golgi", "ER"]))
        out = m.unknown_to_nan(original)
        assert list(out.cat.categories) == list(original.cat.categories)
        assert not out.isna().any()

    def test_categorical_round_trip_restores_the_category_set(self):
        original = pd.Series(pd.Categorical(["Golgi", None, "ER"]))
        back = m.unknown_to_nan(m.nan_to_unknown(original))
        assert list(back.cat.categories) == list(original.cat.categories)
        assert list(pd.isna(back)) == list(pd.isna(original))


# ==============================================================================
# Sanity checks on an incoming artifact
# ==============================================================================


class TestLooksRemapped:
    """``pRoloc::remap`` replaces exprs with PCA scores and renames fractions to PC1..PCn,
    which is invisible in the object's shape."""

    def test_pc_names_are_detected(self):
        assert m.looks_remapped(["PC1", "PC2", "PC3"])

    @pytest.mark.parametrize(
        "names",
        [
            ["Fraction.1", "Fraction.2"],
            ["PC1", "Fraction.2"],  # a mixture is not a remap
            ["PCa", "PCb"],
            [],
        ],
    )
    def test_other_names_are_not(self, names):
        assert not m.looks_remapped(names)


class TestPrologNotes:
    """The quirk table is advisory only and cannot change what data crosses over."""

    @pytest.mark.parametrize(
        "column", ["perTurbe.all.scores", "svm.all.scores", "knn.all.scores"]
    )
    def test_known_upstream_misnamings_have_notes(self, column):
        assert column in m.PROLOC_NOTES
        assert isinstance(m.PROLOC_NOTES[column], str)

    def test_notes_for_selects_only_present_columns(self):
        notes = m.notes_for(["markers", "svm", "svm.all.scores"])
        assert list(notes) == ["svm.all.scores"]

    def test_notes_for_is_empty_when_nothing_matches(self):
        assert m.notes_for(["markers", "svm"]) == {}


# ==============================================================================
# The spec block -- the one thing that must accompany the data
# ==============================================================================


class TestSpecBlock:
    def test_round_trip(self):
        block = m.build_spec_block(
            layer=None,
            obsm_colnames={"svm.all.scores": ["ER", "Golgi"]},
            dropped=["layers:pvals", "obsp:connectivities"],
        )
        parsed = m.read_spec_block(block)
        assert parsed["spec"] == m.SPEC_VERSION
        assert parsed["layer"] is None
        assert parsed[m.OBSM_COLNAMES_KEY] == {"svm.all.scores": ["ER", "Golgi"]}
        assert parsed["dropped"] == ["layers:pvals", "obsp:connectivities"]

    def test_no_marker_column_is_nominated(self):
        """`fcol` is a per-call argument in pRoloc, not a property of the object.

        An MSnSet can hold `markers`, `markers.orig`, `pd.markers` and more at once -- verified
        against pRoloc, where each is independently usable as `fcol`. Recording one would
        impose a restriction neither framework has.
        """
        block = m.build_spec_block(layer=None)
        assert not any("fcol" in key for key in block)

    def test_empty_layer_means_x(self):
        block = m.build_spec_block(layer=None)
        assert block["msnset_exprs_layer"] == ""
        assert m.read_spec_block(block)["layer"] is None

    def test_named_layer_survives(self):
        block = m.build_spec_block(layer="log_intensities")
        assert m.read_spec_block(block)["layer"] == "log_intensities"

    def test_no_name_maps_are_written(self):
        """Names cross verbatim, so there is no map to store."""
        block = m.build_spec_block(layer=None)
        assert not any("py_names" in key or "r_names" in key for key in block)

    def test_missing_spec_is_readable(self):
        """A plain h5ad from elsewhere has no spec block and must still be importable."""
        parsed = m.read_spec_block({})
        assert parsed["spec"] is None
        assert parsed["major"] is None
        assert parsed[m.OBSM_COLNAMES_KEY] == {}
        assert parsed["dropped"] == []

    def test_newer_major_version_raises(self):
        with pytest.raises(m.SpecVersionError, match="Upgrade grassp"):
            m.read_spec_block({"msnset_spec": "grassp-msnset/99"})

    def test_newer_major_version_warns_when_not_strict(self):
        with pytest.warns(UserWarning, match="Upgrade grassp"):
            m.read_spec_block({"msnset_spec": "grassp-msnset/99"}, strict=False)

    def test_malformed_spec_raises(self):
        with pytest.raises(m.SpecVersionError, match="Unrecognised"):
            m.read_spec_block({"msnset_spec": "not-a-spec"})

    def test_same_major_version_is_accepted(self):
        assert m.check_spec(m.SPEC_VERSION) == m.spec_major(m.SPEC_VERSION)

    def test_numpy_arrays_are_tolerated(self):
        """h5ad readers hand back numpy arrays, not lists."""
        block = m.build_spec_block(
            layer=None,
            obsm_colnames={"k": ["ER", "Golgi"]},
            dropped=["layers:pvals"],
        )
        block[m.OBSM_COLNAMES_KEY]["k"] = np.array(["ER", "Golgi"], dtype=object)
        block["msnset_dropped"] = np.array(["layers:pvals"], dtype=object)
        parsed = m.read_spec_block(block)
        assert parsed[m.OBSM_COLNAMES_KEY]["k"] == ["ER", "Golgi"]
        assert parsed["dropped"] == ["layers:pvals"]
