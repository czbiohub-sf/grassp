"""What :func:`grassp.io.read_prolocdata` makes of *real* pRolocdata objects.

Every other test of the reader mocks the ``rdata`` package and so never touches its conversion
of R's serialisation format -- which is exactly where the reader breaks (``hyperLOPIT2015``
was unreadable until 2026-10-05 and nothing noticed). The ``fixtures/mini_*`` files are genuine
pRolocdata objects cut down to a few dozen proteins by ``fixtures/make_prolocdata_fixtures.R``,
subset rather than rebuilt so that fData, pData, experimentData and the S4 class versions are
exactly as pRolocdata serialised them. Each was chosen for a parser quirk it carries; the R
script's header says which.

These are regression tests: the values pinned here were read off the fixtures with the reader
as of 2026-10-05 and checked against the full objects in R 4.6.1 / pRolocdata 1.50.0.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from grassp.io import read

FIXTURES = Path(__file__).parent / "fixtures"

#: Dataset name -> file. Both extensions occur in pRolocdata, and the reader must take either.
FILES = {
    "hyperLOPIT2015": "mini_hyperLOPIT2015.RData",
    "dunkley2006": "mini_dunkley2006.RData",
    "itzhak2016stcSILAC": "mini_itzhak2016stcSILAC.rda",
    "tan2009r1": "mini_tan2009r1.RData",
}
SHAPES = {
    "hyperLOPIT2015": (34, 20),
    "dunkley2006": (24, 16),
    "itzhak2016stcSILAC": (30, 30),
    "tan2009r1": (28, 4),
}


def _read(name, **kwargs):
    path = FIXTURES / FILES[name]
    if not path.exists():  # pragma: no cover - the fixtures are committed
        pytest.skip(f"{path.name} is missing; regenerate it with make_prolocdata_fixtures.R")
    return read.read_prolocdata(str(path), **kwargs)


@pytest.fixture(scope="module")
def datasets(real_rdata):
    """All four slices, read with NaN kept so the missing-value tests can see it."""
    with real_rdata():
        return {name: _read(name, replace_nan=False) for name in FILES}


@pytest.fixture(scope="module")
def hyperlopit(datasets):
    return datasets["hyperLOPIT2015"]


@pytest.fixture(scope="module")
def dunkley(datasets):
    return datasets["dunkley2006"]


@pytest.fixture(scope="module")
def itzhak(datasets):
    return datasets["itzhak2016stcSILAC"]


@pytest.fixture(scope="module")
def tan(datasets):
    return datasets["tan2009r1"]


# ==============================================================================
# What holds for every pRolocdata object
# ==============================================================================


class TestEveryDataset:
    @pytest.mark.parametrize("name", list(FILES))
    def test_shape_and_orientation(self, datasets, name):
        """``exprs`` is already proteins x fractions: no transpose, and the slice's size."""
        adata = datasets[name]
        assert adata.shape == SHAPES[name]
        assert adata.X.dtype == float

    @pytest.mark.parametrize("name", list(FILES))
    def test_the_r_object_name_is_the_dataset_name(self, datasets, name):
        """Not the file name: pRolocdata's object is called ``dunkley2006`` whatever the file."""
        assert datasets[name].uns["dataset_name"] == name

    @pytest.mark.parametrize("name", list(FILES))
    def test_names_are_plain_python_strings(self, datasets, name):
        adata = datasets[name]
        for label in [
            *adata.obs_names,
            *adata.var_names,
            *adata.obs.columns,
            *adata.var.columns,
        ]:
            assert type(label) is str, (label, type(label))

    @pytest.mark.parametrize("name", list(FILES))
    def test_the_unknown_sentinel_is_gone_from_every_column(self, datasets, name):
        """Every grassp annotator picks markers with ``.notna()``; "unknown" must not survive."""
        obs = datasets[name].obs
        assert not (obs.astype(str) == "unknown").any().any()
        for column in obs.columns:
            if isinstance(obs[column].dtype, pd.CategoricalDtype):
                assert "unknown" not in obs[column].cat.categories, column

    @pytest.mark.parametrize("name", list(FILES))
    def test_six_unlabelled_proteins_per_slice(self, datasets, name):
        """The generator draws exactly six "unknown" rows; they must arrive as NaN."""
        assert datasets[name].obs["markers"].isna().sum() == 6

    @pytest.mark.parametrize("name", list(FILES))
    def test_marker_columns_get_colours(self, datasets, name):
        assert "markers_colors" in datasets[name].uns

    @pytest.mark.parametrize("name", list(FILES))
    def test_miape_metadata_arrives(self, datasets, name):
        """The slots are there for all four; itzhak2016stcSILAC left every one of them blank."""
        metadata = datasets[name].uns["MIAPE_metadata"]
        assert ".__classVersion__" not in metadata
        assert {"name", "lab", "title", "pubMedIds"} <= set(metadata)
        expected_lab = (
            "" if name == "itzhak2016stcSILAC" else "Cambridge Centre for Proteomics (CCP)"
        )
        assert str(metadata["lab"][0]) == expected_lab


# ==============================================================================
# hyperLOPIT2015: a nested data.frame in fData, and .RData
# ==============================================================================


class TestHyperLOPIT2015:
    def test_the_nested_tagm_frame_is_flattened_in_place(self, hyperlopit):
        """``fData$TAGM`` is a data.frame; it arrives as ``TAGM.<col>``, where ``TAGM`` stood.

        This is the layout that made the whole file unreadable before
        :func:`grassp.io.read._flatten_nested_frames` existed.
        """
        columns = list(hyperlopit.obs.columns)
        assert columns[-7:] == [
            "markers2015",
            "TAGM.tagm.map.allocation",
            "TAGM.tagm.map.probability",
            "TAGM.tagm.mcmc.allocation",
            "TAGM.tagm.mcmc.probability",
            "TAGM.tagm.mcmc.outlier",
            "TAGM.tagm.mcmc.shannon",
        ]
        assert len(columns) == 25 + 6

    def test_inner_values_land_in_the_right_rows(self, hyperlopit):
        obs = hyperlopit.obs
        assert obs.loc["Q05793", "TAGM.tagm.map.allocation"] == "Extracellular matrix"
        assert obs.loc["Q01320", "TAGM.tagm.map.allocation"] == "Nucleus - Chromatin"
        assert pd.isna(obs.loc["Q01320", "markers"])  # allocated, but not a marker
        assert obs.loc["Q01320", "TAGM.tagm.mcmc.shannon"] == pytest.approx(3.399768e-09)

    def test_inner_dtypes_are_what_prolocdata_stored(self, hyperlopit):
        """``tagm.map.probability`` is a *character* column upstream -- not grassp's to fix."""
        obs = hyperlopit.obs
        assert pd.api.types.is_float_dtype(obs["TAGM.tagm.mcmc.probability"])
        assert pd.api.types.is_float_dtype(obs["TAGM.tagm.mcmc.outlier"])
        assert pd.api.types.is_float_dtype(obs["TAGM.tagm.mcmc.shannon"])
        assert obs["TAGM.tagm.map.probability"].dtype == object
        assert obs.loc["Q01320", "TAGM.tagm.map.probability"] == "0.999999999494105"

    def test_class_names_with_punctuation_survive(self, hyperlopit):
        classes = set(hyperlopit.obs["markers"].dropna())
        assert len(classes) == 14
        assert "Endoplasmic reticulum/Golgi apparatus" in classes
        assert "Nucleus - Non-chromatin" in classes

    def test_r_integer_columns_become_nullable_ints(self, hyperlopit):
        """R's integer NA has no numpy equivalent, so rdata picks pandas' ``Int32``."""
        assert str(hyperlopit.obs["peptides.rep1"].dtype) == "Int32"
        assert hyperlopit.obs.loc["Q05793", "peptides.rep1"] == 61

    def test_miape_points_at_the_paper(self, hyperlopit):
        assert str(hyperlopit.uns["MIAPE_metadata"]["pubMedIds"][0]) == "26754106"

    def test_processing_log_is_the_published_one(self, hyperlopit):
        assert hyperlopit.uns["processing"] == [
            "Loaded on Wed Sep 19 15:10:03 2018.",
            "Normalised to sum of intensities.",
        ]

    def test_pdata_becomes_var_with_its_dtypes(self, hyperlopit):
        var = hyperlopit.var
        assert list(var.columns) == [
            "Replicate",
            "TMT.Reagent",
            "Acquisiton.Method",  # sic, pRolocdata's spelling
            "Gradient.Fraction",
            "Iodixonal.Density",
        ]
        assert list(hyperlopit.var_names[:3]) == ["X126.rep1", "X127N.rep1", "X127C.rep1"]
        assert isinstance(var["Acquisiton.Method"].dtype, pd.CategoricalDtype)
        assert var.loc["X127N.rep1", "Iodixonal.Density"] == 6.0

    def test_expression_values(self, hyperlopit):
        assert hyperlopit.X[0, :5].round(3).tolist() == [0.002, 0.047, 0.11, 0.08, 0.198]
        # "Normalised to sum of intensities" -- per replicate, and there are two of them
        assert np.allclose(hyperlopit.X.sum(axis=1), 2.0, atol=0.01)
        assert np.allclose(hyperlopit.X[:, :10].sum(axis=1), 1.0, atol=0.01)


# ==============================================================================
# dunkley2006: factor columns next to a character markers column
# ==============================================================================


class TestDunkley2006:
    def test_r_factors_become_categoricals_without_the_sentinel(self, dunkley):
        """ "unknown" was a *level* of these factors; it must be removed, not left as a phantom."""
        obs = dunkley.obs
        for column in ["assigned", "new", "pd.2013", "pd.markers", "markers.orig"]:
            assert isinstance(obs[column].dtype, pd.CategoricalDtype), column
        assert list(obs["markers.orig"].cat.categories) == [
            "ER",
            "Golgi",
            "PM",
            "mit/plastid",
            "vacuole",
        ]
        assert obs["markers.orig"].isna().sum() == 17
        assert "Phenotype 1" in obs["pd.2013"].cat.categories

    def test_the_character_markers_column_is_plain_object(self, dunkley):
        obs = dunkley.obs
        assert obs["markers"].dtype == object
        assert obs.loc["AT1G56340", "markers"] == "ER lumen"
        assert obs.loc["AT1G56340", "markers.orig"] == "ER"
        assert pd.isna(obs.loc["AT2G16530", "markers.orig"])  # "unknown" in a factor
        assert obs.loc["AT2G16530", "markers"] == "ER membrane"

    def test_var_dtypes(self, dunkley):
        var = dunkley.var
        assert list(var.columns) == ["membrane.prep", "fraction", "replicate"]
        assert str(var["membrane.prep"].dtype) == "Int32"
        assert isinstance(var["replicate"].dtype, pd.CategoricalDtype)
        assert list(dunkley.var_names[:3]) == ["M1F1A", "M1F4A", "M1F7A"]

    def test_processing_log(self, dunkley):
        assert dunkley.uns["processing"] == [
            "Loaded on Thu Jul 16 22:53:08 2015.",
            "Normalised to sum of intensities.",
            "Added markers from  'mrk' marker vector. Thu Jul 16 22:53:08 2015",
        ]

    def test_expression_values(self, dunkley):
        assert dunkley.X[0, :4].round(4).tolist() == [0.3367, 0.3033, 0.2011, 0.1588]


# ==============================================================================
# itzhak2016stcSILAC: missing quantitation, odd column names, no processing log
# ==============================================================================


class TestItzhak2016:
    def test_missing_values_are_kept_when_asked(self, itzhak):
        assert int(np.isnan(itzhak.X).sum()) == 140
        assert int(np.isnan(itzhak.X).any(axis=1).sum()) == 8
        # P15954 is missing one whole map (5 fractions) twice over
        row = itzhak["P15954"].X[0]
        assert np.isnan(row[10:15]).all() and np.isnan(row[20:25]).all()
        assert not np.isnan(row[:10]).any()

    def test_missing_values_become_zero_by_default(self, real_rdata):
        with real_rdata():
            adata = _read("itzhak2016stcSILAC")
        assert not np.isnan(adata.X).any()
        assert adata["P15954"].X[0, 10:15].tolist() == [0.0] * 5

    def test_column_names_with_spaces_are_kept_verbatim(self, itzhak):
        """Including the double space pRolocdata has in one of them."""
        assert "Organellar  markers with sub-compartments" in itzhak.obs.columns
        assert "Profiled in how many maps?" in itzhak.obs.columns
        assert list(itzhak.var_names[:2]) == ["log H/L MAP1_03K", "log H/L MAP1_06K"]

    def test_an_empty_processing_log_is_absent_not_empty(self, itzhak):
        assert "processing" not in itzhak.uns

    def test_values(self, itzhak):
        obs = itzhak.obs
        assert obs.loc["Q5JTZ9", "Lead gene name"] == "AARS2"
        assert obs.loc["Q5JTZ9", "Organellar  markers with sub-compartments"] == "Mito_Matrix"
        assert obs.loc["Q5JTZ9", "Profiled in how many maps?"] == 6.0
        assert set(obs["markers"].dropna()) >= {"Ergic/cisGolgi", "ER_high curvature"}


# ==============================================================================
# tan2009r1: four fractions, several factor columns
# ==============================================================================


class TestTan2009:
    def test_four_fractions(self, tan):
        assert list(tan.var_names) == ["X114", "X115", "X116", "X117"]
        assert list(tan.var.columns) == ["Fractions"]
        assert isinstance(tan.var["Fractions"].dtype, pd.CategoricalDtype)

    def test_factor_level_order_survives(self, tan):
        """R's level order, uppercase before lowercase, not a re-sort on the Python side."""
        assert list(tan.obs["markers.tl"].cat.categories) == [
            "ER",
            "Golgi",
            "Nucleus",
            "PM",
            "Proteasome",
            "Ribosome 40S",
            "Ribosome 60S",
            "mitochondrion",
        ]
        assert list(tan.obs["PLSDA"].cat.categories) == ["ER/Golgi", "PM", "mitochondrion"]
        assert tan.obs["PLSDA"].isna().sum() == 21

    def test_mixed_dtypes_in_fdata(self, tan):
        obs = tan.obs
        assert str(obs["No.peptide.IDs"].dtype) == "Int32"
        assert pd.api.types.is_float_dtype(obs["Mascot.score"])
        assert obs["pd.2013"].dtype == object  # character upstream, unlike dunkley2006's
        assert obs.loc["Q7KMP8", "Mascot.score"] == pytest.approx(84.8)
        assert obs.loc["Q7KMP8", "pd.2013"] == "Phenotype 2"
        assert obs.loc["Q7KMP8", "markers"] == "Proteasome"
        assert pd.isna(obs.loc["Q7KMP8", "PLSDA"])

    def test_processing_log(self, tan):
        assert tan.uns["processing"] == [
            "Added markers from  'mrk' marker vector. Thu Jul 16 22:53:44 2015"
        ]
