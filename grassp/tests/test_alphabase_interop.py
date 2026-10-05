"""Guard rails for reading search-engine output through alphabase / alphapepttools.

grassp deliberately does **not** wrap ``alphapepttools.io.read_pg_table``: alphabase's
standardized reader output is meant to become a shared format across proteomics tools, so the
``alphapepttools_tutorial`` notebook shows the handful of manual steps that take it into
grassp's proteins-by-samples layout instead. Those steps encode assumptions about what the
reader returns. If an alphabase or alphapepttools release changes them, the tutorial's prose
and its transformation silently drift. These tests pin both down.

* :class:`TestReaderContract` asserts each property of the reader output that the tutorial
  text describes.
* :class:`TestMatchesProtdata` applies the tutorial's transformation and checks the result is
  identical to what :func:`grassp.io.read_maxquant` (protdata) produces from the same file.

``to_grassp`` below mirrors ``read_maxquant_apt`` in the notebook. Keep the two in sync.

Everything runs offline on a small synthetic ``proteinGroups.txt``. The whole module skips
when alphapepttools is not installed (``pip install grassp[dev]`` brings it in).
"""

from __future__ import annotations
import re

import anndata as ad
import numpy as np
import pandas as pd
import pytest

apt = pytest.importorskip("alphapepttools")
from alphabase.pg_reader import pg_reader_provider  # noqa: E402

import grassp as gr  # noqa: E402

# ==============================================================================
# The tutorial's transformation
# ==============================================================================

#: Columns the DC tutorial filters on that alphabase does not map by default. Same as the
#: notebook's ``EXTRA_COLUMNS`` minus the QC statistics, which are irrelevant here.
EXTRA_COLUMNS = {
    "only_identified_by_site": "Only identified by site",
    "potential_contaminant": "Potential contaminant",
}

FLAG_COLUMNS = ("is_decoy", "only_identified_by_site", "potential_contaminant")

#: (alphabase measurement name or regex, column prefix to strip) for each MaxQuant quantity.
QUANTITIES = {
    "lfq": ("lfq", "LFQ intensity "),
    "raw": ("raw", "Intensity "),
    "ibaq": ("ibaq", "iBAQ "),
    "msms": (r"^MS/MS count .+$", "MS/MS count "),
}


def read_apt(path, regex, prefix, extra=EXTRA_COLUMNS):
    """One alphapepttools read with the sample prefix stripped, as in the notebook."""
    a = apt.io.read_pg_table(
        path,
        search_engine="maxquant",
        measurement_regex=regex,
        additional_column_mapping=extra,
    )
    a.obs_names = a.obs_names.str.removeprefix(prefix)
    return a


def to_grassp(path, quantity="lfq", layers=("msms",), index_column="uniprot_ids"):
    """The notebook's ``read_maxquant_apt``: alphapepttools reads -> grassp layout."""
    main = read_apt(path, *QUANTITIES[quantity])
    adata = ad.AnnData(X=main.X.T.astype(np.float32), obs=main.var.copy(), var=main.obs.copy())
    adata.obs_names = adata.obs[index_column].astype(str)
    adata.obs_names.name = None

    for name in layers:
        layer = read_apt(path, *QUANTITIES[name])
        assert (layer.obs_names == adata.var_names).all()
        assert (layer.var[index_column].values == adata.obs[index_column].values).all()
        adata.layers[name] = layer.X.T.astype(np.float32)

    for col in FLAG_COLUMNS:
        if adata.obs[col].dtype != bool:
            adata.obs[col] = adata.obs[col].eq("+")
    adata.uns["RawInfo"] = {
        "Search_Engine": "MaxQuant",
        "reader": "alphapepttools",
        "filter_columns": list(FLAG_COLUMNS),
    }
    return adata


# ==============================================================================
# Synthetic proteinGroups.txt
# ==============================================================================

SAMPLES = ["Rep1_01K", "Rep1_Cyt", "Rep2_01K", "Rep2_cyt"]  # lower-case cyt is a real quirk

# Six protein groups: two ordinary, one without a protein name (the reason alphabase cannot
# index by name), one decoy, one contaminant, one only identified by site.
PROTEINS = pd.DataFrame(
    {
        "Protein IDs": [
            "P0DN26;A0A075B759",
            "A0A096LP01",
            "P0DPI2;A0A0B4J2D5",
            "REV__P12345",
            "CON__P02769",
            "Q9Y6Y8",
        ],
        "Majority protein IDs": [
            "P0DN26",
            "A0A096LP01",
            "P0DPI2",
            "REV__P12345",
            "CON__P02769",
            "Q9Y6Y8",
        ],
        "Protein names": [
            "Peptidyl-prolyl cis-trans isomerase",
            np.nan,
            np.nan,
            np.nan,
            "Albumin",
            "Protein transport protein Sec23A",
        ],
        "Gene names": ["PPIAL4E;PPIAL4A", "LINC00493", np.nan, np.nan, "ALB", "SEC23A"],
        "Peptides": [3, 1, 2, 1, 12, 20],
        "Razor + unique peptides": [3, 1, 2, 1, 12, 20],
        "Sequence coverage [%]": [10.5, 3.0, 7.2, 4.4, 40.1, 55.9],
        "Mol. weight [kDa]": [18.0, 12.3, 20.1, 30.0, 69.3, 86.1],
        "Q-value": [0.0, 0.002, 0.0, 0.01, 0.0, 0.0],
        "Score": [50.1, 6.2, 20.3, 5.0, 323.3, 400.0],
    }
)


def _quantity_block(prefix, rng, integer=False):
    """Per-sample columns for one MaxQuant quantity, with some zeros as MaxQuant writes them."""
    values = rng.integers(0, 5, size=(len(PROTEINS), len(SAMPLES))) * (
        1 if integer else rng.uniform(1e5, 1e8)
    )
    if not integer:
        values = values.round(1)
    return pd.DataFrame(values, columns=[f"{prefix}{s}" for s in SAMPLES])


@pytest.fixture(scope="module")
def pg_file(tmp_path_factory):
    """A MaxQuant proteinGroups.txt with the column layout MaxQuant 2.x writes."""
    rng = np.random.default_rng(0)
    intensity = _quantity_block("Intensity ", rng)
    lfq = _quantity_block("LFQ intensity ", rng)
    msms = _quantity_block("MS/MS count ", rng, integer=True)
    ibaq = _quantity_block("iBAQ ", rng)

    summary = pd.DataFrame(
        {
            # experiment-wide totals share the prefix of the per-sample columns
            "Intensity": intensity.sum(axis=1),
            "iBAQ": ibaq.sum(axis=1),
            "MS/MS count": msms.sum(axis=1),
        }
    )
    flags = pd.DataFrame(
        {
            "Only identified by site": [np.nan, np.nan, np.nan, np.nan, np.nan, "+"],
            "Reverse": [np.nan, np.nan, np.nan, "+", np.nan, np.nan],
            "Potential contaminant": [np.nan, np.nan, np.nan, np.nan, "+", np.nan],
            "id": range(len(PROTEINS)),
        }
    )
    df = pd.concat([PROTEINS, intensity, lfq, msms, ibaq, summary, flags], axis=1)

    path = tmp_path_factory.mktemp("maxquant") / "proteinGroups.txt"
    df.to_csv(path, sep="\t", index=False)
    return path


@pytest.fixture(scope="module")
def apt_lfq(pg_file):
    """The reader output as the tutorial first shows it: defaults, no extra columns."""
    return apt.io.read_pg_table(pg_file, search_engine="maxquant", measurement_regex="lfq")


@pytest.fixture(scope="module")
def via_protdata(pg_file):
    return gr.io.read_maxquant(
        pg_file, intensity_column_prefixes=["LFQ intensity ", "MS/MS count "]
    )


@pytest.fixture(scope="module")
def via_apt(pg_file):
    return to_grassp(pg_file)


# ==============================================================================
# What the reader returns (the tutorial's numbered list)
# ==============================================================================


class TestReaderContract:
    """Each test corresponds to a claim in the tutorial text about the reader output."""

    def test_maxquant_reader_is_registered(self):
        assert "maxquant" in apt.io.list_available_reader("pg_reader")

    def test_default_mapping_provides_uniprot_ids_and_decoy_flag(self):
        mapping = pg_reader_provider.get_reader("maxquant").column_mapping
        assert mapping["uniprot_ids"] == "Protein IDs"
        assert mapping["is_decoy"] == "Reverse"

    def test_named_lfq_regex_selects_per_sample_lfq_columns_only(self):
        regex = pg_reader_provider.get_reader("maxquant").get_preconfigured_regex()["lfq"]
        pattern = re.compile(regex)
        assert pattern.search("LFQ intensity Rep1_01K")
        assert not pattern.search("Intensity Rep1_01K")
        assert not pattern.search("LFQ intensity L Rep1_01K"), "isotope-label channel"

    def test_orientation_is_samples_by_proteins(self, apt_lfq):
        assert apt_lfq.shape == (len(SAMPLES), len(PROTEINS))

    def test_sample_names_keep_the_column_prefix(self, apt_lfq):
        assert apt_lfq.obs_names.tolist() == [f"LFQ intensity {s}" for s in SAMPLES]

    def test_protein_index_falls_back_to_integers(self, apt_lfq):
        # `Protein names` is the first mapped column and is not unique, so alphapepttools
        # cannot use it as var index and numbers the rows instead ...
        assert apt_lfq.var_names.tolist() == [str(i) for i in range(len(PROTEINS))]
        # ... while the accessions travel as a column, unique and complete
        assert apt_lfq.var["uniprot_ids"].tolist() == PROTEINS["Protein IDs"].tolist()
        assert apt_lfq.var["uniprot_ids"].is_unique

    def test_only_the_four_mapped_metadata_columns_survive(self, apt_lfq):
        assert apt_lfq.var.columns.tolist() == ["proteins", "uniprot_ids", "genes", "is_decoy"]

    def test_decoy_flag_is_boolean(self, apt_lfq):
        assert apt_lfq.var["is_decoy"].dtype == bool
        assert apt_lfq.var["is_decoy"].tolist() == [False, False, False, True, False, False]

    def test_additional_columns_arrive_verbatim(self, pg_file):
        a = read_apt(pg_file, *QUANTITIES["lfq"])
        assert set(EXTRA_COLUMNS) <= set(a.var.columns)
        assert a.var["potential_contaminant"].tolist()[4] == "+"
        assert a.var["potential_contaminant"].isna().sum() == len(PROTEINS) - 1

    def test_missing_values_are_zero_not_nan(self, apt_lfq):
        assert not np.isnan(apt_lfq.X).any()
        assert (apt_lfq.X == 0).any()

    def test_summary_columns_are_not_mistaken_for_samples(self, pg_file):
        # "Intensity" / "MS/MS count" without a sample name are experiment-wide totals
        for regex, prefix in (QUANTITIES["raw"], QUANTITIES["msms"]):
            a = read_apt(pg_file, regex, prefix)
            assert a.obs_names.tolist() == SAMPLES


# ==============================================================================
# After the tutorial's transformation, protdata and alphapepttools agree
# ==============================================================================


class TestMatchesProtdata:
    def test_same_proteins_same_order(self, via_apt, via_protdata):
        assert via_apt.obs_names.tolist() == via_protdata.obs_names.tolist()

    def test_same_samples_same_order(self, via_apt, via_protdata):
        assert via_apt.var_names.tolist() == via_protdata.var_names.tolist()

    def test_lfq_intensities_identical(self, via_apt, via_protdata):
        assert via_apt.X.dtype == via_protdata.X.dtype == np.float32
        np.testing.assert_array_equal(via_apt.X, via_protdata.X)

    def test_msms_layer_identical(self, via_apt, via_protdata):
        np.testing.assert_array_equal(
            via_apt.layers["msms"], via_protdata.layers["MS_MS count"]
        )

    @pytest.mark.parametrize(
        "apt_col, protdata_col",
        [
            ("is_decoy", "Reverse"),
            ("potential_contaminant", "Potential contaminant"),
            ("only_identified_by_site", "Only identified by site"),
        ],
    )
    def test_flags_identical(self, via_apt, via_protdata, apt_col, protdata_col):
        assert via_apt.obs[apt_col].dtype == bool
        assert via_apt.obs[apt_col].tolist() == via_protdata.obs[protdata_col].eq("+").tolist()

    def test_contaminant_removal_keeps_the_same_proteins(self, via_apt, via_protdata):
        a = via_apt.copy()
        gr.pp.remove_contaminants(a)  # columns come from uns["RawInfo"], set by to_grassp
        p = via_protdata.copy()
        gr.pp.remove_contaminants(
            p,
            filter_columns=["Only identified by site", "Reverse", "Potential contaminant"],
            filter_value="+",
        )
        assert a.n_obs == len(PROTEINS) - 3
        assert a.obs_names.tolist() == p.obs_names.tolist()

    @pytest.mark.parametrize("quantity", ["raw", "ibaq"])
    def test_other_quantities_match_too(self, pg_file, quantity):
        prefix = QUANTITIES[quantity][1]
        expected = gr.io.read_maxquant(pg_file, intensity_column_prefixes=[prefix])
        got = to_grassp(pg_file, quantity=quantity, layers=())
        assert got.var_names.tolist() == expected.var_names.tolist()
        np.testing.assert_array_equal(got.X, expected.X)
