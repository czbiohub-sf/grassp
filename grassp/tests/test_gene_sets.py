"""The bundled gene-set accessors, and the one wiring they own.

``_load_gmt``'s bundled-default branch resolves through ``gene_sets_curated``, so the
species map lives in one place; the tests below pin both the accessors and that the
delegation kept the tools' behaviour identical.
"""

import re

import pytest

import grassp as gr

from grassp.datasets.gene_sets import (
    _apply_min_genes,
    _read_gmt,
    _resolve_path,
    provenance,
)
from grassp.tools.enrichment import _load_gmt

SPECIES_TERM_COUNTS = {"hsap": 24, "mmus": 24, "scer": 23}
SL_TERM_COUNTS = {"hsap": 240, "mmus": 241, "scer": 107}


class TestGeneSetsCurated:
    @pytest.mark.parametrize("species,n_terms", sorted(SPECIES_TERM_COUNTS.items()))
    def test_every_species_resolves(self, species, n_terms):
        sets = gr.ds.gene_sets_curated(species)
        assert len(sets) == n_terms
        assert all(isinstance(genes, list) and genes for genes in sets.values())

    def test_human_is_the_default(self):
        assert gr.ds.gene_sets_curated() == gr.ds.gene_sets_curated("hsap")

    def test_carries_the_complex_terms_that_are_not_sl_nodes(self):
        """The three Complex Portal terms are what makes this set 'curated' rather than
        a UniProt SL subset -- they have no SL node behind them."""
        sets = gr.ds.gene_sets_curated()
        for term in ("40S cytosolic ribosome", "60S cytosolic ribosome", "Proteasome"):
            assert term in sets
            assert term not in gr.ds.gene_sets_uniprot_sl()

    def test_min_genes_drops_small_terms(self):
        everything = gr.ds.gene_sets_curated()
        floored = gr.ds.gene_sets_curated(min_genes=500)
        assert set(floored) < set(everything)
        assert all(len(genes) >= 500 for genes in floored.values())

    def test_min_genes_is_keyword_only(self):
        with pytest.raises(TypeError):
            gr.ds.gene_sets_curated("hsap", 100)


class TestGeneSetsUniprotSl:
    @pytest.mark.parametrize("species,n_terms", sorted(SL_TERM_COUNTS.items()))
    def test_every_species_resolves(self, species, n_terms):
        assert len(gr.ds.gene_sets_uniprot_sl(species)) == n_terms

    def test_is_finer_than_the_curated_set(self):
        assert len(gr.ds.gene_sets_uniprot_sl()) > len(gr.ds.gene_sets_curated())

    def test_root_keeps_the_subtree(self):
        nuclear = gr.ds.gene_sets_uniprot_sl(root="Nucleus")
        assert {"Nucleolus", "Nuclear pore complex", "Cajal body"} <= set(nuclear)
        assert "Mitochondrion matrix" not in nuclear
        assert len(nuclear) < len(gr.ds.gene_sets_uniprot_sl())

    def test_root_accepts_an_accession(self):
        assert gr.ds.gene_sets_uniprot_sl(root="SL-0191") == gr.ds.gene_sets_uniprot_sl(
            root="Nucleus"
        )

    def test_root_follows_is_a_as_well_as_part_of(self):
        """Nucleolus is *part of* Nucleus, Acrosome *is a* secretory vesicle. Following
        only one relation silently drops half of any subtree."""
        vesicles = gr.ds.gene_sets_uniprot_sl(root="Cytoplasmic vesicle")
        assert "Secretory vesicle" in vesicles  # part_of edge
        assert "Clathrin-coated vesicle" in vesicles  # is_a edge

    def test_root_includes_itself_when_present(self):
        assert "Mitochondrion envelope" in gr.ds.gene_sets_uniprot_sl(
            root="Mitochondrion envelope"
        )

    def test_unknown_root_raises(self):
        with pytest.raises(ValueError, match="not a UniProt SL term name or accession"):
            gr.ds.gene_sets_uniprot_sl(root="Nukleus")

    def test_root_and_min_genes_compose(self):
        sets = gr.ds.gene_sets_uniprot_sl(root="Nucleus", min_genes=100)
        assert sets
        assert set(sets) < set(gr.ds.gene_sets_uniprot_sl(root="Nucleus"))
        assert all(len(genes) >= 100 for genes in sets.values())


class TestSharedHelpers:
    @pytest.mark.parametrize(
        "fn", [gr.ds.gene_sets_curated, gr.ds.gene_sets_uniprot_sl], ids=["curated", "sl"]
    )
    def test_unknown_species_raises_with_the_options(self, fn):
        with pytest.raises(
            ValueError, match=r"species must be one of \['hsap', 'mmus', 'scer'\]"
        ):
            fn("xxxx")

    def test_min_genes_below_one_raises(self):
        with pytest.raises(ValueError, match="min_genes must be at least 1"):
            gr.ds.gene_sets_curated(min_genes=0)

    def test_read_gmt_keeps_the_description_column(self):
        """``root=`` needs the SL accession the description column holds."""
        parsed = _read_gmt(_resolve_path("uniprot_subcell", "hsap"))
        description, genes = parsed["Nucleolus"]
        assert description.startswith("SL-")
        assert genes

    def test_apply_min_genes_is_a_noop_for_none(self):
        sets = {"a": ["x"], "b": ["y", "z"]}
        assert _apply_min_genes(sets, None) == sets


class TestToolsStillResolveTheBundledDefault:
    @pytest.mark.parametrize("species,n_terms", sorted(SPECIES_TERM_COUNTS.items()))
    def test_load_gmt_default_matches_the_accessor(self, species, n_terms):
        assert _load_gmt(None, species=species) == gr.ds.gene_sets_curated(species)
        assert len(_load_gmt(None, species=species)) == n_terms

    def test_public_load_gmt_agrees(self):
        assert gr.tl.load_gmt(None, species="hsap") == gr.ds.gene_sets_curated()

    def test_unknown_species_still_raises_from_the_tools(self):
        with pytest.raises(ValueError, match="species must be one of"):
            _load_gmt(None, species="nope")

    def test_mutating_the_result_cannot_corrupt_a_later_call(self):
        """The accessor re-reads the GMT rather than handing out a shared cache."""
        first = gr.ds.gene_sets_curated()
        first.pop("Nucleus")
        first["Cytoplasm"].clear()
        second = gr.ds.gene_sets_curated()
        assert "Nucleus" in second
        assert second["Cytoplasm"]


class TestGeneSetsCompartments:
    def test_resolves(self):
        sets = gr.ds.gene_sets_compartments()
        assert len(sets) == 874
        assert all(genes for genes in sets.values())

    def test_labels_are_upper_case(self):
        """Both published analyses have COMPARTMENTS terms upper-cased; downstream
        colour lookup is case-insensitive but cross-vocabulary joins are not."""
        assert all(term == term.upper() for term in gr.ds.gene_sets_compartments())

    def test_the_build_floor_of_five_genes_held(self):
        assert min(len(g) for g in gr.ds.gene_sets_compartments().values()) >= 5

    def test_generic_terms_are_absent(self):
        """The whole point of the drop-list: CYTOPLASM and friends name no place."""
        sets = gr.ds.gene_sets_compartments()
        for term in ("CYTOPLASM", "MEMBRANE", "ORGANELLE", "EXTRACELLULAR EXOSOME"):
            assert term not in sets

    def test_description_column_carries_the_go_id(self):
        parsed = _read_gmt(_resolve_path("compartments", "hsap"))
        description, _ = parsed["NUCLEOLUS"]
        assert description.startswith("GO:")

    def test_is_bundled_for_human_only(self):
        for species in ("mmus", "scer"):
            with pytest.raises(ValueError, match="bundled for \\['hsap'\\] only"):
                gr.ds.gene_sets_compartments(species)


class TestGeneSetsGoCc:
    def test_resolves(self):
        assert len(gr.ds.gene_sets_go_cc()) == 1829

    def test_labels_carry_the_go_id(self):
        sets = gr.ds.gene_sets_go_cc()
        assert "nucleolus (GO:0005730)" in sets
        assert all(re.search(r"\(GO:\d{7}\)$", term) for term in sets)

    def test_cytoplasm_is_blacklisted(self):
        """It replaces the published population filter, which removed exactly this one
        term against both published populations."""
        assert "cytoplasm (GO:0005737)" not in gr.ds.gene_sets_go_cc()

    def test_structural_generics_are_absent(self):
        sets = gr.ds.gene_sets_go_cc()
        for term in (
            "membrane (GO:0016020)",
            "organelle (GO:0043226)",
            "cellular_component (GO:0005575)",
        ):
            assert term not in sets

    def test_is_finer_than_every_other_bundled_vocabulary(self):
        assert (
            len(gr.ds.gene_sets_go_cc())
            > len(gr.ds.gene_sets_compartments())
            > len(gr.ds.gene_sets_uniprot_sl())
            > len(gr.ds.gene_sets_curated())
        )

    def test_is_bundled_for_human_only(self):
        with pytest.raises(ValueError, match="bundled for \\['hsap'\\] only"):
            gr.ds.gene_sets_go_cc("mmus")


class TestProvenance:
    """The vendored vocabularies are built from rolling downloads, so the build date is
    the only version COMPARTMENTS has. These keep it recorded and quoted accurately."""

    @pytest.mark.parametrize("stem", ["compartments", "go_cc"])
    def test_sidecar_exists_and_is_complete(self, stem):
        record = provenance(stem)
        for field in ("gmt", "built", "n_terms", "source", "url", "species", "recipe"):
            assert field in record, f"{stem} provenance is missing {field!r}"
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", record["built"])

    @pytest.mark.parametrize(
        "stem,fn",
        [("compartments", gr.ds.gene_sets_compartments), ("go_cc", gr.ds.gene_sets_go_cc)],
    )
    def test_sidecar_term_count_matches_the_gmt(self, stem, fn):
        assert provenance(stem)["n_terms"] == len(fn())

    @pytest.mark.parametrize(
        "stem,fn",
        [("compartments", gr.ds.gene_sets_compartments), ("go_cc", gr.ds.gene_sets_go_cc)],
    )
    def test_docstring_quotes_the_real_build_date(self, stem, fn):
        """A docstring that states a build date goes stale the moment the vocabulary is
        rebuilt, and a stale date is worse than none -- it is what someone will cite."""
        built = provenance(stem)["built"]
        assert built in fn.__doc__, (
            f"{fn.__name__} docstring does not quote its build date {built!r}; "
            f"rerun marker_curation/fetch_{stem}.py and update the docstring"
        )

    def test_docstring_quotes_the_go_release(self):
        release = provenance("go_cc")["release"]
        assert release in gr.ds.gene_sets_go_cc.__doc__

    def test_unknown_source_raises(self):
        with pytest.raises(FileNotFoundError, match="No provenance sidecar"):
            provenance("uniprot_subcell")
