"""Bundled compartment gene sets, one accessor per source.

Each source is its own function rather than a ``source=`` argument on a shared one,
because the filtering that produces them barely overlaps: the curated set is defined by
a hand-written term map plus Complex Portal queries, the UniProt SL set by an ancestor
walk over the controlled vocabulary. A single entry point would carry one mutually
exclusive keyword per source and silently ignore three of them on every call.

What the accessors *share* -- species resolution, GMT parsing, the ``min_genes`` floor
-- lives in the private helpers here, so a new source adds a docstring and its own
filter rather than a fourth copy of the parser. :func:`grassp.tools.enrichment._load_gmt`
resolves its bundled default through :func:`gene_sets_curated`, so the species map has
one home.
"""

from __future__ import annotations
import gzip
import json
import pickle

from pathlib import Path
from typing import Literal

#: Species codes, matching ``pp.add_markers`` and the ``species=`` argument of the
#: tools, rather than the ``human``/``mouse``/``yeast`` tokens used in the filenames.
Species = Literal["hsap", "mmus", "scer"]

_SPECIES_TO_FILE_TOKEN: dict[str, str] = {
    "hsap": "human",
    "mmus": "mouse",
    "scer": "yeast",
}

_EXTERNAL = Path(__file__).parent / "external"


def _resolve_path(stem: str, species: str, available: set[str] | None = None) -> Path:
    """Map ``(stem, species code)`` onto a bundled GMT, or raise with the options.

    ``available`` narrows the species for sources that are only vendored for some of
    them -- COMPARTMENTS and GO CC are human-only -- so the caller gets "bundled for
    hsap only" rather than a FileNotFoundError on a path they never chose.
    """
    if species not in _SPECIES_TO_FILE_TOKEN:
        raise ValueError(
            f"species must be one of {sorted(_SPECIES_TO_FILE_TOKEN)}, got {species!r}"
        )
    if available is not None and species not in available:
        raise ValueError(f"{stem!r} is bundled for {sorted(available)} only, got {species!r}")
    path = _EXTERNAL / f"{stem}_{_SPECIES_TO_FILE_TOKEN[species]}.gmt"
    if not path.exists():  # pragma: no cover - only reachable from a broken install
        raise FileNotFoundError(f"Bundled gene sets missing: {path}")
    return path


def provenance(stem: str, species: str = "hsap") -> dict:
    """How a vendored vocabulary was built: source, release, build date, recipe.

    Written by the ``marker_curation`` scripts next to the GMT. COMPARTMENTS has no
    upstream version to cite -- Jensen lab publish a rolling file -- so ``built`` is the
    only handle on which vocabulary a result came from, which is why the accessors quote
    it in their docstrings and a test keeps the two in step.
    """
    path = _resolve_path(stem, species).with_suffix(".json")
    if not path.exists():
        raise FileNotFoundError(f"No provenance sidecar for {stem!r} ({species}): {path}")
    return json.loads(path.read_text())


def _read_gmt(path: Path) -> dict[str, tuple[str, list[str]]]:
    """Parse a GMT into ``{term: (description, genes)}``, keeping the description.

    The description column is what makes the UniProt SL sets navigable -- it holds the
    SL accession, which is the key into the controlled vocabulary and so into the
    hierarchy. Dropping it, as a plain ``{term: genes}`` parser does, is what would force
    ``root=`` to re-derive the accession by name lookup.
    """
    gene_sets: dict[str, tuple[str, list[str]]] = {}
    with path.open() as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            term, description, *genes = parts
            gene_sets[term] = (description, [g for g in genes if g])
    return gene_sets


def _apply_min_genes(
    gene_sets: dict[str, list[str]], min_genes: int | None
) -> dict[str, list[str]]:
    """Drop terms below the size floor."""
    if min_genes is None:
        return gene_sets
    if min_genes < 1:
        raise ValueError(f"min_genes must be at least 1, got {min_genes}")
    return {term: genes for term, genes in gene_sets.items() if len(genes) >= min_genes}


def gene_sets_curated(
    species: Species = "hsap",
    *,
    min_genes: int | None = None,
) -> dict[str, list[str]]:
    """Curated compartments that subcellular proteomics can usually resolve.

    Manually curated lists of compartments that a fractionation or proximity
    labelling experiment tends to separate into distinct profiles. They are the
    recommended starting point for an initial annotation, because they are the easiest to interpret,
    they are robust across datasets, and each carries enough members each compartment.

    These gene sets were created the following way:

    * ~90 fine-grained UniProt SL terms mapped onto ~20 coarse compartment labels by a
      hand-written term map;
    * large complexes that these experiments routinely resolve as their own cluster,
      the 40S and 60S cytosolic ribosomes and the proteasome, were taken from EBI Complex
      Portal cross-references or a UniProt keyword where no Complex Portal entry exists;

    Gene identifiers are gene-name tokens of reviewed UniProt entries for the species.

    Parameters
    ----------
    species
        One of ``"hsap"`` (human, 24 terms), ``"mmus"`` (mouse, 24 terms) or
        ``"scer"`` (yeast, 23 terms).
    min_genes
        Drop terms with fewer than this many genes. ``None`` (default) keeps all of
        them; every term in this set is large enough for most purposes.

    Returns
    -------
    dict[str, list[str]]
        Mapping of compartment label to gene names, ready to pass as ``gene_sets`` to
        :func:`grassp.tl.mgsa`, :func:`grassp.tl.calculate_cluster_enrichment` or
        :func:`grassp.tl.merge_clusters_go`. Those functions already use this set as
        their default, so an explicit call is only needed to inspect or subset it.

    See Also
    --------
    gene_sets_uniprot_sl : the full, unconsolidated UniProt SL vocabulary.
    grassp.pp.add_markers : published marker sets, as ``.obs`` label columns.

    Examples
    --------
    >>> import grassp as gr
    >>> sets = gr.ds.gene_sets_curated()
    >>> len(sets)
    24
    >>> sorted(sets)[:4]
    ['40S cytosolic ribosome', '60S cytosolic ribosome', 'Cell membrane', 'Centrosome']
    >>> len(gr.ds.gene_sets_curated("scer"))
    23

    Notes
    -----
    Regenerated by ``python marker_curation/fetch_consolidated_terms.py``, which writes
    ``uniprot_subcell_consolidated_{species}.gmt``.
    """
    path = _resolve_path("uniprot_subcell_consolidated", species)
    gene_sets = {term: genes for term, (_, genes) in _read_gmt(path).items()}
    return _apply_min_genes(gene_sets, min_genes)


def _vocabulary_index() -> tuple[dict, dict[str, str]]:
    """The SL controlled vocabulary plus a name -> accession index."""
    with gzip.open(_EXTERNAL / "uniprot_subcell_vocab.pkl.gz", "rb") as f:
        vocab = pickle.load(f)
    return vocab, {entry["ID"]: acc for acc, entry in vocab.items()}


def gene_sets_uniprot_sl(
    species: Species = "hsap",
    *,
    root: str | None = None,
    min_genes: int | None = None,
) -> dict[str, list[str]]:
    """The full UniProt subcellular-location vocabulary, one term per node.

    Where :func:`gene_sets_curated` collapses the vocabulary onto compartments an
    experiment can resolve, this keeps every fine-grained term as its own set: 240 terms
    for human, 241 for mouse, 107 for yeast. Use it when a coarse annotation is already
    in place and a specific sub-compartment question needs finer terms. Be aware that
    many of these terms are not separable in a fractionation profile -- ``Nucleolus``
    and ``Nucleoplasm`` co-migrate -- so an assignment to one of them is a statement
    about the annotation, not about the data.

    Terms are dropped from the raw vocabulary when they are host-cell locations, viral
    particles, structures absent from the species, or generic containers broad enough to
    absorb most of the proteome (``Membrane``, ``Endomembrane system``). Terms with no
    reviewed protein in the species are dropped too, which removes organelles the
    species does not have -- plastids for the mammals, and conversely keeps Vacuole and
    the spindle pole body for yeast.

    Parameters
    ----------
    species
        One of ``"hsap"``, ``"mmus"`` or ``"scer"``.
    root
        Keep only the terms at or below this one in the hierarchy, given as an SL term
        name (``"Nucleus"``) or accession (``"SL-0191"``). Both is-a and part-of edges
        are followed. ``None`` (default) returns every term.
    min_genes
        Drop terms with fewer than this many genes. Worth setting here: the full
        vocabulary contains many terms with a handful of members, which are unstable
        under enrichment testing.

    Returns
    -------
    dict[str, list[str]]
        Mapping of SL term name to gene names.

    See Also
    --------
    gene_sets_curated : the coarse, experiment-resolvable compartments to start from.
    grassp.datasets.uniprot_cc.uniprot_subcellular_vocabulary : the term definitions and
        hierarchy behind these sets.

    Examples
    --------
    >>> import grassp as gr
    >>> len(gr.ds.gene_sets_uniprot_sl())
    240
    >>> nuclear = gr.ds.gene_sets_uniprot_sl(root="Nucleus")
    >>> sorted(nuclear)[:4]
    ['Cajal body', 'Gem', 'Nuclear body', 'Nuclear pore complex']
    >>> len(gr.ds.gene_sets_uniprot_sl(root="Mitochondrion", min_genes=100))
    7

    Notes
    -----
    Membership comes from UniProt's ``cc_scl_term`` query, which **matches down the
    hierarchy**: the genes listed for ``Nucleus`` include those annotated only to
    ``Nucleolus``. Terms therefore contain their descendants rather than partitioning
    them, so treating the sets as disjoint classes, or as independent draws in a
    leave-one-out evaluation, leaks membership between parent and child.

    Regenerated by ``python marker_curation/fetch_hierarchical_goterms.py``.
    """
    path = _resolve_path("uniprot_subcell", species)
    parsed = _read_gmt(path)

    if root is not None:
        vocab, name_to_acc = _vocabulary_index()
        root_acc = root if root in vocab else name_to_acc.get(root)
        if root_acc is None:
            raise ValueError(
                f"root {root!r} is not a UniProt SL term name or accession. "
                "See grassp.datasets.uniprot_cc.uniprot_subcellular_vocabulary()."
            )

        def _is_under(accession: str) -> bool:
            if accession == root_acc:
                return True
            seen, stack = {accession}, [accession]
            while stack:
                entry = vocab.get(stack.pop(), {})
                # is-a and part-of both: the SL hierarchy splits containment across the
                # two inconsistently (Nucleolus is *part of* Nucleus, Acrosome *is a*
                # secretory vesicle), so one relation alone loses half the subtree.
                for parent_name in entry.get("HI", []) + entry.get("HP", []):
                    parent_acc = name_to_acc.get(parent_name)
                    if parent_acc is None or parent_acc in seen:
                        continue
                    if parent_acc == root_acc:
                        return True
                    seen.add(parent_acc)
                    stack.append(parent_acc)
            return False

        parsed = {term: value for term, value in parsed.items() if _is_under(value[0])}

    gene_sets = {term: genes for term, (_, genes) in parsed.items()}
    return _apply_min_genes(gene_sets, min_genes)


def gene_sets_compartments(
    species: Species = "hsap",
    *,
    min_genes: int | None = None,
) -> dict[str, list[str]]:
    """COMPARTMENTS curated subcellular localisation, 874 terms for human.

    COMPARTMENTS (Binder et al. 2014, https://compartments.jensenlab.org) integrates
    localisation evidence from several channels; this is the **knowledge** channel, the
    curated one -- manually assigned GO cellular-component terms carrying a 1-5 star
    confidence, already propagated over the ontology. Kept at confidence >= 3 and >= 5
    genes, with generic and non-locational terms removed (whole-cell aggregators,
    topology qualifiers, "region of" groupings, the extracellular-exosome dump) and
    near-duplicate gene sets collapsed at Jaccard >= 0.8, keeping the canonical
    organelle name.

    Sits between :func:`gene_sets_curated` and :func:`gene_sets_go_cc` in granularity.
    Term labels are upper-case, as in both published grassp analyses; the GO id is in
    the GMT's description column.

    **Vendored build: 2026-09-24.** COMPARTMENTS publishes no versioned archive, only a
    rolling download, so this build date is the only version this vocabulary has --
    quote it when reporting results. :func:`grassp.datasets.gene_sets.provenance`
    returns the full record, and ``marker_curation/fetch_compartments.py`` rebuilds it.

    Parameters
    ----------
    species
        Only ``"hsap"`` is bundled; the knowledge channel for other species is not
        curated to a comparable standard.
    min_genes
        Drop terms with fewer than this many genes. The build already applies a floor of
        5; raising it further is worthwhile for enrichment testing.

    Returns
    -------
    dict[str, list[str]]
        Mapping of upper-case compartment name to gene symbols.

    See Also
    --------
    gene_sets_curated : the coarse, experiment-resolvable compartments to start from.
    gene_sets_go_cc : the finer GO cellular-component vocabulary.

    Examples
    --------
    >>> import grassp as gr
    >>> sets = gr.ds.gene_sets_compartments()
    >>> len(sets)
    874
    >>> len(gr.ds.gene_sets_compartments(min_genes=20))
    404

    Notes
    -----
    13 term names contain ``/``, which h5ad rejects as an obsm column name; annotators
    fall back to ``uns[f"{key}_probabilities_categories"`` for those.
    """
    path = _resolve_path("compartments", species, available={"hsap"})
    gene_sets = {term: genes for term, (_, genes) in _read_gmt(path).items()}
    return _apply_min_genes(gene_sets, min_genes)


def gene_sets_go_cc(
    species: Species = "hsap",
    *,
    min_genes: int | None = None,
) -> dict[str, list[str]]:
    """Cleaned, propagated GO cellular component, 1829 terms for human.

    The finest of the bundled vocabularies. Raw GO CC is not usable as a label set --
    it mixes places a protein can be with groupings that only organise the ontology,
    and GOA's high-throughput evidence codes dump most of the proteome into
    ``extracellular exosome`` -- so this is the cleaned form: locational qualifiers
    only, high-throughput evidence dropped, propagated over ``is_a`` + ``part_of``,
    generic terms removed, grouping terms whose descendants span three or more major
    organelles removed, and near-duplicates collapsed at Jaccard >= 0.9 keeping the
    most specific term.

    At this granularity most terms are not separable in a fractionation profile, so
    set ``min_term_size`` when diffusing: of the 1829 terms only ~380 have 20 or more
    members in a typical 9,500-protein map, and a call to one of the remaining 1,400 is
    a statement about the annotation rather than about the data.

    Labels are ``name (GO:id)``, matching both published grassp analyses; the bare GO id
    is also in the GMT's description column.

    **Vendored build: 2026-09-24, from GO release 2026-06-19 (ontology 2026-06-15).**
    :func:`grassp.datasets.gene_sets.provenance` returns the full record, and
    ``marker_curation/fetch_go_cc.py`` rebuilds it against another release.

    Parameters
    ----------
    species
        Only ``"hsap"`` is bundled.
    min_genes
        Drop terms with fewer than this many genes. This is a floor on the *vocabulary*;
        for a floor relative to the object being annotated -- usually what you want --
        pass ``min_term_size`` to :func:`grassp.tl.independent_diffusion` instead.

    Returns
    -------
    dict[str, list[str]]
        Mapping of ``name (GO:id)`` to gene symbols.

    See Also
    --------
    gene_sets_curated : the coarse, experiment-resolvable compartments to start from.
    grassp.tl.independent_diffusion : one-vs-rest diffusion, which these sets are for.

    Examples
    --------
    >>> import grassp as gr
    >>> sets = gr.ds.gene_sets_go_cc()
    >>> len(sets)
    1829
    >>> "nucleolus (GO:0005730)" in sets
    True

    Notes
    -----
    The published builds additionally dropped terms covering over half of the dataset
    being annotated, which made the vocabulary dataset-dependent. Audited against both
    published populations that filter removed exactly one term, ``cytoplasm``, so
    GO:0005737 is simply blacklisted here and the vocabulary is the same artifact for
    every dataset. 35 term names contain ``/``, which h5ad rejects as an obsm column
    name.
    """
    path = _resolve_path("go_cc", species, available={"hsap"})
    gene_sets = {term: genes for term, (_, genes) in _read_gmt(path).items()}
    return _apply_min_genes(gene_sets, min_genes)
