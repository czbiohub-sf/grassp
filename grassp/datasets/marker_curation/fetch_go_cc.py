"""Build a cleaned, propagated human GO cellular-component vocabulary.

Raw GO CC is not usable as a label vocabulary: it mixes places a protein can be with
groupings that merely organise the ontology, and GOA's high-throughput evidence codes
dump most of the proteome into ``extracellular exosome``. The recipe below is the one
both published builds used, reconciled and pinned:

1. GAF rows with a locational qualifier (:data:`KEEP_QUALIFIERS`; an empty qualifier is
   legacy and allowed), dropping ``NOT`` and the high-throughput evidence codes in
   :data:`DROP_EVIDENCE`;
2. propagate each annotation up ``is_a`` + ``part_of``, the closure GO CC propagation is
   defined over;
3. drop generic terms: GO's own ``gocheck_do_not_annotate`` subset, plus the structural
   roots it misses (:data:`GENERIC_TERMS`);
4. spatial dispersion -- a subcellular location lives under one major organelle, so drop
   grouping terms that sit under no major yet whose descendants span
   :data:`MAJOR_SPAN` or more of them (functional complex types, topological groupings);
5. collapse Jaccard >= :data:`JACCARD` near-duplicates, keeping the most specific term.

**Deviation from the published builds, and why.** Both of them additionally dropped
terms covering more than half of a *population* -- the map's own gene symbols in one
case, a fixed 9,518-gene reference list in the other -- which made the vocabulary a
function of the dataset being annotated and so impossible to vendor. Auditing that
filter against both populations showed it removes exactly one term, ``cytoplasm``
(GO:0005737, at 0.756 of the population), with the nearest survivor far below the cut at
0.463 (``nucleus``). It is a one-term blacklist wearing a threshold's clothes, so
GO:0005737 simply joins :data:`GENERIC_TERMS` and the population argument is gone. The
sibling COMPARTMENTS recipe already blacklists ``CYTOPLASM`` by name, twice. Applied at
the same step the population cut ran, steps 4 and 5 see identical input, so the build is
unchanged for those two populations.

Callers who do want a dataset-relative cut should express it at annotation time, where
the AnnData is in hand, rather than by rebuilding the vocabulary.

Outputs, into ``../external/``:

- ``go_cc_human.gmt`` -- term label ``name (GO:id)`` as both published vocabularies have
  it, with the bare GO id in the description column;
- ``go_cc_human.json`` -- build date, GO release and every parameter above.

Run from anywhere:
    python marker_curation/fetch_go_cc.py
    python marker_curation/fetch_go_cc.py --obo /path/to/go-basic.obo --gaf /path/to/HUMAN-uniprot.gaf.gz
"""

from __future__ import annotations
import argparse
import gzip
import re
import sys

from collections import defaultdict
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _vocab_utils import (  # noqa: E402
    EXTERNAL,
    collapse_near_duplicates,
    download,
    write_gmt,
    write_provenance,
)

#: The GO release both published maps were built from. Pinned rather than
#: ``current.geneontology.org``: the curation repo tracked the rolling release and its
#: vocabulary had already drifted 61 terms away from the paper's.
GO_RELEASE = "2026-06-19"
OBO_URL = f"https://release.geneontology.org/{GO_RELEASE}/ontology/go-basic.obo"
GAF_URL = f"https://release.geneontology.org/{GO_RELEASE}/annotations/gaf/HUMAN-uniprot.gaf.gz"

KEEP_QUALIFIERS = {"located_in", "part_of", "is_active_in"}
DROP_EVIDENCE = {"HDA", "HTP", "ND"}  # high-throughput; drops the exosome dump

#: Structural roots the ``gocheck_do_not_annotate`` subset misses, plus ``cytoplasm``
#: (see the module docstring): membrane-enclosed lumen, organelle, protein-containing
#: complex, membrane, cellular_component, cellular anatomical entity, cytoplasm.
GENERIC_TERMS = {
    "GO:0031974",
    "GO:0043226",
    "GO:0032991",
    "GO:0016020",
    "GO:0005575",
    "GO:0110165",
    "GO:0005737",
}

MAJOR_SPAN = 3
JACCARD = 0.9

#: nucleus, mitochondrion, ER, Golgi, endosome, lysosome, peroxisome, plasma membrane,
#: cytosol, extracellular, vacuole, cytoskeleton.
MAJORS = {
    "GO:0005634",
    "GO:0005739",
    "GO:0005783",
    "GO:0005794",
    "GO:0005768",
    "GO:0005764",
    "GO:0005777",
    "GO:0005886",
    "GO:0005829",
    "GO:0005576",
    "GO:0005773",
    "GO:0005856",
}


def parse_obo(path: Path):
    """``(name, parents, cc, do_not_annotate)`` over non-obsolete cellular-component terms."""
    name, namespace, parents, subsets = {}, {}, {}, {}
    for block in path.read_text().split("\n[Term]\n")[1:]:
        term_id = term_name = term_ns = None
        par: set[str] = set()
        subs: list[str] = []
        obsolete = False
        for line in block.split("\n"):
            if line.startswith("["):
                break
            if line.startswith("id: GO:"):
                term_id = line[4:].strip()
            elif line.startswith("name: "):
                term_name = line[6:].strip()
            elif line.startswith("namespace: "):
                term_ns = line[11:].strip()
            elif line.startswith("is_a: GO:"):
                par.add(line.split()[1])
            elif line.startswith("relationship: part_of GO:"):
                par.add(line.split()[2])
            elif line.startswith("subset: "):
                subs.append(line[8:].strip())
            elif line.startswith("is_obsolete: true"):
                obsolete = True
        if term_id and not obsolete:
            name[term_id] = term_name
            namespace[term_id] = term_ns
            parents[term_id] = par
            subsets[term_id] = subs
    cc = {t for t in namespace if namespace[t] == "cellular_component"}
    do_not_annotate = {
        t
        for t in cc
        if "gocheck_do_not_annotate" in subsets[t]
        or "gocheck_do_not_manually_annotate" in subsets[t]
    }
    return name, parents, cc, do_not_annotate


def obo_data_version(path: Path) -> str | None:
    """The ``data-version`` header, e.g. ``releases/2026-06-15``."""
    with path.open() as handle:
        for line in handle:
            if line.startswith("data-version:"):
                return line.split(":", 1)[1].strip()
            if line.startswith("[Term]"):
                break
    return None


def build(obo: Path, gaf: Path) -> tuple[dict[str, list[str]], dict[str, str]]:
    name, parents, cc, do_not_annotate = parse_obo(obo)

    @lru_cache(maxsize=None)
    def ancestors(term: str) -> frozenset[str]:
        out: set[str] = set()
        for parent in parents.get(term, ()):
            if parent in cc:
                out.add(parent)
                out |= ancestors(parent)
        return frozenset(out)

    # (1) GAF, keeping locational qualifiers and dropping high-throughput evidence.
    direct: dict[str, set[str]] = defaultdict(set)
    n_lines = n_kept = 0
    with gzip.open(gaf, "rt", errors="replace") as handle:
        for line in handle:
            if not line or line[0] == "!":
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 15 or f[8] != "C":
                continue
            n_lines += 1
            if "NOT" in f[3] or f[6] in DROP_EVIDENCE:
                continue
            if f[3] and f[3] not in KEEP_QUALIFIERS:
                continue  # an empty qualifier is legacy, and allowed
            if f[4] in cc:
                direct[f[2]].add(f[4])
                n_kept += 1
    print(f"[gocc] GAF CC lines {n_lines} -> kept {n_kept}; {len(direct)} symbols")

    # (2) propagate over is_a + part_of.
    term_to_symbol: dict[str, set[str]] = defaultdict(set)
    for symbol, terms in direct.items():
        for term in set(terms).union(*(ancestors(t) for t in terms)):
            term_to_symbol[term].add(symbol)
    gmt = {t: sorted(s) for t, s in term_to_symbol.items()}
    print(f"[gocc] propagated: {len(gmt)} terms")

    # (3) generic terms: GO's own do-not-annotate subset and the roots it misses.
    gmt = {t: g for t, g in gmt.items() if t not in do_not_annotate and t not in GENERIC_TERMS}
    print(f"[gocc] after generic removal: {len(gmt)} terms")

    # (4) spatial dispersion.
    span: dict[str, set[str]] = defaultdict(set)
    for term in cc:
        majors = (ancestors(term) | {term}) & MAJORS
        if majors:
            for anc in ancestors(term) | {term}:
                span[anc] |= majors
    gmt = {
        t: g
        for t, g in gmt.items()
        if ((ancestors(t) | {t}) & MAJORS) or len(span.get(t, set())) < MAJOR_SPAN
    }
    print(f"[gocc] after the spatial-dispersion filter: {len(gmt)} terms")

    # (5) collapse near-duplicates, keeping the most specific term.
    labelled = {f"{name[t]} ({t})": g for t, g in gmt.items()}
    labelled = collapse_near_duplicates(labelled, JACCARD, most_specific=True)
    print(f"[gocc] after collapsing Jaccard >= {JACCARD}: {len(labelled)} terms")

    go_ids = {label: re.search(r"\((GO:\d+)\)$", label).group(1) for label in labelled}
    return labelled, go_ids


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--obo", type=Path, default=None, help="Existing go-basic.obo.")
    parser.add_argument(
        "--gaf", type=Path, default=None, help="Existing HUMAN-uniprot.gaf.gz."
    )
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=EXTERNAL)
    args = parser.parse_args()

    cache = Path(__file__).resolve().parent / "_cache"
    obo = args.obo or download(OBO_URL, cache / "go-basic.obo", args.force_download)
    gaf = args.gaf or download(GAF_URL, cache / "HUMAN-uniprot.gaf.gz", args.force_download)

    data_version = obo_data_version(Path(obo))
    print(f"[gocc] ontology {data_version}")

    sets, go_ids = build(Path(obo), Path(gaf))
    gmt_path = args.output_dir / "go_cc_human.gmt"
    write_gmt(gmt_path, sets, descriptions=go_ids)
    write_provenance(
        gmt_path,
        source="GO cellular component + GOA human (GAF)",
        url=GAF_URL,
        homepage="https://geneontology.org",
        release=GO_RELEASE,
        ontology_data_version=data_version,
        species="hsap",
        gene_ids="symbol",
        recipe={
            "keep_qualifiers": sorted(KEEP_QUALIFIERS),
            "drop_evidence": sorted(DROP_EVIDENCE),
            "propagate": ["is_a", "part_of"],
            "generic_terms": sorted(GENERIC_TERMS),
            "major_span": MAJOR_SPAN,
            "jaccard": JACCARD,
            "jaccard_keeps": "most specific",
        },
    )


if __name__ == "__main__":
    main()
