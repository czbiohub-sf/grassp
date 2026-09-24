"""Build the COMPARTMENTS curated subcellular vocabulary for human.

COMPARTMENTS (Binder et al. 2014, https://compartments.jensenlab.org) integrates
subcellular localisation evidence from several channels. The **knowledge** channel is
the curated one: manually assigned GO cellular-component terms with a 1-5 star
confidence, already propagated over the ontology.

Recipe, reconciling the two ports this replaces -- the pan-human map's
``_vocabularies.py`` and the curation repo's ``build_simplified_compartments.py``. The
first rebuilt from the live download but had reverse-engineered its own base; the second
started from a frozen GMT whose builder was lost and applied only the regex pass. This
script does the whole thing from the download, with both generic-term passes:

1. keep rows with confidence >= :data:`MIN_CONFIDENCE` stars;
2. key on the GO term name, collect gene symbols, drop terms under
   :data:`MIN_GENES` members;
3. drop the explicitly listed generic / non-locational terms (:data:`DROP_TERMS`);
4. drop anything matching the generic-name patterns (:data:`GENERIC_PATTERNS`), which
   catches terms that enter COMPARTMENTS in later releases;
5. collapse byte-identical gene sets, then Jaccard >= :data:`JACCARD` near-duplicates,
   keeping the shortest (most canonical) organelle name per cluster.

Term labels stay upper-case, as both published vocabularies have them. The description
column carries the GO id, which the previous builds discarded.

Outputs, into ``../external/``:

- ``compartments_human.gmt``
- ``compartments_human.json`` -- build date, source URL and every parameter above

COMPARTMENTS publishes no versioned archive, only a rolling file, so this vocabulary
cannot be pinned to a release. The build date in the sidecar is the only version it has;
that is precisely why the result is vendored rather than downloaded on demand.

Run from anywhere:
    python marker_curation/fetch_compartments.py
    python marker_curation/fetch_compartments.py --input /path/to/human_compartment_knowledge_full.tsv
"""

from __future__ import annotations
import argparse
import re
import sys

from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _vocab_utils import (  # noqa: E402
    EXTERNAL,
    collapse_near_duplicates,
    deduplicate_identical,
    download,
    write_gmt,
    write_provenance,
)

URL = "https://download.jensenlab.org/human_compartment_knowledge_full.tsv"

MIN_CONFIDENCE = 3  # stars, out of 5; the cut the published vocabulary used
MIN_GENES = 5
JACCARD = 0.8

#: Generic and non-locational terms, read off the published vocabulary: whole-cell
#: aggregators, topology qualifiers, "region/part of" groupings and the
#: extracellular-exosome dump, none of which name a place a protein can be assigned to.
DROP_TERMS = {
    "APICAL PART OF CELL",
    "BASAL PART OF CELL",
    "BOUNDING MEMBRANE OF ORGANELLE",
    "CELL CORTEX REGION",
    "CELL PERIPHERY",
    "CELLULAR ANATOMICAL ENTITY",
    "CELLULAR_COMPONENT",
    "CHROMOSOMAL REGION",
    "CYTOPLASM",
    "CYTOPLASMIC REGION",
    "CYTOSOLIC REGION",
    "ENDOMEMBRANE SYSTEM",
    "ENDOPLASMIC RETICULUM SUBCOMPARTMENT",
    "EXTRACELLULAR EXOSOME",
    "EXTRACELLULAR MEMBRANE-BOUNDED ORGANELLE",
    "EXTRACELLULAR ORGANELLE",
    "EXTRACELLULAR REGION",
    "EXTRACELLULAR VESICLE",
    "GOLGI APPARATUS SUBCOMPARTMENT",
    "INTEGRAL COMPONENT OF MEMBRANE",
    "INTRACELLULAR ANATOMICAL STRUCTURE",
    "INTRACELLULAR MEMBRANE-BOUNDED ORGANELLE",
    "INTRACELLULAR NON-MEMBRANE-BOUNDED ORGANELLE",
    "INTRACELLULAR ORGANELLE",
    "INTRACELLULAR ORGANELLE LUMEN",
    "INTRINSIC COMPONENT OF MEMBRANE",
    "MEMBRANE",
    "MEMBRANE-BOUNDED ORGANELLE",
    "MEMBRANE-ENCLOSED LUMEN",
    "NON-MEMBRANE-BOUNDED ORGANELLE",
    "NUCLEAR LUMEN",
    "ORGANELLE",
    "ORGANELLE ENVELOPE",
    "ORGANELLE ENVELOPE LUMEN",
    "ORGANELLE INNER MEMBRANE",
    "ORGANELLE LUMEN",
    "ORGANELLE MEMBRANE",
    "ORGANELLE OUTER MEMBRANE",
    "ORGANELLE SUBCOMPARTMENT",
    "PERINUCLEAR REGION OF CYTOPLASM",
    "PLASMA MEMBRANE REGION",
    "PROTEIN-CONTAINING COMPLEX",
    "SUPRAMOLECULAR COMPLEX",
}

#: A second, rule-based pass over whatever the explicit list misses. Kept as regexes
#: because it has to hold for terms that enter COMPARTMENTS in later releases.
GENERIC_PATTERNS = [
    r"SUBCOMPARTMENT",
    r"\bSIDE OF\b",
    r"COMPONENT OF MEMBRANE",
    r"SUPRAMOLECULAR",
    r"^PROTEIN-CONTAINING COMPLEX$",
    r"MEMBRANE-BOUNDED ORGANELLE",
    r"^(INTRACELLULAR )?ORGANELLE$",
    r"^ORGANELLE LUMEN$",
    r"MEMBRANE-ENCLOSED LUMEN",
    r"^(INTRACELLULAR )?ORGANELLE LUMEN$",
    r"ANATOMICAL STRUCTURE",
    r"^CATALYTIC COMPLEX$",
    r"^CELL PERIPHERY$",
    r"^INTRACELLULAR$",
    r"^CYTOPLASM$",
]


def parse_knowledge_channel(path: Path) -> tuple[dict[str, list[str]], dict[str, str]]:
    """``(term -> gene symbols, term -> GO id)`` above the confidence cut.

    Columns are ENSP, gene symbol, GO id, GO term name, source, evidence code and the
    1-5 confidence. Gene symbols rather than the ENSP ids, because every downstream
    grassp vocabulary is keyed on symbols.
    """
    genes: dict[str, list[str]] = defaultdict(list)
    seen: dict[str, set[str]] = defaultdict(set)
    go_ids: dict[str, str] = {}
    with path.open() as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 7:
                continue
            try:
                confidence = int(fields[6])
            except ValueError:
                continue
            if confidence < MIN_CONFIDENCE:
                continue
            term, symbol = fields[3].upper(), fields[1]
            go_ids.setdefault(term, fields[2])
            if symbol not in seen[term]:
                seen[term].add(symbol)
                genes[term].append(symbol)
    return dict(genes), go_ids


def build(path: Path) -> tuple[dict[str, list[str]], dict[str, str]]:
    sets, go_ids = parse_knowledge_channel(path)
    print(f"[compartments] confidence >= {MIN_CONFIDENCE}: {len(sets)} terms")

    sets = {t: g for t, g in sets.items() if len(g) >= MIN_GENES}
    print(f"[compartments] >= {MIN_GENES} genes: {len(sets)} terms")

    sets = {t: g for t, g in sets.items() if t not in DROP_TERMS}
    print(f"[compartments] after the generic-term list: {len(sets)} terms")

    sets = {
        t: g for t, g in sets.items() if not any(re.search(p, t) for p in GENERIC_PATTERNS)
    }
    print(f"[compartments] after the generic-term patterns: {len(sets)} terms")

    sets = deduplicate_identical(sets)
    print(f"[compartments] after dropping identical memberships: {len(sets)} terms")

    sets = collapse_near_duplicates(sets, JACCARD, most_specific=False)
    print(f"[compartments] after collapsing Jaccard >= {JACCARD}: {len(sets)} terms")
    return sets, go_ids


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Existing human_compartment_knowledge_full.tsv; downloaded if omitted.",
    )
    parser.add_argument(
        "--force-download", action="store_true", help="Re-download even if cached."
    )
    parser.add_argument("--output-dir", type=Path, default=EXTERNAL)
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    source = args.input or download(
        URL, here / "_cache" / "human_compartment_knowledge_full.tsv", args.force_download
    )

    sets, go_ids = build(Path(source))
    gmt_path = args.output_dir / "compartments_human.gmt"
    write_gmt(gmt_path, sets, descriptions=go_ids)
    write_provenance(
        gmt_path,
        source="COMPARTMENTS knowledge channel (Binder et al. 2014)",
        url=URL,
        homepage="https://compartments.jensenlab.org",
        release=None,  # rolling file; no versioned archive exists
        species="hsap",
        gene_ids="symbol",
        recipe={
            "min_confidence": MIN_CONFIDENCE,
            "min_genes": MIN_GENES,
            "drop_terms": len(DROP_TERMS),
            "generic_patterns": len(GENERIC_PATTERNS),
            "jaccard": JACCARD,
            "jaccard_keeps": "shortest name",
        },
    )


if __name__ == "__main__":
    main()
