"""Fetch the full *hierarchical* set of UniProt subcellular-location terms
(cellular architecture) for human, mouse, and yeast — without consolidation.

Where ``fetch_custom_goterms.py`` maps ~90 fine-grained UniProt SL terms onto
~20 consolidated compartments, this script keeps every fine-grained term as its
own node and preserves the UniProt SL hierarchy (``is-a`` + ``part-of``). The
result is a browsable compartment graph rather than a flat marker set.

Scope — intracellular architecture, the boundary (plasma) membrane system, and
the extracellular / secreted space. For each species a term is dropped when it
is:

- a host-cell location (name starts with ``Host``) — pathogen annotations;
- a viral particle, a location deposited into another cell, or a structure
  absent from the target species that survives only via a few mis-annotated
  proteins — the term *or any is-a/part-of ancestor* is in the species'
  exclude-ancestors set (:data:`EXCLUDE_ANCESTORS`, plus
  :data:`EXCLUDE_ANCESTORS_MAMMAL` for human/mouse);
- a too-broad or non-native *container* term, dropped by exact name so its
  children/siblings survive (:data:`EXCLUDE_TERMS`, plus
  :data:`EXCLUDE_TERMS_MAMMAL` for human/mouse): the generic roots Membrane /
  Endomembrane system (which aggregate most of the proteome via hierarchical
  matching), Cell envelope (→ keep child Cell membrane), Septate junction
  (→ keep sibling Paranodal septate junction);
- without any reviewed protein *in that species* — auto-removes organelles with
  no counterpart in the species (for mammals: Plastid, Thylakoid, Acidocalcisome,
  Cell wall, …; the same gate keeps genuine yeast structures such as Cell wall,
  Bud, Ascus when the script is run for yeast).

Species-aware exclusions: the Vacuole (+ membrane) and the Spindle pole body are
real budding-yeast organelles (203 / 171 / 35 reviewed yeast proteins), so they
are excluded only for the mammals (human, mouse) and kept for yeast.

Reuses:

- :func:`grassp.datasets.uniprot_cc.uniprot_subcellular_vocabulary` for the
  controlled vocabulary and its hierarchy, and
  :func:`grassp.datasets.uniprot_cc.find_roots` for top-level containers;
- :func:`fetch_custom_goterms.fetch_term_genes` (sibling script) for the
  UniProt reviewed-gene-token query.

Note: the vocabulary from ``subcell.txt`` only contains ``Cellular component``
terms — ``Topology`` / ``Orientation`` entries use ``IT`` / ``IO`` identifiers
instead of ``ID`` and are skipped by the parser — so no category filter is
needed here.

Outputs, for each species. Only the GMT is shipped (written to
``../grassp/datasets/external/``, alongside the ``consolidated_goterms_{species}``
data); the node/edge hierarchy CSVs are curation intermediates and stay in this
``marker_curation/`` directory.

- ``./uniprot_subcell_{species}_nodes.csv`` — one row per kept term with its raw
  is-a / part-of parents, GO xref, top-level part-of container(s) and the number
  of reviewed gene tokens;
- ``./uniprot_subcell_{species}_edges.csv`` — one row per hierarchy edge
  (child -> parent, ``relation`` in {``is_a``, ``part_of``}) between kept terms;
- ``../grassp/datasets/external/uniprot_subcell_{species}.gmt`` — one row per
  kept term (keyed on the SL term name, description = SL accession) listing its
  directly-annotated reviewed gene names.

Run from anywhere:
    python marker_curation/fetch_hierarchical_goterms.py
    python marker_curation/fetch_hierarchical_goterms.py --species yeast
"""

from __future__ import annotations
import argparse
import sys
import time

from pathlib import Path

import pandas as pd
import requests

from grassp.datasets.uniprot_cc import find_roots, uniprot_subcellular_vocabulary

# Reuse the sibling script's UniProt gene-token query + polite defaults.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_custom_goterms import REQUEST_SLEEP, UNIPROT_HEADERS, fetch_term_genes  # noqa: E402

# ------------------------------------------------------------------- config
# Species name -> NCBI taxonomy id (as used by UniProt's model_organism filter)
# and whether it is a mammal (mammals get the extra exclusions below). Yeast is
# S. cerevisiae S288C, matching fetch_custom_goterms.py.
SPECIES: dict[str, dict] = {
    "human": {"taxon": 9606, "mammal": True},
    "mouse": {"taxon": 10090, "mammal": True},
    "yeast": {"taxon": 559292, "mammal": False},
}

# --- exclusions applied to EVERY species -----------------------------------
# A term is excluded if the term itself, or any of its is-a/part-of ancestors,
# has one of these names — i.e. the whole subtree is out of scope. Covers viral
# particles, locations deposited into another cell, and structures that survive
# the gene gate only via a few host/mis-annotated proteins in every one of the
# three species (apicomplexan micronemes / parasitophorous vacuole, fission-yeast
# cell tip). Extracellular / secreted terms are intentionally NOT here — the
# Secreted subtree is kept as a first-class location and its non-native members
# are removed by the reviewed-protein gate instead.
EXCLUDE_ANCESTORS: set[str] = {
    "Virion",
    "Tegument",
    "Target cell",
    "Microneme",
    "Parasitophorous vacuole",
    "Cell tip",
}

# Exact-name exclusions: drop only this node, keep its children/siblings. Two
# rationales:
#  (1) non-native *container* terms whose proteins really belong to a separate,
#      properly-named term that must survive:
#        Cell envelope    -> bacterial covering; genes belong to child "Cell membrane"
#        Septate junction -> invertebrate; keep sibling "Paranodal septate junction"
#  (2) generic type-roots too broad to be a useful location — via hierarchical
#      cc_scl_term matching they aggregate most of the proteome (Membrane ~17k,
#      Endomembrane system ~13k human genes). Their specific descendants
#      (Mitochondrion membrane, ER membrane, Cell membrane, …) are kept.
EXCLUDE_TERMS: set[str] = {
    "Cell envelope",
    "Septate junction",
    "Membrane",
    "Endomembrane system",
}

# --- exclusions applied to MAMMALS only (human, mouse) ---------------------
# These are genuine budding-yeast organelles (Vacuole 203, Vacuole membrane 171,
# Spindle pole body 35 reviewed yeast proteins) but are absent in mammals, where
# they survive only via a handful of mis-annotated proteins. Excluded for the
# mammals; kept for yeast.
EXCLUDE_ANCESTORS_MAMMAL: set[str] = {"Spindle pole body"}
EXCLUDE_TERMS_MAMMAL: set[str] = {"Vacuole", "Vacuole membrane"}


# ---------------------------------------------------------------- functions
def exclusion_sets(mammal: bool) -> tuple[set[str], set[str]]:
    """Return ``(exclude_ancestors, exclude_terms)`` effective for a species."""
    ancestors = set(EXCLUDE_ANCESTORS)
    terms = set(EXCLUDE_TERMS)
    if mammal:
        ancestors |= EXCLUDE_ANCESTORS_MAMMAL
        terms |= EXCLUDE_TERMS_MAMMAL
    return ancestors, terms


def collect_ancestor_names(name: str, name_to_acc: dict[str, str], vocab: dict) -> set[str]:
    """Return every ancestor term name reachable from ``name`` by walking both
    the is-a (``HI``) and part-of (``HP``) relationships upward.

    Parent names that are not themselves vocabulary entries (they can appear as
    free-text hierarchy labels) are still returned so they can be matched
    against the exclude-ancestors set."""
    ancestors: set[str] = set()
    stack = [name]
    while stack:
        current = stack.pop()
        acc = name_to_acc.get(current)
        if acc is None:
            continue
        entry = vocab[acc]
        for parent in (*entry.get("HI", []), *entry.get("HP", [])):
            if parent not in ancestors:
                ancestors.add(parent)
                stack.append(parent)
    return ancestors


def in_scope(
    name: str,
    name_to_acc: dict[str, str],
    vocab: dict,
    exclude_ancestors: set[str],
    exclude_terms: set[str],
) -> bool:
    """True when a term is in scope for a species: not a host location, not an
    exact-name exclusion, and neither the term nor any is-a/part-of ancestor is
    in ``exclude_ancestors``."""
    if name.lower().startswith("host"):
        return False
    if name in exclude_terms:  # exact-name drop, children/siblings unaffected
        return False
    if name in exclude_ancestors:
        return False
    return not (collect_ancestor_names(name, name_to_acc, vocab) & exclude_ancestors)


def structural_keep(
    vocab: dict, exclude_ancestors: set[str], exclude_terms: set[str]
) -> dict[str, str]:
    """Return ``{accession: term_name}`` for terms that pass the structural
    (non-gene) filters for a species."""
    name_to_acc = {entry["ID"]: acc for acc, entry in vocab.items()}
    return {
        acc: entry["ID"]
        for acc, entry in vocab.items()
        if in_scope(entry["ID"], name_to_acc, vocab, exclude_ancestors, exclude_terms)
    }


def build_tables(
    vocab: dict,
    taxon_id: int,
    session: requests.Session,
    exclude_ancestors: set[str],
    exclude_terms: set[str],
    dry_run: bool = False,
    max_terms: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[str]]]:
    """Fetch genes for every structurally-kept term in the given taxon, drop
    terms with no reviewed protein, and assemble the nodes table, edges table,
    and a ``{term_name: [genes]}`` mapping for the GMT."""
    name_to_acc = {entry["ID"]: acc for acc, entry in vocab.items()}
    candidates = structural_keep(vocab, exclude_ancestors, exclude_terms)
    print(
        f"  structural filter: {len(candidates)} / {len(vocab)} terms kept "
        f"(dropped {len(vocab) - len(candidates)} host/viral/non-native/too-broad)"
    )

    items = sorted(candidates.items(), key=lambda kv: kv[1])
    if max_terms is not None:
        items = items[:max_terms]

    genes_by_term: dict[str, list[str]] = {}
    node_records: list[dict] = []
    for acc, term in items:
        if dry_run:
            genes: list[str] = []
        else:
            try:
                genes = fetch_term_genes(term, taxon_id, session)
            except requests.RequestException as exc:
                print(f"    ERROR  {term!r}: {exc}")
                continue
            time.sleep(REQUEST_SLEEP)
        n_genes = len(set(genes))
        if not dry_run and n_genes == 0:
            # No reviewed protein in this species -> not part of its architecture.
            continue
        genes_by_term[term] = sorted(set(genes))
        entry = vocab[acc]
        top_part_of = [vocab[r]["ID"] for r in find_roots(acc, vocab, relationship="HP")]
        node_records.append(
            {
                "SL_id": acc,
                "Term": term,
                "is_a": "|".join(entry.get("HI", [])),
                "part_of": "|".join(entry.get("HP", [])),
                "top_part_of": "|".join(top_part_of),
                "GO": "|".join(entry.get("GO", [])),
                "n_genes": n_genes,
            }
        )

    nodes = pd.DataFrame(
        node_records,
        columns=["SL_id", "Term", "is_a", "part_of", "top_part_of", "GO", "n_genes"],
    )

    # Edges only between terms that survived (both endpoints kept).
    kept_names = set(nodes["Term"])
    edge_records: list[dict] = []
    for acc in nodes["SL_id"]:
        entry = vocab[acc]
        child = entry["ID"]
        for relation, parents in (
            ("is_a", entry.get("HI", [])),
            ("part_of", entry.get("HP", [])),
        ):
            for parent in parents:
                if parent in kept_names:
                    edge_records.append(
                        {
                            "child_id": acc,
                            "child": child,
                            "parent_id": name_to_acc.get(parent, ""),
                            "parent": parent,
                            "relation": relation,
                        }
                    )
    edges = pd.DataFrame(
        edge_records,
        columns=["child_id", "child", "parent_id", "parent", "relation"],
    )
    return nodes, edges, genes_by_term


def write_gmt(
    genes_by_term: dict[str, list[str]], term_to_acc: dict[str, str], path: Path
) -> None:
    """Write a GMT keyed on the (unconsolidated) SL term name; the description
    field holds the SL accession so the hierarchy id survives into the GMT."""
    with path.open("w") as f:
        for term in sorted(genes_by_term):
            genes = genes_by_term[term]
            if not genes:
                continue
            f.write("\t".join([term, term_to_acc.get(term, "Uniprot_SL"), *genes]) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--species",
        nargs="+",
        choices=sorted(SPECIES),
        default=sorted(SPECIES),
        help="species to fetch (default: all)",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="re-download the UniProt SL vocabulary instead of using the cache",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="skip UniProt gene queries; only report the structural filter",
    )
    parser.add_argument(
        "--max-terms",
        type=int,
        default=None,
        help="limit the number of terms queried per species (for quick testing)",
    )
    args = parser.parse_args()

    here = Path(__file__).resolve().parent  # marker_curation (node/edge CSVs)
    external = here.parent / "external"  # shipped data (GMTs only)

    vocab = uniprot_subcellular_vocabulary(force_download=args.force_download)
    print(f"loaded UniProt SL vocabulary: {len(vocab)} cellular-component terms")

    session = requests.Session()
    session.headers.update(UNIPROT_HEADERS)

    for species in args.species:
        cfg = SPECIES[species]
        exclude_ancestors, exclude_terms = exclusion_sets(cfg["mammal"])
        print(f"\n=== {species} (taxon {cfg['taxon']}, mammal={cfg['mammal']}) ===")
        nodes, edges, genes_by_term = build_tables(
            vocab,
            cfg["taxon"],
            session,
            exclude_ancestors,
            exclude_terms,
            dry_run=args.dry_run,
            max_terms=args.max_terms,
        )
        print(
            f"  kept {len(nodes)} terms, {len(edges)} hierarchy edges, "
            f"{sum(len(g) for g in genes_by_term.values())} (term, gene) pairs"
        )

        nodes_path = here / f"uniprot_subcell_{species}_nodes.csv"
        edges_path = here / f"uniprot_subcell_{species}_edges.csv"
        nodes.to_csv(nodes_path, index=False)
        edges.to_csv(edges_path, index=False)
        print(f"  -> {nodes_path}")
        print(f"  -> {edges_path}")

        if not args.dry_run:
            term_to_acc = dict(zip(nodes["Term"], nodes["SL_id"]))
            gmt_path = external / f"uniprot_subcell_{species}.gmt"
            write_gmt(genes_by_term, term_to_acc, gmt_path)
            print(f"  -> {gmt_path}")


if __name__ == "__main__":
    main()
