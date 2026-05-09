"""Fetch reviewed UniProt subcellular-location markers for a curated set of
compartments across human, mouse, and yeast.

For each (species, fine-grained term) pair, queries UniProt for reviewed
entries whose ``cc_scl_term`` matches the term, splits multi-name gene-name
fields into individual rows, deduplicates, and writes:

- ``./consolidated_goterms_{species}.csv`` with columns
  ``Compartment, Compartment_consolidated, Gene_name``
- ``../grassp/datasets/external/consolidated_goterms_{species}.gmt`` keyed
  on the consolidated compartment label.

Run from anywhere:
    python marker_curation/fetch_custom_goterms.py
"""

from __future__ import annotations
import time

from pathlib import Path

import pandas as pd
import requests

# ---------------------------------------------------------------------- terms
# Map: fine-grained UniProt SL term -> consolidated compartment label.
TERM_MAP: dict[str, str] = {
    "Autophagosome": "Autophagosome",
    "COPI-coated vesicle": "Golgi apparatus",
    "COPII-coated vesicle": "Golgi apparatus",
    "Caveola": "Cell membrane",
    "Cell cortex": "Cell cortex",
    "Cell membrane": "Cell membrane",
    "Cell projection": "Cell cortex",
    "Cell surface": "Cell membrane",
    "Centriolar satellite": "Centrosome",
    "Centriole": "Centrosome",
    "Centrosome": "Centrosome",
    "Cilium": "Cilium",
    "Cilium axoneme": "Cilium",
    "Cilium basal body": "Cilium",
    "Clathrin-coated pit": "Endosome",
    "Clathrin-coated vesicle": "Endosome",
    "Coated pit": "Endosome",
    "Cytoplasm": "Cytoplasm",
    "Cytoplasmic granule": "Cytoplasm",
    "Cytoskeleton": "Cytoskeleton",
    "Cytosol": "Cytoplasm",
    "Early endosome": "Endosome",
    "Endoplasmic reticulum": "Endoplasmic reticulum",
    "Endoplasmic reticulum-Golgi intermediate compartment": "Golgi apparatus",
    "Endosome": "Endosome",
    "Filopodium": "Cell cortex",
    "Focal adhesion": "Cell cortex",
    "Golgi apparatus": "Golgi apparatus",
    "Golgi stack": "Golgi apparatus",
    "Late endosome": "Endosome",
    "Lipid droplet": "Lipid droplet",
    "Lysosome": "Lysosome",
    "Microtubule organizing center": "Centrosome",
    "Mitochondrion": "Mitochondrion",
    "Mitochondrion envelope": "Mitochondrion",
    "Mitochondrion inner membrane": "Mitochondrion",
    "Mitochondrion matrix": "Mitochondrion",
    "Mitochondrion membrane": "Mitochondrion",
    "Mitochondrion outer membrane": "Mitochondrion",
    "Multivesicular body": "Endosome",
    "Nuclear body": "Nucleus",
    "Nucleolus": "Nucleolus",
    "Nucleoplasm": "Nucleus",
    "Nucleus": "Nucleus",
    "Nucleus lamina": "Nucleus",
    "Nucleus membrane": "Nucleus membrane",
    "Nucleus speckle": "Nucleus",
    "P-body": "RNA granules",
    "PML body": "Nucleus",
    "Peroxisome": "Peroxisome",
    "Phagocytic cup": "Cell cortex",
    "Phagosome": "Endosome",
    "Rough endoplasmic reticulum": "Endoplasmic reticulum",
    "Smooth endoplasmic reticulum": "Endoplasmic reticulum",
    "Stress fiber": "Cytoskeleton",
    "Stress granule": "RNA granules",
    "cis-Golgi network": "Golgi apparatus",
    "trans-Golgi network": "Golgi apparatus",
}

# Species name -> NCBI taxonomy id used by UniProt's `model_organism` filter.
# Yeast uses 559292 (S. cerevisiae S288C reference proteome).
SPECIES: dict[str, int] = {
    "human": 9606,
    "mouse": 10090,
    "yeast": 559292,
}

UNIPROT_URL = "https://rest.uniprot.org/uniprotkb/stream"
UNIPROT_HEADERS = {"User-Agent": "grassp-marker-curation/0.1"}
REQUEST_SLEEP = 0.3  # seconds between calls — be polite to UniProt


# ----------------------------------------------------------------- functions
def fetch_term_genes(term: str, taxon_id: int, session: requests.Session) -> list[str]:
    """Return gene-name tokens for reviewed UniProt entries with the given
    subcellular-location term in the given taxon. Each protein's
    space-separated ``Gene Names`` field is split into individual tokens
    (primary + synonyms), so a protein with several gene names produces
    several entries."""
    query = f'((cc_scl_term:"{term}") AND (reviewed:true)) ' f"AND (model_organism:{taxon_id})"
    params = {
        "fields": "accession,id,gene_names",
        "format": "tsv",
        "query": query,
    }
    response = session.get(UNIPROT_URL, params=params, timeout=60)
    response.raise_for_status()
    lines = response.text.strip().split("\n")
    if len(lines) < 2:
        return []
    genes: list[str] = []
    # First line is the header (Entry, Entry Name, Gene Names).
    for line in lines[1:]:
        fields = line.split("\t")
        if len(fields) < 3:
            continue
        gene_names_field = fields[2]
        for token in gene_names_field.split():
            if token:
                genes.append(token)
    return genes


def fetch_species(species: str, taxon_id: int, session: requests.Session) -> pd.DataFrame:
    """Loop over every term in :data:`TERM_MAP`, collect a long-format
    DataFrame with one row per gene name, and deduplicate."""
    records: list[dict[str, str]] = []
    for term, consolidated in TERM_MAP.items():
        try:
            genes = fetch_term_genes(term, taxon_id, session)
        except requests.RequestException as exc:
            print(f"  ERROR  {term!r} ({species}): {exc}")
            continue
        for gene in genes:
            records.append(
                {
                    "Compartment": term,
                    "Compartment_consolidated": consolidated,
                    "Gene_name": gene,
                }
            )
        print(f"  {term:55s}  {len(genes):5d} gene tokens")
        time.sleep(REQUEST_SLEEP)
    df = pd.DataFrame(
        records, columns=["Compartment", "Compartment_consolidated", "Gene_name"]
    )
    df = df.drop_duplicates(
        subset=["Compartment", "Compartment_consolidated", "Gene_name"]
    ).reset_index(drop=True)
    return df


def write_gmt(df: pd.DataFrame, path: Path) -> None:
    """Write a GMT file (matching ``consolidated_goterms.gmt``) keyed on the
    consolidated compartment label, one row per consolidated compartment with
    all unique gene names."""
    with path.open("w") as f:
        for compartment, group in df.groupby("Compartment_consolidated", sort=True):
            genes = sorted(set(group["Gene_name"]))
            f.write("\t".join([compartment, "Uniprot_SL", *genes]) + "\n")


def main() -> None:
    here = Path(__file__).resolve().parent
    # repo_root = here.parent
    # out_external = repo_root / "grassp" / "datasets" / "external"
    # out_external.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update(UNIPROT_HEADERS)

    for species, taxon_id in SPECIES.items():
        print(f"\n=== {species} (taxon {taxon_id}) ===")
        df = fetch_species(species, taxon_id, session)

        csv_path = here / f"consolidated_goterms_{species}.csv"
        df.to_csv(csv_path, index=False)
        print(f"  -> {csv_path}  ({len(df)} unique gene rows)")

        gmt_path = here.parent / "external" / f"consolidated_goterms_{species}.gmt"

        write_gmt(df, gmt_path)
        n_compartments = df["Compartment_consolidated"].nunique()
        print(f"  -> {gmt_path}  ({n_compartments} consolidated compartments)")


if __name__ == "__main__":
    main()
