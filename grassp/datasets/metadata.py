from pathlib import Path

import pandas as pd


def subcellular_annotations() -> pd.DataFrame:
    """Load the subcellular annotations.

    Returns
    -------
    pd.DataFrame
        The subcellular annotations.
    """
    return pd.read_csv(
        Path(__file__).parent / "external/subcellular_annotations.tsv",
        sep="\t",
        index_col=0,
    )


def _read_gmt(path: Path) -> pd.DataFrame:
    """Read a gmt file.

    Parameters
    ----------
    path : Path
        The path to the gmt file.

    Returns
    -------
    pd.DataFrame
        The gmt file.
    """
    go_terms = []
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            term_id = parts[0]
            # description = parts[1]
            genes = parts[2:]
            for gene in genes:
                if gene:  # Skip empty entries
                    go_terms.append([gene, term_id])
    # Create DataFrame with gene-term pairs
    go_df = pd.DataFrame(go_terms, columns=["genesymbol", "geneset"])
    return go_df


def uniprot_compartment_goterms() -> pd.DataFrame:
    """Load the uniprot compartment goterms.

    Returns
    -------
    pd.DataFrame
        The uniprot compartment goterms.
    """

    return _read_gmt(
        Path(__file__).parent / "external/custom_goterms_genes_reviewed.gmt",
    )
