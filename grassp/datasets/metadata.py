import pandas as pd
from pathlib import Path


def subcellular_annotations() -> pd.DataFrame:
    return pd.read_csv(
        Path(__file__).parent / "external/subcellular_annotations.tsv",
        sep="\t",
        index_col=0,
    )


def uniprot_compartment_goterms_path() -> str:
    return str(Path(__file__).parent / "external/custom_goterms_genes_reviewed.tsv")
