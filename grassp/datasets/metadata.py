import pandas as pd
from pathlib import Path


def subcellular_annotations() -> pd.DataFrame:
    return pd.read_csv(Path(__file__).parent / "external/subcellular_annotations.tsv", sep="\t")
