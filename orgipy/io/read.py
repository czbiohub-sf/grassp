from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import alphastats

import re

import anndata


def read(loader: alphastats.BaseLoader) -> anndata.AnnData:
    import alphastats

    alphastats.DataSet._check_loader(
        1, loader
    )  # need to put ugly one because this is not specified as a staticmethod in alphapeptstats
    rawinput = loader.rawinput
    software = loader.software
    index_column = loader.index_column
    intensity_column = loader.intensity_column
    intensity_regex = re.compile(intensity_column.replace("[sample]", ".*"))
    filter_columns = loader.filter_columns
    # evidence_df = loader.evidence_df
    gene_names = loader.gene_names

    df = rawinput.copy()

    # get the intensity columns
    if isinstance(intensity_column, str):
        intensity_regex = re.compile(intensity_column.replace("[sample]", ".*"))
        intensity_col_mask = df.columns.map(lambda x: intensity_regex.search(x) is not None)
    else:
        intensity_col_mask = df.columns.isin(intensity_column)

    # Convert to anndata object
    df = df.transpose()
    var = df[~intensity_col_mask].transpose()
    var.set_index(index_column, inplace=True)
    adata = anndata.AnnData(X=df[intensity_col_mask], var=var)
    adata.obs["Intensity_col"] = adata.obs.index
    sample_regex = re.compile(intensity_column.replace("[sample]", ""))
    adata.obs["Sample_name"] = adata.obs.index.str.replace(sample_regex, "", regex=True)
    adata.obs.set_index(keys="Sample_name", drop=False, inplace=True)

    # Add properties of the experiment to uns
    adata.uns["RawInfo"] = {
        "software": software,
        "filter_columns": filter_columns,
        "gene_names": gene_names,
    }
    return adata
