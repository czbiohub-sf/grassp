from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import alphastats
    from typing import Union

import re
import urllib

import anndata
import numpy as np
import pandas as pd
import protdata


def read_alphastats(
    loader: alphastats.BaseLoader,
    x_dtype: Union[np.dtype, type, int, float, None] = None,
    proteins_as_obs: bool = False,
) -> anndata.AnnData:
    """Read proteomics data into an AnnData object.

    Parameters
    ----------
    loader
        A loader object from alphastats that contains the raw proteomics data and metadata
    x_dtype
        Data type to use for the intensity matrix, by default None
    proteins_as_obs
        If True, proteins will be stored in obs (rows) rather than var (columns), by default False


    Notes
    -----
    The loader object must contain:
    - rawinput: DataFrame with protein data
    - software: Name of proteomics software used
    - index_column: Column name containing protein identifiers
    - intensity_column: Column name pattern for intensity values
    - filter_columns: Columns used for filtering
    - gene_names: Gene name mapping information
    """

    try:
        import alphastats
    except ImportError:
        raise Exception(
            "To read alphastats, please install the `alphastats` python package (pip install alphastats)."
        )

    alphastats.DataSet._check_loader(
        1, loader
    )  # need to put ugly 1 because this is not specified as a staticmethod in alphapeptstats
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
    var = df.loc[:, ~intensity_col_mask]
    X = df.loc[:, intensity_col_mask]
    X = X.replace(np.nan, 0)
    obs = pd.DataFrame(index=X.columns)
    var.set_index(index_column, inplace=True)
    var.index = var.index.astype(str)
    adata = anndata.AnnData(X=X.to_numpy(dtype=x_dtype).T, var=var, obs=obs)
    adata.obs["Intensity_col"] = adata.obs.index
    sample_regex = re.compile(intensity_column.replace("[sample]", ""))
    adata.obs["Sample_name"] = adata.obs.index.str.replace(sample_regex, "", regex=True)
    adata.obs.set_index(keys="Sample_name", drop=False, inplace=True)
    obs.index = obs.index.astype(str)

    # Proteins could either be in the rows or columns
    if proteins_as_obs:
        adata = adata.T

    # Add properties of the experiment to uns
    adata.uns["RawInfo"] = {
        "software": software,
        "filter_columns": filter_columns,
        "gene_names": gene_names,
    }
    adata.uns["proteins_as_obs"] = proteins_as_obs
    return adata


def read_prolocdata(file_name: str) -> anndata.AnnData:
    """
    Read a prolocdata file and return an AnnData object.
    Parameters
    ----------
    file_name : str
        The path to the prolocdata file or a URL.
    Returns
    -------
    adata : AnnData
    """
    try:
        import rdata
    except ImportError:
        raise Exception(
            "To read prolocdata, please install the `rdata` python package (pip install rdata)."
        )

    parsed_url = urllib.parse.urlparse(file_name)
    if parsed_url.scheme != "":
        with urllib.request.urlopen(file_name) as dataset:
            pdata = rdata.parser.parse_data(dataset.read(), extension="rda")
    else:
        pdata = rdata.parser.parse_file(file_name)
    proloc_classes = {
        "AnnotatedDataFrame": lambda x, y: x,
        "Versions": lambda x, y: x,
        "MSnProcess": lambda x, y: x,
        "MSnSet": lambda x, y: x,
        "MIAPE": lambda x, y: x,
    }
    pdata = rdata.conversion.convert(
        pdata, constructor_dict={**proloc_classes, **rdata.conversion.DEFAULT_CLASS_MAP}
    )
    dataset_name = next(iter(pdata.keys()))
    pdata = pdata[dataset_name]
    # Create AnnData object
    obs = pdata.featureData.data
    var = pdata.phenoData.data
    X = np.array(pdata.assayData.maps[0]["exprs"])
    adata = anndata.AnnData(obs=obs, var=var, X=X)

    # Add metadata
    adata.uns["dataset_name"] = dataset_name
    metadata = {k: v for k, v in vars(pdata.experimentData).items() if len(v) > 0}
    del metadata[".__classVersion__"]
    adata.uns["MIAPE_metadata"] = metadata

    return adata


def read_maxquant(*args, **kwargs) -> anndata.AnnData:
    """
    Wrapper for protdata.read_maxquant that transposes the AnnData object.
    """
    adata = protdata.io.read_maxquant(*args, **kwargs)
    return adata.T


def read_fragpipe(*args, **kwargs) -> anndata.AnnData:
    """
    Wrapper for protdata.read_fragpipe that transposes the AnnData object.
    """
    adata = protdata.io.read_fragpipe(*args, **kwargs)
    return adata.T


def read_diann(*args, **kwargs) -> anndata.AnnData:
    """
    Wrapper for protdata.read_diann that transposes the AnnData object.
    """
    adata = protdata.io.read_diann(*args, **kwargs)
    return adata.T
