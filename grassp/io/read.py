from __future__ import annotations

# import re
import urllib

import anndata
import numpy as np
import pandas as pd
import protdata
import scipy.sparse

from ..util import layer_names
from . import _msnset

# def read_alphastats(
#     loader: alphastats.BaseLoader,
#     x_dtype: Union[np.dtype, type, int, float, None] = None,
# ) -> anndata.AnnData:
#     """Read proteomics data into an AnnData object.

#     Parameters
#     ----------
#     loader
#         A loader object from alphastats that contains the raw proteomics data and metadata
#     x_dtype
#         Data type to use for the intensity matrix, by default None


#     Notes
#     -----
#     The loader object must contain:
#     - rawinput: DataFrame with protein data
#     - software: Name of proteomics software used
#     - index_column: Column name containing protein identifiers
#     - intensity_column: Column name pattern for intensity values
#     - filter_columns: Columns used for filtering
#     - gene_names: Gene name mapping information
#     """

#     try:
#         import alphastats
#     except ImportError:
#         raise Exception(
#             "To read alphastats, please install the `alphastats` python package (pip install alphastats)."
#         )

#     alphastats.DataSet._check_loader(
#         1, loader
#     )  # need to put ugly 1 because this is not specified as a staticmethod in alphapeptstats
#     rawinput = loader.rawinput
#     software = loader.software
#     index_column = loader.index_column
#     intensity_column = loader.intensity_column
#     intensity_regex = re.compile(intensity_column.replace("[sample]", ".*"))
#     filter_columns = loader.filter_columns
#     # evidence_df = loader.evidence_df
#     gene_names = loader.gene_names

#     df = rawinput.copy()

#     # get the intensity columns
#     if isinstance(intensity_column, str):
#         intensity_regex = re.compile(intensity_column.replace("[sample]", ".*"))
#         intensity_col_mask = df.columns.map(lambda x: intensity_regex.search(x) is not None)
#     else:
#         intensity_col_mask = df.columns.isin(intensity_column)

#     # Convert to anndata object
#     var = df.loc[:, ~intensity_col_mask]
#     X = df.loc[:, intensity_col_mask]
#     X = X.replace(np.nan, 0)
#     obs = pd.DataFrame(index=X.columns)
#     var.set_index(index_column, inplace=True)
#     var.index = var.index.astype(str)
#     adata = anndata.AnnData(X=X.to_numpy(dtype=x_dtype).T, var=var, obs=obs)
#     adata.obs["Intensity_col"] = adata.obs.index
#     sample_regex = re.compile(intensity_column.replace("[sample]", ""))
#     adata.obs["Sample_name"] = adata.obs.index.str.replace(sample_regex, "", regex=True)
#     adata.obs.set_index(keys="Sample_name", drop=False, inplace=True)
#     obs.index = obs.index.astype(str)

#     # Proteins could either be in the rows or columns

#     # Add properties of the experiment to uns
#     adata.uns["RawInfo"] = {
#         "software": software,
#         "filter_columns": filter_columns,
#         "gene_names": gene_names,
#     }
#     return adata


def _preprocess_adata(adata: anndata.AnnData) -> anndata.AnnData:
    """Preprocess an AnnData object."""

    # Replace NaNs with 0 in .X
    if isinstance(adata.X, np.ndarray):
        adata.X = np.nan_to_num(adata.X, nan=0)
    elif isinstance(adata.X, scipy.sparse.spmatrix):
        adata.X.data = np.nan_to_num(adata.X.data, nan=0, copy=False)

    # Replace NaNs with 0 in all layers
    for layer in layer_names(adata):
        arr = adata.layers[layer]
        if isinstance(arr, np.ndarray):
            adata.layers[layer] = np.nan_to_num(arr, nan=0, copy=False)
        elif isinstance(arr, scipy.sparse.spmatrix):
            adata.layers[layer].data = np.nan_to_num(arr.data, nan=0, copy=False)

    return adata


def _unknown_sentinel_to_nan(
    adata: anndata.AnnData, *, unknown_label: str = _msnset.UNKNOWN_LABEL, set_colors: bool
) -> None:
    """Convert pRoloc's ``"unknown"`` sentinel to ``NaN`` throughout ``.obs``, in place.

    The one pRoloc convention Python has to handle itself. Every other path goes through the
    companion R package, which converts on its own side of the boundary -- but
    :func:`read_prolocdata` parses ``.rda`` files with the pure-Python ``rdata`` package, so
    there is no R here to do it.

    It touches every text column that contains the sentinel, not a nominated marker column: a
    pRolocdata object routinely carries several (``markers``, ``markers.orig``, ``pd.markers``,
    ``pd.2013``). That is not cosmetic -- every grassp annotator selects its markers with
    ``.notna()``, so an untranslated ``"unknown"`` becomes a spurious compartment class and gets
    trained on. Nothing else is touched: no renaming, no dtype coercion.
    """
    touched: list[str] = []
    for column in adata.obs.columns:
        values = adata.obs[column]
        if not _msnset.is_labelish(values):
            continue
        if (values.dropna().astype(object).astype(str) == str(unknown_label)).any():
            adata.obs[column] = _msnset.unknown_to_nan(values, unknown_label)
            touched.append(column)

    if set_colors and touched:
        from ..preprocessing.annotation import set_sensible_compartment_colors

        set_sensible_compartment_colors(adata, columns=touched)


def _import_rdata():
    """Import the optional ``rdata`` dependency or raise a helpful error."""
    try:
        import rdata
    except ImportError as exc:  # pragma: no cover - exercised via the install extra
        raise ImportError(
            "gr.io.read_prolocdata requires the optional 'rdata' dependency (a pure-Python "
            "reader for R's .rda/.rds files). Install it with "
            "`pip install grassp[proloc]`."
        ) from exc
    return rdata


def read_prolocdata(
    file_name: str,
    allow_nullable_strings: bool = False,
    *,
    replace_nan: bool = True,
    unknown_to_nan: bool = True,
    set_colors: bool = True,
) -> anndata.AnnData:
    """Read a pRolocdata ``MSnSet`` file (``.rda``/``.rds``) into an AnnData object.

    Reads R's serialisation format directly with the pure-Python ``rdata`` package, so no R
    installation is involved. The MSnSet maps onto grassp's layout without a transpose --
    ``exprs()`` is already features-by-fractions -- with ``featureData`` becoming ``.obs``,
    ``phenoData`` becoming ``.var``, and ``experimentData`` becoming
    ``.uns["MIAPE_metadata"]``.

    Parameters
    ----------
    file_name : str
        Path to the file, or a URL (e.g. a raw pRolocdata GitHub link).
    allow_nullable_strings : bool, default False
        If False, convert pandas nullable StringDtype columns in obs/var to
        regular Python object-dtype strings for compatibility with older
        anndata writers (anndata<0.11). If True, keep nullable string dtype.
    replace_nan : bool, default True
        Replace ``NaN`` with ``0`` in ``.X`` and every layer. Kept on by default for backward
        compatibility, but consider ``False``: in fractionation data a missing measurement is
        not a measured zero, and pRoloc offers ``filterNA`` precisely because the distinction
        matters.
    unknown_to_nan : bool, default True
        Convert pRoloc's literal ``"unknown"`` sentinel to ``NaN``. **Leaving this off is
        almost never what you want:** every grassp annotator picks its markers with
        ``.notna()``, so an untranslated ``"unknown"`` is treated as a real compartment and
        gets trained on as one. This is the only path where grassp does the conversion itself
        -- the h5ad round trip leaves it to the R side, where the convention belongs.
    set_colors : bool, default True
        Assign compartment colours to the label columns.

    Returns
    -------
    adata : AnnData
        Proteins in ``.obs``, fractions in ``.var``.

    See Also
    --------
    grassp.io.read_msnset : Read an ``MSnSet`` exported as h5ad by the ``grasspio`` R package.
    grassp.io.write_msnset : Send a grassp object to pRoloc.

    Examples
    --------
    >>> adata = gr.io.read_prolocdata("dunkley2006.rda")   # doctest: +SKIP
    >>> adata.obs["markers"].isna().sum()                  # unlabelled proteins   doctest: +SKIP
    """
    rdata = _import_rdata()

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
    # Handle both cases: pdata is a dict (container) or already the dataset
    if isinstance(pdata, dict):
        dataset_name = next(iter(pdata.keys()))  # Reads the first dataset in the dictionary
        pdata = pdata[dataset_name]
    else:
        dataset_name = file_name
    # Create AnnData object with robust dtype handling
    obs_raw = pdata.featureData.data
    var_raw = pdata.phenoData.data
    # Ensure pandas DataFrames
    obs = (obs_raw if isinstance(obs_raw, pd.DataFrame) else pd.DataFrame(obs_raw)).copy()
    var = (var_raw if isinstance(var_raw, pd.DataFrame) else pd.DataFrame(var_raw)).copy()

    # Normalize column/index names to plain Python strings (avoid numpy.str_)
    obs.columns = obs.columns.map(str)
    obs.index = obs.index.map(str)
    var.columns = var.columns.map(str)
    var.index = var.index.map(str)

    # Infer better dtypes; control whether strings become pandas StringDtype
    # obs = obs.convert_dtypes(convert_string=allow_nullable_strings)
    # var = var.convert_dtypes(convert_string=allow_nullable_strings)

    if not allow_nullable_strings:
        for df in (obs, var):
            for c in df.columns:
                dt = df[c].dtype
                if pd.api.types.is_extension_array_dtype(dt) and pd.api.types.is_string_dtype(
                    dt
                ):
                    df[c] = df[c].astype(object)

    # Expression matrix
    X = np.asarray(pdata.assayData.maps[0]["exprs"], dtype=float)

    # Construct AnnData (expects shape: (n_obs, n_vars) == X.shape)
    adata = anndata.AnnData(obs=obs, var=var, X=X)

    # Add metadata
    adata.uns["dataset_name"] = dataset_name
    adata.uns["file_name"] = file_name
    metadata = {
        k: v
        for k, v in vars(pdata.experimentData).items()
        if hasattr(v, "__len__") and len(v) > 0
    }
    # Remove class version key if present
    metadata.pop(".__classVersion__", None)
    adata.uns["MIAPE_metadata"] = metadata

    # The MSnSet's processing log is provenance worth keeping. Guarded the same way as the
    # MIAPE metadata above, since what `rdata` hands back for any given slot varies.
    processing = getattr(getattr(pdata, "processingData", None), "processing", None)
    if hasattr(processing, "__len__") and len(processing) > 0:
        adata.uns["processing"] = [str(entry) for entry in processing]

    if replace_nan:
        _preprocess_adata(adata)

    if unknown_to_nan:
        _unknown_sentinel_to_nan(adata, set_colors=set_colors)

    return adata


def read_maxquant(*args, **kwargs) -> anndata.AnnData:
    """Read MaxQuant proteinGroups.txt file into an AnnData object.

    This function serves as a wrapper around :func:`~protdata.io.read_maxquant`,
    automatically transposing the data so that proteins are stored in ``.obs``
    (rows) and samples in ``.var`` (columns), following grassp conventions.

    Parameters
    ----------
    *args
        Positional arguments passed to :func:`protdata.io.read_maxquant`.
    **kwargs
        Keyword arguments passed to :func:`protdata.io.read_maxquant`.

    Returns
    -------
    AnnData
        Annotated data object with proteins as observations (rows) and samples
        as variables (columns).

    See Also
    --------
    protdata.io.read_maxquant : Original protdata function for reading MaxQuant files.

    Examples
    --------
    >>> import grassp as gr  # doctest: +SKIP
    >>> adata = gr.io.read_maxquant('proteinGroups.txt')  # doctest: +SKIP
    """
    adata = protdata.io.read_maxquant(*args, **kwargs)
    _preprocess_adata(adata)
    return adata.T


def read_fragpipe(*args, **kwargs) -> anndata.AnnData:
    """Read FragPipe combined_protein.tsv file into an AnnData object.

    This function serves as a wrapper around :func:`~protdata.io.read_fragpipe`,
    automatically transposing the data so that proteins are stored in ``.obs``
    (rows) and samples in ``.var`` (columns), following grassp conventions.

    Parameters
    ----------
    *args
        Positional arguments passed to :func:`protdata.io.read_fragpipe`.
    **kwargs
        Keyword arguments passed to :func:`protdata.io.read_fragpipe`.

    Returns
    -------
    AnnData
        Annotated data object with proteins as observations (rows) and samples
        as variables (columns).

    See Also
    --------
    protdata.io.read_fragpipe : Original protdata function for reading FragPipe files.

    Examples
    --------
    >>> import grassp as gr  # doctest: +SKIP
    >>> adata = gr.io.read_fragpipe('combined_protein.tsv')  # doctest: +SKIP
    """
    adata = protdata.io.read_fragpipe(*args, **kwargs)
    _preprocess_adata(adata)
    return adata.T


def read_diann(*args, **kwargs) -> anndata.AnnData:
    """Read DIA-NN report.tsv file into an AnnData object.

    This function serves as a wrapper around :func:`~protdata.io.read_diann`,
    automatically transposing the data so that proteins are stored in ``.obs``
    (rows) and samples in ``.var`` (columns), following grassp conventions.

    Parameters
    ----------
    *args
        Positional arguments passed to :func:`protdata.io.read_diann`.
    **kwargs
        Keyword arguments passed to :func:`protdata.io.read_diann`.

    Returns
    -------
    AnnData
        Annotated data object with proteins as observations (rows) and samples
        as variables (columns).

    See Also
    --------
    protdata.io.read_diann : Original protdata function for reading DIA-NN files.

    Examples
    --------
    >>> import grassp as gr  # doctest: +SKIP
    >>> adata = gr.io.read_diann('report.tsv')  # doctest: +SKIP
    """
    adata = protdata.io.read_diann(*args, **kwargs)
    _preprocess_adata(adata)
    return adata.T
