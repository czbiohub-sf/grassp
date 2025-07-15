from __future__ import annotations
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from anndata import AnnData

import os
import urllib

import requests
import scanpy

from scanpy._settings import settings

from .. import io


def hein_2024() -> AnnData:
    """Download the Hein 2024 dataset.
    This dataset is described in https://www.cell.com/cell/fulltext/S0092-8674(24)01344-8.

    Returns
    -------
    AnnData
        The Hein 2024 dataset.
    """
    filename = settings.datasetdir / "hein_2024.h5ad"
    url = "https://drive.google.com/uc?export=download&id=1RMPQucHYbQgzIu-GcwoqApvwa8mODDOp"
    return scanpy.read(filename, backup_url=url)


def itzhak_2016() -> AnnData:
    """Download the ITZHAK 2016 dataset.
    This dataset is described in https://elifesciences.org/articles/16950.

    Returns
    -------
    AnnData
        The ITZHAK 2016 dataset.
    """
    filename = settings.datasetdir / "itzhak_2016.h5ad"
    url = "https://drive.google.com/uc?export=download&id=1zNSTVmJ-Xms86_WtDnjUROQpPXUEr2Ux"
    return scanpy.read(filename, backup_url=url)


def hek_dc_2025(
    enrichment: Literal["raw", "enriched"] = "raw",
) -> AnnData:
    """
    Download the unpublished DC fractionation data from Elias lab at Stanford.

    Returns
    -------
    AnnData
        The HEK DC dataset
    """
    if enrichment == "raw":
        filename = settings.datasetdir / "NonEnriched_DC_Processed.h5ad"
        url = "https://public.czbiohub.org/proteinxlocation/internal/NonEnriched_DC_Processed.h5ad"
        return scanpy.read(filename, backup_url=url)
    elif enrichment == "enriched":
        filename = settings.datasetdir / "Enriched_DC_Processed.h5ad"
        url = (
            "https://public.czbiohub.org/proteinxlocation/internal/Enriched_DC_Processed.h5ad"
        )
        return scanpy.read(filename, backup_url=url)
    else:
        raise ValueError("Enrichment argument must be either 'raw' or 'enriched'")


def hek_atps_2025(
    enrichment: Literal["raw", "enriched"] = "raw",
) -> AnnData:
    """
    Download the unpublished ATPS fractionation data from Elias lab at Stanford.

    Returns
    -------
    AnnData
        The HEK ATPS dataset
    """

    if enrichment == "raw":
        filename = settings.datasetdir / "NonEnriched_ATPS_Processed.h5ad"
        url = "https://public.czbiohub.org/proteinxlocation/internal/NonEnriched_ATPS_Processed.h5ad"
        return scanpy.read(filename, backup_url=url)
    elif enrichment == "enriched":
        filename = settings.datasetdir / "Enriched_ATPS_Processed.h5ad"
        url = "https://public.czbiohub.org/proteinxlocation/internal/Enriched_ATPS_Processed.h5ad"
        return scanpy.read(filename, backup_url=url)
    else:
        raise ValueError("Enrichment argument must be either 'raw' or 'enriched'")


def schessner_2023() -> AnnData:
    """Download the Schessner 2023 dataset.
    This dataset is described in https://www.nature.com/articles/s41467-023-41000-7.

    Returns
    -------
    AnnData
        The Schessner 2023 dataset.
    """
    filename = settings.datasetdir / "schlessner_2023.h5ad"
    url = "https://drive.google.com/uc?export=download&id=1JMHWDqLeX3bacvMRQZopg1VJzc0WRvNK"
    return scanpy.read(filename, backup_url=url)


def download_prolocdata(name: str) -> AnnData:
    """Download a prolocdata file from the prolocdata repository.
    To see the list of available files, use `list_prolocdata_files`.
    You can find more information about the prolocdata repository at https://github.com/lgatto/pRolocdata.

    See Also
    --------
    list_prolocdata_files : List all available prolocdata files that can be downloaded.


    Parameters
    ----------
    name : str
        The name of the file to download.

    Returns
    -------
    AnnData
        The downloaded file as an AnnData object.
    """
    parsed_url = urllib.parse.urlparse(name)
    if parsed_url.scheme != "":
        return io.read_prolocdata(parsed_url.geturl())
    else:
        files = list_prolocdata_files()
        if name not in files:
            raise ValueError(f"Invalid prolocdata file: {name}")
        return io.read_prolocdata(files[name])


def list_prolocdata_files() -> dict:
    """List all files in the prolocdata repository.
    You can find more information about the prolocdata repository at https://github.com/lgatto/pRolocdata.

    Returns
    -------
    dict
        A dictionary of file names and their download URLs.
    """
    # GitHub API endpoint for repository contents in the data directory
    api_url = "https://api.github.com/repos/lgatto/pRolocdata/contents/data"

    response = requests.get(api_url)
    response.raise_for_status()

    files = {
        os.path.splitext(item["name"])[0]: item["download_url"]
        for item in response.json()
        if item["type"] == "file"
    }

    return files
