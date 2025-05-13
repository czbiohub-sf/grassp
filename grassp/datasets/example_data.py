from __future__ import annotations
from typing import TYPE_CHECKING

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


def ithzak_2016() -> AnnData:
    """Download the ITHZAK 2016 dataset.
    This dataset is described in https://elifesciences.org/articles/16950.

    Returns
    -------
    AnnData
        The ITHZAK 2016 dataset.
    """
    filename = settings.datasetdir / "ithzak_2016.h5ad"
    url = "https://drive.google.com/uc?export=download&id=1zNSTVmJ-Xms86_WtDnjUROQpPXUEr2Ux"
    return scanpy.read(filename, backup_url=url)


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
