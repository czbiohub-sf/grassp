from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from anndata import AnnData

import scanpy
import requests
import os
from .. import io

from scanpy._settings import settings


def hein_2024() -> AnnData:
    filename = settings.datasetdir / "hein_2024.h5ad"
    url = "https://drive.google.com/uc?export=download&id=1RMPQucHYbQgzIu-GcwoqApvwa8mODDOp"
    return scanpy.read(filename, backup_url=url)


def ithzak_2016() -> AnnData:
    filename = settings.datasetdir / "ithzak_2016.h5ad"
    url = "https://drive.google.com/uc?export=download&id=1zNSTVmJ-Xms86_WtDnjUROQpPXUEr2Ux"
    return scanpy.read(filename, backup_url=url)


def schlessner_2023() -> AnnData:
    filename = settings.datasetdir / "schlessner_2023.h5ad"
    url = "https://drive.google.com/uc?export=download&id=1JMHWDqLeX3bacvMRQZopg1VJzc0WRvNK"
    return scanpy.read(filename, backup_url=url)


def download_prolocdata(name: str) -> AnnData:
    from urllib.parse import urlparse

    parsed_url = urlparse(name)
    if parsed_url.scheme != "":
        return io.read_prolocdata(parsed_url.geturl())
    else:
        files = list_prolocdata_files()
        if name not in files:
            raise ValueError(f"Invalid prolocdata file: {name}")
        return io.read_prolocdata(files[name])


def list_prolocdata_files():
    # GitHub API endpoint for repository contents in the data directory
    api_url = f"https://api.github.com/repos/lgatto/pRolocdata/contents/data"

    response = requests.get(api_url)
    response.raise_for_status()

    files = {
        os.path.splitext(item["name"])[0]: item["download_url"]
        for item in response.json()
        if item["type"] == "file"
    }

    return files
