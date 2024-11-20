from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from anndata import AnnData

import scanpy

from scanpy._settings import settings


def ithzak_2016() -> AnnData:
    filename = settings.datasetdir / "ithzak_2016.h5ad"
    url = "https://drive.google.com/uc?export=download&id=1zNSTVmJ-Xms86_WtDnjUROQpPXUEr2Ux"
    return scanpy.read(filename, backup_url=url)


def schlessner_2023() -> AnnData:
    filename = settings.datasetdir / "schlessner_2023.h5ad"
    url = "https://drive.google.com/uc?export=download&id=1JMHWDqLeX3bacvMRQZopg1VJzc0WRvNK"
    return scanpy.read(filename, backup_url=url)
