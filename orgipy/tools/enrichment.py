from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

import scanpy

rank_proteins_groups = scanpy.tl.rank_genes_groups
