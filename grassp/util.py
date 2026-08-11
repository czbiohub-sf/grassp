"""Small helpers shared across the package.

Orientation invariant
---------------------
Throughout grassp an :class:`~anndata.AnnData` holds **proteins in ``.obs`` (rows) and
samples/fractions in ``.var`` (columns)**. This is the transpose of scanpy's cells-by-genes
convention, and it is not optional: the readers establish it on the way in
(:func:`grassp.io.read_maxquant`, :func:`~grassp.io.read_fragpipe` and
:func:`~grassp.io.read_diann` transpose what ``protdata`` returns, and
:func:`~grassp.io.read_prolocdata` and :func:`~grassp.io.read_msnset` build it directly),
and every tool, plot and IO function assumes it. Where a step genuinely needs the other
orientation it transposes locally, as :func:`grassp.pp.normalize_total` does.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from anndata import AnnData


def layer_names(data: AnnData) -> list[str]:
    """Names of the real layers in ``data``, excluding the main matrix.

    anndata >= 0.13 backs ``.X`` with ``layers[None]``, so iterating over
    ``data.layers`` also yields the main matrix under a ``None`` key. Code that
    copies layers into a freshly built :class:`~anndata.AnnData` must skip it,
    otherwise ``X=`` and ``layers[None]`` are both supplied and anndata rejects
    the pair as inconsistent.
    """
    return [name for name in data.layers.keys() if name is not None]
