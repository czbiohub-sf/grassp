"""Small helpers shared across the package.

Orientation invariant
---------------------
Throughout grassp an :class:`~anndata.AnnData` holds **proteins in ``.obs`` (rows) and
samples/fractions in ``.var`` (columns)**. This is the transpose of scanpy's cells-by-genes
convention, and it is not optional: the readers establish it on the way in
(:func:`grassp.io.read_maxquant`, :func:`~grassp.io.read_fragpipe` and
:func:`~grassp.io.read_diann` transpose what ``protdata`` returns, and
:func:`~grassp.io.read_prolocdata` builds it directly), and every tool, plot and IO function
assumes it. Where a step genuinely needs the other
orientation it transposes locally, as :func:`grassp.pp.normalize_total` does.

Labelled matrices
-----------------
Tools that score every protein against every compartment write an ``(n_obs, n_classes)``
matrix to ``.obsm``. A bare ndarray carries no column names, so the class each column stands
for has to travel separately and be reapplied positionally -- a mismatch there produces
wrong-but-plausible output rather than an error. :func:`set_matrix` therefore stores these as
:class:`pandas.DataFrame` entries, which AnnData supports natively, making the mapping
intrinsic to the stored object. :func:`get_matrix` reads them back and still accepts the bare
ndarrays written by older versions.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Any, Literal, Sequence

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from anndata import AnnData

#: Which axis a labelled matrix is aligned to. ``"obs"`` is ``.obsm``, ``"var"`` is ``.varm``.
Axis = Literal["obs", "var"]


def layer_names(data: AnnData) -> list[str]:
    """Names of the real layers in ``data``, excluding the main matrix.

    anndata >= 0.13 backs ``.X`` with ``layers[None]``, so iterating over
    ``data.layers`` also yields the main matrix under a ``None`` key. Code that
    copies layers into a freshly built :class:`~anndata.AnnData` must skip it,
    otherwise ``X=`` and ``layers[None]`` are both supplied and anndata rejects
    the pair as inconsistent.
    """
    return [name for name in data.layers.keys() if name is not None]


def set_matrix(
    data: AnnData,
    key: str,
    values: np.ndarray | pd.DataFrame,
    columns: Sequence[Any],
    *,
    axis: Axis = "obs",
) -> None:
    """Store a labelled matrix in ``.obsm`` (or ``.varm``) as a column-named DataFrame.

    Parameters
    ----------
    data
        Object to write into.
    key
        Key in ``.obsm``/``.varm``.
    values
        ``(n_obs, len(columns))`` matrix, already in ``data``'s row order. A DataFrame is
        accepted and taken positionally -- its own index and columns are discarded, because
        the caller has already aligned it.
    columns
        One label per column, in column order. Coerced to :class:`str`: on the way to h5ad
        these become HDF5 dataset names, which cannot be anything else.
    axis
        ``"obs"`` writes ``.obsm``, ``"var"`` writes ``.varm``.

    Raises
    ------
    ValueError
        If two labels collide after the string coercion. AnnData accepts duplicate columns
        on assignment and only rejects them when the object is written, so an unguarded
        collision surfaces as a failed ``write_h5ad`` a long way from the call that caused
        it. Failing here names the key and the offending labels instead.
    """
    labels = [str(c) for c in columns]
    duplicates = sorted({name for name in labels if labels.count(name) > 1})
    if duplicates:
        raise ValueError(
            f"Duplicate column labels for {axis}m[{key!r}]: {duplicates}. "
            "Column labels must be unique."
        )
    index = data.obs_names if axis == "obs" else data.var_names
    frame = pd.DataFrame(np.asarray(values), index=index, columns=labels)
    mapping = data.obsm if axis == "obs" else data.varm
    mapping[key] = frame


def get_matrix(
    data: AnnData, key: str, *, axis: Axis = "obs"
) -> tuple[np.ndarray, list[str] | None]:
    """Read a labelled matrix back as ``(values, columns)``.

    Accepts both what :func:`set_matrix` writes and the bare ndarrays written by older
    versions of grassp, so objects saved before the switch keep working.

    Parameters
    ----------
    data
        Object to read from.
    key
        Key in ``.obsm``/``.varm``.
    axis
        ``"obs"`` reads ``.obsm``, ``"var"`` reads ``.varm``.

    Returns
    -------
    values
        The matrix as an ndarray. Always an ndarray, so callers can use numpy reductions --
        ``DataFrame.sum(axis=1)`` returns a Series and ``keepdims`` is unsupported, which
        silently changes the meaning of arithmetic written against arrays.
    columns
        Column labels, or ``None`` when the stored entry is a bare ndarray and therefore
        carries none.
    """
    mapping = data.obsm if axis == "obs" else data.varm
    stored = mapping[key]
    if isinstance(stored, pd.DataFrame):
        return stored.to_numpy(), [str(c) for c in stored.columns]
    return np.asarray(stored), None
