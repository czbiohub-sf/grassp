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
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal, Sequence

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from anndata import AnnData

#: Which axis a labelled matrix is aligned to. ``"obs"`` is ``.obsm``, ``"var"`` is ``.varm``.
Axis = Literal["obs", "var"]

#: The slots :func:`diff_anndata` walks, in the order :meth:`anndata.AnnData.__repr__` prints them.
_DIFFABLE_SLOTS = ("obs", "var", "uns", "obsm", "varm", "layers", "obsp", "varp")


def layer_names(data: AnnData) -> list[str]:
    """Names of the real layers in ``data``, excluding the main matrix.

    anndata >= 0.13 backs ``.X`` with ``layers[None]``, so iterating over
    ``data.layers`` also yields the main matrix under a ``None`` key. Code that
    copies layers into a freshly built :class:`~anndata.AnnData` must skip it,
    otherwise ``X=`` and ``layers[None]`` are both supplied and anndata rejects
    the pair as inconsistent.
    """
    return [name for name in data.layers.keys() if name is not None]


def _slot_keys(data: AnnData, slot: str) -> list[str]:
    """The keys of one slot, as strings, with the main matrix left out of ``layers``."""
    if slot == "layers":
        return [str(name) for name in layer_names(data)]
    if slot == "uns":
        return list(_flatten_uns(data.uns))
    elem = getattr(data, slot)
    keys = elem.columns if slot in ("obs", "var") else elem.keys()
    return [str(key) for key in keys]


def _flatten_uns(mapping: Mapping, prefix: str = "") -> dict[str, Any]:
    """``.uns`` as a flat ``{dotted.path: leaf}`` mapping.

    ``.uns`` is the one slot that nests, and nesting is where its interesting changes happen: a
    round trip through another framework leaves the top-level keys alone while turning a scalar
    two levels down into a one-element array. Reporting only ``neighbors`` would say nothing,
    so the leaves are addressed by path instead.
    """
    flat: dict[str, Any] = {}
    for key, value in mapping.items():
        path = f"{prefix}{key}"
        if isinstance(value, Mapping) and len(value) > 0:
            flat.update(_flatten_uns(value, f"{path}."))
        else:
            flat[path] = value
    return flat


def _describe(value: Any) -> str:
    """A short, comparable description of how a value is stored.

    Type plus whatever else changes silently: the dtype and shape of an array, the length of a
    list. Round trips through other frameworks routinely turn a Python ``int`` into a
    one-element ``int64`` array, or a list into an ndarray, and neither shows up in the type name
    alone.
    """
    if isinstance(value, np.ndarray):
        return f"ndarray[{value.dtype}]{list(value.shape)}"
    if isinstance(value, pd.DataFrame):
        return f"DataFrame{list(value.shape)}"
    if isinstance(value, pd.Series):
        return f"Series[{value.dtype}]{list(value.shape)}"
    if isinstance(value, (list, tuple, Mapping)):
        return f"{type(value).__name__}[{len(value)}]"
    return type(value).__name__


def diff_anndata(a: AnnData, b: AnnData, *, check_dtypes: bool = True) -> pd.DataFrame:
    """What differs *structurally* between two :class:`~anndata.AnnData` objects.

    Walks the same slots :meth:`anndata.AnnData.__repr__` prints and reports, per key, whether it
    was added, removed, or kept but stored differently. Useful for answering "what did that step
    actually do to my object", and for checking what survived a round trip through another
    framework.

    What counts as *stored differently* is whatever changes silently in that slot:

    * ``.obs`` / ``.var`` -- the column's dtype, so a Categorical degraded to ``object`` shows up.
    * ``.obsm`` / ``.varm`` -- the container, so an array that came back as a DataFrame (and
      therefore now names its own columns) shows up.
    * ``.uns`` -- the type, dtype, shape or length of each **leaf**, addressed by dotted path.
      This is the slot that nests, and nesting is where its drift lives: a round trip can leave
      every top-level key in place while turning a scalar two levels down into a one-element
      array, which reads back as ``neighbors.params.n_neighbors: int -> ndarray[int64][1]``.

    Values themselves are **not** compared -- the matrices can be dense, sparse or dask, and two
    objects can hold the same keys with different numbers. Use
    ``anndata.tests.helpers.assert_equal`` when you need that.

    Parameters
    ----------
    a, b
        The before and after objects. Differences are reported as changes from ``a`` to ``b``.
    check_dtypes
        Whether to report the ``"changed"`` rows at all -- the three bullets above, all of which
        are about *how* a key that exists in both is stored. ``False`` leaves only what was added
        and removed, which is the question to ask when the two objects are not expected to store
        things the same way in the first place. The shape row is unaffected.

    Returns
    -------
    One row per difference, with columns ``change`` (``"added"``, ``"removed"`` or
    ``"changed"``), ``slot``, ``key`` and ``detail``. Empty when the two are structurally
    identical, so ``diff_anndata(a, b).empty`` is the question "did anything move".

    Examples
    --------
    >>> import anndata, numpy as np, pandas as pd
    >>> obs = pd.DataFrame({"markers": ["ER", "Golgi"]}, index=["P1", "P2"])
    >>> before = anndata.AnnData(np.zeros((2, 3)), obs=obs)
    >>> before.uns["neighbors"] = {"params": {"n_neighbors": 15}}
    >>> after = before.copy()
    >>> after.obs["svm"] = pd.Categorical(["ER", "ER"])
    >>> del after.obs["markers"]
    >>> after.uns["neighbors"]["params"]["n_neighbors"] = np.array([15])
    >>> diff_anndata(before, after)  # doctest: +NORMALIZE_WHITESPACE
        change slot                           key                   detail
    0    added  obs                           svm
    1  removed  obs                       markers
    2  changed  uns  neighbors.params.n_neighbors  int -> ndarray[int64][1]

    >>> diff_anndata(before, after, check_dtypes=False)  # doctest: +NORMALIZE_WHITESPACE
        change slot      key detail
    0    added  obs      svm
    1  removed  obs  markers
    """
    rows: list[dict[str, str]] = []

    if a.shape != b.shape:
        rows.append(
            {
                "change": "changed",
                "slot": "shape",
                "key": "",
                "detail": f"{a.shape} -> {b.shape}",
            }
        )

    for slot in _DIFFABLE_SLOTS:
        keys_a, keys_b = _slot_keys(a, slot), _slot_keys(b, slot)
        rows.extend(
            {"change": "added", "slot": slot, "key": key, "detail": ""}
            for key in keys_b
            if key not in keys_a
        )
        rows.extend(
            {"change": "removed", "slot": slot, "key": key, "detail": ""}
            for key in keys_a
            if key not in keys_b
        )

        if not check_dtypes:
            continue

        if slot == "uns":
            flat_a, flat_b = _flatten_uns(a.uns), _flatten_uns(b.uns)

        shared = [key for key in keys_a if key in keys_b]
        for key in shared:
            if slot in ("obs", "var"):
                before, after = getattr(a, slot)[key].dtype, getattr(b, slot)[key].dtype
            elif slot in ("obsm", "varm"):
                # An ndarray that came back as a DataFrame has gained its column names, which is
                # a real change in what the object tells you about itself.
                before, after = type(getattr(a, slot)[key]), type(getattr(b, slot)[key])
                before, after = before.__name__, after.__name__
            elif slot == "uns":
                before, after = _describe(flat_a[key]), _describe(flat_b[key])
            else:
                continue
            if str(before) != str(after):
                rows.append(
                    {
                        "change": "changed",
                        "slot": slot,
                        "key": key,
                        "detail": f"{before} -> {after}",
                    }
                )

    return pd.DataFrame(rows, columns=["change", "slot", "key", "detail"])


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
