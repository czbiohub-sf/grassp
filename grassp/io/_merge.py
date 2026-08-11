"""Row alignment for merging a pRoloc artifact onto an existing :class:`~anndata.AnnData`.

Everything here serves :func:`grassp.io.read_proloc_results` -- the *convenience* path, for
keeping an in-memory session going. The primary path is a round trip:
:func:`grassp.io.write_msnset` out, :func:`grassp.io.read_msnset` back, which rebuilds the
object and needs none of this.

The reason it exists at all is that pRoloc workflows shrink objects routinely -- ``filterNA``,
``markerMSnSet``, ``unknownMSnSet``, ``filterZeroRows``, plain ``x[i, j]`` -- so an artifact
frequently covers only some of the proteins it is being merged onto. A silent partial merge is
the failure mode most likely to be mistaken for a scientific result, hence the reporting.
"""

from __future__ import annotations
import warnings

from typing import Literal

import numpy as np
import pandas as pd

#: How to react to a mismatch between an artifact and the object being merged into.
OnMismatch = Literal["error", "warn", "ignore"]


def align_rows(
    source_index: pd.Index,
    target_names: pd.Index,
    on_missing: OnMismatch,
    on_extra: OnMismatch,
    source: str,
) -> int:
    """Report on the overlap between an artifact and its merge target.

    Returns the number of target rows the artifact covers, and raises or warns per
    ``on_missing`` / ``on_extra`` with counts and example IDs.
    """
    target = pd.Index(target_names.map(str))
    source_index = pd.Index(source_index.map(str))

    if not source_index.is_unique:
        duplicated = source_index[source_index.duplicated()].unique().tolist()
        raise ValueError(
            f"Artifact {source} has duplicated protein IDs, so it cannot be merged: "
            f"{duplicated[:5]}{' ...' if len(duplicated) > 5 else ''}. "
            "R rownames must be unique; check the id_column argument."
        )

    missing = target.difference(source_index)
    extra = source_index.difference(target)

    _report(
        on_missing,
        missing.tolist(),
        f"{len(missing)} of {len(target)} proteins in the target object are absent from "
        f"{source}; their new columns will be NaN",
    )
    _report(
        on_extra,
        extra.tolist(),
        f"{len(extra)} proteins in {source} are absent from the target object and were "
        "dropped",
    )
    return len(target) - len(missing)


def _report(mode: OnMismatch, examples: list[str], message: str) -> None:
    """Raise, warn, or stay silent about a row mismatch."""
    if not examples or mode == "ignore":
        return
    shown = ", ".join(map(str, examples[:5]))
    full = f"{message}. Examples: {shown}{' ...' if len(examples) > 5 else ''}"
    if mode == "error":
        raise ValueError(full)
    if mode == "warn":
        warnings.warn(full, stacklevel=3)
        return
    raise ValueError(f"Unknown mismatch mode {mode!r}; expected error, warn, or ignore.")


def reindex_column(
    values: pd.Series, source_index: pd.Index, target_names: pd.Index
) -> pd.Series:
    """Align a column onto the target's ``obs_names``, filling absentees with ``NaN``.

    Reindexing a *Series* preserves its dtype, including Categorical (whose unmatched rows
    become ``NaN`` without disturbing the category set). Going via ``.to_numpy()`` instead
    would silently degrade a Categorical to object, which then breaks anything reaching for
    ``.cat`` -- ``gr.tl.svm_annotation`` among them.
    """
    series = pd.Series(values.to_numpy(), index=pd.Index(source_index.map(str)))
    if isinstance(values.dtype, pd.CategoricalDtype):
        series = series.astype(values.dtype)
    return series.reindex(pd.Index(target_names.map(str)))


def reindex_matrix(
    matrix: np.ndarray, source_index: pd.Index, target_names: pd.Index
) -> np.ndarray:
    """Align an ``.obsm`` matrix onto the target, filling absent rows with zeros.

    Zero rather than ``NaN`` because these are probability matrices, and downstream row-sum
    operations should treat an unmeasured protein as carrying no evidence rather than
    poisoning the whole row. Note that a naive ``argmax`` would still return class 0 for such
    a row -- the label column, which is ``NaN``, is the safe thing to read.
    """
    frame = pd.DataFrame(np.asarray(matrix), index=pd.Index(source_index.map(str)))
    aligned = frame.reindex(pd.Index(target_names.map(str)))
    return aligned.fillna(0.0).to_numpy(dtype=float)
