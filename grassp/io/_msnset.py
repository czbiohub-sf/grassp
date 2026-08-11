"""The exchange contract between grassp :class:`~anndata.AnnData` and pRoloc ``MSnSet``.

The two data models are close enough that almost no translation is needed, and this module
deliberately does almost none. Column names, values and dtypes cross unchanged; what is here
is only what genuinely cannot be carried by h5ad alone.

The contract
------------

===========================================  ===================================================
grassp ``AnnData``                            pRoloc ``MSnSet``
===========================================  ===================================================
``.X`` / ``.layers[layer]``                   ``exprs()`` -- **no transpose**; grassp keeps
                                              proteins in ``.obs``, matching MSnSet's
                                              features-by-fractions orientation.
``.obs_names`` / ``.var_names``               ``featureNames()`` / ``sampleNames()``
``.obs`` scalar columns                       ``fData()`` scalar columns
``.obsm[k]`` + ``uns["obsm_colnames"][k]``    ``fData()[[k]]`` **matrix-valued** column
``.var``                                      ``pData()``
``.varm[k]`` + ``uns["varm_colnames"][k]``    ``pData()[[k]]`` **matrix-valued** column
``.layers[k]``                                extra ``assayData`` elements
``.uns``                                      carried through the h5ad; ``uns["processing"]``
                                              also becomes ``processingData()@processing``
===========================================  ===================================================

The matrix-column rows are the only ones that need any machinery at all. A matrix stored
*inside a single* ``fData`` column is pRoloc's mechanism for per-protein x per-compartment
values -- ``svm.all.scores``, ``tagm.map.joint``, ``bandle.joint``, ``Markers``,
``GOAnnotations`` -- and it maps onto ``.obsm``. ``pData`` is the same ``AnnotatedDataFrame``
class, so ``.varm`` maps onto matrix-valued ``pData`` columns by exactly the same route, and
``.layers`` onto the extra matrices ``assayData`` can already hold. Because ``.obsm``/``.varm``
arrays carry no column names, those travel separately in ``uns["obsm_colnames"]`` and
``uns["varm_colnames"]``.

All three are safe under subsetting, which is what makes them usable rather than merely
storable: ``x[i, j]`` and ``markerMSnSet()`` subset a matrix ``fData`` column with the features,
a matrix ``pData`` column with the samples, and every ``assayData`` element with both --
verified against pRoloc.

The one slot that cannot cross is ``.obsp``/``.varp``: ``eSet`` has no pairwise slot, and
pRoloc has nothing to map one onto. It is listed in ``uns["msnset_dropped"]`` and recomputed
from ``.X`` with one call to :func:`grassp.pp.neighbors`.

What this module deliberately does *not* do
-------------------------------------------

* **No renaming.** pRoloc tolerates non-syntactic ``fData`` and ``pData`` names -- verified by
  running ``svmClassification`` on an ``MSnSet`` whose columns were ``Gene names`` and whose
  fractions were ``Fraction 1``. So names cross verbatim in both directions and there is no
  name map to keep in step.
* **No value translation.** pRoloc encodes unlabelled features as the literal string
  ``"unknown"`` and needs it (``markerMSnSet`` fails outright on ``NA``), but that is pRoloc's
  convention, so the companion R package converts on both boundaries. :func:`nan_to_unknown`
  and :func:`unknown_to_nan` are here for the two places Python has to do it itself: the way
  out of :func:`grassp.io.write_msnset`, and :func:`grassp.io.read_prolocdata`, which parses
  ``.rda`` files with no R involved at all.
* **No dtype coercion.** ``anndataR`` already maps types faithfully in both directions --
  verified by round-tripping a column of each kind: ``float64``/``numeric`` (``NaN`` included),
  ``int64``/``integer``, ``bool``/``logical``, pandas Categorical <-> R ``factor`` with its
  levels and its ordered-ness intact, and nullable ``Int64``/``string``. Nothing here needs to
  help. The single exception is on the R side: ``anndataR`` writes an R *character* ``NA`` as
  the literal string ``"NA"``, so the companion package writes ``NA``-carrying columns as
  factors, whose ``NA`` survives correctly.
* **No list of method names.** Which columns exist and what they mean is pRoloc's business.
"""

from __future__ import annotations
import re
import warnings

from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

#: Version of the exchange contract. Written to ``uns["msnset_spec"]`` and asserted by both
#: sides. Bump the major part for any change an older reader could silently misinterpret.
SPEC_VERSION = "grassp-msnset/1"

#: The literal string pRoloc uses for unlabelled features.
UNKNOWN_LABEL = "unknown"

#: ``uns`` key holding the ``{obsm_key: [column names]}`` mapping.
OBSM_COLNAMES_KEY = "obsm_colnames"

#: The same for ``.varm``, which maps onto matrix-valued ``pData`` columns.
VARM_COLNAMES_KEY = "varm_colnames"

#: ``uns`` keys the contract owns. An exported artifact carries the whole of the source
#: ``.uns``, so these are the names a user's own entries cannot keep.
RESERVED_UNS_KEYS = (
    "msnset_spec",
    "msnset_exprs_layer",
    "msnset_dropped",
    OBSM_COLNAMES_KEY,
    VARM_COLNAMES_KEY,
)

#: Purely advisory notes about columns pRoloc writes under a surprising name, verified in the
#: pRoloc sources. They carry **no behaviour**: such a column is imported under its own name
#: like any other, and the note is attached to the import report so a reader is not left
#: guessing. Adding or removing an entry cannot change what data crosses over.
PROLOC_NOTES: dict[str, str] = {
    "perTurbe.all.scores": (
        "misspelled by pRoloc itself (R/machinelearning-functions-PerTurbo.R:374); "
        "these are perTurbo scores"
    ),
    "svm.all.scores": (
        "ksvmClassification also writes this name "
        "(R/machinelearning-functions-ksvm.R:247), so it may be ksvm rather than svm output"
    ),
    "knn.all.scores": (
        "pRoloc overwrites this matrix's colnames with the single string 'knn.scores' "
        "(R/machinelearning-functions-knn.R:236), so per-class names are unrecoverable"
    ),
}


class SpecVersionError(ValueError):
    """Raised when an artifact's ``msnset_spec`` is incompatible with this module."""


# ---------------------------------------------------------------------------
# The "unknown" sentinel
# ---------------------------------------------------------------------------


def is_labelish(values: pd.Series) -> bool:
    """Whether a column can hold a text sentinel at all.

    Only about dtype, never about content: strings and Categoricals can, numbers and booleans
    cannot. Putting ``"unknown"`` in a float column would corrupt it.

    Examples
    --------
    >>> is_labelish(pd.Series(["Golgi", None]))
    True
    >>> is_labelish(pd.Series([1.0, np.nan]))
    False
    """
    return (
        isinstance(values.dtype, pd.CategoricalDtype)
        or pd.api.types.is_object_dtype(values)
        or pd.api.types.is_string_dtype(values)
    )


def nan_to_unknown(values: pd.Series, unknown_label: str = UNKNOWN_LABEL) -> pd.Series:
    """Replace ``NaN`` with pRoloc's ``"unknown"`` sentinel in one text column.

    pRoloc encodes unlabelled features as the literal string ``"unknown"`` and needs it:
    ``markerMSnSet`` and ``unknownMSnSet`` fail outright on ``NA``. grassp uses ``NaN``.

    Handles the three dtypes a label column can have. Categoricals need ``"unknown"`` added as
    a category before it can be assigned, which is easy to get wrong by hand; numeric and
    boolean columns are returned untouched.

    Examples
    --------
    >>> nan_to_unknown(pd.Series(["Golgi", None])).tolist()
    ['Golgi', 'unknown']
    >>> nan_to_unknown(pd.Series([1.0, np.nan])).tolist()   # numbers are left alone
    [1.0, nan]
    """
    if not is_labelish(values):
        return values
    if isinstance(values.dtype, pd.CategoricalDtype):
        out = values.copy()
        if unknown_label not in out.cat.categories:
            out = out.cat.add_categories([unknown_label])
        return out.fillna(unknown_label)
    return values.astype(object).fillna(unknown_label)


def unknown_to_nan(values: pd.Series, unknown_label: str = UNKNOWN_LABEL) -> pd.Series:
    """Replace pRoloc's ``"unknown"`` sentinel with ``NaN`` in one column.

    The inverse of :func:`nan_to_unknown`, needed only by :func:`grassp.io.read_prolocdata`,
    which parses ``.rda`` files directly with no R involved. On the h5ad path the companion R
    package converts on the way out.

    Without it, every grassp annotator -- which selects markers with ``.notna()`` -- treats
    ``"unknown"`` as a real compartment and happily trains on it.

    For a Categorical the category is *removed*, not merely blanked. ``rdata`` maps an R factor
    to a Categorical, so this is the common case for a pRolocdata marker column, and leaving
    ``"unknown"`` behind as an unused category is not cosmetic: anything that iterates
    ``.cat.categories`` -- :func:`grassp.pp.set_sensible_compartment_colors`, scanpy's legends --
    would show a phantom compartment with no members.

    Examples
    --------
    >>> out = unknown_to_nan(pd.Series(pd.Categorical(["Golgi", "unknown"])))
    >>> list(out.cat.categories)
    ['Golgi']
    """
    if isinstance(values.dtype, pd.CategoricalDtype):
        if unknown_label not in values.cat.categories:
            return values.copy()
        # remove_categories blanks the affected rows to NaN as it goes
        return values.cat.remove_categories([unknown_label])
    out = pd.Series(values, copy=True)
    return out.mask(out.astype(object).astype(str) == str(unknown_label))


# ---------------------------------------------------------------------------
# Sanity checks on an incoming artifact
# ---------------------------------------------------------------------------

_PC_NAME = re.compile(r"^PC\d+$")


def looks_remapped(var_names: Sequence[str]) -> bool:
    """Whether an artifact's fractions look like principal components rather than fractions.

    ``pRoloc::remap`` projects an ``MSnSetList`` into a shared PCA space, *replacing*
    ``exprs()`` with the component scores and renaming ``sampleNames`` to ``PC1``..``PCn``. An
    object that has been through it no longer holds fractionation profiles, which is easy to
    miss because nothing else about its shape changes.

    Examples
    --------
    >>> looks_remapped(["PC1", "PC2", "PC3"])
    True
    >>> looks_remapped(["Fraction.1", "Fraction.2"])
    False
    """
    names = [str(v) for v in var_names]
    if not names:
        return False
    return all(_PC_NAME.match(n) for n in names)


def notes_for(names: Iterable[str]) -> dict[str, str]:
    """Advisory notes for any of ``names`` that pRoloc is known to misname."""
    return {str(name): PROLOC_NOTES[str(name)] for name in names if str(name) in PROLOC_NOTES}


# ---------------------------------------------------------------------------
# Class names for matrix-valued columns
# ---------------------------------------------------------------------------


def _as_names(value) -> list[str]:
    """Coerce whatever an h5ad reader hands back for a name list into ``list[str]``."""
    return [str(name) for name in np.atleast_1d(value).ravel().tolist()]


def declared_colnames(value) -> dict[str, list[str]]:
    """Parse a ``uns["obsm_colnames"]``-shaped mapping, tolerating numpy arrays as values.

    Examples
    --------
    >>> declared_colnames({"k": np.array(["ER", "Golgi"], dtype=object)})
    {'k': ['ER', 'Golgi']}
    >>> declared_colnames(None)
    {}
    """
    return {str(key): _as_names(names) for key, names in dict(value or {}).items()}


def matrix_colnames(
    uns: Mapping, matrices: Mapping, *, uns_key: str = OBSM_COLNAMES_KEY
) -> dict[str, list[str]]:
    """Recover the class names belonging to each matrix of an incoming artifact.

    ``uns[uns_key]`` is the contract and always wins. A matrix that arrived as a DataFrame
    carries its own column names and can fall back on them; a bare array cannot, which is
    exactly why the ``uns`` key exists.

    Examples
    --------
    >>> matrix_colnames({"obsm_colnames": {"svm.all.scores": ["ER", "Golgi"]}}, {})
    {'svm.all.scores': ['ER', 'Golgi']}
    """
    names = declared_colnames(uns.get(uns_key))
    for key, value in matrices.items():
        if str(key) not in names and isinstance(value, pd.DataFrame):
            names[str(key)] = [str(column) for column in value.columns]
    return names


# ---------------------------------------------------------------------------
# The spec block
# ---------------------------------------------------------------------------


def spec_major(spec: str) -> int:
    """Extract the major version from a ``"grassp-msnset/N"`` string."""
    match = re.match(r"^grassp-msnset/(\d+)$", str(spec))
    if match is None:
        raise SpecVersionError(
            f"Unrecognised msnset_spec {spec!r}; expected 'grassp-msnset/<major>'."
        )
    return int(match.group(1))


def check_spec(spec: str | None, *, strict: bool = True) -> int | None:
    """Validate an artifact's spec version against this module.

    ``spec`` is the value of ``uns["msnset_spec"]``, or ``None`` for an artifact that has none
    -- a plain h5ad from elsewhere, which is still readable. A newer major version raises when
    ``strict``, warns otherwise.

    Returns the artifact's major version, or ``None`` when ``spec`` is ``None``.

    Raises
    ------
    SpecVersionError
        If the artifact was written by a newer, incompatible version of the contract.
    """
    if spec is None:
        return None
    found = spec_major(spec)
    ours = spec_major(SPEC_VERSION)
    if found > ours:
        message = (
            f"This artifact was written with {spec!r} but this grassp understands up to "
            f"{SPEC_VERSION!r}. Upgrade grassp (`pip install -U grassp`) to read it "
            "reliably."
        )
        if strict:
            raise SpecVersionError(message)
        warnings.warn(message, stacklevel=2)
    return found


def build_spec_block(
    *,
    layer: str | None,
    obsm_colnames: dict[str, Sequence[str]] | None = None,
    varm_colnames: dict[str, Sequence[str]] | None = None,
    dropped: Iterable[str] = (),
) -> dict[str, object]:
    """Assemble the self-describing ``uns`` block that accompanies an exported artifact.

    Note what is **not** here: no nomination of a marker column. pRoloc's ``fcol`` is a
    per-call argument, not a property of the object -- an ``MSnSet`` can hold any number of
    marker columns at once (``markers``, ``markers.orig``, ``pd.markers``, ``pd.2013``, ...)
    and different functions can be pointed at different ones, exactly as in AnnData. Recording
    a single "the" marker column would impose a restriction neither framework has.
    """
    block: dict[str, object] = {
        "msnset_spec": SPEC_VERSION,
        "msnset_exprs_layer": "" if layer is None else str(layer),
        "msnset_dropped": sorted(str(d) for d in dropped),
    }
    for key, mapping in (
        (OBSM_COLNAMES_KEY, obsm_colnames),
        (VARM_COLNAMES_KEY, varm_colnames),
    ):
        if mapping:
            block[key] = {str(k): [str(c) for c in v] for k, v in mapping.items()}
    return block


def read_spec_block(uns: Mapping, *, strict: bool = True) -> dict[str, object]:
    """Extract and validate the spec block from an artifact's ``.uns``.

    Missing pieces come back as ``None`` / empty, so a plain h5ad with no spec block at all is
    still readable.
    """
    spec = uns.get("msnset_spec")
    spec = None if spec is None else str(spec)
    major = check_spec(spec, strict=strict)

    layer = uns.get("msnset_exprs_layer")
    layer = None if layer in (None, "") else str(layer)

    dropped_raw = uns.get("msnset_dropped")
    dropped = [] if dropped_raw is None else _as_names(dropped_raw)

    return {
        "spec": spec,
        "major": major,
        "layer": layer,
        OBSM_COLNAMES_KEY: declared_colnames(uns.get(OBSM_COLNAMES_KEY)),
        VARM_COLNAMES_KEY: declared_colnames(uns.get(VARM_COLNAMES_KEY)),
        "dropped": dropped,
    }
