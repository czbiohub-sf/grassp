"""Read and write pRoloc ``MSnSet`` objects as h5ad, for round-tripping with R.

grassp and `pRoloc <https://bioconductor.org/packages/pRoloc/>`_ are the Python and R
frameworks for the same science, and their data models are nearly isomorphic -- but there is
no format they share. This module implements the grassp half of a **symmetric h5ad round
trip**; the R half is the companion package ``grasspio``, installed with::

    remotes::install_github("czbiohub-sf/grassp", subdir = "r/grasspio")

The workflow, with grassp doing the preprocessing and pRoloc the classification::

    >>> import grassp as gr
    >>> adata = gr.ds.load_dataset("hek_dc_2025")           # doctest: +SKIP
    >>> gr.pp.add_markers(adata, species="hsap")            # doctest: +SKIP
    >>> gr.io.write_msnset(adata, "experiment.h5ad")        # doctest: +SKIP

then in R::

    library(grasspio)
    x <- grassp_as_msnset("experiment.h5ad")
    x <- svmClassification(x, fcol = "markers", scores = "all")
    x <- tagmMapPredict(x, params = tagmMapTrain(x), probJoint = TRUE)
    grassp_write_msnset(x, "results.h5ad")

and back in Python::

    >>> annotated = gr.io.read_msnset("results.h5ad")       # doctest: +SKIP

That last step is a **round trip**, not an import of a foreign format: ``annotated`` should be
as close to ``adata`` as the two data models physically allow, which is everything except the
neighbour graph. :func:`read_proloc_results` is a convenience for the other case -- when you
still have the object in memory and would rather graft the new columns onto it than rebuild it.

The contract lives in :mod:`grassp.io._msnset`, and is deliberately almost empty: column
names, values and dtypes cross unchanged in both directions. The one thing h5ad cannot carry by
itself is the class names belonging to an ``.obsm`` matrix -- pRoloc stores per-protein x
per-compartment values *inside a single* ``fData`` column (``svm.all.scores``,
``tagm.map.joint``, ``bandle.joint``, ``Markers``), which maps onto ``.obsm``, whose arrays have
no column names -- so those travel in ``uns["obsm_colnames"]``.
"""

from __future__ import annotations
import warnings

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Sequence

if TYPE_CHECKING:
    from anndata import AnnData

import numpy as np
import pandas as pd

from ..util import set_matrix
from . import _merge, _msnset
from ._merge import OnMismatch  # noqa: F401  (re-exported; part of the public signatures)
from ._msnset import SpecVersionError  # noqa: F401  (re-exported for callers catching it)
from ._msnset import OBSM_COLNAMES_KEY, VARM_COLNAMES_KEY

#: Which axis a matrix-valued annotation belongs to. ``"obs"`` is ``.obsm`` <-> matrix ``fData``
#: columns, ``"var"`` is ``.varm`` <-> matrix ``pData`` columns; ``pData`` and ``fData`` are the
#: same ``AnnotatedDataFrame`` class, so the two behave identically.
Axis = Literal["obs", "var"]


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


def _source_label(path: str | Path | AnnData) -> str:
    """A human-readable name for an artifact, for warnings and provenance."""
    import anndata

    return "<AnnData>" if isinstance(path, anndata.AnnData) else str(Path(path))


def read_msnset(
    path: str | Path | AnnData,
    *,
    set_colors: bool = True,
    strict_spec: bool = True,
) -> AnnData:
    """Read a pRoloc artifact back into grassp.

    This is the return leg of the round trip: :func:`write_msnset` out, ``grasspio``'s
    ``grassp_write_msnset`` in R, then this. The artifact is already grassp-shaped, so the work
    is a contract-version check and two sanity warnings -- proteins are in ``.obs``, fractions
    in ``.var``, matrix-valued ``fData``/``pData`` columns in ``.obsm``/``.varm`` with their
    class names in ``uns["obsm_colnames"]``/``uns["varm_colnames"]``, extra ``assayData``
    elements in ``.layers``. Nothing is renamed or converted.

    Everything crosses except ``.obsp``/``.varp``: ``eSet`` has no pairwise slot. Recompute the
    graph with :func:`grassp.pp.neighbors`, which is where it came from anyway.

    Parameters
    ----------
    path
        Path to an h5ad written by ``grasspio``'s ``grassp_write_msnset`` (or by
        :func:`write_msnset`). An in-memory :class:`~anndata.AnnData` is passed through as-is.
    set_colors
        Assign compartment colours via
        :func:`grassp.preprocessing.set_sensible_compartment_colors`.
    strict_spec
        Whether an artifact written by a newer contract version raises rather than warns.

    Returns
    -------
    The artifact as an :class:`~anndata.AnnData`.

    See Also
    --------
    read_proloc_results : Graft an artifact's annotations onto an object you already have.
    list_msnset_results : Inventory an artifact without reading it into a session.

    Examples
    --------
    >>> adata = gr.io.read_msnset("results.h5ad")       # doctest: +SKIP
    >>> adata.obs["markers"].isna().sum()               # unlabelled proteins   doctest: +SKIP
    """
    artifact, source = _read_artifact(path)
    _msnset.check_spec(
        None if "msnset_spec" not in artifact.uns else str(artifact.uns["msnset_spec"]),
        strict=strict_spec,
    )
    if _msnset.looks_remapped(artifact.var_names):
        warnings.warn(
            f"The fractions in {source} are named PC1..PCn, which means this object went "
            "through pRoloc::remap(): its expression matrix holds principal-component "
            "scores, not fractionation profiles. The feature annotations are still valid, "
            "but do not treat the matrix as profiles.",
            stacklevel=2,
        )
    if set_colors:
        from ..preprocessing.annotation import set_sensible_compartment_colors

        set_sensible_compartment_colors(artifact)
    return artifact


def read_proloc_results(
    path: str | Path | AnnData,
    data: AnnData,
    *,
    key_prefix: str = "",
    suffix: str = "",
    on_missing: OnMismatch = "warn",
    on_extra: OnMismatch = "warn",
    set_colors: bool = True,
    id_column: str | None = None,
    strict_spec: bool = True,
    copy: bool = False,
) -> AnnData | None:
    """Graft a pRoloc artifact's annotations onto an :class:`~anndata.AnnData` you already have.

    A convenience wrapper over :func:`read_msnset`, for keeping an in-memory session going:
    prefer the plain round trip (``read_msnset``) unless you specifically want to keep an
    object you have been working on. It **merges** rather than replaces -- ``.X``, ``.layers``,
    ``.obsp``, ``.varp``, ``.var`` and ``.varm`` are never touched, and rows are never reordered
    or dropped -- which is also its limitation: matrices the R side altered do not come back.

    What it does write, it writes under the artifact's own names, so a same-named ``.obs``
    column or ``.obsm`` entry **is** replaced. Since an export carries every ``.obsm`` entry by
    default, that routinely includes your embeddings, with values that made the trip unchanged.
    Pass ``key_prefix`` or ``suffix`` to keep the two apart.

    Beyond aligning rows on ``.obs_names``, it changes nothing. Every ``.obs`` column is copied
    under its own name with its own dtype, and every ``.obsm`` matrix likewise. There is no
    notion of which method produced a column, so ``svm``, ``knn``, ``rf``, ``plsda``, ``nnet``,
    ``perTurbo``, ``ksvm``, ``phenoDisco``, TAGM, BANDLE and anything added to pRoloc later all
    work identically -- and what you see in Python is what ``fvarLabels(x)`` showed in R.

    Parameters
    ----------
    path
        Path to an h5ad written by ``grasspio``'s ``grassp_write_msnset``, or an in-memory
        :class:`~anndata.AnnData`.
    data
        The object to merge onto. Matched on ``.obs_names``.
    key_prefix, suffix
        Affixed to every key written, e.g. ``suffix="_control"`` to keep two conditions apart.
    on_missing, on_extra
        What to do when the artifact lacks proteins present in ``data`` (``on_missing``) or
        contains proteins absent from it (``on_extra``). One of ``"error"``, ``"warn"``
        (default) or ``"ignore"``. Missing proteins get ``NaN`` / all-zero ``.obsm`` rows.
    set_colors
        Assign compartment colours via
        :func:`grassp.preprocessing.set_sensible_compartment_colors`, which decides for itself
        which of the new columns look like compartment annotations.
    id_column
        ``.obs`` column in the artifact holding the protein IDs, if the R side renamed
        features. Defaults to using the artifact's index.
    strict_spec
        Whether an artifact written by a newer contract version raises rather than warns.
    copy
        Return a modified copy instead of writing into ``data``.

    Returns
    -------
    ``None`` when ``copy`` is ``False`` (``data`` is modified in place), otherwise the
    modified copy. It writes:

    - ``.obs[<every column>]`` and ``.obsm[<every matrix>]`` -- under pRoloc's names.
    - ``.uns["obsm_colnames"][...]`` -- class names for every matrix. This is the only place
      they can live, because ``.obsm`` arrays carry no column names.
    - ``.uns["proloc_import"]`` -- provenance: source, contract version, what was copied,
      match counts, and any advisory notes about columns pRoloc is known to misname.

    Notes
    -----
    ``"unknown"`` has already become ``NaN`` by the time results arrive: the companion R
    package converts on the way out, because that sentinel is pRoloc's convention and belongs
    on pRoloc's side. If you are reading a ``.rda`` instead, :func:`read_prolocdata` does the
    same thing itself, since no R is involved there.

    Names are preserved rather than translated, so pass pRoloc's own score column where a
    grassp plotting helper wants a probability:
    ``gr.pl.umap_prob(adata, color="svm.pred", color_prob="svm.scores")``.

    Examples
    --------
    >>> gr.io.read_proloc_results("results.h5ad", adata)            # doctest: +SKIP
    >>> adata.obs["tagm.map.allocation"].value_counts()             # doctest: +SKIP
    >>> adata.obsm["svm.all.scores"].shape                          # doctest: +SKIP
    >>> adata.uns["obsm_colnames"]["svm.all.scores"]                # doctest: +SKIP
    """
    target = data.copy() if copy else data
    artifact = read_msnset(path, set_colors=False, strict_spec=strict_spec)
    source = _source_label(path)

    obs = artifact.obs.copy()
    if id_column is not None:
        if id_column not in obs.columns:
            raise KeyError(
                f"id_column '{id_column}' is not in the artifact. Available columns: "
                f"{sorted(obs.columns)[:20]}"
            )
        obs = obs.set_index(obs[id_column].astype(str))
    obs.index = obs.index.map(str)

    matrices = {str(k): _as_array(v) for k, v in artifact.obsm.items()}
    colnames = _msnset.matrix_colnames(artifact.uns, artifact.obsm)
    n_matched = _merge.align_rows(obs.index, target.obs_names, on_missing, on_extra, source)

    def _key(name: str) -> str:
        return f"{key_prefix}{name}{suffix}"

    written: list[str] = []
    for column in obs.columns:
        aligned = _merge.reindex_column(obs[column], obs.index, target.obs_names)
        aligned.index = target.obs.index
        target.obs[_key(column)] = aligned
        written.append(_key(column))

    colnames_store = target.uns.setdefault(OBSM_COLNAMES_KEY, {})
    for matrix_key, matrix in matrices.items():
        aligned = _merge.reindex_matrix(matrix, obs.index, target.obs_names)
        names = colnames.get(matrix_key)
        if names and len(names) == aligned.shape[1]:
            # Land it with its class names attached, so the mapping survives in the object
            # itself and not only in the uns block.
            set_matrix(target, _key(matrix_key), aligned, names)
        else:
            target.obsm[_key(matrix_key)] = aligned
        if names:
            colnames_store[_key(matrix_key)] = list(names)

    target.uns[_key("proloc_import")] = {
        "source": source,
        "spec": str(artifact.uns.get("msnset_spec") or "none (no contract block)"),
        "obs_columns": written,
        "matrices": {_key(k): list(colnames.get(k, [])) for k in matrices},
        "n_matched": n_matched,
        "n_target": int(target.n_obs),
        "n_artifact": int(len(obs)),
        "notes": _msnset.notes_for(list(obs.columns) + list(matrices)),
    }

    if set_colors and written:
        from ..preprocessing.annotation import set_sensible_compartment_colors

        # This already decides for itself which columns look like compartment annotations
        # (by how many of their values it recognises), so there is nothing to pre-filter.
        set_sensible_compartment_colors(target, columns=written)

    return target if copy else None


def list_msnset_results(
    path: str | Path | AnnData, *, strict_spec: bool = False
) -> dict[str, Any]:
    """Inventory a pRoloc artifact without reading it into a session.

    Reports what the artifact holds and where each piece belongs, so you can look before you
    read or merge. ``strict_spec`` defaults to ``False`` here, since inspecting an artifact you
    cannot read is exactly when you want a report.

    Returns
    -------
    A dict describing the artifact:

    - ``source``, ``spec``, ``n_obs``, ``n_vars``, ``obs_names`` -- provenance, shape and a few
      example IDs.
    - ``obs`` / ``var`` -- ``{column: dtype}`` for the scalar ``fData`` / ``pData`` columns.
    - ``matrices`` -- ``{name: {"shape": [...], "categories": [...]}}`` for the matrix-valued
      ``fData`` columns, which live in ``.obsm``; ``var_matrices`` the same for ``pData``, in
      ``.varm``.
    - ``layers`` -- extra ``assayData`` element names.
    - ``notes`` -- advisory warnings about columns pRoloc is known to misname.
    - ``dropped`` -- slots the writer could not carry, per ``uns["msnset_dropped"]``.

    Examples
    --------
    >>> gr.io.list_msnset_results("results.h5ad")["matrices"]   # doctest: +SKIP
    {'svm.all.scores': {'shape': [2538, 12], 'categories': ['Cytosol', ...]}}
    """
    from ..util import layer_names

    artifact = read_msnset(path, set_colors=False, strict_spec=strict_spec)
    spec = _msnset.read_spec_block(artifact.uns, strict=False)

    def _describe(mapping, uns_key: str) -> dict[str, dict[str, Any]]:
        names = _msnset.matrix_colnames(artifact.uns, mapping, uns_key=uns_key)
        return {
            str(key): {
                "shape": list(np.shape(_as_array(value))),
                "categories": names.get(str(key), []),
            }
            for key, value in mapping.items()
        }

    return {
        "source": _source_label(path),
        "spec": spec["spec"],
        "n_obs": int(artifact.n_obs),
        "n_vars": int(artifact.n_vars),
        "obs_names": list(artifact.obs_names[:5]),
        "obs": {str(c): str(artifact.obs[c].dtype) for c in artifact.obs.columns},
        "var": {str(c): str(artifact.var[c].dtype) for c in artifact.var.columns},
        "matrices": _describe(artifact.obsm, OBSM_COLNAMES_KEY),
        "var_matrices": _describe(artifact.varm, VARM_COLNAMES_KEY),
        # layer_names(), not artifact.layers: anndata >= 0.13 backs .X with layers[None], so
        # iterating .layers directly reports a phantom "None" layer that is really the main
        # matrix.
        "layers": sorted(layer_names(artifact)),
        "notes": _msnset.notes_for(list(artifact.obs.columns) + list(artifact.obsm)),
        "dropped": list(spec["dropped"]),
    }


def _read_artifact(path: str | Path | AnnData) -> tuple[AnnData, str]:
    """Load an h5ad artifact, or pass an in-memory ``AnnData`` through unchanged."""
    import anndata

    if isinstance(path, anndata.AnnData):
        return path, _source_label(path)
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"No such pRoloc artifact: {resolved}")
    return anndata.read_h5ad(resolved), _source_label(resolved)


def _as_array(value) -> np.ndarray:
    """An ``.obsm``/``.varm`` entry as a plain array, whatever AnnData is holding.

    ``.obsm`` legitimately holds a DataFrame or a sparse matrix as well as an ndarray, and both
    would otherwise become a useless 0-d object array under ``np.asarray``.
    """
    if isinstance(value, pd.DataFrame):
        return value.to_numpy()
    return _dense(value) if hasattr(value, "todense") else np.asarray(value)


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def write_msnset(
    data: AnnData,
    path: str | Path,
    *,
    layer: str | None = None,
    layers: Sequence[str] | None = None,
    nan_to_unknown: bool = True,
    obs_columns: Sequence[str] | None = None,
    var_columns: Sequence[str] | None = None,
    obsm_keys: Sequence[str] | None = None,
    varm_keys: Sequence[str] | None = None,
    check_normalized: bool = True,
    write_script: bool = False,
    overwrite: bool = False,
) -> Path:
    """Export a grassp :class:`~anndata.AnnData` as a pRoloc-ready artifact.

    Writes a self-describing h5ad that the companion R package turns into a real ``MSnSet``::

        remotes::install_github("czbiohub-sf/grassp", subdir = "r/grasspio")
        library(grasspio)
        x <- grassp_as_msnset("experiment.h5ad")

    Everything crosses by default -- ``.obs``, ``.var``, every ``.obsm`` and ``.varm`` entry,
    every layer, and the whole of ``.uns`` -- because the expected next step is to bring the
    object back with :func:`read_msnset`. The exceptions are ``.obsp``/``.varp``, which ``eSet``
    has no slot for, and whatever you exclude yourself; both are listed in
    ``uns["msnset_dropped"]``. Column names and dtypes are written out as they are; the one
    edit is ``nan_to_unknown``, which swaps grassp's ``NaN`` for the ``"unknown"`` string
    pRoloc requires.

    Note that no column is nominated as *the* marker column. pRoloc's ``fcol`` is a per-call
    argument, so an ``MSnSet`` can carry as many marker columns as it likes -- ``markers``,
    ``markers.orig``, ``pd.markers`` -- and each function is pointed at whichever it needs,
    exactly as in AnnData. You choose in R, at the call.

    Parameters
    ----------
    data
        Proteins in ``.obs``, fractions in ``.var`` (grassp's orientation, which already
        matches ``MSnSet``'s features-by-fractions ``exprs``).
    path
        Destination ``.h5ad``.
    layer
        Which matrix becomes ``exprs()``, the one pRoloc's functions operate on. ``None`` uses
        ``.X``. If you name a layer here, ``.X`` itself is not carried, and that is recorded in
        ``uns["msnset_dropped"]``.
    layers
        Additional layers to carry across as extra ``assayData`` elements. ``None`` (default)
        sends all of them; pass ``[]`` to send none. An ``MSnSet``'s ``assayData`` is a Biobase
        environment holding any number of equal-dimension matrices, so they survive intact and
        -- importantly -- are subset together with ``exprs`` by ``markerMSnSet``, ``filterNA``
        and ordinary ``[`` indexing. In R they are reachable with
        ``assayDataElementNames(x)`` and ``assayDataElement(x, "pvals")``.
    nan_to_unknown
        Replace ``NaN`` with ``"unknown"`` in every text column of ``.obs`` -- string, object
        and Categorical alike; numeric and boolean columns are left alone. pRoloc encodes
        unlabelled features with that sentinel and needs it: ``markerMSnSet`` and
        ``unknownMSnSet`` fail outright on ``NA``, and a classifier's training set is chosen
        with ``fData(object)[, fcol] != "unknown"``. Applying it to every text column rather
        than to a nominated one is what lets an object carry several marker sets at once.
    obs_columns, var_columns
        Restrict what crosses over. ``None`` exports everything.
    obsm_keys, varm_keys
        ``.obsm`` / ``.varm`` entries to export as matrix-valued ``fData`` / ``pData`` columns.
        ``None`` (default) sends all of them; pass ``[]`` to send none. ``pData`` is the same
        ``AnnotatedDataFrame`` class as ``fData``, so the two use one mechanism, and R subsets
        each correctly along its own axis. Column names are taken from
        ``uns["obsm_colnames"]``/``uns["varm_colnames"]``, else ``uns[f"{key}_categories"]``,
        else the categories of the companion label column, else ``V1..Vn`` -- so a probability
        matrix gets its compartment names even when the ``uns`` key was never written, which is
        how portal datasets are curated.
    check_normalized
        Warn when profiles do not sum to 1 per protein. pRoloc's distance-based methods and
        its plots assume sum-normalised profiles; see :func:`grassp.pp.normalize_total`.
    write_script
        Also write ``<stem>_run_proloc.R`` next to the artifact: the install line, the
        ``grassp_as_msnset`` call, a worked SVM/TAGM/phenoDisco sequence, and the
        ``grassp_write_msnset`` call to send results back.
    overwrite
        Overwrite an existing file rather than raising.

    Returns
    -------
    The path actually written.

    Raises
    ------
    ValueError
        If ``.obs_names`` are not unique or contain blanks (R rownames cannot), if
        ``layer`` is missing, or if the destination exists and ``overwrite`` is ``False``.

    Notes
    -----
    The contract owns the ``uns`` keys in :data:`grassp.io._msnset.RESERVED_UNS_KEYS`; they are
    regenerated on every export rather than copied from ``data.uns``, so an artifact always
    describes itself rather than whatever it was last read from.

    Examples
    --------
    >>> gr.io.write_msnset(adata, "experiment.h5ad", write_script=True)   # doctest: +SKIP
    PosixPath('experiment.h5ad')
    """
    import anndata

    from ..util import layer_names

    destination = Path(path)
    if destination.exists() and not overwrite:
        raise ValueError(f"{destination} already exists. Pass overwrite=True to replace it.")

    _validate_obs_names(data)

    if layer is not None and layer not in data.layers:
        raise KeyError(
            f"layer '{layer}' is not in data.layers. Available: {layer_names(data)}"
        )
    matrix = data.layers[layer] if layer is not None else data.X
    if matrix is None:
        raise ValueError("Nothing to export: the chosen expression matrix is empty.")
    matrix = _dense(matrix)
    _check_matrix(matrix, check_normalized=check_normalized)

    obs = _select_columns(data.obs, obs_columns, "obs_columns", "data.obs")
    if nan_to_unknown:
        for column in obs.columns:
            obs[column] = _msnset.nan_to_unknown(obs[column])
    var = _select_columns(data.var, var_columns, "var_columns", "data.var")

    obsm, obsm_colnames, dropped = _select_matrices(data, axis="obs", keys=obsm_keys)
    varm, varm_colnames, varm_dropped = _select_matrices(data, axis="var", keys=varm_keys)
    dropped.extend(varm_dropped)

    # Every layer except the one used as exprs. An MSnSet's assayData is a Biobase environment
    # holding any number of equal-dimension matrices, so these arrive as additional
    # assayDataElements and subset consistently with exprs on the R side.
    available_layers = [name for name in layer_names(data) if name != layer]
    if layers is None:
        chosen_layers = available_layers
    else:
        missing = [name for name in layers if name not in layer_names(data)]
        if missing:
            raise KeyError(
                f"layers not found in data.layers: {missing}. "
                f"Available: {layer_names(data)}"
            )
        chosen_layers = [name for name in layers if name != layer]
    extra_layers = {name: _dense(data.layers[name]) for name in chosen_layers}

    dropped.extend(f"layers:{name}" for name in available_layers if name not in extra_layers)
    if layer is not None and data.X is not None:
        dropped.append(f"X (replaced by layer={layer!r} as exprs)")
    # The one structural impossibility: eSet has no pairwise slot, and pRoloc has no graph to
    # map one onto. Graphs are derived from .X anyway -- gr.pp.neighbors rebuilds them.
    dropped.extend(f"obsp:{name}" for name in data.obsp)
    dropped.extend(f"varp:{name}" for name in data.varp)

    # Column names cross verbatim. pRoloc tolerates non-syntactic fData/pData names -- checked
    # by running svmClassification on an MSnSet whose columns were `Gene names` and whose
    # fractions were `Fraction 1` -- so there is nothing to mangle and no map to keep in step.
    artifact = anndata.AnnData(X=matrix, obs=obs, var=var)
    for key, value in obsm.items():
        artifact.obsm[key] = value
    for key, value in varm.items():
        artifact.varm[key] = value
    for key, value in extra_layers.items():
        artifact.layers[key] = value

    # The whole of .uns crosses: h5ad holds arbitrary uns, the R side ignores what it does not
    # understand, and a round trip should not lose the neighbour parameters, the PCA variance
    # ratios or a schema version just because this module has never heard of them. The
    # contract's own keys are regenerated rather than copied.
    artifact.uns.update(
        {k: v for k, v in data.uns.items() if k not in _msnset.RESERVED_UNS_KEYS}
    )
    artifact.uns.update(
        _msnset.build_spec_block(
            layer=layer,
            obsm_colnames=obsm_colnames,
            varm_colnames=varm_colnames,
            dropped=dropped,
        )
    )

    artifact.write_h5ad(destination)
    if write_script:
        _write_r_script(destination)
    return destination


def _dense(matrix) -> np.ndarray:
    """A float64 dense array, whatever AnnData is holding."""
    return np.asarray(matrix.todense() if hasattr(matrix, "todense") else matrix, dtype=float)


def _select_columns(
    frame: pd.DataFrame, columns: Sequence[str] | None, argument: str, slot: str
) -> pd.DataFrame:
    """A copy of ``frame``, restricted to ``columns`` if given."""
    frame = frame.copy()
    if columns is None:
        return frame
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise KeyError(f"{argument} not found in {slot}: {missing}")
    return frame.loc[:, list(columns)]


def _validate_obs_names(data: AnnData) -> None:
    """Reject index values R cannot use as rownames."""
    names = pd.Index([str(n) for n in data.obs_names])
    blank = [n for n in names if n.strip() == "" or n.lower() in {"nan", "none"}]
    if blank:
        raise ValueError(
            f"{len(blank)} obs_names are blank or NaN-like, which R cannot use as "
            f"rownames: {blank[:5]}. Filter or rename these proteins first."
        )
    if not names.is_unique:
        duplicated = names[names.duplicated()].unique().tolist()
        raise ValueError(
            f"obs_names must be unique to become R rownames, but "
            f"{len(duplicated)} are duplicated: {duplicated[:5]}. "
            "Deduplicate with `adata.obs_names_make_unique()` or aggregate first."
        )


def _check_matrix(matrix: np.ndarray, *, check_normalized: bool) -> None:
    """Warn about matrix properties that will bite on the pRoloc side."""
    n_nan = int(np.isnan(matrix).sum())
    if n_nan:
        warnings.warn(
            f"The exported matrix has {n_nan} missing values. Several pRoloc methods "
            "require complete profiles (it offers `filterNA`); consider imputing first "
            "with gr.pp.impute_knn.",
            stacklevel=3,
        )
    if not check_normalized:
        return
    with np.errstate(invalid="ignore"):
        sums = np.nansum(matrix, axis=1)
    off = np.abs(sums - 1.0) > 1e-3
    if off.any():
        warnings.warn(
            f"{int(off.sum())} of {len(sums)} protein profiles do not sum to 1 "
            f"(observed range {np.nanmin(sums):.3g}-{np.nanmax(sums):.3g}). pRoloc's "
            "distance-based methods and plots assume sum-normalised profiles; see "
            "gr.pp.normalize_total. Pass check_normalized=False to silence this.",
            stacklevel=3,
        )


#: Suffixes grassp's own annotators append to a label column's name when they store the
#: matching probability matrix. Stripping one recovers the label column, whose categories name
#: the matrix's columns.
_MATRIX_SUFFIXES = ("_probabilities", ".probabilities", "_one_hot_labels")


def _select_matrices(
    data: AnnData, *, axis: Axis, keys: Sequence[str] | None
) -> tuple[dict[str, np.ndarray], dict[str, list[str]], list[str]]:
    """Choose which ``.obsm`` (or ``.varm``) entries to export, and report the rest as dropped.

    One function for both axes, because ``pData`` and ``fData`` are the same
    ``AnnotatedDataFrame`` class -- a matrix column behaves the same way on either.

    ``keys=None`` takes every two-dimensional entry; a non-2D one is dropped, since an
    ``AnnotatedDataFrame`` cannot hold it. Column names are looked for in four places, in
    order of authority, and fall back to ``V1..Vn``:

    1. the entry's own columns, when it is a :class:`~pandas.DataFrame`. Nothing can
       contradict names carried by the matrix itself, which is why grassp's annotators write
       them that way -- see :func:`grassp.util.set_matrix`.
    2. ``uns["obsm_colnames"][key]`` / ``uns["varm_colnames"][key]`` -- an explicit declaration.
    3. ``uns[f"{key}_categories"]``, including after stripping a ``_probabilities`` /
       ``.probabilities`` / ``_one_hot_labels`` suffix -- the convention grassp's own annotators
       write.
    4. the categories of the companion label column, ``obs[stem]`` (or ``var[stem]``) -- which
       is where they actually live for a probability matrix whose ``uns`` key was never
       written. Portal datasets are curated this way:
       ``harmonized_annotation_propagated_probabilities`` has no ``_categories`` entry, but
       ``obs["harmonized_annotation_propagated"]`` is a Categorical of exactly the right width.
    """
    mapping = data.obsm if axis == "obs" else data.varm
    frame = data.obs if axis == "obs" else data.var
    uns_key = OBSM_COLNAMES_KEY if axis == "obs" else VARM_COLNAMES_KEY

    available = {str(k): _as_array(v) for k, v in mapping.items()}
    carried = {
        str(k): [str(c) for c in v.columns]
        for k, v in mapping.items()
        if isinstance(v, pd.DataFrame)
    }
    declared = _msnset.declared_colnames(data.uns.get(uns_key))

    def _stems(key: str) -> list[str]:
        return [key] + [key[: -len(s)] for s in _MATRIX_SUFFIXES if key.endswith(s)]

    def _names_for(key: str, width: int) -> list[str]:
        if len(carried.get(key, ())) == width:
            return carried[key]
        if len(declared.get(key, ())) == width:
            return declared[key]
        for stem in _stems(key):
            candidate = data.uns.get(f"{stem}_categories")
            if candidate is not None:
                names = [str(c) for c in np.atleast_1d(candidate).ravel().tolist()]
                if len(names) == width:
                    return names
        for stem in _stems(key):
            column = frame.get(stem)
            if column is not None and isinstance(column.dtype, pd.CategoricalDtype):
                names = [str(c) for c in column.cat.categories]
                if len(names) == width:
                    return names
        return [f"V{i + 1}" for i in range(width)]

    if keys is None:
        chosen = [key for key, value in available.items() if value.ndim == 2]
    else:
        missing = [key for key in keys if key not in available]
        if missing:
            raise KeyError(
                f"{axis}m keys not found in data.{axis}m: {missing}. "
                f"Available: {sorted(available)}"
            )
        chosen = [str(key) for key in keys]

    matrices = {key: available[key].astype(float) for key in chosen}
    colnames = {key: _names_for(key, available[key].shape[1]) for key in chosen}
    dropped = [f"{axis}m:{key}" for key in available if key not in matrices]
    return matrices, colnames, dropped


def _write_r_script(artifact: Path, marker_key: str = "markers") -> Path:
    """Render a runnable R script beside the artifact.

    ``marker_key`` only fills in the ``fcol =`` arguments in the generated template. It is not
    a property of the artifact -- pRoloc takes ``fcol`` per call, so edit the script if your
    marker column is named something else, or point different steps at different columns.
    """
    script = artifact.with_name(f"{artifact.stem}_run_proloc.R")
    results = artifact.with_name(f"{artifact.stem}_results.h5ad")
    script.write_text(
        f'''#!/usr/bin/env Rscript
# Generated by grassp gr.io.write_msnset(). Run with:  Rscript {script.name}
#
# One-time setup:
#   install.packages(c("remotes", "BiocManager"))
#   BiocManager::install(c("pRoloc", "rhdf5"))
#   remotes::install_github("czbiohub-sf/grassp", subdir = "r/grasspio")

library(grasspio)
library(pRoloc)

x <- grassp_as_msnset("{artifact.name}")
stopifnot(validObject(x))
print(getMarkerClasses(x, fcol = "{marker_key}"))

# --- Support vector machine -------------------------------------------------
# In real work get the hyperparameters from svmOptimisation(), which is a grid search over
# repeated cross-validation and takes minutes:
#   params <- svmOptimisation(x, fcol = "{marker_key}", times = 100,
#                             class.weights = classWeights(x, fcol = "{marker_key}"))
#   x <- svmClassification(x, params, fcol = "{marker_key}", scores = "all")
# The fixed values below just make this script finish quickly.
set.seed(1)  # e1071's probability scaling consumes the RNG
x <- svmClassification(x, fcol = "{marker_key}", sigma = 0.1, cost = 16, scores = "all")

# `scores = "all"` writes the per-class matrix but NOT the scalar score that orgQuants() and
# getPredictions() look for, so derive it.
fData(x)$svm.scores <- apply(fData(x)$svm.all.scores, 1, max)

# `mcol` is which column held the training labels; like fcol it defaults to "markers".
ts <- orgQuants(x, fcol = "svm", scol = "svm.scores", mcol = "{marker_key}", t = 0.75)
ts[is.na(ts)] <- Inf  # a class with too few markers has no quantile
x <- getPredictions(x, fcol = "svm", scol = "svm.scores", mcol = "{marker_key}", t = ts)

# --- k nearest neighbours ---------------------------------------------------
set.seed(1)
x <- knnClassification(x, fcol = "{marker_key}", k = 5)

# --- Left out on purpose ---------------------------------------------------
# tagmMapTrain() currently fails on fractionation data of this shape with "x is not a
# symmetric matrix" (an exact-equality symmetry test upstream in LaplacesDemon), and
# phenoDisco() with the recommended times = 100 runs for hours:
# x <- tagmMapPredict(x, params = tagmMapTrain(x, fcol = "{marker_key}"),
#                     fcol = "{marker_key}", probJoint = TRUE)
# x <- phenoDisco(x, fcol = "{marker_key}", times = 100, GS = 10)

grassp_write_msnset(x, "{results.name}", overwrite = TRUE)
cat("Wrote {results.name}. Back in Python:\\n")
cat('  adata = gr.io.read_msnset("{results.name}")\\n')
'''
    )
    return script
