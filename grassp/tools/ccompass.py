"""Run the C-COMPASS multi-localization neural network on grassp AnnData objects.

`C-COMPASS <https://github.com/ICB-DCM/C-COMPASS>`_ (Haas et al., *Nat. Methods*
2025) predicts, for every protein, a *class contribution* vector over subcellular
compartments that sums to 1 -- the mechanism by which it calls multi-localizing
proteins. Its published implementation is GUI-first, but the actual machine
learning core is reachable as a library. This module bridges the two: it converts
a grassp :class:`~anndata.AnnData` (proteins in ``.obs``, fractions in ``.var``)
into the ``dict[condition_replicate -> DataFrame]`` structures that C-COMPASS's
GUI-free worker :func:`ccompass.MOP.multi_organelle_prediction` consumes, runs the
real published network, and writes the results back into standard grassp slots so
they are interchangeable with the native annotators in
:mod:`grassp.tools.localization`.

C-COMPASS is an *optional* dependency; install it with ``pip install grassp[ccompass]``
(this pulls in ``tensorflow``/``keras``/``keras-tuner``). It needs **Python <= 3.13**: tensorflow
publishes no wheels for 3.14, so the extra cannot be installed there even though the rest of
grassp supports it.
"""

from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from anndata import AnnData

import numpy as np
import pandas as pd

from ..util import set_matrix

#: Name of the compartment/label column that C-COMPASS expects on every profile.
_CLASS_COL = "class"

#: grassp default for the first-hidden-layer size search. C-COMPASS's own default is
#: ``"long"`` (C..F neurons), but the C-COMPASS paper (Haas et al. 2025) uses the
#: ``"short"`` range (C+0.4(F-C)..C+0.6(F-C)), so we default to that for paper fidelity.
_DEFAULT_NN_OPTIMIZATION = "short"


def _resolve_nn_params(nn_params, reliability, core):
    """Resolve ``nn_params`` into a ``NeuralNetworkParametersModel``.

    Accepts ``None`` (grassp defaults), a ``dict`` of overrides, a path to a YAML/JSON
    parameter file (the same pydantic serialization C-COMPASS uses for its own
    ``settings.yaml`` / session files), or an already-built model. For ``None`` /
    ``dict`` / file inputs the first-hidden-layer search defaults to ``"short"`` (the
    paper's setting) unless explicitly overridden; a fully-built model is respected as-is.
    """
    Model = core.NeuralNetworkParametersModel
    if isinstance(nn_params, Model):
        params = nn_params
    else:
        if nn_params is None:
            data: dict = {}
        elif isinstance(nn_params, (str, Path)):
            import yaml

            with open(nn_params) as fh:
                data = yaml.safe_load(fh) or {}
        elif isinstance(nn_params, dict):
            data = dict(nn_params)
        else:
            raise TypeError(
                "nn_params must be None, a dict, a path to a YAML/JSON file, or a "
                f"NeuralNetworkParametersModel, not {type(nn_params).__name__}."
            )
        data.setdefault("NN_optimization", _DEFAULT_NN_OPTIMIZATION)
        params = Model(**data)
    if reliability is not None:
        params = params.model_copy(update={"reliability": reliability})
    return params


def _import_ccompass():
    """Import the optional C-COMPASS modules or raise a helpful error."""
    try:
        from ccompass import MOA, MOP, core  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised via install extra
        raise ImportError(
            "gr.tl.ccompass requires the optional 'ccompass' dependency. "
            "Install it with `pip install grassp[ccompass]` "
            "(this also installs tensorflow/keras). Note that it needs Python <= 3.13, since "
            "tensorflow publishes no wheels for 3.14."
        ) from exc
    return MOP, MOA, core


def ccompass_default_params(as_dict: bool = False, path: str | None = None):
    """Return (and optionally save) the default C-COMPASS hyperparameters used by
    :func:`ccompass`.

    Use this to discover, inspect, and customize the parameters:

    .. code-block:: python

        import grassp as gr

        gr.tl.ccompass_default_params()  # every field + default
        gr.tl.ccompass_default_params(path="params.yaml")  # editable template
        gr.tl.ccompass(adata, marker_key="marker", nn_params="params.yaml")

    These are C-COMPASS's :class:`~ccompass.core.NeuralNetworkParametersModel` defaults
    with grassp's one override, ``NN_optimization="short"`` (the C-COMPASS-paper setting;
    C-COMPASS's own default is ``"long"``). Every field is a valid ``nn_params`` key.

    Parameters
    ----------
    as_dict
        If ``True`` return a plain ``dict`` (``model_dump()``) instead of the pydantic
        model. Ignored when only writing a file.
    path
        If given, write the defaults to this YAML file (a template you can edit and pass
        back as ``nn_params``).

    Returns
    -------
    A :class:`~ccompass.core.NeuralNetworkParametersModel` (or ``dict`` if ``as_dict``).
    """
    _, _, core = _import_ccompass()
    params = _resolve_nn_params(None, None, core)
    if path is not None:
        import yaml

        with open(path, "w") as fh:
            yaml.safe_dump(params.model_dump(), fh, sort_keys=False)
    return params.model_dump() if as_dict else params


def _minmax_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Scale every protein profile (row) to ``[0, 1]`` across its fractions.

    Mirrors C-COMPASS's per-profile ``MinMaxScaler`` step
    (``FDP.pre_post_scaling(..., how="minmax")``). Constant rows map to zeros.
    """
    values = df.to_numpy(dtype=float)
    row_min = np.nanmin(values, axis=1, keepdims=True)
    row_max = np.nanmax(values, axis=1, keepdims=True)
    span = row_max - row_min
    # avoid division by zero for constant / all-nan rows
    span[span == 0] = 1.0
    scaled = (values - row_min) / span
    return pd.DataFrame(scaled, index=df.index, columns=df.columns)


def _build_subcon_frames(
    profiles: pd.DataFrame,
    var: pd.DataFrame,
    condition_key: str | None,
    replicate_key: str | None,
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """Split a protein x fraction table into C-COMPASS condition/replicate frames.

    Returns a ``{"{condition}_Rep.{replicate}" -> DataFrame}`` mapping (the
    ``ConditionReplicate`` convention used throughout C-COMPASS) and the ordered
    list of unique condition names.
    """
    if condition_key is not None:
        conditions = var[condition_key].astype(str)
    else:
        conditions = pd.Series("cond", index=var.index)

    if replicate_key is not None:
        replicates = var[replicate_key].astype(str)
    else:
        replicates = pd.Series("1", index=var.index)

    subcons: dict[str, pd.DataFrame] = {}
    for (cond, rep), group in var.groupby([conditions, replicates], sort=True):
        cols = group.index
        subcon = f"{cond}_Rep.{rep}"
        subcons[subcon] = profiles[cols].copy()

    condition_names = list(dict.fromkeys(conditions.tolist()))
    return subcons, condition_names


def ccompass(
    data: AnnData,
    marker_key: str = "markers",
    *,
    condition_key: str | None = None,
    replicate_key: str | None = None,
    layer: str | None = None,
    scale: bool = True,
    nn_params: Any = None,
    reliability: int | None = None,
    key_added: str = "ccompass",
    max_processes: int = 1,
    aggregate: bool = True,
    set_colors: bool = True,
    inplace: bool = True,
):
    """Predict compartment class contributions with the C-COMPASS neural network.

    Converts ``data`` into the profile dictionaries expected by C-COMPASS, trains
    its multi-organelle network (per condition x replicate, ensembled over
    ``rounds`` x ``subrounds``), and -- when ``aggregate`` is ``True`` -- runs
    C-COMPASS's :func:`~ccompass.MOA.stats_proteome` to obtain reliability-filtered
    class contributions. Results are written into ``data`` following the same
    conventions as :func:`~grassp.tools.competitive_propagation`.

    Parameters
    ----------
    data
        :class:`~anndata.AnnData` with proteins in ``.obs`` and fractions in
        ``.var``. ``.X`` (or ``layers[layer]``) holds the fractionation profiles.
    marker_key
        ``.obs`` column with known compartment labels (e.g. added by
        :func:`grassp.pp.add_markers`). ``NaN`` entries are treated as unlabeled
        proteins to be predicted.
    condition_key, replicate_key
        Optional ``.var`` columns identifying biological condition and replicate.
        When ``replicate_key`` is given, each replicate is trained as an
        independent network and averaged afterwards (C-COMPASS's ``"separate"``
        combination mode). When omitted, all samples are treated as a single
        condition/replicate.
    layer
        Layer to use as the profile matrix. ``None`` uses ``.X``.
    scale
        If ``True`` (default), min-max scale each protein profile to ``[0, 1]``
        across fractions, reproducing C-COMPASS's per-profile normalization.
        Disable if the profiles are already normalized.
    nn_params
        Neural-network / training / fCC hyperparameters. One of: ``None`` (grassp
        defaults), a ``dict`` of overrides, a path to a YAML/JSON parameter file (the
        same pydantic serialization C-COMPASS uses for its own settings/session files),
        or a fully-built :class:`ccompass.core.NeuralNetworkParametersModel`. Every
        C-COMPASS hyperparameter is settable this way (``upsampling``, ``svm_filter``,
        ``mixed_part``, ``NN_epochs``, ``rounds``, ``subrounds``, ``reliability``, …).
        For ``None``/``dict``/file inputs the first-hidden-layer search defaults to
        ``NN_optimization="short"`` (matching the C-COMPASS paper) unless you set it
        explicitly; a fully-built model is used exactly as given. The resolved
        hyperparameters are recorded in ``adata.uns[f"{key_added}_nn_params"]``.
        Call :func:`ccompass_default_params` to inspect every field and its default, or
        to write an editable YAML template.
    reliability
        Percentile (0-100) for the false-positive filter that produces ``fCC`` /
        ``fNN_winner``. Overrides ``nn_params.reliability`` when given.
    key_added
        Prefix for the result slots (default ``"ccompass"``). With more than one
        condition, the condition name is appended (``"{key_added}_{condition}"``).
    max_processes
        Number of worker processes for C-COMPASS (one per condition x round).
    aggregate
        If ``True`` (default), reconcile rounds/replicates via
        :func:`~ccompass.MOA.stats_proteome` and write both raw (``CC``) and
        reliability-filtered (``fCC``) contributions. If ``False``, only average
        the raw per-round network outputs.
    set_colors
        If ``True``, assign compartment colors to the new ``.obs`` label columns
        via :func:`grassp.pp.set_sensible_compartment_colors`.
    inplace
        If ``True`` (default) write into ``data`` and return ``None``; otherwise
        return the C-COMPASS ``class_predictions`` dictionary.

    Returns
    -------
    Modified ``data`` (in place) with, per condition ``suffix`` (``""`` for a
    single condition, else ``"_{condition}"``):

    - ``.obsm[f"{key_added}{suffix}_contributions"]`` -- proteins x compartments
      class-contribution matrix (rows sum to ~1).
    - ``.uns[f"{key_added}{suffix}_categories"]`` -- ordered compartment names.
    - ``.obs[f"{key_added}{suffix}"]`` -- winning compartment (argmax, ``NN_winner``).
    - When ``aggregate``: ``.obsm[f"{key_added}{suffix}_fcontributions"]`` and
      ``.obs[f"{key_added}{suffix}_fwinner"]`` -- reliability-filtered outputs.

    Notes
    -----
    C-COMPASS distinguishes genuine multi-localization from measurement noise by
    training on synthetic mixtures of marker profiles, ensembling over many
    training rounds, and thresholding each contribution against the values that
    markers of *other* compartments receive (the ``reliability`` percentile).
    This differs from grassp's native permutation-null approach in
    :func:`~grassp.tools.resolve_soft_labels`, so results are complementary
    rather than identical.
    """
    MOP, MOA, core = _import_ccompass()

    # ------------------------------------------------------------------
    # 1. Build the protein x fraction profile table.
    # ------------------------------------------------------------------
    profiles = data.to_df(layer=layer)
    if scale:
        profiles = _minmax_rows(profiles)

    # ------------------------------------------------------------------
    # 2. Split into C-COMPASS condition/replicate frames.
    # ------------------------------------------------------------------
    subcons, condition_names = _build_subcon_frames(
        profiles, data.var, condition_key, replicate_key
    )

    # ------------------------------------------------------------------
    # 3. Attach marker classes and split marker / test / full profiles.
    # ------------------------------------------------------------------
    if marker_key not in data.obs:
        raise KeyError(f"marker_key '{marker_key}' not found in data.obs.")
    marker_series = data.obs[marker_key].astype("object")

    fract_marker: dict[str, pd.DataFrame] = {}
    fract_test: dict[str, pd.DataFrame] = {}
    for subcon, frame in subcons.items():
        frame = frame.copy()
        frame[_CLASS_COL] = marker_series.reindex(frame.index).values
        fract_marker[subcon] = frame[frame[_CLASS_COL].notna()]
        fract_test[subcon] = frame[frame[_CLASS_COL].isna()]
        if fract_marker[subcon].empty:
            raise ValueError(
                f"No markers found for '{subcon}'. Check that '{marker_key}' "
                "has labels overlapping the fractionation data."
            )
    fract_full = core.create_fullprofiles(fract_marker, fract_test)

    # C-COMPASS's aggregated statistics (stats_proteome / SVM reconciliation)
    # require at least two replicates per condition -- the same constraint its
    # own preprocessing enforces. Fail fast with an actionable message rather
    # than surfacing an opaque error from deep inside ccompass.
    if aggregate:
        reps_per_condition: dict[str, int] = {}
        for subcon in subcons:
            cond = subcon.rsplit("_Rep.", 1)[0]
            reps_per_condition[cond] = reps_per_condition.get(cond, 0) + 1
        thin = [c for c, n in reps_per_condition.items() if n < 2]
        if thin:
            raise ValueError(
                "C-COMPASS aggregated statistics require >=2 replicates per "
                f"condition, but these have fewer: {thin}. Provide a "
                "`replicate_key` that resolves multiple replicates, or pass "
                "`aggregate=False` to get the averaged network contributions."
            )

    # ------------------------------------------------------------------
    # 4. Resolve neural-network hyperparameters.
    # ------------------------------------------------------------------
    nn_params = _resolve_nn_params(nn_params, reliability, core)

    # ------------------------------------------------------------------
    # 5. Run the GUI-free C-COMPASS worker.
    # ------------------------------------------------------------------
    class_predictions = MOP.multi_organelle_prediction(
        fract_full,
        fract_marker,
        fract_test,
        nn_params,
        max_processes=max_processes,
    )

    if not inplace:
        return class_predictions

    # record the exact hyperparameters used, for provenance/reproducibility
    data.uns[f"{key_added}_nn_params"] = nn_params.model_dump()

    single_condition = len(condition_names) == 1

    # ------------------------------------------------------------------
    # 6. Aggregate and write results back into `data`.
    # ------------------------------------------------------------------
    label_columns: list[str] = []
    if aggregate:
        results = MOA.stats_proteome(
            class_predictions,
            {"class": subcons},
            fract_marker,
            condition_names,
            nn_params.reliability,
        )
        for condition, stats in results.items():
            suffix = "" if single_condition else f"_{condition}"
            key = f"{key_added}{suffix}"
            metrics = stats.metrics
            classnames = list(stats.classnames)

            _write_contributions(data, key, metrics, classnames, "CC_", "_contributions")
            _write_contributions(data, key, metrics, classnames, "fCC_", "_fcontributions")
            data.uns[f"{key}_categories"] = classnames
            _write_labels(data, key, metrics["NN_winner"])
            _write_labels(data, f"{key}_fwinner", metrics["fNN_winner"])
            label_columns += [key, f"{key}_fwinner"]
    else:
        for condition in condition_names:
            suffix = "" if single_condition else f"_{condition}"
            key = f"{key_added}{suffix}"
            cc = _mean_condition_contributions(class_predictions, condition)
            classnames = list(cc.columns)
            set_matrix(
                data,
                f"{key}_contributions",
                cc.reindex(data.obs_names).to_numpy(dtype=float),
                classnames,
            )
            data.uns[f"{key}_categories"] = classnames
            winner = pd.Series(
                np.array(classnames)[cc.to_numpy().argmax(axis=1)],
                index=cc.index,
            )
            _write_labels(data, key, winner)
            label_columns.append(key)

    if set_colors and label_columns:
        from ..preprocessing.annotation import set_sensible_compartment_colors

        set_sensible_compartment_colors(data, columns=label_columns)

    return None


def _write_contributions(
    data: AnnData,
    key: str,
    metrics: pd.DataFrame,
    classnames: list[str],
    prefix: str,
    suffix: str,
) -> None:
    """Store a ``{prefix}{class}`` contribution block from ``metrics`` in obsm.

    The stored columns are the bare ``classnames``, not the ``{prefix}{class}`` names the
    block is selected by: the prefix only disambiguates the two blocks within ``metrics``,
    and the obsm key already does that. Dropping it keeps the column names in step with
    ``uns[f"{key}_categories"]``.
    """
    cols = [f"{prefix}{name}" for name in classnames]
    block = metrics.reindex(columns=cols).reindex(data.obs_names)
    set_matrix(data, f"{key}{suffix}", block.to_numpy(dtype=float), classnames)


def _write_labels(data: AnnData, col: str, winner: pd.Series) -> None:
    """Write a winner Series to ``.obs[col]`` as a category aligned to obs_names."""
    aligned = winner.reindex(data.obs_names)
    data.obs[col] = pd.Categorical(aligned)


def _mean_condition_contributions(class_predictions: dict, condition: str) -> pd.DataFrame:
    """Average per-round network outputs across all replicates of a condition."""
    frames = []
    for subcon, pred in class_predictions.items():
        if not subcon.startswith(condition + "_"):
            continue
        round_dfs = [rr.z_full_df for rr in pred.round_results.values()]
        stacked = np.stack([df.to_numpy(dtype=float) for df in round_dfs])
        frames.append(
            pd.DataFrame(
                stacked.mean(axis=0),
                index=round_dfs[0].index,
                columns=round_dfs[0].columns,
            )
        )
    # align on the union of proteins and average across replicates
    combined = pd.concat(frames)
    return combined.groupby(level=0).mean()
