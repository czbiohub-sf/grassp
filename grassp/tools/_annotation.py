"""The annotation output contract, in one place.

grassp ships seven things that assign compartment labels to proteins -- graph
propagation, one-vs-rest diffusion, an SVM, TAGM, C-COMPASS, and two cluster-enrichment
paths -- and each of them used to invent its own set of output slots. One wrote
``uns[f"{key}_categories"]`` and another did not; one wrote a ``_one_hot_labels``
companion that the confusion matrix then required; one wrote no scalar confidence at all;
TAGM wrote dotted keys. The result was that every *consumer* had to special-case every
*producer*, and nothing in a stored object said whether its probability rows summed to
one.

``util.set_matrix`` fixed the same problem one level down: it made the class each column
stands for intrinsic to the matrix instead of travelling separately. This module does it
for the stage -- what kind of matrix it is, which annotator made it, and what it was
told -- so a consumer can read an annotation without knowing who wrote it.

Slots written by :func:`write_annotation`, for every annotator::

    obsm[f"{key}_probabilities"]   labelled DataFrame, one column per class
    obs[key]                       Categorical primary call, NaN where abstained
    obs[f"{key}_probability"]      float, the probability of that call
    uns[f"{key}_params"]           provenance: method, kind, gt_col, thresholds, ...
    uns[f"{key}_colors"]           in the category order of obs[key]

``kind`` is the load-bearing entry. ``"simplex"`` means the rows sum to one and the
classes competed; ``"per_term"`` means each column is an independent membership
probability and they do not. An entropy computed over a renormalized ``"per_term"``
matrix is meaningless, and a containment resolver applied to a ``"simplex"`` one is too,
so the resolvers check it rather than trusting the caller.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, Sequence

if TYPE_CHECKING:
    from anndata import AnnData

import warnings

import numpy as np
import pandas as pd

from ..util import MULTILOC_SEP, get_matrix, set_matrix

#: Whether a probability matrix's rows sum to one (the classes competed) or each column
#: is an independent membership probability (they did not).
Kind = Literal["simplex", "per_term"]


class MarkerLabels(NamedTuple):
    """The four views of a label column that annotators keep re-deriving."""

    #: the column as a Categorical, with its declared categories preserved
    labels: pd.Series
    #: boolean mask of the annotated rows
    mask: np.ndarray
    #: ``(n_obs, n_categories)`` one-hot, zero on unannotated rows
    one_hot: np.ndarray
    #: the declared vocabulary, in declared order
    categories: pd.Index


def marker_labels(data: AnnData, gt_col: str) -> MarkerLabels:
    """Read a label column into the forms the annotators need.

    Twelve places used to compute some subset of this by hand, which is how the two
    predictors ended up disagreeing about whether an object-dtype column is a valid
    ``gt_col`` and whether a declared-but-unobserved category gets a matrix column.

    The one-hot spans the *declared* categories, so it always agrees with
    ``categories`` even when a class has no members -- that column is simply zero.
    """
    if gt_col not in data.obs:
        raise KeyError(f"Column {gt_col!r} not found in data.obs")
    labels = data.obs[gt_col].astype("category")
    mask = labels.notna().to_numpy()
    if not mask.any():
        raise ValueError(f"No annotated proteins in {gt_col!r} (every value is NaN)")
    categories = labels.cat.categories
    one_hot = (
        pd.get_dummies(labels)
        .reindex(columns=categories, fill_value=False)
        .to_numpy(dtype=float)
    )
    return MarkerLabels(labels=labels, mask=mask, one_hot=one_hot, categories=categories)


def clamp_marker_rows(P: np.ndarray, one_hot: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Pin annotated rows of ``P`` back to their one-hot encoding.

    ``fix_markers`` was implemented three times -- twice inside the propagation and once
    as a Python loop over markers in the SVM path.
    """
    P = np.asarray(P, dtype=float).copy()
    P[mask] = one_hot[mask]
    return P


def resolve_simplex(
    P: np.ndarray,
    categories: Sequence[Any],
    *,
    min_probability: float | None = None,
    unknown_label: str | None = None,
    declared_categories: Sequence[Any] | None = None,
) -> tuple[pd.Categorical, np.ndarray]:
    """Turn a probability matrix into a primary call plus its confidence.

    Four annotators inlined this -- argmax to a label, max to a confidence, threshold to
    NaN -- and two of them did it *inside* an ``if inplace:`` branch, so the threshold
    silently did nothing on the other path. Returning both up front means the caller
    cannot forget one.

    Parameters
    ----------
    P
        ``(n_obs, len(categories))`` probabilities.
    categories
        Class names in column order.
    min_probability
        Abstain (write NaN) where the top probability is below this. ``None`` applies no
        cutoff, which is honest because the full matrix is stored anyway.
    unknown_label
        A class that means "no call". Rows whose top class is this one abstain, while the
        column itself stays in the matrix.
    declared_categories
        Vocabulary for the returned Categorical, if wider than the matrix's columns --
        e.g. an SVM predicts only the classes it saw, but the label column should keep
        the full ``gt_col`` vocabulary so colours and comparisons line up.
    """
    P = np.asarray(P, dtype=float)
    columns = pd.Index([str(c) for c in categories])
    if P.shape[1] != len(columns):
        raise ValueError(
            f"probability matrix has {P.shape[1]} columns but {len(columns)} class names "
            "were given; they must describe the same matrix."
        )
    winner = P.argmax(axis=1)
    probability = P.max(axis=1)
    called = columns.to_numpy()[winner].astype(object)

    if unknown_label is not None:
        called[called == str(unknown_label)] = np.nan
    if min_probability is not None:
        called[probability < min_probability] = np.nan

    vocabulary = pd.Index(
        [str(c) for c in (declared_categories if declared_categories is not None else columns)]
    )
    if unknown_label is not None:
        vocabulary = vocabulary.drop(str(unknown_label), errors="ignore")
    return pd.Categorical(called, categories=vocabulary), probability


def write_annotation(
    data: AnnData,
    key: str,
    P: np.ndarray,
    categories: Sequence[Any],
    *,
    kind: Kind,
    method: str,
    labels: pd.Categorical | None = None,
    probability: np.ndarray | None = None,
    derive_labels: bool = True,
    gt_col: str | None = None,
    min_probability: float | None = None,
    unknown_label: str | None = None,
    declared_categories: Sequence[Any] | None = None,
    extra_params: dict[str, Any] | None = None,
    set_colors: bool = True,
) -> None:
    """Write an annotation into the slots described in the module docstring.

    ``labels`` may be supplied by annotators that decide their call some other way --
    TAGM picks its allocation from the Gaussian part alone, not from the argmax of what
    it stores -- otherwise :func:`resolve_simplex` derives it.

    ``derive_labels=False`` suppresses that derivation, for a ``"per_term"`` matrix where
    an argmax is not a meaningful call: the columns did not compete, so the largest
    membership need not be the right label. Those annotators resolve separately and pass
    ``labels`` in, or leave ``obs[key]`` unwritten. ``obs[f"{key}_probability"]`` is
    still written either way, as the top membership.
    """
    if labels is None and derive_labels:
        labels, derived = resolve_simplex(
            P,
            categories,
            min_probability=min_probability,
            unknown_label=unknown_label,
            declared_categories=declared_categories,
        )
        if probability is None:
            probability = derived
    if probability is None:
        probability = np.asarray(P, dtype=float).max(axis=1)

    set_matrix(data, f"{key}_probabilities", P, categories)
    if labels is not None:
        data.obs[key] = labels
    data.obs[f"{key}_probability"] = np.asarray(probability, dtype=float)
    data.uns[f"{key}_params"] = {
        "method": method,
        "kind": kind,
        "gt_col": gt_col,
        "min_probability": min_probability,
        "unknown_label": unknown_label,
        "label_separator": MULTILOC_SEP,
        **(extra_params or {}),
    }
    if set_colors and labels is not None:
        assign_compartment_colors(data, [key])


def read_annotation(data: AnnData, key: str) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    """Read an annotation back as ``(probabilities, categories, params)``.

    Accepts anything :func:`write_annotation` produced, plus the older shapes: a bare
    ndarray whose names live in ``uns[f"{key}_categories"]`` (what the R bridge writes
    when a class name contains ``"/"``), and objects with no params entry at all.
    """
    P, columns = get_matrix(data, f"{key}_probabilities")
    P = np.asarray(P, dtype=float)
    if columns is None:
        legacy = data.uns.get(f"{key}_categories")
        if legacy is None:
            raise ValueError(
                f"obsm['{key}_probabilities'] carries no column names and there is no "
                f"uns['{key}_categories'] to fall back on, so the class vocabulary "
                "cannot be recovered."
            )
        columns = [str(c) for c in legacy]
    if len(columns) != P.shape[1]:
        raise ValueError(
            f"obsm['{key}_probabilities'] has {P.shape[1]} columns but {len(columns)} "
            "class names were resolved; they must describe the same matrix."
        )
    return P, list(columns), dict(data.uns.get(f"{key}_params", {}))


def require_kind(data: AnnData, key: str, expected: Kind) -> None:
    """Refuse an annotation of the wrong kind, when the object says which it is.

    A ``"per_term"`` matrix renormalized into a simplex and fed to the entropy null
    produces confident-looking calls from a statistic that does not apply to it, and the
    reverse holds for the containment resolver. Objects written before ``_params`` existed
    say nothing, and are allowed through rather than blocked.
    """
    kind = dict(data.uns.get(f"{key}_params", {})).get("kind")
    if kind is not None and kind != expected:
        raise ValueError(
            f"{key!r} holds {kind!r} probabilities but this resolver needs {expected!r}. "
            + (
                "Use grassp.tl.resolve_diffusion for per-term memberships."
                if expected == "simplex"
                else "Use grassp.tl.resolve_soft_labels for simplex probabilities."
            )
        )


def assign_compartment_colors(data: AnnData, columns: Sequence[str]) -> None:
    """Give label columns their canonical compartment colours.

    Imported at call time because the palette lives in ``grassp.preprocessing``, which
    imports from ``grassp.tools``. Four call sites used to carry this dance individually,
    and three others copied ``uns[f"{gt_col}_colors"]`` positionally instead -- which
    painted every compartment with its neighbour's colour whenever the two vocabularies
    were ordered differently.
    """
    try:
        from ..preprocessing import set_sensible_compartment_colors

        set_sensible_compartment_colors(
            data, columns=list(columns), cutoff=0.0, verbose=False, plot_mapping=False
        )
    except Exception as exc:  # pragma: no cover - colouring must never break a run
        warnings.warn(f"could not assign compartment colours ({exc}).")
