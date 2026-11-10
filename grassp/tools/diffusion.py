"""Independent (one-vs-rest) label diffusion for ontology-style annotations.

This is the second of grassp's two graph-annotation families:

* :func:`~grassp.tools.competitive_diffusion` — mutually-exclusive labels that
  **compete** on the kNN graph. Each protein carries a single label; the propagated
  probabilities are cross-class normalized to a **simplex** (rows sum to 1). Use it with
  markers or any single-label, non-overlapping annotation.

* :func:`independent_diffusion` (this module) — **overlapping / hierarchical** labels
  (GO cellular-component, UniProt-SL, COMPARTMENTS). A protein may belong to several
  terms, and terms are nested, so the labels must **not** compete. Each term is diffused
  **independently, one-vs-rest**, giving a per-term membership probability (the rows do
  **not** form a simplex). Label resolution — turning the probability vector into a call
  — happens as a separate, explicit step at the end.

Pipeline (marker-free; only proteins carrying at least one term count as evidence):

1. **Seed** the graph with the vocabulary's multi-hot membership ``Y``; terms with fewer
   than 5 members in the map are not scored.
2. **Diffuse** every scored term at once by label spreading ``F = a S F + (1-a) Y`` with
   ``S = D^{-1/2} W D^{-1/2}`` — one-vs-rest, no cross-term normalization.
3. **Depth** ``a``: one value for the whole map, chosen by k-fold cross-validation over
   the annotated proteins (hide every annotation of one fold, diffuse the rest, rank the
   hidden proteins; best mean average precision over terms). Hiding a fold also removes
   labels around the hidden proteins, which is the situation the method is used in, so
   the chosen depth follows the graph's sparsity and the annotation coverage.
4. **Score** ``s`` — the fraction of a protein's annotated diffused neighbourhood that
   carries the term, leave-one-out (its own seed is removed, so it never reads back its
   own annotation).
5. **Calibrate** ``s`` to a probability by cross-fitted isotonic regression fitted on
   annotated proteins: a per-term curve on ``s`` blended with pooled curves on the
   standardized score ``z = (s - p) / sqrt(p (1-p) / n)`` fitted within log-size strata of
   terms (``n`` = effective number of annotated neighbours). Small terms and proteins with
   thin evidence lean on the pooled curves.
6. **Resolve**: a term is *on* at ``P >= theta``, or at ``P >= theta_child`` if it is
   nested in an on term (nesting = containment >= ``nest_threshold``, read off the gene
   sets); the on terms with no on term nested in them are the multi-location call, and the
   one with the most lineage support is the single call. This is the closed form of an
   MGSA-style penalised likelihood over active terms (see :func:`independent_diffusion`).
"""

from __future__ import annotations
import warnings

from typing import Literal

import numpy as np
import pandas as pd
import scipy.sparse as sp

from anndata import AnnData
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold

from ..util import MULTILOC_SEP, get_matrix
from ._annotation import assign_compartment_colors, require_kind, write_annotation
from ._graph import make_diffuser, neff_kish, symmetric_normalized

#: Candidate diffusion depths for the cross-validated choice of ``alpha``.
ALPHA_GRID = (0.3, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 0.99)
#: Terms need at least this many members in the map to be scored (and cross-fitted).
MIN_MEMBERS = 5

_DEPRECATED = {
    "alphas": "the per-term depth sweep is gone; pass alpha_grid for the cross-validated "
    "choice of one global alpha, or alpha to fix it",
    "n_probe": "the Hutchinson effective-neighbourhood estimate is no longer used",
    "eta": "the resolver is now the closed form of the penalised likelihood; the activation "
    "threshold is theta (0.63 = the break-even of eta=1, eps=0.02)",
    "tau": "the resolver's candidate floor never bound (nothing below 0.5 can be activated)",
    "maxk": "the resolver no longer enumerates active-term subsets",
    "cap": "the resolver no longer enumerates active-term subsets",
}


def _pop_deprecated(kwargs: dict, func: str) -> dict:
    """Warn about and drop removed keyword arguments; reject anything else unknown."""
    out = {}
    for name in list(kwargs):
        if name in _DEPRECATED:
            warnings.warn(
                f"{func}: '{name}' is deprecated and ignored ({_DEPRECATED[name]}).",
                DeprecationWarning,
                stacklevel=3,
            )
            out[name] = kwargs.pop(name)
    if kwargs:
        raise TypeError(f"{func}() got unexpected keyword arguments {sorted(kwargs)}")
    return out


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _resolve_gene_sets(gene_sets, species: str) -> dict[str, list[str]]:
    """Accept a ``{term: [genes]}`` dict, a ``.gmt`` path, an Enrichr library name, or
    ``None`` (the consolidated UniProt subcellular-location sets bundled with grassp)."""
    if isinstance(gene_sets, dict):
        return {t: list(g) for t, g in gene_sets.items()}
    from .enrichment import _load_gmt

    return _load_gmt(gene_sets, species=species)


def _containment(cats, gene_sets) -> np.ndarray:
    """Containment matrix ``C[a, t] = |G_a & G_t| / |G_a|`` over the term categories."""
    G = [set(gene_sets[t]) for t in cats]
    size = np.array([len(g) for g in G])
    genes = sorted(set().union(*G)) if G else []
    gi = {g: k for k, g in enumerate(genes)}
    r, c = [], []
    for i, s in enumerate(G):
        for g in s:
            r.append(i)
            c.append(gi[g])
    M = sp.csr_matrix((np.ones(len(r)), (r, c)), shape=(len(cats), len(genes)))
    return np.clip((M @ M.T).toarray() / np.maximum(size[:, None], 1), 0, 1)


def _map_seeds(gene_sets, pop) -> dict[str, set]:
    """Gene sets restricted to the genes present in the map."""
    return {t: set(g) & pop for t, g in gene_sets.items()}


def _nesting(terms, seeds, nest_threshold) -> np.ndarray:
    """``N[a, b]``: term ``a`` is nested in term ``b`` (containment over the genes present in
    the map ``>= nest_threshold``). Near-duplicates nest both ways; the smaller term (or, at
    equal size, the later one) is kept as the child so that one of them stays a leaf."""
    C = _containment(terms, seeds)
    size = np.array([len(seeds[t]) for t in terms])
    N = C >= nest_threshold
    np.fill_diagonal(N, False)
    for x, y in zip(*np.where(N & N.T)):
        if size[x] > size[y] or (size[x] == size[y] and x < y):
            N[x, y] = False
    return N


def _column_ap(Y, scores) -> np.ndarray:
    """Average precision of every column of ``scores`` against the binary ``Y`` (same tie
    handling as :func:`sklearn.metrics.average_precision_score`, vectorized)."""
    n, m = scores.shape
    o = np.argsort(-scores, axis=0, kind="stable")
    ss = np.take_along_axis(scores, o, 0)
    ys = np.take_along_axis(Y, o, 0).astype(float)
    precision = np.cumsum(ys, 0) / np.arange(1, n + 1)[:, None]
    group_end = np.ones((n, m), bool)
    group_end[:-1] = ss[:-1] != ss[1:]
    idx = np.where(group_end, np.arange(n)[:, None], n - 1)
    end = np.minimum.accumulate(idx[::-1], axis=0)[::-1]
    at_end = np.take_along_axis(precision, end, 0)
    n_pos = ys.sum(0)
    return np.divide((ys * at_end).sum(0), n_pos, out=np.zeros(m), where=n_pos > 0)


# --------------------------------------------------------------------------- #
# steps 3-4: depth and score
# --------------------------------------------------------------------------- #
def _select_alpha(
    Y, Y01, valid, annotated, diffuse, alphas, cv_splits, random_state, min_members
):
    """One global ``alpha`` by k-fold cross-validation of the corrected score.

    For each candidate, every annotation of one fold of annotated proteins is hidden, the
    rest is diffused, and the hidden proteins are scored by the share of the annotated mass
    reaching them that carries each term. The criterion is the mean, over terms that have
    members and non-members among the annotated proteins, of the average precision of those
    out-of-fold scores. Only terms with ``>= min_members`` members take part (and are
    diffused): the choice is global, and many tiny terms would otherwise dominate the mean
    with their noisy per-fold precision. Returns ``(alpha, {alpha: criterion})``.
    """
    ann = np.where(annotated)[0]
    terms = np.where(valid & (Y01.sum(axis=0) >= min_members))[0]
    if not len(terms):  # small vocabulary: use every term
        terms = np.where(valid)[0]
    Ya = Y01[np.ix_(ann, terms)]
    scorable = (Ya.sum(0) > 0) & (Ya.sum(0) < len(ann))
    if len(ann) < cv_splits or not scorable.any():
        return 0.8, {}
    order = np.random.default_rng(random_state).permutation(ann)
    folds = [order[f::cv_splits] for f in range(cv_splits)]
    curve = {}
    for a in alphas:
        oof = np.zeros((Y.shape[0], len(terms)))
        for hide in folds:
            Yf = Y[:, terms].copy()
            Yf[hide] = 0.0
            mask = annotated.astype(float)
            mask[hide] = 0.0
            reach = diffuse(mask, a)[hide]  # hidden: no own seed, so no self term
            F = diffuse(Yf, a)[hide]
            oof[hide] = np.divide(
                F, reach[:, None], out=np.zeros_like(F), where=reach[:, None] > 1e-9
            )
        curve[float(a)] = float(_column_ap(Ya[:, scorable], oof[ann][:, scorable]).mean())
    return max(curve, key=curve.get), curve


def _corrected_score(Y, valid, annotated, diffuse, alpha):
    """Leave-one-out share of the annotated diffused neighbourhood carrying each term."""
    mask = annotated.astype(float)
    F = np.zeros_like(Y)
    F[:, valid] = diffuse(Y[:, valid], alpha)
    reach = (diffuse(mask, alpha) - (1 - alpha) * mask)[:, None]
    s = np.divide(F - (1 - alpha) * Y, reach, out=np.zeros_like(F), where=reach > 1e-9)
    s[:, ~valid] = 0.0
    return s


# --------------------------------------------------------------------------- #
# step 5: calibration
# --------------------------------------------------------------------------- #
def _calibrate(
    s, Y01, valid, annotated, S, *, kappa, evidence_f, cv_splits, random_state, tiebreak
):
    """Cross-fitted, size-aware isotonic calibration of the corrected score.

    Per term, a curve on ``s`` (its own evidence) is blended with pooled curves on
    ``z = (s - p) / sqrt(p (1-p) / n)``, fitted within log-size strata of terms and
    interpolated between neighbouring strata (``p`` = the term's base rate among annotated
    proteins, ``n`` = the 1-hop Kish effective number of annotated neighbours). The blend
    weight ``m / (m + kappa)`` trusts a term's own curve in proportion to its member count
    ``m``, scaled per protein by ``min(1, n / (evidence_f * median n))`` so that proteins
    with thin annotated evidence lean on the pooled curves. Curves are fitted on annotated
    proteins only (unannotated ones are unknowns, not negatives), out of fold.
    """
    n, m = s.shape
    n_eff = np.maximum(neff_kish(S[:, np.where(annotated)[0]]), 1.0)
    p = Y01[annotated].mean(axis=0)
    z = (s - p) / np.sqrt(np.maximum(p * (1 - p), 1e-12) / n_eff[:, None])

    vj = np.where(valid)[0]
    size = Y01.sum(axis=0)
    log_size = np.log(size[vj])
    Q = int(np.clip(len(vj) // 8, 1, 6))
    edges = np.quantile(log_size, np.linspace(0, 1, Q + 1))
    stratum = np.clip(np.searchsorted(edges, log_size, side="right") - 1, 0, Q - 1)
    strata = np.unique(stratum)  # quantile ties can empty a bin
    medians = np.array([np.median(log_size[stratum == q]) for q in strata])
    eye = np.eye(len(strata))
    mix = np.array(
        [[np.interp(x, medians, eye[q]) for q in range(len(strata))] for x in log_size]
    )
    w_term = size / (size + kappa)
    trust = (
        np.minimum(1.0, n_eff / (evidence_f * np.median(n_eff[annotated])))
        if evidence_f
        else np.ones(n)
    )

    P = np.zeros((n, m))
    folds = KFold(cv_splits, shuffle=True, random_state=random_state).split(np.arange(n))
    for train, test in folds:
        train = train[annotated[train]]
        pooled = [
            IsotonicRegression(out_of_bounds="clip").fit(
                z[np.ix_(train, vj[stratum == q])].ravel(),
                Y01[np.ix_(train, vj[stratum == q])].ravel(),
            )
            for q in strata
        ]
        for jj, j in enumerate(vj):
            p_pool = sum(
                mix[jj, q] * curve.predict(z[test, j])
                for q, curve in enumerate(pooled)
                if mix[jj, q] > 0
            )
            p_term = (
                IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
                .fit(s[train, j], Y01[train, j])
                .predict(s[test, j])
            )
            w = w_term[j] * trust[test]
            P[test, j] = w * p_term + (1 - w) * p_pool
    if tiebreak:  # order isotonic plateaus by s
        P[:, vj] = np.clip(P[:, vj] + tiebreak * s[:, vj], 0, 1)
    return P


# --------------------------------------------------------------------------- #
# step 6: resolution
# --------------------------------------------------------------------------- #
def _leaves_and_calls(P, on, N, terms, sizes, primary):
    """Per protein: the on terms with no on term nested in them (the multi-location call,
    ordered by P) and the single call chosen among them."""
    n = P.shape[0]
    labels = np.empty(n, dtype=object)
    leaves = []
    for i in range(n):
        c = np.where(on[i])[0]
        if not len(c):
            labels[i] = None
            leaves.append([])
            continue
        sub = N[np.ix_(c, c)]
        is_leaf = ~sub.any(axis=0)
        lv = c[is_leaf]
        if primary == "lineage":  # own P + P of the on terms it is nested in
            support = P[i, lv] + (sub[is_leaf] * P[i, c][None, :]).sum(1)
            labels[i] = terms[int(lv[np.argmax(support)])]
        else:  # "smallest": the most specific by member count
            labels[i] = terms[int(lv[np.argmin(sizes[lv])])]
        leaves.append([terms[a] for a in lv[np.argsort(-P[i, lv], kind="stable")]])
    return labels, leaves


def _resolve(
    P,
    terms,
    seeds,
    sizes,
    mode,
    *,
    theta,
    theta_child,
    nest_threshold,
    term_threshold,
    min_term_size,
):
    """Turn the per-term probability matrix into ``(labels, leaves)``.

    ``sizes`` is the per-term number of members present in the map; ``min_term_size``
    restricts the terms a protein can be given to those with at least that many members --
    a granularity floor: because the vocabulary is hierarchical, dropping an
    under-represented (small) term leaves its larger ancestors, so the call backs off to the
    most specific term that clears the floor rather than emitting a rare label.
    """
    sizes = np.asarray(sizes)
    eligible = sizes >= max(int(min_term_size), 1)
    Pe = np.where(eligible[None, :], P, 0.0)
    N = _nesting(terms, seeds, nest_threshold)
    if mode == "likelihood":
        on = Pe >= theta
        if theta_child is not None:
            on |= (Pe >= theta_child) & ((on.astype(float) @ N.T.astype(float)) > 0)
        return _leaves_and_calls(Pe, on, N, terms, sizes, "lineage")
    if mode == "specific":
        return _leaves_and_calls(Pe, Pe >= term_threshold, N, terms, sizes, "smallest")
    if mode == "argmax":
        _, leaves = _leaves_and_calls(Pe, Pe >= term_threshold, N, terms, sizes, "lineage")
        best = Pe.argmax(axis=1)
        labels = np.array(
            [terms[b] if Pe[i, b] > 0 else None for i, b in enumerate(best)], dtype=object
        )
        return labels, leaves
    raise ValueError(
        f"resolve must be 'likelihood', 'specific', 'argmax' or None; got {mode!r}"
    )


def _write_calls(adata, key, labels, leaves, out_key=None):
    adata.obs[out_key or key] = pd.Categorical(
        pd.Series(labels, index=adata.obs_names, dtype=object)
    )
    if out_key is None or out_key == key:
        adata.obs[f"{key}_label_compact"] = [
            MULTILOC_SEP.join(x) if x else None for x in leaves
        ]
        assign_compartment_colors(adata, [key])


def resolve_diffusion(
    data: AnnData,
    gene_sets=None,
    key_added: str = "ann_diffusion",
    mode: Literal["likelihood", "specific", "argmax"] = "likelihood",
    min_term_size: int = 0,
    term_threshold: float = 0.5,
    theta: float = 0.63,
    theta_child: float | None = 0.5,
    nest_threshold: float = 0.8,
    gene_key: str = "gene_symbol",
    species: str = "hsap",
    out_key: str | None = None,
    inplace: bool = True,
    **deprecated,
) -> AnnData | None:
    """(Re)resolve stored diffusion probabilities into a per-protein label, in place.

    Reads ``obsm[{key_added}_probabilities]`` (written by :func:`independent_diffusion`) and
    writes ``obs[out_key]``. ``out_key`` defaults to ``key_added`` for likelihood/argmax --
    overwriting the primary call, since that is what re-resolving means, together with
    ``obs[{key_added}_label_compact]`` -- and to ``f"{key_added}_specific"`` for specific,
    so an alternative resolution sits beside the default rather than replacing it. The
    parameters are as in :func:`independent_diffusion`; ``gene_sets`` must be the
    vocabulary that was diffused (it defines the nesting). Re-resolving with the same
    parameters reproduces the stored call exactly. Lets you obtain several resolutions or
    sweep ``min_term_size`` / ``theta`` without re-diffusing. With ``inplace=False`` the
    labels are written to a copy, which is returned.
    """
    _pop_deprecated(deprecated, "resolve_diffusion")
    require_kind(data, key_added, "per_term")
    if not inplace:
        data = data.copy()
    gmt = _resolve_gene_sets(gene_sets, species)
    # The term names are columns of the stored matrix. The uns entry is only a fallback
    # for matrices that carry no column names: those written before grassp labelled them,
    # and those the R bridge demotes when a class name contains "/".
    P, columns = get_matrix(data, f"{key_added}_probabilities")
    P = np.asarray(P, dtype=float)
    terms = list(columns) if columns is not None else list(data.uns[f"{key_added}_categories"])
    seeds = _map_seeds({t: gmt.get(t, ()) for t in terms}, set(data.obs[gene_key].astype(str)))
    sizes = np.array([len(seeds[t]) for t in terms])
    labels, leaves = _resolve(
        P,
        terms,
        seeds,
        sizes,
        mode,
        theta=theta,
        theta_child=theta_child,
        nest_threshold=nest_threshold,
        term_threshold=term_threshold,
        min_term_size=min_term_size,
    )
    if out_key is None:
        out_key = f"{key_added}_specific" if mode == "specific" else key_added
    _write_calls(data, key_added, labels, leaves, out_key=out_key)
    return None if inplace else data


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
def independent_diffusion(
    data: AnnData,
    gene_sets=None,
    gene_key: str = "gene_symbol",
    species: str = "hsap",
    obsp_key: str = "connectivities",
    alpha: float | None = None,
    alpha_grid=ALPHA_GRID,
    alpha_min_members: int = 20,
    calibration: Literal["size_aware", "none"] = "size_aware",
    kappa: float = 30.0,
    evidence_f: float | None = 0.35,
    resolve: Literal["likelihood", "specific", "argmax"] | None = "likelihood",
    theta: float = 0.63,
    theta_child: float | None = 0.5,
    nest_threshold: float = 0.8,
    term_threshold: float = 0.5,
    min_term_size: int = 0,
    cv_splits: int = 5,
    random_state: int = 0,
    key_added: str = "ann_diffusion",
    inplace: bool = True,
    verbose: bool = False,
    **deprecated,
) -> AnnData | None:
    """Annotate a map with overlapping / hierarchical labels by one-vs-rest diffusion.

    Each term in ``gene_sets`` is diffused independently over the kNN graph (no cross-term
    competition), giving a **non-simplex** per-term membership probability, which is then
    resolved into a call. See the module docstring for the pipeline. For mutually-exclusive
    single labels (e.g. markers), use :func:`~grassp.tools.competitive_diffusion` instead.

    Parameters
    ----------
    data
        :class:`~anndata.AnnData` with a neighbour graph in ``obsp[obsp_key]``.
    gene_sets
        The label vocabulary: a ``{term: [gene_ids]}`` dict, a path to a ``.gmt`` file, an
        Enrichr library name (fetched via ``gseapy``), or ``None`` (default) for the
        consolidated UniProt subcellular-location sets bundled with grassp.
    gene_key
        ``obs`` column with the gene identifiers matching ``gene_sets``. When several
        entries share an identifier (isoforms, duplicate accessions), only the first carries
        the gene's annotation and the others are scored as unannotated proteins, so one gene
        is one unit of evidence and a protein cannot vouch for itself through a duplicate;
        the number of such entries is recorded in
        ``uns[{key_added}_params]["duplicate_gene_entries"]``.
    species
        Species of the bundled sets when ``gene_sets`` is ``None`` (and of the GMT loader).
    obsp_key
        Neighbour-graph key in ``obsp`` (default ``"connectivities"``).
    alpha
        Diffusion depth shared by all terms. ``None`` (default) chooses it for this map by
        ``cv_splits``-fold cross-validation over the annotated proteins, among
        ``alpha_grid``: one fold's annotations are hidden, the rest are diffused, and the
        hidden proteins are ranked per term; the value with the best mean average precision
        over terms wins. Sparse graphs (small k) and sparsely annotated maps get deeper
        diffusion. Pass a number to fix it.
    alpha_grid
        Candidates for the cross-validated choice of ``alpha``.
    alpha_min_members
        Only terms with at least this many members in the map take part in the
        cross-validated choice of ``alpha`` (all scored terms if none does). Keeps the
        choice stable and fast on deep vocabularies with many tiny terms.
    calibration
        ``"size_aware"`` (default; see the module docstring) or ``"none"`` (the raw
        corrected score, clipped to ``[0, 1]``).
    kappa
        Support prior of the calibration blend: a term's own curve gets weight
        ``m / (m + kappa)`` for ``m`` members in the map.
    evidence_f
        Proteins whose effective number of annotated neighbours is below
        ``evidence_f`` x the map median lean on the pooled curves in proportion; ``None``
        disables this.
    resolve
        How to turn the probabilities into ``obs[{key_added}]``: ``"likelihood"`` (default,
        see ``theta``), ``"specific"`` (the term with the fewest members among those with
        ``P >= term_threshold`` and no such term nested in them), ``"argmax"``, or ``None``
        to only write probabilities.
    theta, theta_child
        The ``"likelihood"`` resolver: a term is *on* for a protein at ``P >= theta``, or at
        ``P >= theta_child`` if it is nested in an on term. The on terms with no on term
        nested in them form the multi-location call; the one with the most lineage support
        (its P plus the P of the on terms it is nested in) is the single call; no on term,
        no call. This is the exact closed form of an MGSA-style penalised Bernoulli
        likelihood over active terms with error rate ``eps`` and penalty ``eta`` per active
        term: one term pays for itself above ``0.5 + eta / (2 ln((1-eps)/eps))`` (0.63 for
        ``eta = 1``, ``eps = 0.02``), and a child replacing its on parent costs no extra
        penalty, so it wins above 0.5.
    nest_threshold
        Term ``a`` is nested in ``b`` when ``|G_a & G_b| / |G_a| >= nest_threshold`` over the
        genes present in the map. In real vocabularies nested pairs sit at 0.95-1.0 and
        co-annotated ones (e.g. nucleus / cytoplasm) at <= 0.77, so co-localised terms do not
        explain each other away.
    term_threshold
        Per-term membership threshold of the ``"specific"`` and ``"argmax"`` resolvers'
        multi-location call (and the ``"specific"`` candidates). Deliberately *not* called
        ``min_probability``, which means "abstain below this confidence" elsewhere in grassp.
    min_term_size
        Granularity floor: a protein may only be labelled with a term that has at least this
        many members present in the map; the call then backs off to a larger ancestor. ``0``
        (default) applies no floor.
    cv_splits, random_state
        Folds and seed of the cross-validation (choice of ``alpha`` and the cross-fitted
        calibration).
    key_added
        Prefix for the outputs (default ``"ann_diffusion"``).
    inplace
        If ``True`` (default) annotate ``data`` and return ``None``; otherwise operate on and
        return a copy.
    verbose
        Print progress.

    Returns
    -------
    Writes ``obsm[{key_added}_probabilities]`` (per-term calibrated membership, non-simplex,
    with the term names as its columns), ``obs[{key_added}_probability]`` (top membership),
    ``uns[{key_added}_alpha]`` (the diffusion depth used), ``uns[{key_added}_params]``
    (provenance, including ``kind="per_term"`` and the cross-validation curve of ``alpha``),
    and -- when ``resolve`` is set -- ``obs[{key_added}]`` (the single call) and
    ``obs[{key_added}_label_compact]`` (the multi-location call, most probable first).
    Returns the AnnData if ``inplace=False``, else ``None``.

    Removed parameters (``alphas``, ``n_probe``, ``eta``, ``tau``, ``maxk``, ``cap``) are
    accepted with a :class:`DeprecationWarning` and ignored.
    """
    _pop_deprecated(deprecated, "independent_diffusion")
    if calibration not in ("size_aware", "none"):
        raise ValueError(f"calibration must be 'size_aware' or 'none'; got {calibration!r}")
    adata = data if inplace else data.copy()
    if obsp_key not in adata.obsp:
        raise KeyError(f"obsp[{obsp_key!r}] not found; run sc.pp.neighbors first.")
    if gene_key not in adata.obs:
        raise KeyError(f"obs[{gene_key!r}] not found.")

    # 1. seed -- one map entry per gene carries the gene's annotation. Several entries can
    # share a gene (isoforms, duplicate accessions) and then sit as each other's nearest
    # neighbours, so seeding both would let a protein vouch for itself through its twin and
    # would inflate member counts; the extra entries are treated as unannotated.
    gsym = adata.obs[gene_key].astype(str).to_numpy()
    seeds = _map_seeds(_resolve_gene_sets(gene_sets, species), set(gsym))
    terms = list(seeds)
    if not terms:
        raise ValueError("gene_sets is empty.")
    Y = np.column_stack([np.isin(gsym, list(seeds[t])) for t in terms]).astype(float)
    duplicate = pd.Series(gsym).duplicated().to_numpy()
    n_duplicate = int(Y[duplicate].any(axis=1).sum())
    Y[duplicate] = 0.0
    Y01 = Y.astype(int)
    sizes = Y01.sum(axis=0)
    valid = sizes >= MIN_MEMBERS
    if not valid.any():
        raise ValueError(
            f"no term has >= {MIN_MEMBERS} members in the map; nothing to diffuse."
        )
    annotated = Y01.any(axis=1)

    # 2-4. operator, depth, score
    S = symmetric_normalized(adata.obsp[obsp_key])
    diffuse = make_diffuser(S)
    alpha_curve = {}
    if alpha is None:
        alpha, alpha_curve = _select_alpha(
            Y,
            Y01,
            valid,
            annotated,
            diffuse,
            alpha_grid,
            cv_splits,
            random_state,
            alpha_min_members,
        )
    s = _corrected_score(Y, valid, annotated, diffuse, alpha)

    # 5. calibrate
    if calibration == "none":
        P = np.clip(s, 0, 1)
    else:
        P = _calibrate(
            s,
            Y01,
            valid,
            annotated,
            S,
            kappa=kappa,
            evidence_f=evidence_f,
            cv_splits=cv_splits,
            random_state=random_state,
            tiebreak=1e-4,
        )

    # derive_labels=False: an argmax over per-term memberships is not a call, because the
    # terms did not compete for one unit of mass. The resolver below writes obs[key_added].
    write_annotation(
        adata,
        key_added,
        P,
        terms,
        kind="per_term",
        method="one-vs-rest-diffusion",
        derive_labels=False,
        gt_col=gene_key,
        extra_params={
            "alpha": float(alpha),
            "alpha_grid": [float(a) for a in alpha_curve] if alpha_curve else None,
            "alpha_cv_ap": [float(v) for v in alpha_curve.values()] if alpha_curve else None,
            "alpha_min_members": int(alpha_min_members),
            "calibration": calibration,
            "kappa": float(kappa),
            "evidence_f": None if evidence_f is None else float(evidence_f),
            "resolve": resolve,
            "theta": float(theta),
            "theta_child": None if theta_child is None else float(theta_child),
            "nest_threshold": float(nest_threshold),
            "term_threshold": float(term_threshold),
            "min_term_size": int(min_term_size),
            "obsp_key": obsp_key,
            "cv_splits": int(cv_splits),
            "random_state": int(random_state),
            "duplicate_gene_entries": n_duplicate,
        },
    )
    adata.uns[f"{key_added}_alpha"] = float(alpha)

    # 6. resolve
    if resolve is not None:
        labels, leaves = _resolve(
            P,
            terms,
            seeds,
            sizes,
            resolve,
            theta=theta,
            theta_child=theta_child,
            nest_threshold=nest_threshold,
            term_threshold=term_threshold,
            min_term_size=min_term_size,
        )
        _write_calls(adata, key_added, labels, leaves)

    if verbose:
        called = (
            f"; called {float(adata.obs[key_added].notna().mean()):.1%}"
            if resolve is not None
            else ""
        )
        print(
            f"[independent_diffusion] {int(valid.sum())}/{len(terms)} terms scored; "
            f"alpha={alpha:.2f}{' (cross-validated)' if alpha_curve else ''}; "
            f"calibration={calibration}; resolve={resolve}{called}"
        )
    return None if inplace else adata
