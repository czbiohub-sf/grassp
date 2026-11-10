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

Pipeline (per term ``t``, marker-free and self-tuning):

1. **Seed** the graph with the term's multi-hot membership ``y_t``.
2. **Diffuse** by label spreading ``F = a S F + (1-a) y_t`` with
   ``S = D^{-1/2} W D^{-1/2}`` — one-vs-rest, no cross-term normalization.
3. **Honest score** ``s_t`` — the leave-one-out fraction of a protein's *annotated*
   neighbourhood carrying the term (a protein's own seed is removed analytically so it
   never reads back its own annotation; proteins without any annotation are unknowns and
   count neither for nor against a term).
4. **Depth** ``a*_t`` chosen per term by leave-one-out average precision over the
   annotated proteins.
5. **Calibrate** ``s_t`` to a probability with cross-fitted isotonic regression. The
   default ``"size_aware"`` calibration pools across terms on a null-standardized
   (effective-sample-size) score, fit within log-size strata so it is size-conditional,
   then blended toward the per-term curve — this keeps small terms from being
   over-confident without under-calibrating large ones.
6. **Resolve** the probability vector into a call: ``"likelihood"`` (an MGSA-style
   active set in which a term explains the terms it is nested in, so broad terms are
   explained away by specific ones while co-localised, non-nested terms are both kept),
   ``"specific"`` (most-specific term above a threshold), or ``"argmax"``. The
   multi-location call is the active set's leaves.
"""

from __future__ import annotations
import itertools

from typing import Literal

import numpy as np
import pandas as pd
import scipy.sparse as sp

from anndata import AnnData
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import KFold

from ..util import MULTILOC_SEP, get_matrix
from ._annotation import assign_compartment_colors, require_kind, write_annotation
from ._graph import make_diffuser, neff_hutchinson, symmetric_normalized


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _resolve_gene_sets(gene_sets, species: str) -> dict[str, list[str]]:
    """Accept a ``{term: [genes]}`` dict, a ``.gmt`` path, or an Enrichr library name."""
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


# --------------------------------------------------------------------------- #
# calibration
# --------------------------------------------------------------------------- #
def _calibrate(
    OOF,
    Y01,
    valid,
    astar,
    denom_by_a,
    diffuse,
    n,
    terms,
    calibration,
    kappa,
    n_probe,
    cv_splits,
    random_state,
    annotated,
):
    """Map honest scores ``OOF`` to calibrated per-term probabilities. See module docstring.

    Every curve is fitted on annotated proteins only (``annotated``): an unannotated protein
    is an unknown, not a negative example, and a protein with a high score in a sparsely
    annotated region is usually a correct prediction rather than evidence against the score.
    The folds are still drawn over all proteins so every protein is predicted out of fold.
    """
    Pcal = np.zeros((n, len(terms)))
    kf = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    folds = [(tr[annotated[tr]], te) for tr, te in kf.split(np.arange(n))]
    mask = annotated.astype(float)

    def _per_term():
        P = np.zeros((n, len(terms)))
        for j in range(len(terms)):
            if not valid[j]:
                continue
            for tr, te in folds:
                iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
                iso.fit(OOF[tr, j], Y01[tr, j])
                P[te, j] = iso.predict(OOF[te, j])
        return P

    def _zscore():
        # per-alpha effective neighbourhood size n_eff_i (Kish, excluding self) via
        # Hutchinson estimates of diag(M_a) and diag(M_a^2); M_a symmetric so
        # mean_g (M_a g)_i^2 -> sum_j M_ij^2 and mean_g g_i (M_a g)_i -> M_ii. Probes are
        # restricted to annotated proteins, so n_eff counts annotated neighbours only.
        rng = np.random.RandomState(random_state)
        neff_by_a = {
            a: neff_hutchinson(diffuse, a, denom_by_a[a], n_probe=n_probe, rng=rng, mask=mask)
            for a in sorted(set(astar.values()))
        }
        Z = np.zeros((n, len(terms)))
        for j, t in enumerate(terms):
            if not valid[j]:
                continue
            p0 = Y01[annotated, j].mean()
            se = np.sqrt(p0 * (1 - p0) / np.maximum(neff_by_a[astar[t]], 1e-6))
            Z[:, j] = (OOF[:, j] - p0) / np.maximum(se, 1e-9)
        return Z

    def _pooled(Z):
        P = np.zeros((n, len(terms)))
        vj = np.where(valid)[0]
        for tr, te in folds:
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
            iso.fit(Z[np.ix_(tr, vj)].ravel(), Y01[np.ix_(tr, vj)].ravel())
            P[np.ix_(te, vj)] = iso.predict(Z[np.ix_(te, vj)].ravel()).reshape(
                len(te), len(vj)
            )
        return P

    def _size_aware(Z):
        vj = np.where(valid)[0]
        m = Y01.sum(axis=0)
        logm = np.log(m[vj])
        Q = int(np.clip(len(vj) // 8, 1, 6))
        qe = np.quantile(logm, np.linspace(0, 1, Q + 1))
        qe[0] -= 1e-6
        qe[-1] += 1e-6
        strat = np.clip(np.digitize(logm, qe) - 1, 0, Q - 1)
        # drop empty strata (ties in log-size collapse quantile bins for small vocabs)
        # and renumber so every stratum index 0..Q-1 has >= 1 term.
        present = sorted(set(strat.tolist()))
        remap = {s: i for i, s in enumerate(present)}
        strat = np.array([remap[s] for s in strat])
        Q = len(present)
        smed = np.array([np.median(logm[strat == s]) for s in range(Q)])
        order = np.argsort(smed)
        smed_s = smed[order]
        logm_full = np.log(np.maximum(m, 1))
        w = m / (m + kappa)
        P = np.zeros((n, len(terms)))
        for tr, te in folds:
            isos = {}
            for s in range(Q):
                tt = vj[strat == s]
                iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4)
                iso.fit(Z[np.ix_(tr, tt)].ravel(), Y01[np.ix_(tr, tt)].ravel())
                isos[s] = iso
            for j in vj:
                it = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
                it.fit(Z[tr, j], Y01[tr, j])
                p_term = it.predict(Z[te, j])
                x = logm_full[j]
                k = int(np.searchsorted(smed_s, x))
                if k <= 0:
                    p_sc = isos[order[0]].predict(Z[te, j])
                elif k >= Q:
                    p_sc = isos[order[-1]].predict(Z[te, j])
                else:
                    f = (x - smed_s[k - 1]) / (smed_s[k] - smed_s[k - 1] + 1e-9)
                    p_sc = (1 - f) * isos[order[k - 1]].predict(Z[te, j]) + f * isos[
                        order[k]
                    ].predict(Z[te, j])
                P[te, j] = w[j] * p_term + (1 - w[j]) * p_sc
        return P

    if calibration in ("none", None):
        Pcal = OOF.copy()
        Pcal[:, ~valid] = 0.0
    elif calibration == "per_term":
        Pcal = _per_term()
    elif calibration == "pooled":
        Pcal = _pooled(_zscore())
    elif calibration == "shrunk":
        w = Y01.sum(axis=0) / (Y01.sum(axis=0) + kappa)
        Pcal = w[None, :] * _per_term() + (1 - w[None, :]) * _pooled(_zscore())
    elif calibration == "size_aware":
        Pcal = _size_aware(_zscore())
    else:
        raise ValueError(
            "calibration must be 'size_aware', 'shrunk', 'pooled', 'per_term' or 'none'; "
            f"got {calibration!r}"
        )
    return Pcal


# --------------------------------------------------------------------------- #
# resolution
# --------------------------------------------------------------------------- #
def _nesting(cats, gene_sets, nest_threshold) -> np.ndarray:
    """``N[a, b]``: term ``a`` is nested in term ``b`` (containment ``>= nest_threshold``).

    Near-duplicates nest both ways; the smaller term (or, at equal size, the later one) is
    kept as the child so that one of them stays a leaf.
    """
    C = _containment(cats, gene_sets)
    size = np.array([len(set(gene_sets[t])) for t in cats])
    N = C >= nest_threshold
    np.fill_diagonal(N, False)
    for x, y in zip(*np.where(N & N.T)):
        if size[x] > size[y] or (size[x] == size[y] and x < y):
            N[x, y] = False
    return N


def _leaves(active, N, P_i, cats):
    """The active terms that contain no other active term, most probable first."""
    lv = [a for a in active if not any(N[b, a] for b in active if b != a)]
    return [cats[a] for a in sorted(lv, key=lambda a: -P_i[a])], lv


def _resolve(
    P,
    cats,
    gene_sets,
    sizes,
    mode,
    term_threshold,
    eta,
    tau,
    maxk,
    cap,
    min_term_size,
    nest_threshold,
):
    """Turn the per-term probability matrix into ``(labels, leaves)``.

    ``labels`` is the single call per protein (``None`` = no call); ``leaves`` the
    multi-location call: the called terms that contain no other called term, so parents
    are implied rather than listed and two non-nested terms mean co-localisation.

    ``sizes`` is the per-term number of members present in the map. ``min_term_size``
    restricts the labels a protein can be given to terms with at least that many members —
    an interpretable **granularity floor**: because the vocabulary is hierarchical, dropping
    an under-represented (small) term leaves its larger ancestors, which by containment still
    explain the neighbourhood, so the resolver gracefully backs off to the most-specific term
    that clears the floor rather than emitting a rare, poorly-supported label.

    In ``"likelihood"`` mode a term explains another only when it is **nested** in it
    (containment ``>= nest_threshold`` over the genes present in the map). Real
    vocabularies put nested pairs at 0.95-1.0 and merely co-annotated ones (e.g. nucleus /
    cytoplasm) at ``<= 0.77``, so co-localised terms no longer explain each other away and a
    dual-localised protein gets both. The single call is the leaf whose removal from the
    active set costs the most likelihood, i.e. the leaf explaining the most membership.
    """
    n = P.shape[0]
    sizes = np.asarray(sizes)
    eligible = sizes >= max(int(min_term_size), 1)
    labels = np.array([None] * n, dtype=object)
    leaves: list[list[str]] = [[] for _ in range(n)]
    if not eligible.any():
        return labels, leaves
    N = _nesting(cats, gene_sets, nest_threshold)
    tt = np.asarray(cats)
    if mode in ("argmax", "specific"):
        Pe = np.where(eligible[None, :], P, -1.0)
        for i in range(n):
            cand = np.where((P[i] >= term_threshold) & eligible)[0]
            leaves[i], lv = _leaves(cand, N, P[i], cats)
            if mode == "argmax":
                labels[i] = tt[Pe[i].argmax()] if Pe[i].max() > 0 else None
            else:
                labels[i] = cats[lv[np.argmin(sizes[lv])]] if len(lv) else None
        return labels, leaves
    if mode == "likelihood":
        C = _containment(cats, gene_sets)
        Cl = np.where(C >= nest_threshold, C, 0.0)  # containment counts only when nested
        elig_idx = np.where(eligible)[0]
        eps = 0.02

        def loglik(sub, cand, pobs):
            ph = np.clip(
                Cl[np.ix_(sub, cand)].max(0) if len(sub) else np.zeros(len(cand)),
                eps,
                1 - eps,
            )
            return np.sum(pobs * np.log(ph) + (1 - pobs) * np.log(1 - ph))

        for i in range(n):
            cand = np.where((P[i] >= tau) & eligible)[0]
            if len(cand) == 0:  # fall back to best eligible term
                cand = np.array([elig_idx[P[i, elig_idx].argmax()]])
            if len(cand) > cap:
                cand = cand[np.argsort(P[i, cand])[::-1][:cap]]
            pobs = P[i, cand]
            best, bA = -1e18, ()
            for A in [()] + [
                x
                for k in range(1, maxk + 1)
                for x in itertools.combinations(range(len(cand)), k)
            ]:
                v = loglik(cand[list(A)], cand, pobs) - eta * len(A)
                if v > best:
                    best, bA = v, A
            sel = cand[list(bA)]
            if not len(sel):
                continue
            leaves[i], lv = _leaves(sel, N, P[i], cats)
            full = loglik(sel, cand, pobs)
            gain = {
                a: full - loglik(np.array([b for b in sel if b != a]), cand, pobs) for a in lv
            }
            labels[i] = cats[max(lv, key=gain.get)]
        return labels, leaves
    raise ValueError(
        f"resolve must be 'likelihood', 'specific', 'argmax' or None; got {mode!r}"
    )


def _write_calls(adata, key, labels, leaves, out_key=None):
    """The contract slot ``obs[out_key]`` (so ``annotation_f1_score(pred_col=key)`` works)
    and ``obs[f"{out_key}_label_compact"]``, the leaves joined by the shared separator."""
    out_key = key if out_key is None else out_key
    adata.obs[out_key] = pd.Categorical(labels)
    adata.obs[f"{out_key}_label_compact"] = [
        MULTILOC_SEP.join(lv) if lv else None for lv in leaves
    ]
    assign_compartment_colors(adata, [out_key])


def resolve_diffusion(
    data: AnnData,
    gene_sets,
    key_added: str = "ann_diffusion",
    mode: Literal["likelihood", "specific", "argmax"] = "likelihood",
    min_term_size: int = 0,
    term_threshold: float = 0.5,
    eta: float = 1.0,
    tau: float = 0.4,
    maxk: int = 3,
    cap: int = 12,
    nest_threshold: float = 0.8,
    gene_key: str = "gene_symbol",
    species: str = "hsap",
    out_key: str | None = None,
    inplace: bool = True,
) -> AnnData | None:
    """(Re)resolve stored diffusion probabilities into a per-protein label, in place.

    Reads ``obsm[{key_added}_probabilities]`` (written by
    :func:`independent_diffusion`) and writes ``obs[out_key]`` and
    ``obs[f"{out_key}_label_compact"]``. ``out_key`` defaults to ``key_added`` for
    likelihood/argmax -- overwriting the primary call, since that is what re-resolving
    means -- and to ``f"{key_added}_specific"`` for specific, so an alternative resolution
    sits beside the default rather than replacing it. The resolver parameters are as in
    :func:`independent_diffusion`; ``gene_sets`` must be the vocabulary that was diffused
    (it defines the nesting). Re-resolving with the same parameters reproduces the stored
    call exactly.
    Lets you obtain several resolutions (e.g. likelihood *and* specific) or sweep
    ``min_term_size`` without re-diffusing. With ``inplace=False`` the labels are written
    to a copy, which is returned.
    """
    require_kind(data, key_added, "per_term")
    if not inplace:
        data = data.copy()
    gmt = _resolve_gene_sets(gene_sets, species)
    # The term names are columns of the stored matrix. The uns entry is only a fallback
    # for matrices that carry no column names: those written before grassp labelled them,
    # and those the R bridge demotes when a class name contains "/".
    P, columns = get_matrix(data, f"{key_added}_probabilities")
    P = np.asarray(P, dtype=float)
    cats = list(columns) if columns is not None else list(data.uns[f"{key_added}_categories"])
    pop = set(data.obs[gene_key].astype(str))
    # the same map-restricted sets independent_diffusion resolved with, so the nesting (and
    # therefore the call) is identical and re-resolving with the same parameters is a no-op
    seeds = {t: set(gmt.get(t, ())) & pop for t in cats}
    sizes = np.array([len(seeds[t]) for t in cats])
    labels, leaves = _resolve(
        P,
        cats,
        seeds,
        sizes,
        mode,
        term_threshold,
        eta,
        tau,
        maxk,
        cap,
        min_term_size,
        nest_threshold,
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
    gene_sets,
    gene_key: str = "gene_symbol",
    species: str = "hsap",
    obsp_key: str = "connectivities",
    alphas=None,
    calibration: Literal["size_aware", "shrunk", "pooled", "per_term", "none"] = "size_aware",
    kappa: float = 30.0,
    resolve: Literal["likelihood", "specific", "argmax"] | None = "likelihood",
    term_threshold: float = 0.5,
    min_term_size: int = 0,
    eta: float = 1.0,
    tau: float = 0.4,
    maxk: int = 3,
    cap: int = 12,
    nest_threshold: float = 0.8,
    cv_splits: int = 5,
    n_probe: int = 96,
    random_state: int = 0,
    key_added: str = "ann_diffusion",
    inplace: bool = True,
    verbose: bool = False,
) -> AnnData | None:
    """Annotate a map with overlapping / hierarchical labels by one-vs-rest diffusion.

    Each term in ``gene_sets`` is diffused independently over the kNN graph (no cross-term
    competition), giving a **non-simplex** per-term membership probability. See the module
    docstring for the full pipeline. For mutually-exclusive single labels (e.g. markers),
    use :func:`~grassp.tools.competitive_diffusion` instead.

    Parameters
    ----------
    data
        :class:`~anndata.AnnData` with a neighbour graph in ``obsp[obsp_key]``.
    gene_sets
        The label vocabulary: a ``{term: [gene_ids]}`` dict, a path to a ``.gmt`` file, or
        an Enrichr library name (fetched via ``gseapy``).
    gene_key
        ``obs`` column with the gene identifiers matching ``gene_sets`` (default
        ``"gene_symbol"``). When several entries share an identifier (isoforms, duplicate
        accessions), only the first carries the gene's annotation and the others are scored
        as unannotated proteins, so one gene is one unit of evidence and a protein cannot
        vouch for itself through a duplicate; the count is recorded in
        ``uns[{key_added}_params]["duplicate_gene_entries"]``.
    species
        Species tag passed to the GMT loader when ``gene_sets`` is a path/name.
    obsp_key
        Neighbour-graph key in ``obsp`` (default ``"connectivities"``).
    alphas
        Candidate diffusion depths; the per-term optimum ``a*`` is chosen by leave-one-out
        average precision. Default ``np.linspace(0.1, 0.9, 19)``.
    calibration
        Score→probability calibration (default ``"size_aware"``). See module docstring;
        ``"none"`` returns the raw honest score.
    kappa
        Support-weight prior for ``"size_aware"``/``"shrunk"`` blending (``w = m/(m+kappa)``).
    resolve
        How to turn the probability vector into ``obs[{key_added}]``:
        ``"likelihood"`` (default; containment-link active set), ``"specific"``
        (most-specific term with ``P >= term_threshold``), ``"argmax"``, or ``None`` to
        skip and only write probabilities.
    term_threshold
        Per-term membership threshold: a term is a candidate for a protein when its
        calibrated membership is at least this. Used by the ``"specific"`` resolver and
        for the compact multi-label set. Deliberately *not* called
        ``min_probability`` -- that name means "abstain below this confidence" everywhere
        else in grassp, which is a different question from "does this protein belong to
        this term".
    min_term_size
        Granularity floor: a protein may only be labelled with a term that has at least this
        many members present in the map. Because the vocabulary is hierarchical, an
        under-represented term is dropped in favour of its larger ancestor (which still
        explains the neighbourhood by containment), so raising it yields fewer, better-
        supported labels without abstaining. ``0`` (default) applies no floor.
    eta, tau, maxk, cap
        Likelihood-resolver parameters: term penalty, candidate floor, max active-set
        size, and candidate cap.
    nest_threshold
        Term ``a`` is nested in ``b`` when ``|G_a & G_b| / |G_a| >= nest_threshold`` over
        the genes present in the map; only nested terms explain each other in the
        ``"likelihood"`` resolver, and the multi-location call lists the called terms
        that contain no other called term. Real vocabularies put nested pairs at 0.95-1.0
        and co-annotated ones (e.g. nucleus / cytoplasm) at ``<= 0.77``.
    cv_splits, n_probe, random_state
        Cross-fit folds, Hutchinson probes for ``n_eff``, and RNG seed.
    key_added
        Prefix for the outputs (default ``"ann_diffusion"``).
    inplace
        If ``True`` (default) annotate ``data`` and return ``None``; otherwise operate on
        and return a copy. Named to match every other annotator -- this was the one that
        spelled it ``copy``, inverted.
    verbose
        Print progress.

    Returns
    -------
    Writes ``obsm[{key_added}_probabilities]`` (per-term calibrated membership, non-simplex,
    with the term names as its columns), ``uns[{key_added}_alpha]`` (per-term ``a*``),
    ``obs[{key_added}_probability]`` (top membership), ``uns[{key_added}_params]``
    (provenance, including ``kind="per_term"``), and — when ``resolve`` is set —
    ``obs[{key_added}]`` (the resolved call) and ``obs[{key_added}_label_compact]`` (the
    multi-location call: the called terms that contain no other called term, most
    probable first, joined by the shared separator).
    Returns the AnnData if ``inplace=False``, else ``None``.
    """
    adata = data if inplace else data.copy()
    if alphas is None:
        alphas = np.linspace(0.1, 0.9, 19)
    if obsp_key not in adata.obsp:
        raise KeyError(f"obsp[{obsp_key!r}] not found; run sc.pp.neighbors first.")
    if gene_key not in adata.obs:
        raise KeyError(f"obs[{gene_key!r}] not found.")

    n = adata.n_obs
    gsym = adata.obs[gene_key].astype(str).to_numpy()
    pop = set(gsym)
    gmt = _resolve_gene_sets(gene_sets, species)
    seeds = {t: (set(g) & pop) for t, g in gmt.items()}
    terms = list(seeds)
    if not terms:
        raise ValueError("gene_sets is empty after intersecting with the population.")

    Y0 = np.zeros((n, len(terms)))
    for j, t in enumerate(terms):
        Y0[np.isin(gsym, list(seeds[t])), j] = 1.0
    # One map entry per gene carries the gene's annotation. Several entries can share a
    # gene (isoforms, duplicate accessions) and then sit as each other's nearest
    # neighbours, so seeding both would let a protein vouch for itself through its twin
    # and would inflate member counts; the extra entries are treated as unannotated.
    duplicate = pd.Series(gsym).duplicated().to_numpy()
    n_duplicate = int((Y0[duplicate] > 0).any(axis=1).sum())
    Y0[duplicate] = 0.0
    Y01 = (Y0 > 0).astype(int)

    # Unannotated proteins are unknowns, not negatives: the reference mass is the diffused
    # mass of the *annotated* proteins, the self term is removed only where there is one,
    # and alpha* and the calibration are judged on annotated proteins only. Otherwise every
    # protein in a sparsely annotated region is penalised for neighbours that carry no label.
    annotated = Y01.any(axis=1)
    mask = annotated.astype(float)
    S = symmetric_normalized(adata.obsp[obsp_key])
    diffuse = make_diffuser(S)
    denom_by_a = {a: diffuse(mask, a) for a in alphas}

    def honest(y, a, den):
        F = diffuse(y, a)
        dd = den - (1 - a) * mask
        return np.divide(F - (1 - a) * y, dd, out=np.zeros_like(F), where=dd > 1e-9)

    # per-term depth a* by leave-one-out average precision; ties -> smaller alpha
    OOF = np.zeros((n, len(terms)))
    astar: dict[str, float] = {}
    valid = np.zeros(len(terms), dtype=bool)
    for j, t in enumerate(terms):
        yi = Y01[:, j]
        if yi.sum() < cv_splits:
            continue
        best_a, best_ap = None, -1.0
        for a in alphas:
            ap = average_precision_score(
                yi[annotated], honest(Y0[:, j], a, denom_by_a[a])[annotated]
            )
            if ap > best_ap + 1e-9:
                best_ap, best_a = ap, a
        astar[t] = best_a
        OOF[:, j] = honest(Y0[:, j], best_a, denom_by_a[best_a])
        valid[j] = True
    if not valid.any():
        raise ValueError(
            f"no term has >= cv_splits ({cv_splits}) members in the map; nothing to calibrate."
        )

    Pcal = _calibrate(
        OOF,
        Y01,
        valid,
        astar,
        denom_by_a,
        diffuse,
        n,
        terms,
        calibration,
        kappa,
        n_probe,
        cv_splits,
        random_state,
        annotated,
    )

    # derive_labels=False: an argmax over per-term memberships is not a call, because
    # the terms did not compete for one unit of mass. The resolver below decides, and
    # writes obs[key_added] itself.
    write_annotation(
        adata,
        key_added,
        Pcal,
        terms,
        kind="per_term",
        method="one-vs-rest-diffusion",
        derive_labels=False,
        gt_col=gene_key,
        extra_params={
            "term_threshold": term_threshold,
            "calibration": calibration,
            "resolve": resolve,
            "min_term_size": int(min_term_size),
            "obsp_key": obsp_key,
            "kappa": float(kappa),
            "random_state": int(random_state),
            "duplicate_gene_entries": n_duplicate,
        },
    )
    adata.uns[f"{key_added}_alpha"] = np.array(
        [astar.get(t, np.nan) for t in terms], dtype=float
    )
    maxp = np.asarray(adata.obs[f"{key_added}_probability"], dtype=float)

    sizes = Y01.sum(axis=0)  # per-term members (genes) present in the map
    if resolve is not None:
        labels, leaves = _resolve(
            Pcal,
            terms,
            seeds,
            sizes,
            resolve,
            term_threshold,
            eta,
            tau,
            maxk,
            cap,
            min_term_size,
            nest_threshold,
        )
        _write_calls(adata, key_added, labels, leaves)

    if verbose:
        av = [a for a in astar.values()]
        print(
            f"[independent_diffusion] {int(valid.sum())}/{len(terms)} terms scored; "
            f"a* {min(av):.2f}-{max(av):.2f}; calibration={calibration}; resolve={resolve}; "
            f"coverage(maxp>0)={float((maxp > 0).mean()):.3f}"
        )
    return None if inplace else adata
