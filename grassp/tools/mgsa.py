"""Model-based Gene Set Analysis (MGSA), a standalone Python reimplementation.

This module reimplements the Bayesian model of Bauer, Gagneur & Robinson,
*GOing Bayesian: model-based gene set analysis of genome-scale data*, Nucleic
Acids Research 38(11):3523-3532 (2010), following the semantics of the
Bioconductor ``mgsa`` package.

Unlike over-representation tests (hypergeometric / Fisher), MGSA explains the
whole set of observed ("study") genes jointly with a small number of *active*
gene sets. Overlapping or redundant sets therefore compete to explain the same
genes, so a set is only reported as active if it explains observations that a
competing set does not. This is the behaviour that lets MGSA decide whether,
e.g., "Lipid droplet" stays active alongside "Endoplasmic reticulum" or is
explained away by it.

Model
-----
Each gene set :math:`i` carries a hidden on/off indicator
:math:`T_i \\sim \\mathrm{Bernoulli}(p)`. A gene is *hidden-active*
(:math:`H_g = 1`) iff it belongs to at least one active set (a deterministic
OR). Observations :math:`O_g` are emitted with two error rates,

.. math::

    P(O_g = 1 \\mid H_g = 0) = \\alpha \\quad (\\text{false-positive rate}), \\\\
    P(O_g = 1 \\mid H_g = 1) = 1 - \\beta \\quad (\\beta = \\text{false-negative rate}).

With contingency counts over the population (first digit = observation, second
= hidden state) ``n11`` (TP), ``n10`` (FP), ``n01`` (FN), ``n00`` (TN), the
data log-likelihood plus the set prior is

.. math::

    \\log P(o \\mid T, \\alpha, \\beta, p) =
        n_{10}\\log\\alpha + n_{00}\\log(1-\\alpha)
      + n_{11}\\log(1-\\beta) + n_{01}\\log\\beta
      + n_a\\log p + n_i\\log(1-p),

where :math:`n_a`/:math:`n_i` are the numbers of active/inactive sets. Uniform
priors are placed on :math:`\\alpha, \\beta, p`, each discretised over a grid.

Inference
---------
A single Metropolis-Hastings chain samples jointly over the set states and the
discrete :math:`(\\alpha, \\beta, p)` grid values. Set-state moves are either a
single set flip or an exchange (deactivate one active + activate one inactive);
parameter moves resample one of :math:`\\alpha, \\beta, p` from its grid. The
contingency counts are updated incrementally on each set toggle (O(set size)),
never recomputed from scratch. The marginal posterior
:math:`P(T_i = 1 \\mid o)` is the fraction of recorded samples in which set
:math:`i` is active, averaged over independent restarts.

The inner sampler is JIT-compiled with :mod:`numba` when it is importable and
falls back to an identical pure-Python implementation otherwise. numba is
strongly recommended: it is ~200x faster than the fallback (the fallback exists
for correctness/portability, not production use).
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterable, Literal, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd

from .enrichment import _load_gmt

if TYPE_CHECKING:
    from anndata import AnnData

try:  # optional acceleration; the un-jitted functions are identical in behaviour
    from numba import njit as _njit

    _HAVE_NUMBA = True
except Exception:  # pragma: no cover - exercised only when numba is absent

    def _njit(func):
        return func

    _HAVE_NUMBA = False


__all__ = [
    "mgsa",
    "load_gmt",
    "MgsaResult",
    "calculate_mgsa",
    "mgsa_to_cluster_distribution",
]


# --------------------------------------------------------------------------- #
# Gene-set loading
# --------------------------------------------------------------------------- #
def load_gmt(
    source: Union[str, Mapping[str, Sequence[str]], None] = None,
    species: str = "hsap",
    deduplicate_terms: bool = True,
) -> dict[str, list[str]]:
    """Resolve a gene-set source into a ``{term: [gene, ...]}`` dict.

    Parameters
    ----------
    source
        One of:

        - ``dict`` / mapping — returned as a plain ``{name: list(genes)}`` dict.
        - existing file path — parsed as GMT (tab-separated; column 0 = term
          name, column 1 = description/source and is discarded, columns 2+ =
          gene identifiers).
        - ``None`` or a non-path string — delegated to
          :func:`grassp.tools.enrichment._load_gmt` (uses the consolidated
          UniProt subcellular-location sets bundled with grassp, or a gseapy
          library name), if grassp is importable.
    species
        Passed through to grassp when ``source`` is ``None``; one of
        ``"hsap"``, ``"mmus"``, ``"scer"``.
    deduplicate_terms
        If ``True`` (default), collapse terms with identical gene membership
        (keeping the first-seen name); see
        :func:`grassp.tools.enrichment._deduplicate_gene_sets`.

    Returns
    -------
    dict[str, list[str]]
        Mapping of term name to list of gene identifiers.
    """
    if isinstance(source, Mapping):
        source = {str(k): list(v) for k, v in source.items()}
    # Delegate dict / GMT-path / gseapy-library / None -> bundled-default resolution
    # to grassp's single loader so GMT parsing and dedup are not duplicated.
    return _load_gmt(source, species=species, deduplicate_terms=deduplicate_terms)


# --------------------------------------------------------------------------- #
# JIT core (numba-accelerated when available)
# --------------------------------------------------------------------------- #
# ``counts`` layout: [n11, n10, n01, n00, n_active, n_inactive]
@_njit
def _toggle(i, active, hidden_count, observed, members_flat, set_ptr, partition,
            pos_of_set, counts):
    """Flip set ``i`` on<->off, updating the partition and contingency counts."""
    if active[i] == 0:
        # activate: move i from the inactive region into the active region
        p = pos_of_set[i]
        j = counts[5] - 1  # last inactive slot
        sj = partition[j]
        partition[j] = i
        partition[p] = sj
        pos_of_set[i] = j
        pos_of_set[sj] = p
        counts[5] -= 1
        counts[4] += 1
        active[i] = 1
        for t in range(set_ptr[i], set_ptr[i + 1]):
            g = members_flat[t]
            hidden_count[g] += 1
            if hidden_count[g] == 1:  # 0 -> 1 transition
                if observed[g]:
                    counts[0] += 1  # n11++
                    counts[1] -= 1  # n10--
                else:
                    counts[2] += 1  # n01++
                    counts[3] -= 1  # n00--
    else:
        # deactivate: move i from the active region into the inactive region
        p = pos_of_set[i]
        j = counts[5]  # first active slot
        sj = partition[j]
        partition[j] = i
        partition[p] = sj
        pos_of_set[i] = j
        pos_of_set[sj] = p
        counts[5] += 1
        counts[4] -= 1
        active[i] = 0
        for t in range(set_ptr[i], set_ptr[i + 1]):
            g = members_flat[t]
            hidden_count[g] -= 1
            if hidden_count[g] == 0:  # 1 -> 0 transition
                if observed[g]:
                    counts[0] -= 1  # n11--
                    counts[1] += 1  # n10++
                else:
                    counts[2] -= 1  # n01--
                    counts[3] += 1  # n00++


@_njit
def _score(counts, ai, bi, pi, la, lma, lb, lmb, lp, lmp, lpa, lpb, lpp):
    """Joint log-score (data log-likelihood + set prior + parameter priors)."""
    return (
        counts[1] * la[ai]
        + counts[3] * lma[ai]
        + counts[0] * lmb[bi]
        + counts[2] * lb[bi]
        + counts[4] * lp[pi]
        + counts[5] * lmp[pi]
        + lpa[ai]
        + lpb[bi]
        + lpp[pi]
    )


@_njit
def _chain_core(n_sets, N, members_flat, set_ptr, observed, la, lma, lb, lmb,
                lp, lmp, lpa, lpb, lpp, n_alpha, n_beta, n_p, n_steps, burn_in, thin,
                flip_freq, seed):
    """Run one MCMC restart; return activity/parameter histograms and MAP state."""
    np.random.seed(seed)

    active = np.zeros(n_sets, dtype=np.int8)
    hidden_count = np.zeros(N, dtype=np.int64)
    partition = np.arange(n_sets).astype(np.int64)
    pos_of_set = np.arange(n_sets).astype(np.int64)

    counts = np.zeros(6, dtype=np.int64)
    lo = 0
    for g in range(N):
        if observed[g]:
            lo += 1
    counts[1] = lo          # n10: every observed gene starts as a false positive
    counts[3] = N - lo      # n00: every other gene starts as a true negative
    counts[5] = n_sets      # n_inactive

    ai = int(np.random.random() * n_alpha)
    bi = int(np.random.random() * n_beta)
    pi = int(np.random.random() * n_p)
    cur_score = _score(counts, ai, bi, pi, la, lma, lb, lmb, lp, lmp, lpa, lpb, lpp)

    activity = np.zeros(n_sets, dtype=np.int64)
    ah = np.zeros(n_alpha, dtype=np.int64)
    bh = np.zeros(n_beta, dtype=np.int64)
    ph = np.zeros(n_p, dtype=np.int64)
    nsamples = 0
    n_accept = 0

    map_score = -1.0e300
    map_active = np.zeros(n_sets, dtype=np.int8)
    map_ai = ai
    map_bi = bi
    map_pi = pi

    for step in range(n_steps):
        old_score = cur_score
        old_ni = counts[5]
        old_na = counts[4]
        old_nbhd = n_sets + old_ni * old_na

        if np.random.random() < flip_freq:
            # ------- set-state move -------
            idx = int(np.random.random() * old_nbhd)
            if idx < n_sets:
                _toggle(idx, active, hidden_count, observed, members_flat,
                        set_ptr, partition, pos_of_set, counts)
                mv = 0
                s1 = idx
                s2 = -1
            else:
                r = idx - n_sets
                ik = r // old_na          # which inactive set to activate
                ak = r % old_na           # which active set to deactivate
                s_add = partition[ik]
                s_rem = partition[old_ni + ak]
                _toggle(s_add, active, hidden_count, observed, members_flat,
                        set_ptr, partition, pos_of_set, counts)
                _toggle(s_rem, active, hidden_count, observed, members_flat,
                        set_ptr, partition, pos_of_set, counts)
                mv = 1
                s1 = s_add
                s2 = s_rem

            new_nbhd = n_sets + counts[5] * counts[4]
            new_score = _score(counts, ai, bi, pi, la, lma, lb, lmb, lp, lmp, lpa, lpb, lpp)
            log_acc = new_score - old_score + math.log(old_nbhd) - math.log(new_nbhd)
            if log_acc >= 0.0 or math.log(np.random.random()) < log_acc:
                cur_score = new_score
                n_accept += 1
            else:  # undo (toggling is its own inverse)
                _toggle(s1, active, hidden_count, observed, members_flat,
                        set_ptr, partition, pos_of_set, counts)
                if mv == 1:
                    _toggle(s2, active, hidden_count, observed, members_flat,
                            set_ptr, partition, pos_of_set, counts)
        else:
            # ------- parameter move -------
            w = int(np.random.random() * 3)
            old_idx = 0
            if w == 0:
                old_idx = ai
                ai = int(np.random.random() * n_alpha)
            elif w == 1:
                old_idx = bi
                bi = int(np.random.random() * n_beta)
            else:
                old_idx = pi
                pi = int(np.random.random() * n_p)

            new_score = _score(counts, ai, bi, pi, la, lma, lb, lmb, lp, lmp, lpa, lpb, lpp)
            log_acc = new_score - old_score  # neighbourhood unchanged
            if log_acc >= 0.0 or math.log(np.random.random()) < log_acc:
                cur_score = new_score
                n_accept += 1
            else:
                if w == 0:
                    ai = old_idx
                elif w == 1:
                    bi = old_idx
                else:
                    pi = old_idx

        # ------- record -------
        if step >= burn_in and (step % thin) == 0:
            nsamples += 1
            ni = counts[5]
            for k in range(ni, n_sets):
                activity[partition[k]] += 1
            ah[ai] += 1
            bh[bi] += 1
            ph[pi] += 1
            if cur_score > map_score:
                map_score = cur_score
                for s in range(n_sets):
                    map_active[s] = active[s]
                map_ai = ai
                map_bi = bi
                map_pi = pi

    return (activity, ah, bh, ph, nsamples, n_accept, map_score, map_active,
            map_ai, map_bi, map_pi)


# --------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------- #
@dataclass
class MgsaResult:
    """Container for MGSA posterior summaries and diagnostics.

    Attributes
    ----------
    sets_results
        Per-set report indexed by set name with columns ``inPopulation``,
        ``inStudySet``, ``estimate`` (posterior activity probability, mean over
        restarts) and ``std_error`` (standard deviation over restarts).
    alpha_post, beta_post, p_post
        Posterior over each parameter grid with columns ``value``,
        ``estimate`` and ``std_error``.
    map_estimate
        The maximum-a-posteriori configuration: ``{"sets": [names],
        "alpha": float, "beta": float, "p": float, "log_score": float}``.
    diagnostics
        Sampler settings and MCMC diagnostics (acceptance rates, sample counts,
        seeds, population/study sizes, per-restart marginals, numba flag).
    """

    sets_results: pd.DataFrame
    alpha_post: pd.DataFrame
    beta_post: pd.DataFrame
    p_post: pd.DataFrame
    map_estimate: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        top = self.sets_results.sort_values("estimate", ascending=False).head(5)
        lines = ["MgsaResult(", "  top sets by posterior activity:"]
        for name, row in top.iterrows():
            lines.append(f"    {name:<28s} {row['estimate']:.3f}")
        d = self.diagnostics
        lines.append(
            f"  population={d.get('population_size')}, "
            f"study_in_population={d.get('study_set_size_in_population')}, "
            f"restarts={d.get('n_restarts')}, steps={d.get('n_steps')}"
        )
        lines.append(")")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def _default_grids(n_sets: int):
    alpha_grid = np.linspace(0.01, 0.3, 10)
    # beta (false-negative rate) ceiling raised to 0.95 so that a small cluster can
    # still activate a large, diffuse compartment (which implies a high FN rate)
    # rather than being forced to the empty MAP. This diverges from the Bioconductor
    # default (0.8); pair it with `beta_prior` if you want to discourage high beta.
    beta_grid = np.linspace(0.1, 0.95, 10)
    top = min(20, max(1, n_sets // 3))
    p_grid = np.linspace(1, top, 10) / n_sets
    return alpha_grid, beta_grid, p_grid


def _log_prior_weights(grid: np.ndarray, prior) -> np.ndarray:
    """Log prior weights over a parameter grid.

    ``prior`` is ``None`` (uniform -> zeros, no effect) or a callable mapping the
    grid values to non-negative (unnormalised) weights, e.g. a decreasing
    ``lambda b: (1 - b) ** k`` to penalise high beta. Normalisation is irrelevant
    (it cancels in the marginalisation and in MCMC acceptance), so the raw log is
    returned; a zero weight maps to ``-inf`` (that grid point is excluded).
    """
    if prior is None:
        return np.zeros(grid.size)
    w = np.asarray(prior(grid), dtype=np.float64)
    if w.shape != grid.shape:
        raise ValueError("prior(grid) must return one weight per grid value.")
    if np.any(w < 0):
        raise ValueError("prior weights must be non-negative.")
    with np.errstate(divide="ignore"):
        return np.log(w)


def _n_configs(m: int, kmax: int) -> int:
    """Number of set configurations with at most ``kmax`` active among ``m``."""
    return int(sum(math.comb(m, j) for j in range(min(kmax, m) + 1)))


def _lse(vals: np.ndarray) -> float:
    mx = float(vals.max())
    return mx + float(np.log(np.exp(vals - mx).sum()))


def _exact_solve(cand_masks, o_mask, N, n_obs, n_total, alpha_grid, beta_grid,
                 p_grid, max_active, lprior_a=None, lprior_b=None, lprior_p=None):
    """Exact marginal posteriors by enumerating configurations with <= ``max_active``
    active candidate sets, with ``alpha``/``beta``/``p`` integrated analytically.

    The likelihood depends on a configuration only through the size of the active
    gene union ``a`` and its overlap with the study set ``b``; ``alpha``, ``beta``
    and ``p`` factorise and are summed over their grids in closed form. Returns
    per-candidate marginals plus exact parameter posteriors and the MAP state.
    """
    m = len(cand_masks)
    eps = 1e-12
    ag = np.clip(alpha_grid, eps, 1 - eps)
    bg = np.clip(beta_grid, eps, 1 - eps)
    pg = np.clip(p_grid, eps, 1 - eps)
    la, lma = np.log(ag), np.log1p(-ag)
    lb, lmb = np.log(bg), np.log1p(-bg)
    lp, lmp = np.log(pg), np.log1p(-pg)
    # optional (log) priors on alpha/beta/p grids; zeros => uniform
    lpa = np.zeros(ag.size) if lprior_a is None else np.asarray(lprior_a, float)
    lpb = np.zeros(bg.size) if lprior_b is None else np.asarray(lprior_b, float)
    lpp = np.zeros(pg.size) if lprior_p is None else np.asarray(lprior_p, float)

    # p-factor depends only on the number of active sets (0..max_active)
    fp = np.empty(max_active + 1)
    psoft = []
    for na in range(max_active + 1):
        pv = na * lp + (n_total - na) * lmp + lpp
        fp[na] = _lse(pv)
        psoft.append(np.exp(pv - fp[na]))

    # cache the (alpha, beta) factor by the sufficient stats (a, b)
    ab_cache: dict = {}

    def ab_factor(a, b):
        r = ab_cache.get((a, b))
        if r is not None:
            return r
        n11, n10, n01 = b, n_obs - b, a - b
        n00 = N - a - n10
        av = n10 * la + n00 * lma + lpa
        bv = n11 * lmb + n01 * lb + lpb
        fa, fb = _lse(av), _lse(bv)
        r = (fa, fb, np.exp(av - fa), np.exp(bv - fb))
        ab_cache[(a, b)] = r
        return r

    # pass 1: find the maximum log-weight (for a numerically safe shift)
    best = [-np.inf]

    def dfs1(start, depth, um):
        a = um.bit_count()
        fa, fb, _, _ = ab_factor(a, (um & o_mask).bit_count())
        lw = fa + fb + fp[depth]
        if lw > best[0]:
            best[0] = lw
        if depth < max_active:
            for i in range(start, m):
                dfs1(i + 1, depth + 1, um | cand_masks[i])

    dfs1(0, 0, 0)
    M = best[0]

    # pass 2: accumulate normaliser and per-set marginals in the hot loop.
    # Parameter posteriors depend only on (a, b) / depth, so their weights are
    # accumulated in scalar buckets and turned into grids once at the end -- this
    # keeps the per-configuration work out of numpy.
    num = np.zeros(m)
    w_ab: dict = {}
    w_depth = np.zeros(max_active + 1)
    combo: list[int] = []
    Z = 0.0
    map_state = {"lw": -np.inf, "combo": (), "a": 0, "b": 0, "depth": 0}

    def dfs2(start, depth, um):
        nonlocal Z
        a = um.bit_count()
        b = (um & o_mask).bit_count()
        key = (a, b)
        fa_fb = ab_scalar.get(key)
        if fa_fb is None:
            fa, fb, _, _ = ab_factor(a, b)
            fa_fb = fa + fb
            ab_scalar[key] = fa_fb
        lw = fa_fb + fp[depth]
        w = math.exp(lw - M)
        Z += w
        for i in combo:
            num[i] += w
        w_ab[key] = w_ab.get(key, 0.0) + w
        w_depth[depth] += w
        if lw > map_state["lw"]:
            map_state.update(lw=lw, combo=tuple(combo), a=a, b=b, depth=depth)
        if depth < max_active:
            for i in range(start, m):
                combo.append(i)
                dfs2(i + 1, depth + 1, um | cand_masks[i])
                combo.pop()

    ab_scalar: dict = {}
    dfs2(0, 0, 0)

    apost = np.zeros(alpha_grid.size)
    bpost = np.zeros(beta_grid.size)
    ppost = np.zeros(p_grid.size)
    for (a, b), w in w_ab.items():
        _, _, asoft, bsoft = ab_factor(a, b)
        apost += w * asoft
        bpost += w * bsoft
    for depth, w in enumerate(w_depth):
        ppost += w * psoft[depth]

    marg = num / Z
    apost /= Z
    bpost /= Z
    ppost /= Z
    # log marginal likelihood (evidence): log sum_configs exp(lw) = M + log(Z),
    # integrated over the alpha/beta/p grids and configs with <= max_active active
    # sets. Suitable for Bayesian model comparison between clusterings.
    log_evidence = M + math.log(Z) if Z > 0 else -np.inf
    # null evidence: the empty configuration (no active set) alone. logE - log_null
    # is a background-cancelling log-Bayes-factor "structure vs noise", the quantity
    # to difference across a partition (a merge score of BF(union) - BF(c1) - BF(c2)).
    fa0, fb0, _, _ = ab_factor(0, 0)
    log_null = fa0 + fb0 + fp[0]
    return marg, apost, bpost, ppost, map_state, len(ab_cache), log_evidence, log_null


def mgsa(
    o: Iterable[str],
    sets: Union[Mapping[str, Sequence[str]], str, None],
    population: Optional[Iterable[str]] = None,
    *,
    alpha_grid: Optional[Sequence[float]] = None,
    beta_grid: Optional[Sequence[float]] = None,
    p_grid: Optional[Sequence[float]] = None,
    alpha_prior: Optional[callable] = None,
    beta_prior: Optional[callable] = None,
    p_prior: Optional[callable] = None,
    method: str = "auto",
    max_active: int = 4,
    exact_max_configs: int = 2_000_000,
    n_steps: int = 1_000_000,
    n_restarts: int = 1,
    burn_in: Optional[int] = None,
    thin: int = 100,
    flip_freq: float = 0.8,
    species: str = "hsap",
    deduplicate_terms: bool = True,
    seed: Optional[int] = None,
) -> MgsaResult:
    """Run model-based gene set analysis.

    Parameters
    ----------
    o
        The study set: identifiers of the observed ("active") genes.
    sets
        Gene sets as a ``{name: [genes]}`` mapping, or anything accepted by
        :func:`load_gmt` (a GMT path, gseapy library name, or ``None`` for the
        grassp-bundled sets).
    population
        The universe of gene identifiers. If ``None``, the union of all set
        members is used. Study genes outside the population are dropped (and the
        count is logged).
    alpha_grid, beta_grid, p_grid
        Discrete grids for the false-positive rate, false-negative rate and
        set-activation prior. Defaults: ``alpha = linspace(0.01, 0.3, 10)``,
        ``beta = linspace(0.1, 0.95, 10)`` and
        ``p = linspace(1, min(20, floor(n_sets/3)), 10) / n_sets``. The beta
        ceiling (0.95) is raised from the Bioconductor default (0.8) so a small
        cluster can activate a large, sparsely-covered compartment.
    alpha_prior, beta_prior, p_prior
        Optional priors over the corresponding grids. ``None`` (default) is a
        uniform prior (matching Bioconductor). Otherwise a callable mapping the
        grid values to non-negative (unnormalised) weights — e.g.
        ``beta_prior=lambda b: (1 - b) ** k`` to *penalise* high false-negative
        rates while still allowing them when the likelihood (enough observed
        hits) outweighs the penalty. Applied to both the exact and MCMC paths.
    method
        Inference method. ``"exact"`` enumerates all configurations with at most
        ``max_active`` active sets and integrates ``alpha``/``beta``/``p``
        analytically, giving variance-free marginals (see Notes). ``"mcmc"``
        runs the Metropolis-Hastings sampler mirroring the R package. ``"auto"``
        (default) uses ``"exact"`` when the enumeration is feasible
        (candidate sets pruned to those overlapping the study set; config count
        <= ``exact_max_configs``) and falls back to ``"mcmc"`` otherwise.
    max_active
        Exact method: maximum number of simultaneously active sets to enumerate.
        The posterior is sparse (the ``p`` prior penalises many active sets), so
        a small cap is effectively exact; raise it until marginals stop changing.
    exact_max_configs
        Exact method: if the number of configurations would exceed this, ``auto``
        falls back to MCMC (and ``method="exact"`` raises).
    n_steps
        MCMC steps per restart.
    n_restarts
        Number of independent chains; posterior estimates are averaged and the
        standard error is their spread. Use >= 5 for meaningful standard errors.
    burn_in
        Steps discarded before recording. Defaults to ``n_steps // 2``.
    thin
        Record one sample every ``thin`` steps after burn-in.
    flip_freq
        Probability of proposing a set-state move (vs. a parameter move).
    species
        Forwarded to :func:`load_gmt` when ``sets`` needs resolving.
    deduplicate_terms
        Forwarded to :func:`load_gmt`; if ``True`` (default) sets with identical
        gene membership are collapsed to a single term before analysis, so
        synonymous/duplicate sets do not split the posterior mass.
    seed
        MCMC base RNG seed. Restart ``r`` uses ``seed + r``, making the whole run
        reproducible. If ``None``, non-deterministic seeds are drawn.

    Returns
    -------
    MgsaResult
        Posterior summaries for the sets and for ``alpha``/``beta``/``p``, the
        MAP configuration, and diagnostics.

    Notes
    -----
    The exact method enumerates every configuration with up to ``max_active``
    active sets (over all sets) and integrates the parameters analytically; it is
    exact up to the ``max_active`` cap, and ``std_error`` is 0 (no Monte-Carlo
    noise). It is feasible for modest set counts (``C(n_sets, <= max_active)``
    configurations); for large collections ``auto`` falls back to MCMC.

    The MCMC RNG stream differs from the reference C implementation (a Mersenne
    Twister with a different call order), so MCMC results agree with the
    Bioconductor package only up to Monte-Carlo error, not bit-for-bit.
    """
    sets = load_gmt(sets, species=species, deduplicate_terms=deduplicate_terms)
    set_names = list(sets.keys())
    n_sets = len(set_names)
    if n_sets == 0:
        raise ValueError("`sets` is empty.")

    # ---- build the population index ----
    if population is None:
        pop = sorted({g for genes in sets.values() for g in genes})
    else:
        pop = list(dict.fromkeys(population))  # de-dup, keep order
    gene_index = {g: i for i, g in enumerate(pop)}
    N = len(pop)
    if N == 0:
        raise ValueError("Empty population.")

    # ---- observations ----
    o = list(o)
    observed = np.zeros(N, dtype=np.int8)
    n_obs_in_pop = 0
    for g in o:
        idx = gene_index.get(g)
        if idx is not None:
            if observed[idx] == 0:
                n_obs_in_pop += 1
            observed[idx] = 1
    n_obs_dropped = len(set(o)) - n_obs_in_pop
    if n_obs_dropped > 0:
        warnings.warn(
            f"mgsa: {n_obs_dropped} of {len(set(o))} study genes are not in the "
            f"population and were dropped."
        )

    # ---- set membership (restricted to the population, de-duplicated) ----
    members_lists = []
    in_population = np.zeros(n_sets, dtype=np.int64)
    in_study = np.zeros(n_sets, dtype=np.int64)
    for si, name in enumerate(set_names):
        seen = set()
        members = []
        for g in sets[name]:
            idx = gene_index.get(g)
            if idx is not None and idx not in seen:
                seen.add(idx)
                members.append(idx)
        members_lists.append(members)
        in_population[si] = len(members)
        in_study[si] = sum(1 for m in members if observed[m])

    # ---- grids ----
    ag, bg, pg = _default_grids(n_sets)
    alpha_grid = np.asarray(alpha_grid if alpha_grid is not None else ag, dtype=np.float64)
    beta_grid = np.asarray(beta_grid if beta_grid is not None else bg, dtype=np.float64)
    p_grid = np.asarray(p_grid if p_grid is not None else pg, dtype=np.float64)

    # ---- optional priors on alpha/beta/p (None => uniform) ----
    lprior_a = _log_prior_weights(alpha_grid, alpha_prior)
    lprior_b = _log_prior_weights(beta_grid, beta_prior)
    lprior_p = _log_prior_weights(p_grid, p_prior)

    # ---- choose inference method ----
    # Exact enumeration is over ALL sets: a set with no observed members can still
    # carry ~prior posterior mass when it is redundant with an active set (its
    # genes are already explained), so it cannot be pruned without error.
    cand_idx = list(range(n_sets))
    n_cfg = _n_configs(n_sets, max_active)
    if method == "auto":
        method = "exact" if n_cfg <= exact_max_configs else "mcmc"
    elif method == "exact" and n_cfg > exact_max_configs:
        raise ValueError(
            f"Exact enumeration needs {n_cfg} configurations "
            f"({len(cand_idx)} candidate sets, max_active={max_active}), exceeding "
            f"exact_max_configs={exact_max_configs}. Lower max_active or use "
            f"method='mcmc'."
        )
    elif method not in ("exact", "mcmc"):
        raise ValueError(f"Unknown method {method!r}; use 'auto', 'exact' or 'mcmc'.")

    common_diag = {
        "population_size": N,
        "study_set_size_in_population": int(n_obs_in_pop),
        "study_genes_dropped": int(n_obs_dropped),
        "n_sets": n_sets,
        "method": method,
    }

    # ---- exact enumeration branch ----
    if method == "exact":
        o_mask = 0
        for i in np.flatnonzero(observed):
            o_mask |= 1 << int(i)
        cand_masks = []
        for si in cand_idx:
            mask = 0
            for idx in members_lists[si]:
                mask |= 1 << idx
            cand_masks.append(mask)

        marg_c, apost, bpost, ppost, map_state, n_ab, log_evidence, log_null = _exact_solve(
            cand_masks, o_mask, N, int(n_obs_in_pop), n_sets,
            alpha_grid, beta_grid, p_grid, max_active,
            lprior_a, lprior_b, lprior_p,
        )
        estimate = np.zeros(n_sets)
        for j, si in enumerate(cand_idx):
            estimate[si] = marg_c[j]

        sets_results = pd.DataFrame(
            {
                "inPopulation": in_population,
                "inStudySet": in_study,
                "estimate": estimate,
                "std_error": np.zeros(n_sets),
            },
            index=set_names,
        )
        alpha_post = pd.DataFrame({"value": alpha_grid, "estimate": apost,
                                   "std_error": np.zeros(alpha_grid.size)})
        beta_post = pd.DataFrame({"value": beta_grid, "estimate": bpost,
                                  "std_error": np.zeros(beta_grid.size)})
        p_post = pd.DataFrame({"value": p_grid, "estimate": ppost,
                               "std_error": np.zeros(p_grid.size)})

        eps = 1e-12
        ac, bc, pc = (np.clip(alpha_grid, eps, 1 - eps), np.clip(beta_grid, eps, 1 - eps),
                      np.clip(p_grid, eps, 1 - eps))
        av = (n_obs_in_pop - map_state["b"]) * np.log(ac) + \
            (N - map_state["a"] - (n_obs_in_pop - map_state["b"])) * np.log1p(-ac) + lprior_a
        bv = map_state["b"] * np.log1p(-bc) + \
            (map_state["a"] - map_state["b"]) * np.log(bc) + lprior_b
        pv = map_state["depth"] * np.log(pc) + \
            (n_sets - map_state["depth"]) * np.log1p(-pc) + lprior_p
        map_estimate = {
            "sets": [set_names[cand_idx[i]] for i in map_state["combo"]],
            "alpha": float(alpha_grid[int(av.argmax())]),
            "beta": float(beta_grid[int(bv.argmax())]),
            "p": float(p_grid[int(pv.argmax())]),
            "log_score": float(map_state["lw"]),
        }
        diagnostics = {
            **common_diag,
            "max_active": int(max_active),
            "n_configs": n_cfg,
            "n_distinct_ab": n_ab,
            "log_evidence": float(log_evidence),
            "log_null": float(log_null),
            "log_bayes_factor": float(log_evidence - log_null),
        }
        return MgsaResult(
            sets_results=sets_results,
            alpha_post=alpha_post,
            beta_post=beta_post,
            p_post=p_post,
            map_estimate=map_estimate,
            diagnostics=diagnostics,
        )

    # ---- MCMC branch ----
    set_ptr = np.zeros(n_sets + 1, dtype=np.int64)
    for si in range(n_sets):
        set_ptr[si + 1] = set_ptr[si] + len(members_lists[si])
    members_flat = np.empty(int(set_ptr[-1]), dtype=np.int64)
    for si in range(n_sets):
        if members_lists[si]:
            members_flat[set_ptr[si]:set_ptr[si + 1]] = members_lists[si]

    eps = 1e-12
    la = np.log(np.clip(alpha_grid, eps, 1 - eps))
    lma = np.log(np.clip(1 - alpha_grid, eps, 1 - eps))
    lb = np.log(np.clip(beta_grid, eps, 1 - eps))
    lmb = np.log(np.clip(1 - beta_grid, eps, 1 - eps))
    lp = np.log(np.clip(p_grid, eps, 1 - eps))
    lmp = np.log(np.clip(1 - p_grid, eps, 1 - eps))
    # replace -inf (zero-weight grid points) with a large negative so numba math is finite
    lpa = np.where(np.isfinite(lprior_a), lprior_a, -1e300)
    lpb = np.where(np.isfinite(lprior_b), lprior_b, -1e300)
    lpp = np.where(np.isfinite(lprior_p), lprior_p, -1e300)

    n_alpha = alpha_grid.size
    n_beta = beta_grid.size
    n_p = p_grid.size

    if burn_in is None:
        burn_in = n_steps // 2
    burn_in = int(burn_in)

    # ---- seeds ----
    if seed is None:
        base_rng = np.random.default_rng()
        seeds = [int(x) for x in base_rng.integers(0, 2**31 - 1, size=n_restarts)]
    else:
        seeds = [int(seed) + r for r in range(n_restarts)]

    # ---- run restarts ----
    sets_marg = np.zeros((n_sets, n_restarts))
    alpha_marg = np.zeros((n_alpha, n_restarts))
    beta_marg = np.zeros((n_beta, n_restarts))
    p_marg = np.zeros((n_p, n_restarts))
    accept_rates = np.zeros(n_restarts)
    nsamples_per = np.zeros(n_restarts, dtype=np.int64)

    best_map_score = -np.inf
    best_map = None

    for r in range(n_restarts):
        (activity, ah, bh, ph, nsamples, n_accept, map_score, map_active,
         map_ai, map_bi, map_pi) = _chain_core(
            n_sets, N, members_flat, set_ptr, observed, la, lma, lb, lmb, lp,
            lmp, lpa, lpb, lpp, n_alpha, n_beta, n_p, int(n_steps), burn_in, int(thin),
            float(flip_freq), int(seeds[r]),
        )
        ns = max(nsamples, 1)
        sets_marg[:, r] = activity / ns
        alpha_marg[:, r] = ah / ns
        beta_marg[:, r] = bh / ns
        p_marg[:, r] = ph / ns
        accept_rates[r] = n_accept / n_steps
        nsamples_per[r] = nsamples
        if map_score > best_map_score:
            best_map_score = map_score
            best_map = (np.asarray(map_active).copy(), map_ai, map_bi, map_pi)

    ddof = 1 if n_restarts > 1 else 0

    sets_results = pd.DataFrame(
        {
            "inPopulation": in_population,
            "inStudySet": in_study,
            "estimate": sets_marg.mean(axis=1),
            "std_error": sets_marg.std(axis=1, ddof=ddof),
        },
        index=set_names,
    )

    def _post_df(grid, marg):
        return pd.DataFrame(
            {
                "value": grid,
                "estimate": marg.mean(axis=1),
                "std_error": marg.std(axis=1, ddof=ddof),
            }
        )

    alpha_post = _post_df(alpha_grid, alpha_marg)
    beta_post = _post_df(beta_grid, beta_marg)
    p_post = _post_df(p_grid, p_marg)

    map_active_arr, map_ai, map_bi, map_pi = best_map
    map_estimate = {
        "sets": [set_names[i] for i in range(n_sets) if map_active_arr[i]],
        "alpha": float(alpha_grid[map_ai]),
        "beta": float(beta_grid[map_bi]),
        "p": float(p_grid[map_pi]),
        "log_score": float(best_map_score),
    }

    diagnostics = {
        **common_diag,
        "n_steps": int(n_steps),
        "n_restarts": int(n_restarts),
        "burn_in": burn_in,
        "thin": int(thin),
        "flip_freq": float(flip_freq),
        "seeds": seeds,
        "acceptance_rate": accept_rates,
        "nsamples_per_restart": nsamples_per,
        "sets_mcmc_post": sets_marg,
        "numba": _HAVE_NUMBA,
    }

    return MgsaResult(
        sets_results=sets_results,
        alpha_post=alpha_post,
        beta_post=beta_post,
        p_post=p_post,
        map_estimate=map_estimate,
        diagnostics=diagnostics,
    )


# AnnData / grassp integration
# --------------------------------------------------------------------------- #
def calculate_mgsa(
    data: "AnnData",
    cluster_key: str = "leiden",
    gene_name_key: str = "Gene_name_canonical",
    gene_sets: str | Mapping[str, Sequence[str]] | None = None,
    species: Literal["hsap", "mmus", "scer"] = "hsap",
    obs_key_added: str = "Cell_compartment_mgsa",
    method: str = "auto",
    max_active: int = 4,
    alpha_prior: Optional[callable] = None,
    beta_prior: Optional[callable] = None,
    min_posterior: float = 0.5,
    deduplicate_terms: bool = True,
    n_steps: int = 1_000_000,
    n_restarts: int = 5,
    seed: Optional[int] = 0,
    posterior_uns_key: Optional[str] = None,
    return_result: bool = True,
    inplace: bool = True,
    verbose: bool = True,
    **mgsa_kwargs,
) -> Optional[pd.DataFrame]:
    """Model-based gene-set analysis (MGSA) per cluster.

    The MGSA analogue of :func:`~grassp.tl.calculate_cluster_enrichment`: for each
    category in ``data.obs[cluster_key]`` it runs :func:`mgsa` with the cluster's
    genes as the study set and *all* genes as the population, then records the
    most probable active compartment. Unlike a per-term hypergeometric test, MGSA
    explains each cluster's genes *jointly*, so overlapping compartments compete
    and a redundant one (e.g. Lipid droplet vs. Endoplasmic reticulum) is only
    reported active if it explains genes the other cannot.

    Parameters
    ----------
    data
        AnnData with proteins as observations.
    cluster_key
        Categorical column in ``data.obs`` with cluster labels.
    gene_name_key
        Column in ``data.obs`` with gene identifiers (matching the gene sets).
    gene_sets
        Gene-set source passed to :func:`load_gmt`; ``None`` uses the bundled
        consolidated UniProt compartment sets for ``species``.
    species
        Species code for the bundled gene sets when ``gene_sets is None``.
    obs_key_added
        Column to write the top active compartment per cluster to (``NaN`` if the
        top posterior is below ``min_posterior``).
    method, max_active
        Forwarded to :func:`mgsa`. ``method="auto"`` (default) uses the exact
        enumeration for the small consolidated compartment vocabulary — fast and
        variance-free — and only falls back to MCMC for GO-sized gene-set libraries.
        ``max_active`` caps the exact enumeration (see :func:`mgsa`).
    min_posterior
        Posterior-activity threshold for assigning the top compartment label.
    deduplicate_terms
        Forwarded to :func:`load_gmt`; if ``True`` (default) sets with identical
        gene membership are collapsed to one term, so synonymous/duplicate
        compartments (e.g. ``PEROXISOME`` ≡ ``MICROBODY`` in fine ontologies) do
        not split the posterior mass and dilute the top compartment below
        ``min_posterior``.
    n_steps, n_restarts, seed
        Forwarded to :func:`mgsa`; only used when the run falls back to
        ``method="mcmc"`` (ignored for the exact path).
    posterior_uns_key
        ``uns`` key for the full (cluster x compartment) posterior activity matrix.
        Defaults to ``f"{obs_key_added}_posterior"``.
    return_result
        If ``True`` return the posterior DataFrame.
    inplace
        If ``True`` annotate ``data`` in place.
    verbose
        Print per-cluster progress.
    **mgsa_kwargs
        Extra keyword arguments forwarded to :func:`mgsa` (e.g. grids, ``thin``).

    Returns
    -------
    Optional[pandas.DataFrame]
        The (cluster x compartment) posterior activity matrix if
        ``return_result`` else ``None``. When ``inplace`` the top-compartment
        labels are written to ``data.obs[obs_key_added]``, the marginal posterior
        matrix to ``data.uns[posterior_uns_key]``, and the per-cluster MAP
        active-set indicator (0/1) to ``data.uns[f"{obs_key_added}_map"]``. The MAP
        matrix is the recommended input to
        :func:`mgsa_to_cluster_distribution` (``use_map=True``).
    """
    sets = load_gmt(gene_sets, species=species, deduplicate_terms=deduplicate_terms)
    set_names = list(sets.keys())
    population = data.obs[gene_name_key].astype(str).tolist()
    groups = data.obs.groupby(cluster_key, observed=True)

    posterior_rows: dict = {}
    map_rows: dict = {}
    top_terms: dict = {}
    evidence_rows: dict = {}
    for name, grp in groups:
        study = grp[gene_name_key].astype(str).tolist()
        res = mgsa(
            study,
            sets,
            population=population,
            method=method,
            max_active=max_active,
            alpha_prior=alpha_prior,
            beta_prior=beta_prior,
            n_steps=n_steps,
            n_restarts=n_restarts,
            seed=seed,
            **mgsa_kwargs,
        )
        est = res.sets_results["estimate"].reindex(set_names)
        posterior_rows[name] = est
        # MAP active-set indicator (the mode of the joint posterior): 1 for sets
        # active in the single best-scoring configuration, 0 otherwise.
        map_active = pd.Series(0, index=set_names, dtype=int)
        map_active[res.map_estimate["sets"]] = 1
        map_rows[name] = map_active
        # per-cluster (log) evidence and null, for the model-comparison merge.
        d = res.diagnostics
        evidence_rows[name] = {
            "log_evidence": d.get("log_evidence", np.nan),
            "log_null": d.get("log_null", np.nan),
        }
        top = est.idxmax()
        top_terms[name] = top if est.loc[top] >= min_posterior else np.nan
        if verbose:
            print(
                f"[mgsa] cluster {name}: top={top_terms[name]} "
                f"(p={float(est.loc[top]):.3f}), MAP={res.map_estimate['sets']}"
            )

    posterior = pd.DataFrame(posterior_rows).T.reindex(columns=set_names)
    posterior.index.name = cluster_key
    map_matrix = pd.DataFrame(map_rows).T.reindex(columns=set_names).fillna(0).astype(int)
    map_matrix.index.name = cluster_key
    evidence = pd.DataFrame(evidence_rows).T[["log_evidence", "log_null"]]
    evidence.index.name = cluster_key

    if inplace:
        obs_df = data.obs
        obs_df[obs_key_added] = groups[cluster_key].transform(
            lambda x: top_terms[x.name]
        )
        data.uns[posterior_uns_key or f"{obs_key_added}_posterior"] = posterior
        data.uns[f"{obs_key_added}_map"] = map_matrix
        data.uns[f"{obs_key_added}_evidence"] = evidence

    if return_result:
        return posterior
    return None


def mgsa_to_cluster_distribution(
    posterior: pd.DataFrame,
    map_matrix: pd.DataFrame | None = None,
    use_map: bool = True,
    unknown_label: str | None = "unknown",
    inactivity: Literal["product", "complement"] = "product",
    drop_empty: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """Convert a per-cluster MGSA posterior matrix into a soft-label distribution.

    MGSA returns per-set *marginal activity probabilities* ``q_t`` (each in
    ``[0, 1]``; several compartments can be active simultaneously), whereas the
    soft-label seed used by :func:`~grassp.tl.soft_cluster_annotation` is a
    per-cluster distribution over compartments (plus an explicit ``unknown``
    class) that sums to 1.

    The recommended mode (``use_map=True``) uses the **MAP active set** as a filter
    and the marginals as weights: only compartments in the MAP (the single best
    joint configuration) receive mass, weighted by their marginals. This is
    important because a completely redundant subset term (e.g. ``Nucleolus`` inside
    an active ``Nucleus``) has a *marginal* equal to the prior activation rate
    ``p`` — it is never in the MAP, so the filter removes that ``≈p`` residual that
    would otherwise inflate the distribution. With ``use_map=False`` all marginals
    are used (the residuals leak in).

    After (optional) MAP masking, ``Q[c, t] ∝ q_t`` and the leftover mass goes to
    ``unknown`` via the probability that *no* (admissible) compartment is active:

    - ``inactivity="product"`` (default): ``unknown ∝ ∏_t (1 - q_t)`` (over the
      admissible sets; an independence approximation of "no set active").
    - ``inactivity="complement"``: ``unknown ∝ 1 - max_t q_t``.

    everything renormalized so each cluster row sums to 1. A confidently
    single-compartment cluster becomes peaked, a genuine multi-compartment MAP is
    split by its marginals (equal marginals -> uniform), and a cluster with an
    empty MAP / nothing active puts its mass on ``unknown``.

    Parameters
    ----------
    posterior
        (cluster x compartment) matrix of MGSA marginal activity probabilities, as
        produced by :func:`calculate_mgsa` (``data.uns[..._posterior]``).
    map_matrix
        (cluster x compartment) 0/1 indicator of the MAP active set per cluster
        (``data.uns[..._map]``). Required when ``use_map=True``.
    use_map
        If ``True`` (default) mask the marginals to the MAP active set before
        building the distribution, removing redundant-subset ``≈p`` residuals.
    unknown_label
        Name of the background/unknown class, or ``None`` to omit it.
    inactivity
        How to compute the unknown/inactivity mass (see above).
    drop_empty
        Drop compartments that never carry mass, tightening the vocabulary.

    Returns
    -------
    Q : pandas.DataFrame
        Row-stochastic (cluster x category) matrix, ``unknown`` last if present.
    categories : list of str
        Column order, suitable as ``seed_categories`` for
        :func:`~grassp.tl.knn_annotation`.
    """
    q = posterior.clip(lower=0.0, upper=1.0).astype(float).fillna(0.0)
    if use_map:
        if map_matrix is None:
            raise ValueError(
                "use_map=True requires map_matrix (data.uns[f'{obs_key_added}_map'] "
                "from calculate_mgsa)."
            )
        mask = (
            map_matrix.reindex(index=q.index, columns=q.columns).fillna(0).astype(float) > 0
        )
        q = q.where(mask, 0.0)
    if inactivity == "product":
        unk = (1.0 - q).prod(axis=1)
    elif inactivity == "complement":
        unk = 1.0 - q.max(axis=1)
    else:
        raise ValueError(f"inactivity must be 'product' or 'complement', got {inactivity!r}")

    denom = q.sum(axis=1) + (unk if unknown_label is not None else 0.0)
    Q = q.div(denom.where(denom > 0, np.nan), axis=0)
    if unknown_label is not None:
        Q[unknown_label] = (unk / denom.where(denom > 0, np.nan))
    Q = Q.fillna(0.0)

    # Clusters with no active mass -> all unknown (or uniform if no unknown class).
    zero = Q.sum(axis=1) == 0
    if zero.any():
        if unknown_label is not None:
            Q.loc[zero, unknown_label] = 1.0
        else:
            Q.loc[zero, :] = 1.0 / Q.shape[1]

    term_cols = list(posterior.columns)
    if drop_empty:
        term_cols = [c for c in term_cols if (Q[c] > 0).any()]
    categories = term_cols + ([unknown_label] if unknown_label is not None else [])
    Q = Q[categories]
    Q = Q.div(Q.sum(axis=1), axis=0)
    return Q, categories
