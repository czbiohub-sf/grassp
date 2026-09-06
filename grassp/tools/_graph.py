"""The neighbour-graph operator shared by both annotation families.

:func:`~grassp.tools.competitive_diffusion` and
:func:`~grassp.tools.independent_diffusion` differ in what they do *with* the diffused
labels -- the first cross-normalizes them to a simplex, the second keeps them per-term --
but the diffusion itself is the same operator, and it used to be written out twice. The
copies had drifted: one compared the iteration's L1 change against ``tol`` and the other
against ``tol * n_columns``, so the same nominal tolerance meant different things and one
of them silently tightened as the label vocabulary grew.

This module holds the single implementation. Nothing here knows about compartments,
markers or probabilities; it deals in an affinity matrix and a real-valued matrix to
spread over it.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from anndata import AnnData

import warnings

import numpy as np
import scipy.sparse as sp

#: Default iteration cap and per-column tolerance for :func:`spread`.
DEFAULT_MAX_ITER = 60
DEFAULT_TOL = 1e-4


def symmetric_normalized(W) -> sp.csr_matrix:
    """Symmetric-normalized affinity ``S = D^{-1/2} W D^{-1/2}`` (Zhou et al. 2003).

    Rows with zero degree get a zero scaling rather than a division by zero, so an
    isolated protein contributes nothing instead of poisoning the matrix with NaN.
    """
    d = np.asarray(W.sum(axis=1)).ravel()
    d_inv_sqrt = np.zeros_like(d, dtype=float)
    nonzero = d > 0
    d_inv_sqrt[nonzero] = 1.0 / np.sqrt(d[nonzero])
    scale = sp.diags(d_inv_sqrt)
    return (scale @ W @ scale).tocsr()


def affinity(data: AnnData, obsp_key: str = "connectivities", *, cache: bool = True):
    """Resolve ``obsp[obsp_key]`` into the affinity matrix to diffuse over.

    ``"connectivities"`` (or any other key) is used as stored. ``"distances"`` is turned
    into a Gaussian RBF affinity ``W = exp(-d² / 2σ²)`` restricted to the existing
    sparsity pattern, with σ the median nonzero distance -- a data-driven width narrower
    than UMAP's fuzzy-union connectivities, which gives more boundary-localized
    smoothing. That matrix is cached at ``obsp["W_spreading"]`` for inspection.

    Parameters
    ----------
    data
        Object carrying the neighbour graph.
    obsp_key
        Key in ``.obsp``.
    cache
        Whether to write the derived RBF matrix back to ``obsp["W_spreading"]``. Only
        applies when ``obsp_key == "distances"``.
    """
    if obsp_key not in data.obsp:
        raise KeyError(f"obsp[{obsp_key!r}] not found; run grassp.pp.neighbors first.")
    if obsp_key != "distances":
        return data.obsp[obsp_key]

    distances = data.obsp[obsp_key]
    sigma = float(np.median(distances.data)) if distances.nnz > 0 else 1.0
    W = distances.copy()
    W.data = np.exp(-(distances.data**2) / (2.0 * sigma**2))
    if cache:
        data.obsp["W_spreading"] = W
    return W


def spread(
    S,
    Y0: np.ndarray,
    *,
    alpha: float,
    max_iter: int = DEFAULT_MAX_ITER,
    tol: float = DEFAULT_TOL,
    verbose: bool = False,
    warn_context: str | None = None,
) -> np.ndarray:
    """Label-spreading fixed point ``F = alpha * S @ F + (1 - alpha) * Y0``.

    Convergence is measured **per column**: the L1 change between iterations is compared
    against ``tol * Y0.shape[1]``, so the criterion reads "mean absolute change per class
    is below ``tol``" and does not silently tighten as the vocabulary grows. Comparing an
    absolute L1 sum instead -- the other convention this replaces -- made the iteration
    count scale with the number of classes for no gain: on a 20-class map it took 19
    iterations where the per-column test took 10, and the two agreed on every label with
    a maximum probability difference of 2e-4.

    Parameters
    ----------
    S
        Affinity operator, normally from :func:`symmetric_normalized`.
    Y0
        ``(n_obs, n_classes)`` seed, re-injected at every step.
    alpha
        Soft clamp in ``[0, 1]``: small values keep the result near ``Y0``, values near
        1 let it drift toward the graph.
    max_iter, tol
        Iteration cap and per-column tolerance.
    verbose
        Print the L1 change each iteration.
    warn_context
        If given, the name to use in a warning when ``max_iter`` is reached before
        convergence. ``None`` iterates silently, which is what the per-term alpha sweep
        in :func:`~grassp.tools.independent_diffusion` needs -- it calls this hundreds of
        times and a warning per call would bury the run.
    """
    if not 0 <= alpha <= 1:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    Y0 = np.asarray(Y0, dtype=float)
    width = Y0.shape[1] if Y0.ndim > 1 else 1
    threshold = tol * width
    F = Y0.copy()
    for iteration in range(max_iter):
        F_next = alpha * np.asarray(S @ F) + (1 - alpha) * Y0
        change = np.abs(F_next - F).sum()
        F = F_next
        if verbose:
            print(f"Diff: {change:.3f}, Iteration {iteration} completed")
        if change < threshold:
            if verbose:
                print(f"Diff: {change:.3f}, Converged")
            return F
    if warn_context is not None:
        warnings.warn(
            f"{warn_context}: max_iter={max_iter} reached without convergence (tol={tol})."
        )
    return F


def neff_kish(T) -> np.ndarray:
    """Kish effective neighbourhood size ``(Σw)² / Σw²`` per row of an affinity matrix.

    Counts how many neighbours a protein effectively has once their weights are taken
    into account, which is what the analytic entropy null needs to know how much
    smoothing each protein received. Rows with no weight get 0.
    """
    w = np.asarray(T.sum(axis=1)).ravel()
    w2 = np.asarray(T.multiply(T).sum(axis=1)).ravel()
    return np.where(w2 > 0, w**2 / w2, 0.0)


def neff_hutchinson(
    diffuse: Callable[[np.ndarray, float], np.ndarray],
    alpha: float,
    denominator: np.ndarray,
    *,
    n_probe: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Kish effective sample size of the *diffused* operator, excluding self.

    Unlike :func:`neff_kish` this measures the neighbourhood a protein actually sees at
    diffusion depth ``alpha``, which is not available in closed form -- the operator is a
    matrix power series, never materialized. Estimated with Hutchinson probes: for a
    symmetric ``M``, ``mean_g (M g)_i²`` converges to ``Σ_j M_ij²`` and
    ``mean_g g_i (M g)_i`` to ``M_ii``, which is enough to remove the diagonal from both
    the sum and the sum of squares.

    Parameters
    ----------
    diffuse
        ``diffuse(Y, alpha)`` applying the operator, as returned by :func:`make_diffuser`.
    alpha
        Diffusion depth to measure at.
    denominator
        ``diffuse(ones, alpha)``, the row sums including self.
    n_probe
        Number of Rademacher probe vectors. More probes, less variance.
    rng
        Random state for the probes.
    """
    probes = rng.choice([-1.0, 1.0], size=(len(denominator), n_probe))
    diffused = diffuse(probes, alpha)
    diagonal = (probes * diffused).mean(axis=1)
    off_sum = np.maximum(denominator - diagonal, 1e-9)
    off_sq = np.maximum((diffused**2).mean(axis=1) - diagonal**2, 1e-12)
    return off_sum**2 / off_sq


def make_diffuser(S, *, max_iter: int = DEFAULT_MAX_ITER, tol: float = DEFAULT_TOL):
    """Return a ``diffuse(Y, alpha)`` closure over :func:`spread`.

    The per-term alpha sweep applies the same operator at many depths, so binding ``S``
    once and varying only ``alpha`` keeps the call sites readable.
    """

    def diffuse(Y, alpha, max_iter=max_iter, tol=tol):
        return spread(S, Y, alpha=alpha, max_iter=max_iter, tol=tol)

    return diffuse
