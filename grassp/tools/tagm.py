from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from anndata import AnnData

import numpy as np
import pandas as pd

from numpy.linalg import det, eigvals, inv
from scipy.special import gammaln  # pylint: disable=no-name-in-module
from scipy.stats import multivariate_normal

# ----------------------------
# Helper functions
# ----------------------------


def multivariate_t_logpdf(X, delta, sigma, df):
    """
    Compute the log pdf of a multivariate t-distribution.

    Parameters:
        X      : (n, d) array of observations.
        delta  : (d,) location vector.
        sigma  : (d, d) scale matrix.
        df     : degrees of freedom.

    Returns:
        logpdf : (n,) array of log-density values.
    """
    d = X.shape[1]
    X_delta = X - delta
    inv_sigma = inv(sigma)
    Q = np.sum((X_delta @ inv_sigma) * X_delta, axis=1)
    log_norm = gammaln((df + d) / 2) - (
        gammaln(df / 2) + 0.5 * d * np.log(df * np.pi) + 0.5 * np.log(det(sigma))
    )
    return log_norm - ((df + d) / 2) * np.log(1 + Q / df)


def log_multivariate_gamma(a, d):
    """Compute the log multivariate gamma function for dimension d."""
    return (d * (d - 1) / 4) * np.log(np.pi) + np.sum(
        [gammaln(a + (1 - j) / 2) for j in range(1, d + 1)]
    )


def dinvwishart_logpdf(Sigma, nu, S):
    """
    Compute the log pdf of an inverse-Wishart distribution.

    Parameters:
        Sigma : (d,d) matrix (the sample covariance).
        nu    : degrees of freedom.
        S     : (d,d) scale matrix.

    Returns:
        log_pdf : scalar log-density.
    """
    d = Sigma.shape[0]
    sign_S, logdet_S = np.linalg.slogdet(S)
    sign_Sigma, logdet_Sigma = np.linalg.slogdet(Sigma)
    const = -0.5 * nu * logdet_S - (nu * d / 2) * np.log(2) - log_multivariate_gamma(nu / 2, d)
    log_pdf = const - ((nu + d + 1) / 2) * logdet_Sigma - 0.5 * np.trace(inv(Sigma) @ S)
    return log_pdf


def dbeta_log(x, a, b):
    """Log pdf of the Beta distribution."""
    return (
        (a - 1) * np.log(x)
        + (b - 1) * np.log(1 - x)
        - (gammaln(a) + gammaln(b) - gammaln(a + b))
    )


def ddirichlet_log(x, alpha):
    """
    Log pdf of a Dirichlet distribution.

    Parameters:
        x     : probability vector (must sum to 1).
        alpha : vector of concentration parameters.
    """
    alpha = np.asarray(alpha)
    return gammaln(np.sum(alpha)) - np.sum(gammaln(alpha)) + np.sum((alpha - 1) * np.log(x))


# ----------------------------
# TAGM MAP training
# ----------------------------


def tagm_map_train(
    adata: AnnData,
    gt_col: str = "markers",
    method: str = "MAP",
    numIter: int = 100,
    mu0: np.ndarray | None = None,
    lambda0: float = 0.01,
    nu0: int | None = None,
    S0: np.ndarray | None = None,
    beta0: np.ndarray | None = None,
    u: int = 2,
    v: int = 10,
    seed: int | None = None,
    inplace: bool = True,
) -> dict | None:
    """Train a *TAGM-MAP* (T-Augmented Gaussian Mixture, MAP variant) model.

    The training procedure follows Crook *et al.* 2018 and estimates
    component parameters for known *marker* classes while accommodating an
    outlier component modelled by a multivariate *t* distribution.

    Workflow
    --------
    1. Split observations into *labelled* (markers) and *unlabelled*
       according to ``adata.obs[gt_col]``.
    2. Compute empirical hyper-priors if not supplied.
    3. Run an EM algorithm to obtain maximum-a-posteriori (MAP) estimates of
       component means, covariances and mixing proportions.
    4. Store the fitted parameters in ``adata.uns['tagm.map.params']`` when
       ``inplace`` is ``True``.

    Parameters
    ----------
    adata
        :class:`anndata.AnnData` with proteins as observations and fractions
        as variables.
    gt_col
        Column name identifying marker proteins (default ``"markers"``).
    method
        Currently only ``"MAP"`` is implemented; reserved for future.
    numIter
        EM iterations (default ``100``).
    mu0, lambda0, nu0, S0, beta0
        Hyper-parameters of the Normal–Inverse-Wishart–Dirichlet prior.  If
        ``None`` sensible empirical defaults are inferred.
    u, v
        Beta prior parameters for the outlier mixing proportion.
    seed
        Random seed for reproducibility.
    inplace
        If ``True`` (default) write parameters to ``adata`` and return
        ``None``; otherwise return the parameter dictionary.

    Returns
    -------
    MAP parameter dictionary when ``inplace`` is ``False``.
    """
    # Split data into marker (labelled) and unknown (unlabelled) subsets.
    marker_idx = adata.obs[gt_col].notna()
    unknown_idx = adata.obs[gt_col].isna()
    adata_markers = adata[marker_idx].copy()
    adata_unknown = adata[unknown_idx].copy()

    # Get marker classes
    markers = np.sort(adata_markers.obs[gt_col].unique())
    K = len(markers)

    # Get expression data
    mydata = np.asarray(adata_markers.X)
    X = np.asarray(adata_unknown.X)
    N, D = mydata.shape

    # Set random seed
    if seed is not None:
        np.random.seed(seed)

    # Set empirical priors
    if nu0 is None:
        nu0 = D + 2
    if S0 is None:
        overall_mean = np.mean(mydata)
        var_vector = np.sum((mydata - overall_mean) ** 2, axis=0) / N
        S0 = np.diag(var_vector) / (K ** (1 / D))
    if mu0 is None:
        mu0 = np.mean(mydata, axis=0)
    if beta0 is None:
        beta0 = np.ones(K)

    priors = {"mu0": mu0, "lambda0": lambda0, "nu0": nu0, "S0": S0, "beta0": beta0}

    # Precompute marker statistics
    nk = np.array([np.sum(adata_markers.obs[gt_col] == m) for m in markers])
    xk = np.array([np.mean(mydata[adata_markers.obs[gt_col] == m], axis=0) for m in markers])

    lambdak = lambda0 + nk  # vector (K,)
    nuk = nu0 + nk  # vector (K,)
    mk = (nk[:, None] * xk + lambda0 * mu0) / lambdak[:, None]  # shape (K, D)

    # Compute sk for each class (loop over classes; hard to vectorize completely)
    sk = np.zeros((K, D, D))
    for j, m in enumerate(markers):
        idx = (adata_markers.obs[gt_col] == m).values
        if np.sum(idx) > 0:
            x_j = mydata[idx]
            diff = x_j - xk[j]
            sk[j] = S0 + (diff.T @ diff) + lambda0 * np.outer(mu0 - xk[j], mu0 - xk[j])

    betak = beta0 + nk

    # Initialize parameters
    muk = mk.copy()
    sigmak = np.array([sk[j] / (nuk[j] + D + 1) for j in range(K)])
    pik = (betak - 1) / (np.sum(betak) - K)

    # Global parameters
    all_data = np.asarray(adata.X)
    M = np.mean(all_data, axis=0)
    V = np.cov(all_data, rowvar=False) / 2
    if np.min(np.linalg.eigvals(V)) < np.finfo(float).eps:
        V = V + np.eye(D) * 1e-6
    eps = (u - 1) / (u + v - 2)

    # EM algorithm
    n_unlab = X.shape[0]
    loglike = np.zeros(numIter)

    for t in range(numIter):
        # E-step: vectorized log density calculations
        log_pdf_normal = np.array(
            [multivariate_normal.logpdf(X, mean=muk[k], cov=sigmak[k]) for k in range(K)]
        ).T
        log_pdf_t = multivariate_t_logpdf(X, M, V, df=4)

        # Compute responsibilities efficiently
        log_a = np.log(pik + 1e-10) + np.log(1 - eps) + log_pdf_normal
        log_b = np.log(pik + 1e-10) + np.log(eps) + log_pdf_t[:, None]

        # Log-sum-exp trick
        max_log = np.maximum(np.max(log_a, axis=1), np.max(log_b, axis=1))
        a = np.exp(log_a - max_log[:, None])
        b = np.exp(log_b - max_log[:, None])
        norm = np.sum(a + b, axis=1, keepdims=True)
        a /= norm
        b /= norm

        w = a + b
        r = np.sum(w, axis=0)

        # M-step: update parameters
        eps = (u + np.sum(b) - 1) / (n_unlab + u + v - 2)
        sum_a = np.sum(a, axis=0)

        # Update means and weights
        xbar = (a.T @ X) / (sum_a[:, None] + 1e-10)
        pik = (r + betak - 1) / (n_unlab + np.sum(betak) - K)

        # Update component parameters
        lambda_new = lambdak + sum_a
        nu_new = nuk + sum_a
        m_new = (sum_a[:, None] * xbar + lambdak[:, None] * mk) / lambda_new[:, None]

        # Update covariances efficiently
        for j in range(K):
            diff = X - xbar[j]
            TS = (a[:, j][:, None] * diff).T @ diff
            vv = (lambdak[j] * sum_a[j]) / lambda_new[j]
            S_new = sk[j] + vv * np.outer(xbar[j] - mk[j], xbar[j] - mk[j]) + TS
            sigmak[j] = S_new / (nu_new[j] + D + 2)

        muk = m_new

        # Compute log-likelihood efficiently
        ll = (
            np.sum(a * log_pdf_normal)
            + np.sum(w * np.log(pik + 1e-10))
            + np.sum([dinvwishart_logpdf(sigmak[j], nu0, S0) for j in range(K)])
            + np.sum(
                [multivariate_normal.logpdf(muk[j], mean=mu0, cov=sigmak[j]) for j in range(K)]
            )
            + np.sum(a) * np.log(1 - eps)
            + np.sum(b) * np.log(eps)
            + np.sum(np.sum(b, axis=1) * log_pdf_t)
            + dbeta_log(eps, u, v)
            + ddirichlet_log(pik, np.full(K, beta0[0] / K))
        )

        loglike[t] = ll

    posteriors = {
        "mu": muk,
        "sigma": sigmak,
        "weights": pik,
        "epsilon": eps,
        "logposterior": loglike,
    }
    params = {
        "method": method,
        "gt_col": gt_col,
        "seed": seed,
        "markers": markers,
        "priors": priors,
        "posteriors": posteriors,
        "datasize": {"data": adata.X.shape},
    }

    if inplace:
        adata.uns["tagm.map.params"] = params
    else:
        return params


# ----------------------------
# TAGM MAP prediction
# ----------------------------


def tagm_map_predict(
    adata: AnnData,
    params: dict | None = None,
    probJoint: bool = False,
    probOutlier: bool = True,
    inplace: bool = True,
) -> pd.DataFrame | None:
    """
    Predict sub-cellular localization for unlabelled proteins using a fitted
    *TAGM-MAP* model.

    Given a set of maximum-a-posteriori (MAP) parameters obtained from
    :func:`tagm_map_train`, the function computes, for every protein with an
    unknown label, the posterior probability of originating from each
    compartment-specific Gaussian as well as from the global multivariate
    *t* outlier component.  Proteins are allocated to the compartment whose
    *Gaussian* posterior is largest and the results are written to
    ``adata.obs`` / ``adata.obsm``.

    Workflow
    --------
    1. Retrieve MAP parameters from *params* or
       ``adata.uns['tagm.map.params']``. This includes the marker column name
       (``gt_col``) used during training.
    2. Split observations into *labelled* (marker) and *unlabelled* sets via
       ``adata.obs[gt_col]``.
    3. Compute posterior probabilities for every component and the outlier
       model.
    4. Allocate unlabelled proteins to the compartment with the highest
       Gaussian posterior probability.
    5. Store allocations, per-protein probabilities and, optionally, the
       joint probability matrix/outlier probability.

    Parameters
    ----------
    adata
        :class:`anndata.AnnData` with proteins as observations and fractions
        as variables. The marker column (``gt_col``) must match the one used
        during training with :func:`tagm_map_train`.
    params
        Parameter dictionary as returned by :func:`tagm_map_train`.  If
        ``None`` (default) the parameters are read from
        ``adata.uns['tagm.map.params']``. The marker column name (``gt_col``)
        is automatically read from the stored parameters.
    probJoint
        If ``True`` also store the joint probability matrix in
        ``adata.obs['tagm.map.joint']`` (default *False*).
    probOutlier
        If ``True`` (default) store the probability of belonging to the outlier
        component in ``adata.obs['tagm.map.outlier']``.
    inplace
        If ``True`` (default) modify *adata* in place and return *None*;
        otherwise return a :class:`~pandas.DataFrame` with the predictions.

    Returns
    -------
    When ``inplace`` is ``False`` a DataFrame with the allocation,
    allocation probability and outlier probability is returned;
    otherwise ``None``.
    """

    if params is None:
        try:
            params = adata.uns["tagm.map.params"]
        except KeyError:
            raise ValueError(
                "No parameters found. Please provide either 'params' or run tagm_map_train first."
            )
    posteriors = params["posteriors"]
    eps = posteriors["epsilon"]
    mu = posteriors["mu"]  # shape (K, D)
    sigma = posteriors["sigma"]  # shape (K, D, D)
    weights = posteriors["weights"]  # shape (K,)
    markers = params["markers"]  # Get markers from trained model
    K = len(markers)
    gt_col = params["gt_col"]  # Always use the gt_col from training

    # Split data.
    marker_idx = adata.obs[gt_col].notna()
    unknown_idx = adata.obs[gt_col].isna()
    adata_markers = adata[marker_idx].copy()
    adata_unknown = adata[unknown_idx].copy()

    X = np.asarray(adata_unknown.X)
    D = X.shape[1]

    # Global parameters (from entire data)
    all_data = np.asarray(adata.X)
    M = np.mean(all_data, axis=0)
    V = np.cov(all_data, rowvar=False) / 2
    if np.min(eigvals(V)) < np.finfo(float).eps:
        V = V + np.eye(D) * 1e-6

    # Compute a and b for unknown data.
    a = np.zeros((adata.shape[0], K))
    b = np.zeros((adata.shape[0], K))
    for j in range(K):
        a[:, j] = (
            np.log(weights[j] + 1e-10)
            + np.log(1 - eps)
            + multivariate_normal.logpdf(adata.X, mean=mu[j], cov=sigma[j])
        )
        b[:, j] = (
            np.log(weights[j] + 1e-10)
            + np.log(eps)
            + multivariate_t_logpdf(adata.X, M, V, df=4)
        )
    ab = np.hstack([a, b])
    c_const = np.max(ab, axis=1, keepdims=True)
    ab = ab - c_const
    ab = np.exp(ab) / np.sum(np.exp(ab), axis=1, keepdims=True)
    a = ab[:, :K]
    b = ab[:, K : 2 * K]
    predictProb = a + b  # overall probability for each component

    # For allocation, use the Gaussian part (a) to choose the marker with highest probability.
    pred_indices = np.argmax(a, axis=1)
    pred_labels = [str(markers[i]) for i in pred_indices]
    # Build a DataFrame for the predictions.
    organelleAlloc = pd.DataFrame(
        {
            "pred": pred_labels,
            "prob": [a[i, pred_indices[i]] for i in range(adata.shape[0])],
        },
        index=adata.obs_names,
    )

    # Combine predictions for all data.
    pred_all = organelleAlloc["pred"]
    prob_all = organelleAlloc["prob"]
    # Outlier probability: for all data, use row sum of b.
    outlier_all = pd.Series(np.sum(b, axis=1), index=adata.obs_names)

    # Ensure the order matches adata.obs.
    pred_all = pred_all.loc[adata.obs_names]
    prob_all = prob_all.loc[adata.obs_names]
    outlier_all = outlier_all.loc[adata.obs_names]

    if inplace:
        adata.obs["tagm.map.allocation"] = pred_all
        if f"{params['gt_col']}_colors" in adata.uns:
            adata.uns["tagm.map.allocation_colors"] = adata.uns[f"{params['gt_col']}_colors"]
        adata.obs["tagm.map.probability"] = prob_all

        adata.obsm["tagm.map.probabilities"] = a
        if probJoint:
            # Create joint probability matrix for markers.
            marker_prob = np.zeros((adata_markers.n_obs, K))
            for i, lbl in enumerate(adata_markers.obs[gt_col]):
                j = np.where(markers == lbl)[0][0]
                marker_prob[i, j] = 1  # vectorized alternative is possible.
            joint = np.vstack([predictProb, marker_prob])
            # Store as a list of arrays (one per observation).
            adata.obs["tagm.map.joint"] = list(joint)
        if probOutlier:
            adata.obs["tagm.map.outlier"] = outlier_all
    else:
        return pd.DataFrame(
            {
                "pred": pred_all,
                "prob": prob_all,
                "outlier": outlier_all,
            },
            index=adata.obs_names,
        )
