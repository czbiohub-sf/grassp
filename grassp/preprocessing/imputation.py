from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Union

    from numpy.random import RandomState

    AnyRandom = Union[int, RandomState, None]


import numpy as np
import scipy.sparse

from anndata import AnnData

from ..util import confirm_proteins_as_obs


def impute_gaussian(
    data: AnnData,
    width: float = 0.3,
    distance: float = 1.8,
    per_sample: bool = True,
    random_state: AnyRandom = 0,
    inplace: bool = True,
) -> np.ndarray | None:
    """Impute missing values using a Gaussian distribution.

    This function imputes missing values (zeros) in the data matrix using a Gaussian
    distribution. The parameters of the Gaussian are derived from the observed (non-zero)
    values, with the mean shifted downward by a specified number of standard deviations.

    Parameters
    ----------
    data
        Annotated data matrix with proteins as observations (rows).
    width
        Width of the Gaussian distribution used for imputation, as a fraction of the
        standard deviation of observed values. Default is 0.3.
    distance
        Number of standard deviations below the mean of observed values to center the
        imputation distribution. Default is 1.8.
    per_sample
        If True, calculate parameters separately for each sample (column).
        If False, use global parameters. Default is True.
    random_state
        Seed for random number generation. Default is 0.
    inplace
        If True, modify data in place. If False, return a copy. Default is True.

    Returns
    -------
    numpy.ndarray or None
        If inplace=False, returns the imputed data matrix.
        If inplace=True, returns None and modifies the input data.

    Notes
    -----
    This implements a simple but effective imputation strategy commonly used in
    proteomics data analysis. Missing values are assumed to be below detection limit
    and are imputed from a Gaussian distribution with parameters derived from the
    observed values but shifted downward.
    """
    confirm_proteins_as_obs(data)
    np.random.seed(random_state)

    if not inplace:
        data = data.copy()
    X = data.X
    if X is None:
        raise ValueError("data.X is None")

    # Convert to dense if sparse
    if scipy.sparse.issparse(X):
        X = X.toarray()

    zero_mask = X != 0
    n_zeros = X.size - zero_mask.sum()

    if per_sample:
        pmean = np.ma.array(X, mask=~zero_mask).mean(axis=0)
        stdev = np.ma.array(X, mask=~zero_mask).std(axis=0)
    else:
        pmean = np.ma.array(X, mask=~zero_mask).mean()
        stdev = np.ma.array(X, mask=~zero_mask).std()

    imp_mean = pmean - distance * stdev
    imp_stdev = stdev * width

    if per_sample:
        imputed_values = np.random.normal(loc=imp_mean, scale=imp_stdev, size=X.shape)
        X[np.invert(zero_mask)] = imputed_values[np.invert(zero_mask)]
    else:
        imputed_values = np.random.normal(loc=imp_mean, scale=imp_stdev, size=n_zeros)
        X[np.invert(zero_mask)] = imputed_values
    if not inplace:
        return X


# ProDA

# import numpy as np
# from scipy.optimize import minimize, Bounds
# from scipy.stats import norm, invgamma, t


# def pd_lm(
#     y,
#     dropout_curve_position,
#     dropout_curve_scale,
#     location_prior_mean,
#     location_prior_scale,
#     variance_prior_scale,
#     variance_prior_df,
#     location_prior_df=3,
#     verbose=False,
# ):
#     """
#     Fit a probabilistic dropout model with location and variance priors.

#     Args:
#         y: numpy array of observations (can contain NaN)
#         dropout_curve_position: float, position parameter for dropout curve
#         dropout_curve_scale: float, scale parameter for dropout curve
#         location_prior_mean: float, prior mean
#         location_prior_scale: float, prior scale
#         variance_prior_scale: float, prior scale for variance
#         variance_prior_df: float, prior degrees of freedom for variance
#         location_prior_df: float, degrees of freedom for location prior (default: 3)
#         verbose: bool, whether to print messages
#     """
#     # Setup
#     y = np.asarray(y)
#     is_missing = np.isnan(y)
#     yo = y[~is_missing]
#     n_obs = len(yo)
#     n_total = len(y)

#     # Initial values
#     beta_init = location_prior_mean
#     sigma2_init = variance_prior_df * variance_prior_scale / (variance_prior_df + 2)

#     def dropout_probability(mu):
#         """Calculate dropout probability using normal CDF"""
#         return norm.cdf((mu - dropout_curve_position) / dropout_curve_scale)

#     def objective(params):
#         """Negative log likelihood with priors and dropout"""
#         beta, sigma2 = params
#         if sigma2 <= 0:
#             return np.inf

#         # Log likelihood for observed values
#         ll_obs = np.sum(norm.logpdf(yo, loc=beta, scale=np.sqrt(sigma2)))

#         # Dropout likelihood
#         print("dropout")
#         print(beta)
#         print(dropout_probability(beta))
#         p_dropout = dropout_probability(beta)
#         ll_dropout = np.sum(np.log(p_dropout) * is_missing)

#         # Location prior (Student's t)
#         ll_loc = np.sum(
#             t.logpdf(
#                 np.repeat(beta, n_total),
#                 df=location_prior_df,
#                 loc=location_prior_mean,
#                 scale=np.sqrt(location_prior_scale),
#             )
#         )
#         # ll_loc = -0.5 * ((beta - location_prior_mean) ** 2) / location_prior_scale**2

#         # Variance prior (Inverse chi-squared)
#         ll_var = invgamma.logpdf(
#             sigma2,
#             a=variance_prior_df / 2,
#             scale=variance_prior_scale * variance_prior_df / 2,
#         ) + np.log(sigma2)
#         # ll_var = -(variance_prior_df + 2) * 0.5 * np.log(
#         # sigma2
#         # ) - variance_prior_df * variance_prior_scale / (2 * sigma2)

#         print(ll_obs, ll_dropout, ll_loc, ll_var)
#         return -(ll_obs + ll_dropout + ll_loc + ll_var)

#     def gradient(params):
#         """Analytic gradient of negative log likelihood"""

#         # Unpack parameters
#         beta, sigma2 = params
#         sign(sd) * exp(dnorm(x, mu, abs(sd), log=TRUE) - invprobit(x, mu, sd, log=TRUE))
#         x = (beta - dropout_curve_position) / dropout_curve_scale
#         imr = norm.logpdf(x) - np.log(1 - norm.cdf(x))
#         print("imr:", imr)
#         # Calculate beta gradients
#         dbeta_p = (
#             -(location_prior_df + 1)
#             * (beta - location_prior_mean)
#             / (
#                 location_prior_df * location_prior_scale
#                 + (beta - location_prior_mean) ** 2
#             )
#         )

#         dbeta_o = -np.sum(beta - yo) / sigma2
#         dbeta_m = np.sum(imr)
#         print(dbeta_p, dbeta_o, dbeta_m)

#         # Calculate sigma2 gradients
#         dsig2_p = (
#             -(1 + variance_prior_df / 2) / sigma2
#             + variance_prior_df * variance_prior_scale / (2 * sigma2**2)
#             + 1 / sigma2
#         )

#         dsig2_o = np.sum((beta - yo) ** 2 - sigma2) / (2 * sigma2**2)
#         dsig2_m = -np.sum(
#             (beta - dropout_curve_position) / (2 * dropout_curve_scale**2) * imr
#         )

#         print("grad")
#         print(dbeta_p, dbeta_o, dbeta_m)
#         print(dsig2_p, dsig2_o, dsig2_m)
#         return np.array([dbeta_p + dbeta_o + dbeta_m, dsig2_p + dsig2_o + dsig2_m])

#     def hessian(params):
#         """Analytic Hessian of negative log likelihood"""
#         beta, sigma2 = params
#         if sigma2 <= 0:
#             return np.array([[np.inf, np.inf], [np.inf, np.inf]])

#         # Compute dropout-related terms
#         p_dropout = dropout_probability(beta)
#         z = (beta - dropout_curve_position) / dropout_curve_scale
#         pdf_z = norm.pdf(z)

#         # Second derivatives for dropout
#         d2_beta_dropout = (
#             -1
#             / dropout_curve_scale**2
#             * np.sum(
#                 (is_missing / p_dropout - (~is_missing) / (1 - p_dropout))
#                 * (-z * pdf_z)
#                 + (is_missing / p_dropout**2 + (~is_missing) / (1 - p_dropout) ** 2)
#                 * pdf_z**2
#             )
#         )

#         # Complete second derivatives
#         d2_beta2 = -n_obs / sigma2 + d2_beta_dropout - 1 / location_prior_scale**2

#         d2_sigma2 = (
#             -np.sum((yo - beta) ** 2) / sigma2**3
#             + 0.5 * n_obs / sigma2**2
#             + (variance_prior_df + 2) * 0.5 / sigma2**2
#             - variance_prior_df * variance_prior_scale / sigma2**3
#         )

#         d2_beta_sigma2 = -np.sum(yo - beta) / sigma2**2

#         H = np.array([[d2_beta2, d2_beta_sigma2], [d2_beta_sigma2, d2_sigma2]])

#         return -H

#     # Optimize with analytic derivatives
#     bounds = Bounds([0, 1e-10], [np.inf, np.inf])
#     result = minimize(
#         objective,
#         x0=[beta_init, sigma2_init],
#         method="BFGS",
#         jac=gradient,
#         # hess=hessian,
#         bounds=bounds,
#     )
#     # result = minimize(
#     #     objective,
#     #     x0=[beta_init, sigma2_init],
#     #     method="L-BFGS-B",
#     #     # jac=gradient,
#     #     # hess=hessian,
#     #     bounds=bounds,
#     # )

#     if not result.success and verbose:
#         print(f"Warning: Optimization did not converge: {result.message}")

#     print(objective(result.x))
#     # Extract results
#     fit_beta, fit_sigma2 = result.x

#     # Get variance-covariance matrix from Hessian
#     try:
#         Var_coef = np.linalg.inv(hessian(result.x))
#     except np.linalg.LinAlgError:
#         if verbose:
#             print("Warning: Could not invert Hessian")
#         Var_coef = np.diag([np.inf, np.inf])

#     # Calculate effective sample size and degrees of freedom
#     fit_sigma2_var = Var_coef[1, 1]
#     n_approx = 2 * fit_sigma2**2 / fit_sigma2_var - variance_prior_df
#     df_approx = n_approx - 1 + variance_prior_df

#     # Calculate unbiased variance estimate
#     if df_approx > 30 * len(y) and df_approx > 100:
#         df_approx = np.inf
#         s2_approx = variance_prior_scale
#     else:
#         s2_approx = fit_sigma2 * n_approx / (n_approx - 1)

#     # Calculate correction factors for variance-covariance matrix
#     try:
#         var_beta = Var_coef[0, 0]
#         correction = np.sqrt(8 * var_beta)
#         Var_coef_corrected = correction * Var_coef * correction
#     except:
#         if verbose:
#             print("Warning: Could not compute correction factors")
#         Var_coef_corrected = Var_coef

#     return {
#         "coefficients": fit_beta,
#         "sigma2": fit_sigma2,
#         "coef_variance_matrix": Var_coef_corrected,
#         "n_approx": n_approx,
#         "df": df_approx,
#         "s2": s2_approx,
#         "n_obs": n_obs,
#         "dropout_prob": dropout_probability(fit_beta),
#     }


# # Example usage:
# if __name__ == "__main__":
#     # Generate example data
#     np.random.seed(42)
#     y = np.random.normal(loc=10, scale=2, size=20)
#     y[np.random.rand(20) < 0.3] = np.nan  # add missing values

#     # Fit model
#     result = pd_lm(
#         y=y,
#         dropout_curve_position=8,
#         dropout_curve_scale=-1,
#         location_prior_mean=9,
#         location_prior_scale=3,
#         variance_prior_scale=1,
#         variance_prior_df=2,
#         verbose=True,
#     )

#     print("\nResults:")
#     print(f"Estimated mean: {result['coefficients']:.3f}")
#     print(f"Estimated variance: {result['sigma2']:.3f}")
#     print(f"Standard error: {np.sqrt(result['coef_variance_matrix'][0,0]):.3f}")
#     print(f"Effective sample size: {result['n_approx']:.1f}")
#     print(f"Degrees of freedom: {result['df']:.1f}")
#     print(f"Dropout probability: {result['dropout_prob']:.3f}")
