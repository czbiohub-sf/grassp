# import numpy as np
# from scipy.stats import t, norm
# from typing import Union, Optional
# import warnings


# def invprobit(
#     x: np.ndarray,
#     mu: Union[float, np.ndarray],
#     sigma: Union[float, np.ndarray],
#     log: bool = False,
#     oneminus: bool = False,
# ) -> np.ndarray:
#     """
#     Inverse probit function (Gaussian CDF)

#     Parameters
#     ----------
#     x : array-like
#         Input values
#     mu : float or array-like
#         Mean parameter
#     sigma : float or array-like
#         Scale parameter
#     log : bool, default=False
#         Return log probability
#     oneminus : bool, default=False
#         Return 1 - probability instead

#     Returns
#     -------
#     np.ndarray
#         Probabilities or log probabilities
#     """
#     z = (x - mu) / sigma
#     if oneminus:
#         p = norm.cdf(-z)
#     else:
#         p = norm.cdf(z)

#     if log:
#         return np.log(p)
#     return p


# def dt_scaled(
#     x: np.ndarray, df: int, mean: float = 0, sd: float = 1, log: bool = False
# ) -> np.ndarray:
#     """
#     Density of scaled t-distribution

#     Parameters
#     ----------
#     x : array-like
#         Input values
#     df : int
#         Degrees of freedom
#     mean : float, default=0
#         Location parameter
#     sd : float, default=1
#         Scale parameter
#     log : bool, default=False
#         Return log density

#     Returns
#     -------
#     np.ndarray
#         Density or log density values
#     """
#     z = (x - mean) / sd
#     if log:
#         return t.logpdf(z, df) - np.log(sd)
#     return t.pdf(z, df) / sd


# def invert_hessian_matrix(coef_hessian, p, verbose=False):
#     """
#     Robustly invert a Hessian matrix, handling numerical stability issues.

#     Args:
#         coef_hessian: numpy array - The Hessian matrix to invert
#         p: int - Dimension of the matrix
#         verbose: bool - Whether to print warning messages

#     Returns:
#         numpy array - The inverted matrix (variance-covariance matrix)
#     """
#     # Initialize variance-covariance matrix with infinity on diagonal
#     Var_coef = np.diag([np.inf] * p)

#     # Find very small diagonal entries
#     very_small_entry = np.where(np.diag(coef_hessian) < 1e-10)[0]

#     # If all diagonal entries are practically zero
#     if len(very_small_entry) == p:
#         return Var_coef

#     # Make copy to avoid modifying original
#     coef_hessian = coef_hessian.copy()

#     # Replace very small diagonal entries
#     for idx in very_small_entry:
#         coef_hessian[idx, idx] = 1e-10

#     # Cap very large values
#     coef_hessian[coef_hessian > 1e10] = 1e10

#     try:
#         Var_coef = np.linalg.inv(coef_hessian)
#     except np.linalg.LinAlgError as err:
#         if verbose:
#             print(f"Warning: {err}")

#     return Var_coef


# def has_intercept(X: np.ndarray) -> bool:
#     """Check if design matrix has intercept column"""
#     return np.any(np.all(X == 1, axis=0))


# def calculate_sigma2_parameters(
#     fit_sigma2: float,
#     fit_sigma2_var: float,
#     variance_prior_scale: Optional[float],
#     variance_prior_df: Optional[float],
#     moderate_variance: bool,
#     n: int,
#     p: int,
# ) -> dict:
#     """
#     Calculate sigma2 parameters for variance moderation

#     Parameters
#     ----------
#     fit_sigma2 : float
#         Fitted variance parameter
#     fit_sigma2_var : float
#         Variance of fitted variance parameter
#     variance_prior_scale : float or None
#         Scale parameter for variance prior
#     variance_prior_df : float or None
#         Degrees of freedom for variance prior
#     moderate_variance : bool
#         Whether to use variance moderation
#     n : int
#         Sample size
#     p : int
#         Number of parameters

#     Returns
#     -------
#     dict
#         Dictionary containing n_approx, df_approx, s2_approx
#     """
#     if moderate_variance and variance_prior_scale is not None:
#         n_approx = 1 / fit_sigma2_var * fit_sigma2**2
#         df_approx = n_approx - p
#         s2_approx = fit_sigma2 * (n_approx - p) / n_approx
#     else:
#         n_approx = n
#         df_approx = n - p
#         s2_approx = fit_sigma2

#     return {"n_approx": n_approx, "df_approx": df_approx, "s2_approx": s2_approx}


# def calculate_skew_correction_factors(
#     y: np.ndarray,
#     yo: np.ndarray,
#     X: np.ndarray,
#     Xm: np.ndarray,
#     Xo: np.ndarray,
#     beta: np.ndarray,
#     sigma2: float,
#     Var_coef: np.ndarray,
#     rho: np.ndarray,
#     zetastar: np.ndarray,
#     location_prior_mean: Optional[float],
#     location_prior_scale: Optional[float],
#     variance_prior_df: Optional[float],
#     variance_prior_scale: Optional[float],
#     location_prior_df: int,
#     moderate_location: bool,
#     moderate_variance: bool,
# ) -> np.ndarray:
#     """Calculate correction factors for skewness in parameter estimates"""
#     # Implementation depends on specific requirements
#     # This is a placeholder that returns ones
#     return np.ones(len(y))
