from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from anndata import AnnData
    from typing import List

import markov_clustering as mc
import networkx as nx
import numpy as np
import pandas as pd
import scanpy as sc

from numpy.linalg import det, eigvals, inv
from scipy.special import gammaln  # pylint: disable=no-name-in-module
from scipy.stats import multivariate_normal


def _get_clusters(matrix):
    # get the attractors - non-zero elements of the matrix diagonal
    attractors = matrix.diagonal().nonzero()[0]

    col = np.zeros(matrix.shape[0])

    # the nodes in the same row as each attractor form a cluster
    for i, attractor in enumerate(attractors):
        idx = matrix.getrow(attractor).nonzero()[1]
        col[idx] = i
    return col


def markov_clustering(adata: AnnData, resolution: float = 1.2, key_added: str = "mc_cluster"):
    """
    Perform Markov Clustering on the connectivity matrix.

    Parameters
    ----------
    adata
        The annotated data matrix.
    resolution
        The inflation parameter for the Markov Clustering (it is called resolution here for consistency with other clustering methods). Default is 1.2.
    key_added
        The key to add the cluster labels to in adata.obs. Default is "mc_cluster".
    """
    if "connectivities" not in adata.obsp:
        raise ValueError(
            "Connectivities matrix not found in adata.obsp, run `sc.pp.neighbors` first"
        )
    result = mc.run_mcl(adata.obsp["connectivities"], inflation=resolution)
    adata.obs[key_added] = _get_clusters(result)
    adata.obs[key_added] = adata.obs[key_added].astype("category")


# Find a good resolution for the leiden clustering
# For this we use the fact that we have good ground truth labels for mitochondria
# The following code will start with low resolution and increse it until the clustering splits mitochondria apart


def _majority_cluster_fraction(series):
    return series.value_counts().max() / series.value_counts().sum()


def leiden_mito_sweep(
    data: AnnData,
    starting_resolution: float = 0.5,
    resolution_increments: float = 0.5,
    min_mito_fraction: float = 0.9,
    increment_threshold: float = 0.005,
    protein_ground_truth_column: str = "protein_ground_truth",
    **leiden_kwargs,
) -> None:
    """Find optimal leiden clustering resolution based on mitochondrial protein clustering.

    Performs a binary search to find the highest resolution that keeps mitochondrial
    proteins clustered together above a minimum fraction threshold.

    Parameters
    ----------
    data
        Annotated data matrix with proteins as observations (rows)
    starting_resolution
        Initial resolution parameter for leiden clustering
    resolution_increments
        Step size for adjusting resolution during binary search
    min_mito_fraction
        Minimum fraction of mitochondrial proteins that should be in the same cluster
    increment_threshold
        Minimum step size before stopping binary search
    protein_ground_truth_column
        Column in data.obs containing protein localization annotations
    **leiden_kwargs
        Additional keyword arguments passed to scanpy.tl.leiden()

    Returns
    -------
    None
        Modifies data.obs['leiden'] and data.uns['leiden'] inplace
    """

    over_before = True
    mito_majority_fraction = 1
    data.obs["leiden"] = "0"
    while resolution_increments > increment_threshold:
        # Binary search for the highest resolution that is over the fraction by less than precision_threshold
        leiden_col = data.obs["leiden"]
        last_mito_majority_fraction = mito_majority_fraction
        sc.tl.leiden(data, resolution=starting_resolution, **leiden_kwargs)
        test_col = data.obs.loc[
            data.obs[protein_ground_truth_column] == "mitochondria", "leiden"
        ]
        mito_majority_fraction = _majority_cluster_fraction(test_col)
        print(f"Resolution: {starting_resolution}, Increment: {resolution_increments}")
        print(f"Majority mito cluster fraction: {mito_majority_fraction}")

        currently_over = mito_majority_fraction > min_mito_fraction
        if over_before != currently_over:
            resolution_increments /= 2
        starting_resolution += (
            resolution_increments if currently_over else -resolution_increments
        )
        over_before = currently_over

    # Set the leiden clusters to the last resolution
    if not currently_over:
        data.obs["leiden"] = leiden_col
        data.uns["leiden"]["params"]["resolution"] = (
            starting_resolution - resolution_increments
        )
        data.uns["leiden"]["mito_majority_fraction"] = last_mito_majority_fraction
    else:
        data.uns["leiden"]["mito_majority_fraction"] = mito_majority_fraction


def _get_knn_annotation_df(
    data: AnnData, obs_ann_col: str, exclude_category: str | List[str] | None = None
) -> pd.DataFrame:
    """
    Get a dataframe with a column of .obs repeated for each protein.
    """
    nrow = data.obs.shape[0]
    obs_ann = data.obs[obs_ann_col]
    if isinstance(exclude_category, str):
        exclude_category = [exclude_category]
    if exclude_category is not None:
        obs_ann.replace(exclude_category, np.nan, inplace=True)

    df = pd.DataFrame(np.tile(obs_ann, (nrow, 1)))
    return df


def knn_annotation(
    data: AnnData,
    obs_ann_col: str,
    key_added: str = "consensus_graph_annotation",
    exclude_category: str | List[str] | None = None,
    inplace: bool = True,
) -> AnnData | None:
    """Annotate proteins based on their k-nearest neighbors.

    For each protein, looks at its k-nearest neighbors and assigns the most common
    annotation among them.

    Parameters
    ----------
    data
        Annotated data matrix with proteins as observations (rows)
    obs_ann_col
        Column in data.obs containing annotations to propagate
    key_added
        Key under which to add the annotations in data.obs
    exclude_category
        Category or list of categories to exclude from annotation propagation
    inplace
        If True, modify the data in place. If False, return the modified data.

    Returns
    -------
    None | AnnData
        Modified AnnData object with new annotations in .obs[key_added]
    """
    df = _get_knn_annotation_df(data, obs_ann_col, exclude_category)

    conn = data.obsp["distances"]
    mask = ~(conn != 0).todense()  # This avoids expensive conn == 0 for sparse matrices
    df[mask] = np.nan

    majority_cluster = df.mode(axis=1, dropna=True).loc[
        :, 0
    ]  # take the first if there are ties
    data.obs[key_added] = majority_cluster.values
    return data if not inplace else None


def to_knn_graph(
    data: AnnData,
    node_label_column: str | None = None,
    neighbors_key: str | None = None,
    obsp: str | None = None,
) -> nx.Graph:
    """Convert AnnData object to a NetworkX graph.

    Parameters
    ----------
    data
        Annotated data matrix with proteins as observations (rows)
    node_label_column
        Column in data.obs to use as node labels. If None, use observation names
    neighbors_key
        The key passed to sc.pp.neighbors. If not specified, the default key is used
    obsp
        Key in data.obsp where adjacency matrix is stored. Takes precedence over neighbors_key

    Returns
    -------
    G
        NetworkX graph with nodes labeled according to node_label_column
    """

    if node_label_column is None:
        node_labels = data.obs_names
    else:
        node_labels = data.obs[node_label_column]

    # Convert the adjacency matrix to a networkx graph
    adjacency = sc._utils._choose_graph(data, obsp, neighbors_key=neighbors_key)
    G = nx.from_scipy_sparse_array(adjacency)

    # Relabel the nodes with the cell names
    G = nx.relabel_nodes(G, {i: node_labels[i] for i in G.nodes})

    return G


def _get_n_nearest_neighbors(G, node, n=10):
    # Ensure the node exists in the graph
    if node not in G:
        raise ValueError(f"Node {node} not in graph")

    neighbors = G[node]
    # Sort neighbors by edge weight in descending order and get the top n
    closest_neighbors = sorted(neighbors.items(), key=lambda x: x[1]["weight"], reverse=True)[
        :n
    ]
    closest_neighbor_nodes = [neighbor for neighbor, _ in closest_neighbors]

    return closest_neighbor_nodes


def get_n_nearest_neighbors(G, node: str, order: int = 1, n: int = 10):
    """Get n nearest neighbors up to a specified order.

    Parameters
    ----------
    G
        NetworkX graph
    node
        Node to find neighbors for
    order
        Order of neighbors to find (1 = direct neighbors, 2 = neighbors of neighbors, etc.)
    n
        Number of nearest neighbors to find at each step

    Returns
    -------
    set
        Set of nodes that are neighbors up to the specified order
    """
    all_neighbors = {node}
    current_neighbors = {node}
    i = 0
    while i < order:
        next_neighbors = set()
        for neighbor in current_neighbors:
            neighbors = _get_n_nearest_neighbors(G, neighbor, n)
            next_neighbors.update(neighbors)
        current_neighbors = next_neighbors
        all_neighbors.update(current_neighbors)
        i += 1
    return all_neighbors


def _modified_jaccard_coeff(
    row: pd.Series,
    organelle_counts: pd.Series,
    norm_degrees_to_def_top_partites: bool = True,
    min_partite_deg: int = 3,
):
    if norm_degrees_to_def_top_partites:
        counts = row
        counts[counts < min_partite_deg] = 0
        counts_norm = counts / organelle_counts[counts.index]
        counts_norm = counts_norm.sort_values(ascending=False)
        counts = counts[counts_norm.index]
        nl = counts.iloc[:2].replace(0, np.nan)
    else:
        nl = row.nlargest(2)

    d1, d2 = nl.values
    k1, k2 = nl.index
    # p_second_largest = organelle_counts[k2] / organelle_counts.sum()
    # p_interfacial = -scipy.stats.binom.logsf(d2-1, d1+d2, p_second_largest)
    jaccard = (d1 + d2) / (organelle_counts[k1] + organelle_counts[k2] - (d1 + d2))
    # d1overd2 = d2/d1#(d1 + d2) / (organelle_counts[k1] + organelle_counts[k2] - (d1 + d2))
    return jaccard, d1, d2, k1, k2


def calculate_interfacialness_score(
    data: AnnData,
    compartment_annotation_column: str,
    neighbors_key: str | None = None,
    obsp: str | None = None,
    exclude_category: str | List[str] | None = None,
) -> AnnData:
    """Calculate interfacialness scores for proteins based on their neighborhood annotations.

    For each protein, examines its nearest neighbors and calculates a modified Jaccard coefficient
    between the two most frequent compartment annotations in the neighborhood. This provides a
    measure of how "interfacial" a protein is between different cellular compartments.

    Parameters
    ----------
    data
        Annotated data matrix with proteins as observations (rows)
    compartment_annotation_column
        Column in data.obs containing compartment annotations
    neighbors_key
        Key for neighbors in data.uns. If not specified, will look for neighbors in obsp
    obsp
        Key for neighbors in data.obsp. Only used if neighbors_key not specified
    exclude_category
        Category or list of categories to exclude from the analysis

    Returns
    -------
    data
        Original AnnData object with added columns in .obs:
        - jaccard_score: Modified Jaccard coefficient measuring interfacialness
        - jaccard_d1: Number of neighbors with most frequent annotation
        - jaccard_d2: Number of neighbors with second most frequent annotation
        - jaccard_k1: Most frequent compartment annotation
        - jaccard_k2: Second most frequent compartment annotation
    """

    if compartment_annotation_column not in data.obs.columns:
        raise ValueError(
            f"Compartment annotation column {compartment_annotation_column} not found in .obs"
        )

    # Get full protein x protein matrix filled with annotations
    df = _get_knn_annotation_df(
        data, compartment_annotation_column, exclude_category=exclude_category
    )
    # Mask non-neighbors with np.nan
    adjacency = sc._utils._choose_graph(data, obsp, neighbors_key=neighbors_key)
    mask = ~(adjacency != 0).todense()  # This avoids expensive conn == 0 for sparse matrices
    df[mask] = np.nan
    vc = df.apply(lambda x: x.value_counts(dropna=True), axis=1).fillna(0).astype("Int64")

    # For each protein, calculate the modified jaccard coefficient
    organelle_counts = data.obs[compartment_annotation_column].value_counts()
    res = vc.apply(
        _modified_jaccard_coeff,
        axis=1,
        result_type="expand",
        organelle_counts=organelle_counts,
    )
    res.columns = [
        "jaccard_score",
        "jaccard_d1",
        "jaccard_d2",
        "jaccard_k1",
        "jaccard_k2",
    ]
    res["jaccard_d2"].replace(np.nan, 0, inplace=True)  # nans come from zero counts
    res["jaccard_score"].replace(np.nan, 0, inplace=True)  # nans come from zero counts

    # Annotate the data with the interfacialness scores
    res.index = data.obs.index
    data.obs = pd.concat([data.obs, res], axis=1)

    return data


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
    adata,
    gt_col="markers",
    method="MAP",
    numIter=100,
    mu0=None,
    lambda0=0.01,
    nu0=None,
    S0=None,
    beta0=None,
    u=2,
    v=10,
    seed=None,
    inplace=True,
):
    """
    Train the TAGM MAP model on an AnnData object.

    The function splits the data into labelled (marker) and unlabelled subsets
    based on adata.obs[gt_col]. It then computes MAP estimates for a T-augmented
    Gaussian mixture model using an EM algorithm.

    Parameters
    ----------
    adata : AnnData
        AnnData object containing the proteomics data.
    gt_col : str, optional
        Column in adata.obs with marker definitions (default "markers").
    numIter : int, optional
        Number of EM iterations (default 100).
    mu0 : np.ndarray, optional
        Prior mean vector (default: column means of marker data).
    lambda0 : float, optional
        Prior shrinkage (default 0.01).
    nu0 : int, optional
        Prior degrees of freedom (default: D+2).
    S0 : np.ndarray, optional
        Prior inverse-Wishart scale matrix (default: empirical).
    beta0 : np.ndarray, optional
        Prior Dirichlet concentration (default: ones for each class).
    u : int, optional
        Beta prior parameter for outlier probability (default u=2).
    v : int, optional
        Beta prior parameter for outlier probability (default v=10).
    seed : int, optional
        Random seed.
    inplace : bool, optional
        If True, modifies the input AnnData object in place.

    Returns
    -------
    dict | None
        Dictionary of MAP parameters. If inplace is True, the input AnnData object is modified in place.
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
    adata,
    params=None,
    gt_col=None,
    probJoint=False,
    probOutlier=True,
    inplace=True,
) -> pd.DataFrame | None:
    """
    Predict subcellular localization for unknown proteins using MAP parameters.

    This function calculates the probabilities for each component (both Gaussian and
    t-distribution contributions) for each unlabelled protein (where adata.obs[gt_col] is NA).
    It assigns the label from the Gaussian part with the highest probability and stores
    the predictions and probabilities in adata.obs.

    Parameters
    ----------
    adata : AnnData
        AnnData object containing the proteomics data.
    params : dict, optional
        MAP parameters as returned by tagm_map_train. If None, the function will try to
        retrieve them from adata.uns["tagm.map.params"].
    gt_col : str, optional
        Column in adata.obs with marker labels (default is "markers").
    probJoint : bool, optional
        If True, also compute and store the joint probability matrix (default is False).
    probOutlier : bool, optional
        If True, store the probability of being an outlier (default is True).
    inplace : bool, optional
        If True, modify the input AnnData object in place.

    Returns
    -------
    pd.DataFrame | None
        The DataFrame with new observation columns containing the predictions and probabilities. If inplace is True, the input AnnData object is modified in place.
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
    gt_col = params["gt_col"] if gt_col is None else gt_col

    # Split data.
    marker_idx = adata.obs[gt_col].notna()
    unknown_idx = adata.obs[gt_col].isna()
    adata_markers = adata[marker_idx].copy()
    adata_unknown = adata[unknown_idx].copy()
    markers = np.sort(adata_markers.obs[gt_col].unique())
    K = len(markers)

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
