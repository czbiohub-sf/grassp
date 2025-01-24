from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from anndata import AnnData
    from typing import List

import networkx as nx
import numpy as np
import pandas as pd
import scanpy as sc

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
) -> AnnData:
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

    Returns
    -------
    data
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
    return data


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
    closest_neighbors = sorted(
        neighbors.items(), key=lambda x: x[1]["weight"], reverse=True
    )[:n]
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
    mask = ~(
        adjacency != 0
    ).todense()  # This avoids expensive conn == 0 for sparse matrices
    df[mask] = np.nan
    vc = (
        df.apply(lambda x: x.value_counts(dropna=True), axis=1)
        .fillna(0)
        .astype("Int64")
    )

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
