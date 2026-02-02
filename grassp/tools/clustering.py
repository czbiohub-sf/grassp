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

from .localization import _get_knn_annotation_df


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
    """Run *Markov Clustering* (MCL) on the neighbour graph.

    The algorithm operates on the *connectivities* matrix written by
    :func:`scanpy.pp.neighbors` and assigns a cluster label to every
    observation.  Labels are stored as a categorical column in
    ``adata.obs[key_added]``.

    Parameters
    ----------
    adata
        Annotated data matrix which already contains ``adata.obsp['connectivities']``.
    resolution
        Inflation parameter of the MCL algorithm. Larger values yield more, smaller
        clusters.  Typical range: ``1.2`` – ``5``.
    key_added
        Observation key used to store the resulting cluster labels
        (default ``"mc_cluster"``).

    Returns
    -------
    None
        The function modifies ``adata`` in place.
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
    """Automated sweep to pick a *Leiden* resolution that keeps mitochondria intact.

    Starting from ``starting_resolution`` the function iteratively increases
    or decreases the resolution (binary-search style) until *mitochondrial*
    proteins—identified via ``protein_ground_truth_column``—remain grouped in
    a single Leiden cluster with at least ``min_mito_fraction`` of all
    mitochondrial proteins.

    The final resolution and the observed fraction are stored in
    ``data.uns['leiden']['params']['resolution']`` and
    ``data.uns['leiden']['mito_majority_fraction']``.

    Parameters
    ----------
    data
        :class:`anndata.AnnData` object *after* :func:`scanpy.pp.neighbors` with proteins as
        observations.
    starting_resolution
        Initial Leiden resolution from which to begin the sweep.
    resolution_increments
        Step size added (or subtracted) each iteration; halved whenever the
        sweep crosses the ``min_mito_fraction`` threshold.
    min_mito_fraction
        Required proportion of mitochondrial proteins that must share a
        cluster (default ``0.9``).
    increment_threshold
        When the increment falls below this value the search terminates.
    protein_ground_truth_column
        Observation column holding the curated ground-truth localisation with
        the category ``"mitochondria"``.
    **leiden_kwargs
        Extra parameters forwarded to :func:`scanpy.tl.leiden` (e.g.
        ``random_state``).

    Returns
    -------
    Updates ``data.obs['leiden']`` and ``data.uns['leiden']`` in place.
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


def to_knn_graph(
    data: AnnData,
    node_label_column: str | None = None,
    neighbors_key: str | None = None,
    obsp: str | None = None,
) -> nx.Graph:
    """Convert the *k*-NN graph stored in ``AnnData`` to a ``networkx`` graph.

    Parameters
    ----------
    data
        :class:`anndata.AnnData` that has been processed with
        :func:`scanpy.pp.neighbors` (or equivalent) so that either
        ``adata.obsp[obsp]`` or ``adata.uns[neighbors_key]`` exists.
    node_label_column
        Observation column whose values become node labels in the resulting
        graph.  If ``None`` (default), ``data.obs_names`` is used.
    neighbors_key
        Key under which Scanpy stored neighbour information (defaults to
        ``'neighbors'``).  Ignored if *obsp* is provided.
    obsp
        Name of a pre-computed adjacency/connectivity matrix in ``adata.obsp``.
        Takes precedence over *neighbors_key*.

    Returns
    -------
    networkx.Graph
        Undirected, weighted graph where edge weights correspond to the
        connectivities/distances of the *k*-NN graph.
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
    """Return the set of closest neighbours for a node in a graph.

    Starting from ``node`` the procedure repeatedly fetches the ``n`` strongest
    (highest weight) edges for every node discovered so far.  After *order*
    iterations the union of visited nodes is returned.

    Parameters
    ----------
    G
        Weighted :class:`networkx.Graph` – typically the output of
        :func:`to_knn_graph`.
    node
        Start node.
    order
        How many *hops* away to expand.  ``1`` (default) returns the direct
        *k*-NN; ``2`` also includes neighbours of neighbours, and so on.
    n
        Number of neighbours per node considered at each expansion step.

    Returns
    -------
    set[str]
        Node identifiers within the specified neighbourhood radius (including
        the *start* node).
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
    """Quantify *interfacialness* of proteins across compartment boundaries.

    The score is based on a **modified Jaccard index** computed from each
    protein’s immediate neighbourhood:

    1. For a given protein count how many of its neighbours belong to each
       compartment (categories in ``compartment_annotation_column``).
    2. Sort counts and take the two highest: ``d1`` and ``d2`` for
       compartments ``k1`` and ``k2``.
    3. Compute

       ``score = (d1 + d2) / (N_k1 + N_k2 - (d1 + d2))``

       where ``N_k`` is the total number of proteins annotated as compartment
       *k* in the dataset.  High scores indicate that a protein sits at an
       interface between two compartments.

    New columns are appended to ``data.obs`` with the *jaccard_* prefix.

    Parameters
    ----------
    data
        :class:`anndata.AnnData` with a neighbour graph and compartment
        annotations for each protein.
    compartment_annotation_column
        Observation column containing the ground-truth compartment labels.
    neighbors_key, obsp
        Specify which neighbour graph to use (mirrors Scanpy conventions).
    exclude_category
        One or multiple category labels (e.g. *'unknown'*) to ignore when
        counting neighbours.

    Returns
    -------
    :class:'~anndata.AnnData` object with additional columns:

        ``jaccard_score``
            Interfacialness score.
        ``jaccard_d1``, ``jaccard_d2``
            Counts of the two dominating neighbour compartments.
        ``jaccard_k1``, ``jaccard_k2``
            Corresponding compartment labels.
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
