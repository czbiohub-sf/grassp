from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from anndata import AnnData
    from typing import List, Literal

import markov_clustering as mc
import networkx as nx
import numpy as np
import pandas as pd
import scanpy as sc

from .localization import _neighbor_label_matrix


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


def _interfacial_partites(
    row: pd.Series,
    organelle_counts: pd.Series,
    flavor: str,
    min_partite_deg: int = 3,
):
    """Pick a protein's two partite compartments and score it; see calculate_interfacialness_score.

    ``row`` holds the protein's neighbour count per compartment.
    """
    # Only compartments holding at least min_partite_deg of the neighbours can be partites;
    # rank those by degree normalised to compartment size.
    eligible = row[row >= min_partite_deg]
    ranked = (eligible / organelle_counts[eligible.index]).sort_values(ascending=False)
    nl = eligible[ranked.index[:2]]

    # With fewer than two partites the missing ones are reported as missing, as in the paper's
    # implementation. Ranking every compartment and zeroing the ineligible ones used to name an
    # arbitrary zero-count compartment (usually the first category) as k2, and as k1 too when
    # none qualified.
    d1, d2 = list(nl.values) + [np.nan] * (2 - len(nl))
    k1, k2 = list(nl.index) + [np.nan] * (2 - len(nl))
    if pd.isna(k2):
        return np.nan, d1, d2, k1, k2
    if flavor == "evenness":
        score = 2 * min(d1, d2) / (d1 + d2)
    else:  # "hein2024"
        score = (d1 + d2) / (organelle_counts[k1] + organelle_counts[k2] - (d1 + d2))
    return score, d1, d2, k1, k2


def calculate_interfacialness_score(
    data: AnnData,
    compartment_annotation_column: str,
    neighbors_key: str | None = None,
    obsp: str | None = None,
    exclude_category: str | List[str] | None = None,
    flavor: Literal["evenness", "hein2024"] = "evenness",
) -> AnnData:
    """Quantify *interfacialness*: how much a protein sits between two compartments.

    Both flavors look at each protein's direct neighbours in the neighbour graph and pick the
    same two *partite* compartments:

    1. Count the protein's neighbours in each compartment of
       ``compartment_annotation_column`` (``d_c`` for compartment *c*).
    2. Only compartments holding at least 3 of the neighbours are eligible.
    3. Rank the eligible compartments by ``d_c / N_c``, where ``N_c`` is the number of proteins
       annotated as *c* in the dataset, and take the top two: ``k1`` and ``k2`` with
       neighbour counts ``d1`` and ``d2``.

    Proteins with fewer than two eligible compartments score 0. The flavors differ only in how
    ``d1`` and ``d2`` are turned into a score.

    ``flavor="evenness"`` (default)
        ``score = 2 * min(d1, d2) / (d1 + d2)``: how evenly the protein's neighbours split
        between the two compartments. 1 is an even split; an *r*:1 split scores
        ``2 / (r + 1)`` (0.67 for 2:1, 0.5 for 3:1, 0.2 for 9:1). Compartment size does not
        enter the score, so the boundary between two large compartments (e.g. cytosol and
        nucleus) scores like the boundary between two small ones. The score is not
        long-tailed, so call proteins interfacial with a fixed split, e.g.
        ``score >= 2/3`` (at least 2:1), rather than with an outlier rule.

        Exclude catch-all labels such as ``"unclassified"`` (``exclude_category``): proteins
        without a clear localization lie between clusters, so being next to them would
        otherwise read as an even split.

    ``flavor="hein2024"``
        The modified Jaccard coefficient of Hein et al., *Cell* (2025), Fig. 3D
        (https://www.cell.com/cell/fulltext/S0092-8674(24)01344-8),
        ``score = (d1 + d2) / (N_k1 + N_k2 - (d1 + d2))``.
        This reproduces the paper's implementation
        (https://github.com/czbiohub-sf/Organelle_IP_analyses_and_figures, script
        ``utils/Jaccard_coefficient.py``), which used the ``connectivities`` graph, the
        consensus graph-based annotation with ``"unclassified"`` counted as a compartment,
        and called proteins interfacial above the 1.5 IQR upper bound of all scores (0.046).

        **This score is dominated by compartment size.** ``d1 + d2`` is bounded by the number
        of neighbours (a few dozen), while ``N_k1 + N_k2`` ranges from tens to thousands of
        proteins, so the score mostly tracks ``1 / (N_k1 + N_k2)``. A protein split evenly
        between cytosol and nucleus can score at most ~0.008 (with 26 neighbours), far below
        the paper's cutoff, while a protein inside a small compartment with only three
        neighbours in another small one can exceed it. How evenly the neighbours split does
        not enter the score, and proteins with many neighbours in the symmetrised graph
        score higher.

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
        One or multiple category labels (e.g. *'unclassified'*) to ignore when
        counting neighbours.
    flavor
        ``"evenness"`` (default) or ``"hein2024"``; see above.

    Returns
    -------
    :class:`~anndata.AnnData` object with additional columns:

        ``interfacial_score``
            Interfacialness score of the chosen ``flavor``; 0 with fewer than two partites.
        ``interfacial_d1``, ``interfacial_d2``
            Neighbour counts in the two partite compartments.
        ``interfacial_k1``, ``interfacial_k2``
            Corresponding compartment labels; missing when fewer than two (or no)
            compartments hold at least three of the protein's neighbours.

    and ``uns["interfacial_params"]`` recording ``flavor``, the annotation column and
    ``exclude_category``.
    """

    if compartment_annotation_column not in data.obs.columns:
        raise ValueError(
            f"Compartment annotation column {compartment_annotation_column} not found in .obs"
        )
    if flavor not in ("evenness", "hein2024"):
        raise ValueError(f"flavor must be 'evenness' or 'hein2024', got {flavor!r}")

    # Get full protein x protein matrix filled with annotations
    df = _neighbor_label_matrix(
        data, compartment_annotation_column, exclude_category=exclude_category
    )
    # Mask non-neighbors with np.nan
    adjacency = sc._utils._choose_graph(data, obsp, neighbors_key=neighbors_key)
    mask = ~(adjacency != 0).todense()  # This avoids expensive conn == 0 for sparse matrices
    df[mask] = np.nan
    vc = df.apply(lambda x: x.value_counts(dropna=True), axis=1).fillna(0).astype("Int64")

    # For each protein, pick the two partites and score it
    organelle_counts = data.obs[compartment_annotation_column].value_counts()
    res = vc.apply(
        _interfacial_partites,
        axis=1,
        result_type="expand",
        organelle_counts=organelle_counts,
        flavor=flavor,
    )
    res.columns = [
        "interfacial_score",
        "interfacial_d1",
        "interfacial_d2",
        "interfacial_k1",
        "interfacial_k2",
    ]
    # result_type="expand" over mixed numbers and labels gives every column object dtype. Give
    # the numeric ones a real dtype: write_h5ad cannot store numbers in an object column, and
    # fillna's silent downcast of object columns -- which used to do this for two of them -- is
    # deprecated in pandas. NaNs come from zero counts; interfacial_d1 keeps its missing values,
    # so it gets the nullable integer dtype.
    res["interfacial_score"] = pd.to_numeric(res["interfacial_score"]).fillna(0)
    res["interfacial_d1"] = pd.to_numeric(res["interfacial_d1"]).astype("Int64")
    res["interfacial_d2"] = pd.to_numeric(res["interfacial_d2"]).fillna(0).astype("int64")

    # Annotate the data with the interfacialness scores
    res.index = data.obs.index
    data.obs = pd.concat([data.obs, res], axis=1)
    data.uns["interfacial_params"] = {
        "flavor": flavor,
        "compartment_annotation_column": compartment_annotation_column,
        "exclude_category": exclude_category,
    }

    return data
