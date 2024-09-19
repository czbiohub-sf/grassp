from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from anndata import AnnData
    from typing import List

import networkx as nx
import numpy as np
import pandas as pd
import scanpy as sc


def knn_annotation(
    data: AnnData,
    orig_ann_col: str,
    key_added: str = "consensus_graph_annotation",
    exclude_category: str | List[str] | None = None,
) -> AnnData:
    nrow = data.obs.shape[0]
    orig_ann = data.obs[orig_ann_col]
    if isinstance(exclude_category, str):
        exclude_category = [exclude_category]
    orig_ann.replace(exclude_category, np.nan, inplace=True)

    df = pd.DataFrame(np.tile(orig_ann, (nrow, 1)))
    conn = data.obsp['distances']
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
    closest_neighbors = sorted(neighbors.items(), key=lambda x: x[1]['weight'], reverse=True)[
        :n
    ]
    closest_neighbor_nodes = [neighbor for neighbor, _ in closest_neighbors]

    return closest_neighbor_nodes


def get_n_nearest_neighbors(G, node: str, order: int = 1, n: int = 10):
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
