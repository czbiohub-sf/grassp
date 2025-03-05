from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from anndata import AnnData
    from typing import List, Sequence

from functools import reduce

import numpy as np
import umap


def align_adatas(
    data_list: List[AnnData], intersect_obs: bool = True, intersect_var: bool = True
) -> List[AnnData]:
    """Align multiple AnnData objects by intersecting their observations and variables.

    Parameters
    ----------
    data_list
        List of AnnData objects to align
    intersect_obs
        Whether to intersect observations (rows) across all objects
    intersect_var
        Whether to intersect variables (columns) across all objects

    Returns
    -------
    list
        List of aligned AnnData objects with matching observations and variables
    """

    obs = [data.obs_names for data in data_list]
    if intersect_obs:
        obs_intersect = reduce(lambda x, y: x.intersection(y), obs)
    else:
        obs_intersect = None
    if intersect_var:
        var = [data.var_names for data in data_list]
        var_intersect = reduce(lambda x, y: x.intersection(y), var)
    else:
        var_intersect = None
    data_sub_list = []
    for data in data_list:
        data_sub = data.copy()
        if var_intersect is not None:
            data_sub = data_sub[:, var_intersect]
        if obs_intersect is not None:
            data_sub = data_sub[obs_intersect, :]
        data_sub_list.append(data_sub)

    return data_sub_list


def aligned_umap(
    data_list: List[AnnData],
    align_data: bool = True,
    return_data_objects: bool = True,
    n_neighbors: int = 20,
    metric: str = "euclidean",
    min_dist: float = 0.1,
    alignment_regularisation: float = 0.002,
    n_epochs: int = 300,
    random_state: int | None = None,
    verbose: bool = False,
    n_components: int = 2,
) -> umap.AlignedUMAP:
    """Calculate aligned UMAP embeddings for multiple datasets.

    Parameters
    ----------
    data_list
        List of AnnData objects to align
    align_data
        Whether to align data by intersecting observations and variables
    return_data_objects
        Whether to return aligned AnnData objects with UMAP embeddings
    n_neighbors
        Number of neighbors to use for UMAP
    metric
        Distance metric for UMAP
    min_dist
        Minimum distance between points in UMAP embedding
    alignment_regularisation
        Strength of alignment regularization between datasets
    n_epochs
        Number of epochs to optimize embeddings
    random_state
        Random seed for reproducibility
    verbose
        Whether to display progress updates
    n_components
        Number of dimensions for UMAP embedding

    Returns
    -------
    data_sub_list or umap.AlignedUMAP
        If return_data_objects is True, returns list of aligned AnnData objects with UMAP embeddings.
        Otherwise returns fitted AlignedUMAP object.
    """

    # Make sure all anndata objects have the same var_names and obs_names
    if align_data:
        data_sub_list = align_adatas(data_list)
    else:
        data_sub_list = data_list

    var_names = [data.var_names for data in data_sub_list]
    assert all(var_names[0].equals(index) for index in var_names[1:])
    obs_names = [data.obs_names for data in data_sub_list]
    assert all(obs_names[0].equals(index) for index in obs_names[1:])

    embeddings = [data.X for data in data_sub_list]
    constant_relation = {i: i for i in range(data_sub_list[0].shape[0])}
    constant_relations = [constant_relation.copy() for i in range(len(embeddings) - 1)]
    # return constant_relations
    neighbors_mapper = umap.AlignedUMAP(
        n_neighbors=n_neighbors,
        metric=metric,
        min_dist=min_dist,
        alignment_regularisation=alignment_regularisation,
        n_epochs=n_epochs,
        random_state=random_state,
        verbose=verbose,
        n_components=n_components,
    ).fit(embeddings, relations=constant_relations)

    if return_data_objects:
        for i, data in enumerate(data_sub_list):
            data.obsm["X_aligned_umap"] = neighbors_mapper.embeddings_[i]
            data.uns["aligned_umap"] = {
                "params": {
                    "a": neighbors_mapper.mappers_[i].a,
                    "b": neighbors_mapper.mappers_[i].b,
                },
                "alignment_params": {
                    "alignment_regularisation": neighbors_mapper.alignment_regularisation,
                    "n_epochs": neighbors_mapper.n_epochs,
                    "random_state": neighbors_mapper.random_state,
                    "n_components": neighbors_mapper.n_components,
                },
            }
        return data_sub_list
    else:
        return neighbors_mapper


def _remodeling_score(
    embeddings: Sequence[np.ndarray],
) -> np.ndarray:
    assert len(embeddings) == 2
    distances = embeddings[0] - embeddings[1]

    return np.linalg.norm(distances, axis=1)


def remodeling_score(
    data_list: List[AnnData],
    aligned_umap_key: str = "X_aligned_umap",
    key_added: str = "remodeling_score",
) -> List[AnnData]:
    """Get aligned UMAP embeddings from each AnnData object.

    Parameters
    ----------
    data_list
        List of AnnData objects containing aligned UMAP embeddings
    aligned_umap_key
        Key in .obsm where aligned UMAP embeddings are stored

    Returns
    -------
    embeddings
        List of numpy arrays containing aligned UMAP embeddings
    """
    embeddings = [data.obsm[aligned_umap_key] for data in data_list]
    remodeling_score = _remodeling_score(embeddings)
    for i, data in enumerate(data_list):
        data.obs[key_added] = remodeling_score

    return data_list
