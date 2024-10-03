from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from anndata import AnnData
    from typing import List, Sequence

from functools import reduce

import numpy as np
import umap


def align_adatas(data_list: List[AnnData]) -> List[AnnData]:
    var_intersect = reduce(lambda x, y: x.var_names.intersection(y.var_names), data_list)
    obs_intersect = reduce(lambda x, y: x.obs_names.intersection(y.obs_names), data_list)
    data_sub_list = []
    for data in data_list:
        data_sub = data.copy()
        data_sub = data_sub[:, var_intersect]
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
    # Make sure all anndata objects have the same var_names and obs_names
    if align_data:
        data_sub_list = align_adatas(data_list)

    assert reduce(lambda x, y: x.var_names == y.var_names, data_sub_list).all()
    assert reduce(lambda x, y: x.obs_names == y.obs_names, data_sub_list).all()

    embeddings = [data.X for data in data_sub_list]
    constant_relations = [{i: i for i in range(data_sub_list[0].shape[0])}]
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
    embeddings = [data.obsm[aligned_umap_key] for data in data_list]
    remodeling_score = _remodeling_score(embeddings)
    for i, data in enumerate(data_list):
        data.obs[key_added] = remodeling_score

    return data_list
