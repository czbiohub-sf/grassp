from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Union

    from numpy.random import RandomState

    AnyRandom = Union[int, RandomState, None]


import numpy as np

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
    confirm_proteins_as_obs(data)
    np.random.seed(random_state)

    if not inplace:
        data = data.copy()
    X = data.X

    zero_mask = X != 0
    n_zeros = X.size - zero_mask.sum()

    if per_sample:
        pmean = X.mean(axis=0, where=zero_mask)
        stdev = X.std(axis=0, where=zero_mask)
    else:
        pmean = X.mean(where=zero_mask)
        stdev = X.std(where=zero_mask)

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
