from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Union

    from numpy.random import RandomState

    AnyRandom = Union[int, RandomState, None]


import numpy as np

from anndata import AnnData


def impute_gaussian(
    data: AnnData,
    width: float = 0.3,
    distance: float = 1.8,
    per_sample: bool = True,
    random_state: AnyRandom = 0,
) -> AnnData:
    data = data.copy()
    np.random.seed(random_state)

    zero_mask = data.X != 0
    n_zeros = data.X.size - zero_mask.sum()

    if per_sample:
        pmean = data.X.mean(axis=1, where=zero_mask)
        stdev = data.X.std(axis=1, where=zero_mask)
    else:
        pmean = data.X.mean(where=zero_mask)
        stdev = data.X.std(where=zero_mask)

    imp_mean = pmean - distance * stdev
    imp_stdev = stdev * width

    if per_sample:
        imputed_values = np.random.normal(
            loc=imp_mean, scale=imp_stdev, size=data.X.shape[::-1]
        )
        data.X[np.invert(zero_mask)] = imputed_values.T[np.invert(zero_mask)]
    else:
        imputed_values = np.random.normal(loc=imp_mean, scale=imp_stdev, size=n_zeros)
        data.X[np.invert(zero_mask)] = imputed_values
    return data
