import numpy as np

from datar.apis.base import (
    cummax,
    cummin,
    cumprod,
    cumsum,
)


@cummax.register(object)
def _cummax(x):
    return np.maximum.accumulate(x)


@cummin.register(object)
def _cummin(x):
    return np.minimum.accumulate(x)


@cumprod.register(object)
def _cumprod(x):
    return np.cumprod(x)


@cumsum.register(object)
def _cumsum(x):
    return np.cumsum(x)
