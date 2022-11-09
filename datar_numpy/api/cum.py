import numpy as np

from datar.apis.base import (
    cummax,
    cummin,
    cumprod,
    cumsum,
)


@cummax.register(object, backend="numpy")
def _cummax(x):
    return np.maximum.accumulate(x)


@cummin.register(object, backend="numpy")
def _cummin(x):
    return np.minimum.accumulate(x)


@cumprod.register(object, backend="numpy")
def _cumprod(x):
    return np.cumprod(x)


@cumsum.register(object, backend="numpy")
def _cumsum(x):
    return np.cumsum(x)
