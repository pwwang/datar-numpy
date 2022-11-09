import numpy as np
from datar.apis.base import (
    which,
    which_min,
    which_max,
)


@which.register(object, backend="numpy")
def _which(x):
    return np.flatnonzero(x)


@which_min.register(object, backend="numpy")
def _which_min(x):
    return np.argmin(x)


@which_max.register(object, backend="numpy")
def _which_max(x):
    return np.argmax(x)
