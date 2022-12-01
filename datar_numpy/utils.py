from __future__ import annotations

import warnings
from collections import namedtuple
from functools import lru_cache
from numbers import Number
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from numpy._typing import DTypeLike

Version = namedtuple("Version", ["major", "minor", "patch"])


@lru_cache(maxsize=1)
def numpy_version() -> Version:
    """Get the version of numpy"""
    return Version(*map(int, np.__version__.split(".")[:3]))


def is_scalar(x: Any) -> bool:
    """Is x a scalar?
    Using np.ndim() instead of np.isscalar() to allow an object (i.e. a dict)
    a scalar, because it can be used as data in a cell of a dataframe.

    Args:
        x: The object to check

    Returns:
        True if x is a scalar, False otherwise
    """
    if isinstance(x, (list, set, tuple, dict)):
        # np.ndim({'a'}) == 0
        return False

    if isinstance(x, type) and issubclass(x, (np.generic, np.ndarray)):
        # np.ndim(np.generic)
        #   <attribute 'ndim' of 'numpy.generic' objects>
        return True

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", np.VisibleDeprecationWarning)
        return np.ndim(x) == 0


def is_null(x: Any) -> bool | np.ndarray[bool]:
    """Is x None or NA? Like pandas.isnull()

    Args:
        x: The object to check

    Returns:
        If x is scalar, return True if x is None or NA, False otherwise.
        If x is an array, return a boolean array with the same shape as x.
    """

    try:
        return np.isnan(x)
    except TypeError:

        isnull_atomic = (
            lambda x: x is None or (isinstance(x, Number) and np.isnan(x))
        )
        return np.vectorize(isnull_atomic, [bool])(x)


def make_array(x: Any, dtype: DTypeLike = None) -> np.ndarray:
    """Make an array from x"""
    if is_scalar(x):
        return np.array(x, dtype=dtype).ravel()

    if isinstance(x, np.ndarray):
        return x if dtype is None else x.astype(dtype)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", np.VisibleDeprecationWarning)
        out = np.array(x, dtype=dtype)

    if dtype is not None:
        return out

    # Keep NAs with strings
    # np.array(["a", np.nan]) turns into ["a", "nan"]
    # but we want ["a", np.nan]
    na_mask = is_null(x)
    if na_mask.any() and out.dtype.kind == "U":
        out = np.array(x, dtype=object)
    return out


def flatten_slice(x: slice) -> np.ndarray[int]:
    """Flatten a slice into an array of integers"""
    start = x.start or 0
    stop = x.stop or 0
    if x.step == 1:
        stop += 1
    return np.arange(start, stop, x.step)
