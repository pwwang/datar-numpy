import pytest
import numpy as np

SENTINEL = 85258525.85258525


def _isnull(x):
    if x is None:
        return True

    try:
        return np.isnan(x)
    except TypeError:
        return False


def _isscalar(x):
    if isinstance(x, (str, bytes)):
        return True
    try:
        iter(x)
    except TypeError:
        return True
    return False


def assert_iterable_equal(x, y, na=SENTINEL, approx=False):
    x = [na if _isnull(elt) else elt for elt in x]
    y = [na if _isnull(elt) else elt for elt in y]
    if approx is True:
        x = pytest.approx(x)
    elif approx:
        x = pytest.approx(x, rel=approx)
    assert x == y, f"x: {x}, y: {y}"


# pytest modifies ast node for assert
def assert_equal(x, y, na_equal=True, approx=False):
    if _isnull(x) and _isnull(y):
        assert na_equal, f"na_equal is False, x: {x}, y: {y}"
    else:
        y = pytest.approx(y) if approx is True else pytest.approx(y, rel=approx)
        assert x == y, f"x: {x}, y: {y}"
