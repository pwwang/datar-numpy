import pytest
import numpy as np
from datar.base import (
    arg,
    conj,
    mod,
    re_,
    im,
)
from .utils import assert_equal, assert_iterable_equal, _isscalar


@pytest.mark.parametrize("fn, x, expected", [
    (arg, 1j, 1.5707963267948966),
    (arg, [1j], [1.5707963267948966]),
    (arg, np.array(1j), 1.5707963267948966),
    (arg, np.array([1j]), [1.5707963267948966]),
    (arg, np.array([1j, 2j]), [1.5707963267948966, 1.5707963267948966]),
    (conj, 1j, -1j),
    (conj, [1j], [-1j]),
    (conj, np.array(1j), -1j),
    (conj, np.array([1j]), [-1j]),
    (conj, np.array([1j, 2j]), [-1j, -2j]),
    (mod, 1j, 1.0),
    (mod, [1j], [1.0]),
    (mod, np.array(1j), 1.0),
    (mod, np.array([1j]), [1.0]),
    (mod, np.array([1j, 2j]), [1.0, 2.0]),
    (re_, 1j, 0.0),
    (re_, [1j], [0.0]),
    (re_, np.array(1j), 0.0),
    (re_, np.array([1j]), [0.0]),
    (re_, np.array([1j, 2j]), [0.0, 0.0]),
    (im, 1j, 1.0),
    (im, [1j], [1.0]),
    (im, np.array(1j), 1.0),
    (im, np.array([1j]), [1.0]),
    (im, np.array([1j, 2j]), [1.0, 2.0]),
])
def test_complex(fn, x, expected):
    if _isscalar(expected):
        assert_equal(fn(x), expected, approx=True)
    else:
        assert_iterable_equal(fn(x), expected, approx=True)
