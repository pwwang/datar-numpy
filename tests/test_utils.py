import pytest  # noqa
import numpy as np
from datar.core import plugin  # noqa
from datar_numpy.utils import is_scalar, make_array
from .utils import assert_iterable_equal


def test_is_scalar():
    assert is_scalar(1)
    assert is_scalar(np.ndarray)


def test_make_array():
    assert_iterable_equal(make_array(1), [1])
    assert_iterable_equal(make_array([1, 2]), [1, 2])
    assert_iterable_equal(make_array(np.array([1, 2])), [1, 2])
    # assert_iterable_equal(make_array({"a": 1, "b": 2}), ["a", "b"])
    assert_iterable_equal(make_array(["1", "2"], dtype=int), [1, 2])
    assert_iterable_equal(make_array(["1", np.nan]), ["1", np.nan])
