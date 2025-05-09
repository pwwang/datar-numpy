import pytest  # noqa: F401
import numpy as np  # noqa: F401
from datar.base import which, which_max, which_min
from .utils import assert_iterable_equal, assert_equal


def test_which():
    assert_iterable_equal(which([True, False, True]), [0, 2])


def test_which_min():
    assert_equal(which_min([1, 2, 3]), 0)


def test_which_max():
    assert_equal(which_max([1, 2, 3]), 2)
