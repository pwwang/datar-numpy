import pytest

from datar.base import cut
from .utils import assert_iterable_equal


def test_cut():
    out1 = cut([1, 2, 3, 4, 5], 3)
    assert_iterable_equal(out1[0], [1.0, 2.333333333333333], approx=True)
    assert_iterable_equal(out1[1], [1.0, 2.333333333333333], approx=True)
    assert_iterable_equal(out1[2], [2.333333333, 3.6666666666], approx=True)
    assert_iterable_equal(out1[3], [3.6666666666, 5.0], approx=True)
    assert_iterable_equal(out1[4], [3.6666666666, 5.0], approx=True)

    out2 = cut([1, 2, 3, 4, 5], [0, 2, 4, 6])
    assert_iterable_equal(out2[0], [0.0, 2.0], approx=True)
    assert_iterable_equal(out2[1], [2.0, 4.0], approx=True)
    assert_iterable_equal(out2[2], [2.0, 4.0], approx=True)
    assert_iterable_equal(out2[3], [4.0, 6.0], approx=True)
    assert_iterable_equal(out2[4], [4.0, 6.0], approx=True)
