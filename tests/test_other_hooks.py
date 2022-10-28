import pytest
from datar import get_versions
from datar.base import c
from .utils import assert_equal, assert_iterable_equal


def test_get_versions():
    versions = get_versions(prnt=False)
    assert "datar-numpy" in versions
    assert "numpy" in versions


def test_c_getitem():
    assert_equal(c[1], 1)
    assert_equal(c["a"], "a")
    assert_iterable_equal(c[1, 2], (1, 2))
    assert_iterable_equal(c[1:3], [1, 2])
    assert_iterable_equal(c[1:3:1], [1, 2, 3])
    assert_iterable_equal(c[1:3:1, 4], [1, 2, 3, 4])
