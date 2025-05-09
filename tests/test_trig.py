import pytest  # noqa: F401
import numpy as np
from datar.base import (
    acos,
    acosh,
    asin,
    asinh,
    atan,
    atanh,
    cos,
    cosh,
    cospi,
    sin,
    sinh,
    sinpi,
    tan,
    tanh,
    tanpi,
    atan2,
)
from .utils import assert_equal


def test_acos():
    assert_equal(acos(0.5), np.arccos(0.5))


def test_acosh():
    assert_equal(acosh(1.5), np.arccosh(1.5))


def test_asin():
    assert_equal(asin(0.5), np.arcsin(0.5))


def test_asinh():
    assert_equal(asinh(1.5), np.arcsinh(1.5))


def test_atan():
    assert_equal(atan(0.5), np.arctan(0.5))


def test_atanh():
    assert_equal(atanh(0.5), np.arctanh(0.5))


def test_cos():
    assert_equal(cos(0.5), np.cos(0.5))


def test_cosh():
    assert_equal(cosh(0.5), np.cosh(0.5))


def test_cospi():
    assert_equal(cospi(0.5), np.cos(np.pi * 0.5))


def test_sin():
    assert_equal(sin(0.5), np.sin(0.5))


def test_sinh():
    assert_equal(sinh(0.5), np.sinh(0.5))


def test_sinpi():
    assert_equal(sinpi(0.5), np.sin(np.pi * 0.5))


def test_tan():
    assert_equal(tan(0.5), np.tan(0.5))


def test_tanh():
    assert_equal(tanh(0.5), np.tanh(0.5))


def test_tanpi():
    assert_equal(tanpi(0.5), np.tan(np.pi * 0.5))


def test_atan2():
    assert_equal(atan2(0.5, 0.5), np.arctan2(0.5, 0.5))
