import pytest
import numpy as np
from datar.base import (
    ceiling,
    cov,
    floor,
    mean,
    median,
    pmax,
    pmin,
    sqrt,
    var,
    scale,
    min_,
    max_,
    round_,
    sum_,
    abs_,
    prod,
    sign,
    signif,
    trunc,
    exp,
    log,
    log2,
    log10,
    log1p,
    sd,
    weighted_mean,
    quantile,
    proportions,
)
from .utils import assert_equal, assert_iterable_equal, _isscalar


@pytest.mark.parametrize(
    "fn, x, expected",
    [
        (ceiling, 1.1, 2),
        (ceiling, [1.1], [2]),
        (floor, 1.1, 1),
        (floor, [1.1], [1]),
        (round_, 1.1, 1),
        (round_, [1.1], [1]),
        (round_, 1.5, 2),
        (round_, [1.5], [2]),
        (mean, [1, 2, 3], 2),
        (median, [1, 2, 3], 2),
        (sqrt, 4, 2),
        (sqrt, [4], [2]),
        (abs_, -1, 1),
        (abs_, 1, 1),
        (abs_, [-1], [1]),
        (abs_, [1], [1]),
        (sign, -1, -1),
        (sign, 1, 1),
        (sign, [-1], [-1]),
        (sign, [1], [1]),
        (trunc, 1.1, 1),
        (trunc, [1.1], [1]),
        (exp, 1, 2.718281828459045),
        (exp, [1], [2.718281828459045]),
        (log2, 2, 1),
        (log2, [2], [1]),
        (log10, 10, 1),
        (log10, [10], [1]),
        (log1p, 1, 0.6931471805599453),
        (log1p, [1], [0.6931471805599453]),
        (
            proportions,
            [1, 2, 3],
            [0.16666666666666666, 0.3333333333333333, 0.5],
        ),
    ],
)
def test_single_arg(fn, x, expected):
    out = fn(x)
    if _isscalar(expected):
        approx = isinstance(expected, float)
        assert_equal(out, expected, approx=approx)
    else:
        approx = isinstance(expected[0], float)
        assert_iterable_equal(out, expected)


@pytest.mark.parametrize(
    "fn, x, na_rm, expected",
    [
        (min_, [1, 2, 3], False, 1),
        (min_, [1, 2, 3], True, 1),
        (min_, [1, 2, 3, np.nan], False, np.nan),
        (min_, [1, 2, 3, np.nan], True, 1),
        (max_, [1, 2, 3], False, 3),
        (max_, [1, 2, 3], True, 3),
        (max_, [1, 2, 3, np.nan], False, np.nan),
        (max_, [1, 2, 3, np.nan], True, 3),
        (sum_, [1, 2, 3], False, 6),
        (sum_, [1, 2, 3], True, 6),
        (sum_, [1, 2, 3, np.nan], False, np.nan),
        (sum_, [1, 2, 3, np.nan], True, 6),
        (prod, [1, 2, 3], False, 6),
        (prod, [1, 2, 3], True, 6),
        (prod, [1, 2, 3, np.nan], False, np.nan),
        (prod, [1, 2, 3, np.nan], True, 6),
        (sd, [1, 2, 3], False, 1.0),
        (sd, [1, 2, 3], True, 1.0),
        (sd, [1, 2, 3, np.nan], False, np.nan),
        (sd, [1, 2, 3, np.nan], True, 1.0),
    ],
)
def test_single_arg_with_na_rm(fn, x, na_rm, expected):
    out = fn(x, na_rm=na_rm)
    if _isscalar(expected):
        approx = isinstance(expected, float)
        assert_equal(out, expected, approx=approx)
    else:
        approx = isinstance(expected[0], float)
        assert_iterable_equal(out, expected)


def test_cov():
    x = [1, 2, 3]
    y = [4, 5, 6]
    assert_equal(cov(x, y), 1.0, approx=True)

    with pytest.raises(ValueError):
        cov(x)


def test_pmax_pmin():
    x = [1, 5, 3]
    y = [4, 2, 6]
    assert_iterable_equal(pmax(x, y), [4, 5, 6])
    assert_iterable_equal(pmin(x, y), [1, 2, 3])


def test_var():
    x = [1, 2, 3]
    assert_equal(var(x), 1.0, approx=True)
    assert_equal(var(x, ddof=0), 0.6666666666666666, approx=True)


def test_scale():
    x = [1, 2, 3]
    assert_iterable_equal(scale(x), [-1, 0, 1])
    assert_iterable_equal(
        scale(x, center=1), [0.0, 0.632455532, 1.264911064], approx=True
    )
    assert_iterable_equal(scale(x, scale_=2), [-0.5, 0.0, 0.5], approx=True)


def test_signif():
    x = [1.234, 5.678, 9.012]
    assert_iterable_equal(signif(x, 2), [1.2, 5.7, 9.0])
    assert_iterable_equal(signif(x, 1), [1.0, 6.0, 9.0])
    assert_iterable_equal(signif(x, 0), [1.0, 6.0, 9.0])
    assert_iterable_equal(signif(x, -1), [1.0, 6.0, 9.0])


def test_log():
    x = [1, 2, 3]
    assert_iterable_equal(
        log(x),
        [0, 0.6931471805599453, 1.0986122886681098],
        approx=True,
    )
    assert_iterable_equal(
        log(x, base=2),
        [0, 1, 1.5849625007211563],
        approx=True,
    )
    assert_iterable_equal(
        log(x, base=10),
        [0, 0.3010299956639812, 0.47712125471966244],
        approx=True,
    )
    assert_iterable_equal(
        log(x, base=np.e),
        [0, 0.6931471805599453, 1.0986122886681098],
        approx=True,
    )


def test_weighted_mean():
    x = [1, 2, 3]
    x2 = [np.nan] * 3
    w = [1, 2, 3]
    w2 = [1, 2, np.nan]
    w3 = [-1, 0, 1]
    assert_equal(weighted_mean(x, w), 2.333333, approx=True)
    assert_equal(weighted_mean(x, w, na_rm=True), 2.333333, approx=True)
    assert_equal(weighted_mean(x), 2)
    assert_equal(weighted_mean(x, w3), np.nan)
    assert_equal(weighted_mean(x2, w2, na_rm=True), np.nan)


def test_quantile():
    x = [1, 2, 3]
    assert_equal(quantile(x, 0.5), 2)
    assert_equal(quantile(x, 0.5, na_rm=True), 2)
    assert_equal(quantile(x, 0.5, na_rm=False), 2)
    assert_iterable_equal(quantile(x, [0.25, 0.75]), [1.5, 2.5])
    assert_iterable_equal(quantile(x, [0.25, 0.75], na_rm=True), [1.5, 2.5])
    assert_iterable_equal(
        quantile([*x, np.nan], [0.25, 0.75], na_rm=True),
        [1.5, 2.5],
    )
    assert_iterable_equal(quantile(x, [0.25, 0.75], na_rm=False), [1.5, 2.5])
    assert_iterable_equal(
        quantile(x, [0.25, 0.75], na_rm=False, type_=7),
        [1.5, 2.5],
    )
    assert_iterable_equal(
        quantile(x, [0.25, 0.75], na_rm=False, type_=8),
        [1.1666666666666667, 2.8333333333333335],
        approx=True,
    )
