import pytest

from datar.base import (
    bessel_i,
    bessel_j,
    bessel_k,
    bessel_y,
)
from .utils import assert_equal, assert_iterable_equal, _isscalar


@pytest.mark.parametrize(
    "fn, x, nu, expon_scaled, expected",
    [
        (bessel_i, 1, 0, False, 1.2660658777520082),
        (bessel_i, [1, 2], 0, False, [1.2660658777520082, 2.279585302336067]),
        (bessel_i, 1, 1, False, 0.5651591039924851),
        (bessel_i, 1, 0, True, 0.46575960759364043),
        (bessel_i, 2, 0, False, 2.279585302336067),
        (bessel_i, 1, 2, False, 0.1357476697670383),
        (bessel_i, 1, 1, True, 0.2079104153497085),
        (bessel_k, 1, 0, False, 0.42102443824070823),
        (bessel_k, 1, 2, False, 1.6248388986351774),
        (bessel_k, 1, 1, False, 0.6019072301972346),
        (bessel_k, 1, 0, True, 1.1444630798068947),
        (bessel_k, 1, 1, True, 1.636153486263258),
    ],
)
def test_bessel_ik(fn, x, nu, expon_scaled, expected):
    out = fn(x, nu, expon_scaled)
    if _isscalar(x):
        assert_equal(out, expected, approx=True)
    else:
        assert_iterable_equal(out, expected, approx=True)


@pytest.mark.parametrize(
    "fn, x, nu, expected",
    [
        (bessel_j, 1, 0, 0.7651976865579666),
        (bessel_j, 1, 2, 0.1149034849319005),
        (bessel_j, 1, 1, 0.4400505857449335),
        (bessel_y, 1, 0, 0.08825696421567697),
        (bessel_y, 1, 2, -1.6506826068162548),
        (bessel_y, 1, 1, -0.7812128213002888),
    ],
)
def test_bessel_jy(fn, x, nu, expected):
    out = fn(x, nu)
    if _isscalar(x):
        assert_equal(out, expected, approx=True)
    else:
        assert_iterable_equal(out, expected, approx=True)
