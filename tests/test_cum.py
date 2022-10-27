import pytest

from datar.base import (
    cummax,
    cummin,
    cumprod,
    cumsum,
)
from .utils import assert_iterable_equal


@pytest.mark.parametrize(
    "fn, x, expected",
    [
        (cummax, [1, 2, 3], [1, 2, 3]),
        (cummax, [1, 3, 2], [1, 3, 3]),
        (cummax, [3, 2, 1], [3, 3, 3]),
        (cummin, [1, 2, 3], [1, 1, 1]),
        (cummin, [1, 3, 2], [1, 1, 1]),
        (cummin, [3, 2, 1], [3, 2, 1]),
        (cumprod, [1, 2, 3], [1, 2, 6]),
        (cumprod, [1, 3, 2], [1, 3, 6]),
        (cumprod, [3, 2, 1], [3, 6, 6]),
        (cumsum, [1, 2, 3], [1, 3, 6]),
        (cumsum, [1, 3, 2], [1, 4, 6]),
        (cumsum, [3, 2, 1], [3, 5, 6]),
    ],
)
def test_cum(fn, x, expected):
    assert_iterable_equal(fn(x), expected)
