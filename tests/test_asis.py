import pytest
import datetime
import numpy as np
from datar.base import (
    is_atomic,
    is_character,
    is_complex,
    is_double,
    is_integer,
    is_element,
    is_finite,
    is_false,
    is_infinite,
    is_logical,
    is_na,
    is_null,
    is_numeric,
    is_true,
    as_character,
    as_complex,
    as_date,
    as_double,
    as_integer,
    as_logical,
    as_null,
    as_numeric,
)
from .utils import assert_equal, _isscalar, assert_iterable_equal


@pytest.mark.parametrize(
    "fn, x, expected",
    [
        (is_atomic, 1, True),
        (is_atomic, [1], False),
        (is_atomic, np.array(1), True),
        (is_atomic, np.array([1]), False),
        (is_atomic, np.array([1, 2]), False),
        (is_atomic, np.array([[1, 2]]), False),
        (is_character, "a", True),
        (is_character, ["a"], True),
        (is_character, np.array("a"), True),
        (is_character, np.array(["a"]), True),
        (is_character, np.array(["a", "b"]), True),
        (is_character, 1, False),
        (is_character, [1], False),
        (is_character, np.array(1), False),
        (is_character, np.array([1]), False),
        (is_character, np.array([1, 2]), False),
        (is_complex, 1j, True),
        (is_complex, [1j], True),
        (is_complex, np.array(1j), True),
        (is_complex, np.array([1j]), True),
        (is_complex, np.array([1j, 2j]), True),
        (is_complex, 1, False),
        (is_complex, [1], False),
        (is_complex, np.array(1), False),
        (is_complex, np.array([1]), False),
        (is_complex, np.array([1, 2]), False),
        (is_double, 1.0, True),
        (is_double, np.array(1.0), True),
        (is_double, [1.0], True),
        (is_double, np.array(1.0), True),
        (is_double, np.array([1.0]), True),
        (is_double, np.array([1.0, 2.0]), True),
        (is_double, 1, False),
        (is_double, [1], False),
        (is_double, np.array(1), False),
        (is_double, np.array([1]), False),
        (is_double, np.array([1, 2]), False),
        (is_integer, 1, True),
        (is_integer, [1], True),
        (is_integer, np.array(1), True),
        (is_integer, np.array([1]), True),
        (is_integer, np.array([1, 2]), True),
        (is_integer, 1.0, False),
        (is_integer, [1.0], False),
        (is_integer, np.array(1.0), False),
        (is_integer, np.array([1.0]), False),
        (is_integer, np.array([1.0, 2.0]), False),
        (is_false, False, True),
        (is_false, np.array(False), True),
        (is_false, np.array([False]), False),
        (is_false, np.array([False, True]), False),
        (is_true, True, True),
        (is_true, np.array(True), True),
        (is_true, np.array([True]), False),
        (is_true, np.array([True, False]), False),
        (is_logical, True, True),
        (is_logical, False, True),
        (is_logical, np.array(True), True),
        (is_logical, np.array(False), True),
        (is_logical, np.array([True]), True),
        (is_logical, np.array([False]), True),
        (is_logical, np.array([True, False]), True),
        (is_logical, np.array([True, False, True]), True),
        (is_logical, np.array([True, False, 1]), False),
        (is_numeric, 1, True),
        (is_numeric, 1.0, True),
        (is_numeric, 1j, True),
        (is_numeric, np.array(1), True),
        (is_numeric, np.array(1.0), True),
        (is_numeric, np.array(1j), True),
        (is_numeric, np.array([1]), True),
        (is_numeric, np.array([1.0]), True),
        (is_numeric, np.array([1j]), True),
        (is_numeric, np.array([1, 2]), True),
        (is_numeric, np.array([1.0, 2.0]), True),
        (is_numeric, np.array([1j, 2j]), True),
        (is_numeric, np.array([1, 2j]), True),
        (is_numeric, np.array([1, 2.0]), True),
        (is_numeric, np.array([1.0, 2j]), True),
        (is_null, None, True),
        (is_null, np.array(None), True),
        (is_null, np.array([None]), False),
        (is_finite, 1, True),
        (is_finite, 1.0, True),
        (is_finite, 1j, True),
        (is_finite, np.array(1), True),
        (is_finite, np.array(1.0), True),
        (is_finite, np.array(1j), True),
        (is_finite, np.array([1]), [True]),
        (is_finite, np.array([1.0]), [True]),
        (is_finite, np.array([1j]), [True]),
        (is_finite, np.array([1, 2]), [True, True]),
        (is_finite, np.array([1.0, 2.0]), [True, True]),
        (is_finite, np.array([1j, 2j]), [True, True]),
        (is_finite, np.array([1, 2j]), [True, True]),
        (is_finite, np.array([1, 2.0]), [True, True]),
        (is_finite, np.array([1.0, 2j]), [True, True]),
        (is_finite, np.array([1, np.inf]), [True, False]),
        (is_finite, np.array([1, np.nan]), [True, False]),
        (is_finite, np.array([1, np.inf, np.nan]), [True, False, False]),
        (is_infinite, 1, False),
        (is_infinite, 1.0, False),
        (is_infinite, 1j, False),
        (is_infinite, np.array(1), False),
        (is_infinite, np.array(1.0), False),
        (is_infinite, np.array(1j), False),
        (is_infinite, np.array([1]), [False]),
        (is_infinite, np.array([1.0]), [False]),
        (is_infinite, np.array([1j]), [False]),
        (is_infinite, np.array([1, 2]), [False, False]),
        (is_infinite, np.array([1.0, 2.0]), [False, False]),
        (is_infinite, np.array([1j, 2j]), [False, False]),
        (is_infinite, np.array([1, 2j]), [False, False]),
        (is_infinite, np.array([1, 2.0]), [False, False]),
        (is_infinite, np.array([1.0, 2j]), [False, False]),
        (is_infinite, np.array([1, np.inf]), [False, True]),
        (is_infinite, np.array([1, np.nan]), [False, False]),
        (is_infinite, np.array([1, np.inf, np.nan]), [False, True, False]),
        (is_na, np.array([np.nan]), [True]),
        (is_na, np.array([np.nan, np.nan]), [True, True]),
        (is_na, np.array([np.nan, 1]), [True, False]),
    ],
)
def test_is(fn, x, expected):
    if expected in (True, False):
        assert_equal(fn(x), expected)
    else:
        assert_iterable_equal(fn(x), expected)


@pytest.mark.parametrize(
    "fn, x, expected",
    [
        (as_character, 1, "1"),
        (as_character, 1.0, "1.0"),
        (as_character, 1j, "1j"),
        (as_character, np.array(1), "1"),
        (as_character, np.array(1.0), "1.0"),
        (as_character, np.array(1j), "1j"),
        (as_character, np.array([1]), ["1"]),
        (as_character, np.array([1.0]), ["1.0"]),
        (as_character, np.array([1j]), ["1j"]),
        (as_character, np.array([1, 2]), ["1", "2"]),
        (as_character, np.array([1.0, 2.0]), ["1.0", "2.0"]),
        (as_character, np.array([1j, 2j]), ["1j", "2j"]),
        (as_complex, 1, 1),
        (as_complex, 1.0, 1.0),
        (as_complex, 1j, 1j),
        (as_complex, np.array(1), 1),
        (as_complex, np.array(1.0), 1.0),
        (as_complex, np.array(1j), 1j),
        (as_complex, np.array([1]), [1]),
        (as_complex, np.array([1.0]), [1.0]),
        (as_complex, np.array([1j]), [1j]),
        (as_complex, np.array([1, 2]), [1, 2]),
        (as_complex, np.array([1.0, 2.0]), [1.0, 2.0]),
        (as_complex, np.array([1j, 2j]), [1j, 2j]),
        (as_complex, np.array([1, 2j]), [1, 2j]),
        (as_complex, np.array([1, 2.0]), [1, 2.0]),
        (as_complex, np.array([1.0, 2j]), [1.0, 2j]),
        (as_complex, np.array([1, np.inf]), [1, np.inf]),
        (as_double, 1, 1),
        (as_double, 1.0, 1.0),
        (as_double, "1", 1.0),
        (as_integer, 1, 1),
        (as_integer, 1.0, 1),
        (as_integer, "1", 1),
        (as_integer, np.array("1"), 1),
        (as_integer, [np.array("1")], [1]),
        (as_logical, 1, True),
        (as_logical, 1.0, True),
        (as_logical, "1", True),
        (as_logical, 0, False),
        (as_logical, 0.0, False),
        (as_logical, "0", True),
        (as_logical, np.array([1, 0]), [True, False]),
        (as_logical, np.array([1.0, 0.0]), [True, False]),
        (as_null, 1, None),
        (as_null, 1.0, None),
        (as_null, "1", None),
        (as_null, np.array([1, 0]), None),
        (as_numeric, 1, 1),
        (as_numeric, 1.0, 1.0),
        (as_numeric, np.array("1.0"), 1.0),
        (as_numeric, "1.1", 1.1),
        (as_numeric, "1+1j", 1 + 1j),
        (as_numeric, "1", 1.0),
        (as_numeric, np.array(["1", "0"]), [1, 0]),
    ],
)
def test_as(fn, x, expected):
    if _isscalar(expected):
        assert_equal(fn(x), expected)
    else:
        assert_iterable_equal(fn(x), expected)


def test_is_element():
    assert_equal(is_element(1, [1, 2, 3]), True)
    assert_equal(is_element(1, [2, 3]), False)
    assert_equal(is_element(1, np.array([1, 2, 3])), True)
    assert_equal(is_element(1, np.array([2, 3])), False)
    assert_iterable_equal(
        is_element([1, 2], np.array([1, 2, 3])), [True, True]
    )


def test_as_numeric_error():
    with pytest.raises(ValueError):
        as_numeric("a")


@pytest.mark.parametrize(
    "x,format,try_formats,optional,tz,origin,expected",
    [
        (
            ["1jan1960", "2jan1960", "31mar1960", "30jul1960"],
            "%d%b%Y",
            None,
            False,
            0,
            None,
            [
                datetime.date(1960, 1, 1),
                datetime.date(1960, 1, 2),
                datetime.date(1960, 3, 31),
                datetime.date(1960, 7, 30),
            ],
        ),
        (
            np.datetime64("2005-02-25"),
            "%d%b%Y",
            None,
            False,
            0,
            None,
            datetime.date(2005, 2, 25),
        ),
        (
            ["02/27/92", "02/27/92", "01/14/92", "02/28/92", "02/01/92"],
            "%m/%d/%y",
            None,
            False,
            0,
            None,
            [
                datetime.date(1992, 2, 27),
                datetime.date(1992, 2, 27),
                datetime.date(1992, 1, 14),
                datetime.date(1992, 2, 28),
                datetime.date(1992, 2, 1),
            ],
        ),
        (
            32768,
            None,
            None,
            False,
            0,
            "1900-01-01",
            datetime.date(1989, 9, 19),
        ),
        (
            35981,
            None,
            None,
            False,
            0,
            "1899-12-30",
            datetime.date(1998, 7, 5),
        ),
        (
            34519,
            None,
            None,
            False,
            0,
            "1904-01-01",
            datetime.date(1998, 7, 5),
        ),
        (
            734373 - 719529,
            None,
            None,
            False,
            0,
            datetime.datetime(1970, 1, 1),
            datetime.date(2010, 8, 23),
        ),
        (
            [datetime.date(2010, 4, 13)],
            None,
            None,
            False,
            12,
            None,
            [datetime.date(2010, 4, 13)],
        ),
        (
            [datetime.date(2010, 4, 13)],
            None,
            None,
            False,
            12 + 13,
            None,
            [datetime.date(2010, 4, 14)],
        ),
        (
            [datetime.datetime(2010, 4, 13)],
            None,
            None,
            False,
            0 - 10,
            None,
            [datetime.date(2010, 4, 12)],
        ),
    ],
)
def test_as_date(x, format, try_formats, optional, tz, origin, expected):
    out = as_date(
        x,
        format=format,
        try_formats=try_formats,
        optional=optional,
        tz=tz,
        origin=origin,
    )
    if np.ndim(out) == 0:
        assert_equal(out, expected)
    else:
        assert_iterable_equal(out, expected)


def test_as_date_error():
    with pytest.raises(NotImplementedError):
        as_date(1.1)

    with pytest.raises(ValueError):
        as_date("1990-1-1", format="%Y")

    out = as_date("1990-1-1", format="%Y", optional=True)
    assert np.isnan(out)
