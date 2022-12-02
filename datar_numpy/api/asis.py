from __future__ import annotations

from numbers import Number
from typing import Any

import numpy as np
from datar.apis.base import (
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
    # as_date,
    as_double,
    as_integer,
    as_logical,
    as_null,
    as_numeric,
)

from ..utils import is_scalar, is_null as _is_null_, make_array
from .constants import NULL


def _is_type(x, pytype, npdtype):
    if isinstance(x, pytype):
        return True

    if isinstance(x, np.ndarray):
        return np.issubdtype(x.dtype, npdtype)

    x = make_array(x)
    return np.issubdtype(x.dtype, npdtype)


def _as_type(x, pytype, npdtype):
    if is_scalar(x):
        if isinstance(x, np.ndarray):
            x = x.item()
        return pytype(x)

    x = make_array(x)

    if np.issubdtype(x.dtype, npdtype):
        return x

    return x.astype(npdtype)


@is_atomic.register(object, backend="numpy")
def _is_atomic(x: Any) -> bool:
    return is_scalar(x)


@is_character.register(object, backend="numpy")
def _is_character(x: Any) -> bool:
    return _is_type(x, str, np.str_)


@is_complex.register(object, backend="numpy")
def _is_complex(x: Any) -> bool:
    return _is_type(x, complex, np.complex_)


@is_double.register(object, backend="numpy")
def _is_double(x: Any) -> bool:
    return _is_type(x, float, np.float_)


@is_integer.register(object, backend="numpy")
def _is_integer(x: Any) -> bool:
    return _is_type(x, int, np.integer)


@is_element.register(object, backend="numpy")
def _is_element(x: Any, y: Any) -> bool:
    return np.isin(x, y)


@is_finite.register(object, backend="numpy")
def _is_finite(x: Any) -> bool:
    return np.isfinite(x)


@is_false.register(object, backend="numpy")
def _is_false(x: Any) -> bool:
    return x is False or np.array_equal(x, False)


@is_infinite.register(object, backend="numpy")
def _is_infinite(x: Any) -> bool:
    return np.isinf(x)


@is_logical.register(object, backend="numpy")
def _is_logical(x: Any) -> bool:
    return _is_type(x, bool, np.bool_)


@is_na.register(object, backend="numpy")
def _is_na(x: Any) -> bool | np.ndarray[bool]:
    return _is_null_(x)


@is_null.register(object, backend="numpy")
def _is_null(x: Any) -> bool:
    return x is NULL or np.array_equal(x, NULL)


@is_numeric.register(object, backend="numpy")
def _is_numeric(x: Any) -> bool:
    return _is_type(x, Number, np.number)


@is_true.register(object, backend="numpy")
def _is_true(x: Any) -> bool:
    return x is True or np.array_equal(x, True)


@as_character.register(object, backend="numpy")
def _as_character(x: Any) -> str | np.ndarray[str]:
    return _as_type(x, str, np.str_)


@as_complex.register(object, backend="numpy")
def _as_complex(x: Any) -> complex | np.ndarray[complex]:
    return _as_type(x, complex, np.complex_)


@as_double.register(object, backend="numpy")
def _as_double(x: Any) -> float | np.ndarray[float]:
    return _as_type(x, float, np.float_)


@as_integer.register(object, backend="numpy")
def _as_integer(x: Any) -> int | np.ndarray[int]:
    return _as_type(x, int, np.int_)


@as_logical.register(object, backend="numpy")
def _as_logical(x: Any) -> bool | np.ndarray[bool]:
    return _as_type(x, bool, np.bool_)


@as_null.register(object, backend="numpy")
def _as_null(x: Any) -> None:
    return NULL


@as_numeric.register(object, backend="numpy")
def _as_numeric(x: Any) -> Number | np.ndarray[Number]:
    try:
        return _as_type(x, int, np.int_)
    except (TypeError, ValueError):
        pass

    try:
        return _as_type(x, float, np.float_)
    except (TypeError, ValueError):
        pass

    try:
        return _as_type(x, complex, np.complex_)
    except (TypeError, ValueError):
        pass

    raise ValueError(f"Cannot convert {x} to numeric")
