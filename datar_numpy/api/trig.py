import numpy as np
from datar.apis.base import (
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


@acos.register(object, backend="numpy")
def _acos(x):
    return np.arccos(x)


@acosh.register(object, backend="numpy")
def _acosh(x):
    return np.arccosh(x)


@asin.register(object, backend="numpy")
def _asin(x):
    return np.arcsin(x)


@asinh.register(object, backend="numpy")
def _asinh(x):
    return np.arcsinh(x)


@atan.register(object, backend="numpy")
def _atan(x):
    return np.arctan(x)


@atanh.register(object, backend="numpy")
def _atanh(x):
    return np.arctanh(x)


@cos.register(object, backend="numpy")
def _cos(x):
    return np.cos(x)


@cosh.register(object, backend="numpy")
def _cosh(x):
    return np.cosh(x)


@cospi.register(object, backend="numpy")
def _cospi(x):
    return np.cos(np.pi * x)


@sin.register(object, backend="numpy")
def _sin(x):
    return np.sin(x)


@sinh.register(object, backend="numpy")
def _sinh(x):
    return np.sinh(x)


@sinpi.register(object, backend="numpy")
def _sinpi(x):
    return np.sin(np.pi * x)


@tan.register(object, backend="numpy")
def _tan(x):
    return np.tan(x)


@tanh.register(object, backend="numpy")
def _tanh(x):
    return np.tanh(x)


@tanpi.register(object, backend="numpy")
def _tanpi(x):
    return np.tan(np.pi * x)


@atan2.register(object, backend="numpy")
def _atan2(y, x):
    return np.arctan2(y, x)
