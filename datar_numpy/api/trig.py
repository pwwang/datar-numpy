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


@acos.register(object)
def _acos(x):
    return np.arccos(x)


@acosh.register(object)
def _acosh(x):
    return np.arccosh(x)


@asin.register(object)
def _asin(x):
    return np.arcsin(x)


@asinh.register(object)
def _asinh(x):
    return np.arcsinh(x)


@atan.register(object)
def _atan(x):
    return np.arctan(x)


@atanh.register(object)
def _atanh(x):
    return np.arctanh(x)


@cos.register(object)
def _cos(x):
    return np.cos(x)


@cosh.register(object)
def _cosh(x):
    return np.cosh(x)


@cospi.register(object)
def _cospi(x):
    return np.cos(np.pi * x)


@sin.register(object)
def _sin(x):
    return np.sin(x)


@sinh.register(object)
def _sinh(x):
    return np.sinh(x)


@sinpi.register(object)
def _sinpi(x):
    return np.sin(np.pi * x)


@tan.register(object)
def _tan(x):
    return np.tan(x)


@tanh.register(object)
def _tanh(x):
    return np.tanh(x)


@tanpi.register(object)
def _tanpi(x):
    return np.tan(np.pi * x)


@atan2.register(object)
def _atan2(y, x):
    return np.arctan2(y, x)
