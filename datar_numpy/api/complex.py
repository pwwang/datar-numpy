import numpy as np
from datar.apis.base import (
    arg,
    conj,
    mod,
    re_,
    im,
)


@arg.register(object, backend="numpy")
def _arg(x):
    return np.angle(x)


@conj.register(object, backend="numpy")
def _conj(x):
    return np.conj(x)


@mod.register(object, backend="numpy")
def _mod(x):
    return np.absolute(x)


@re_.register(object, backend="numpy")
def _re_(x):
    return np.real(x)


@im.register(object, backend="numpy")
def _im(x):
    return np.imag(x)
