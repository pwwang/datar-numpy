import numpy as np
from datar.apis.base import (
    arg,
    conj,
    mod,
    re_,
    im,
)


@arg.register(object)
def _arg(x):
    return np.angle(x)


@conj.register(object)
def _conj(x):
    return np.conj(x)


@mod.register(object)
def _mod(x):
    return np.absolute(x)


@re_.register(object)
def _re_(x):
    return np.real(x)


@im.register(object)
def _im(x):
    return np.imag(x)
