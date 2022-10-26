import numpy as np
from datar.apis.base import (
    beta,
    lgamma,
    digamma,
    trigamma,
    choose,
    factorial,
    gamma,
    lfactorial,
    lchoose,
    lbeta,
    psigamma,
)

from .bessel import _get_special_func_from_scipy


@beta.register(object)
def _beta(x, y):
    return _get_special_func_from_scipy("beta")(x, y)


@lgamma.register(object)
def _lgamma(x):
    return _get_special_func_from_scipy("gammaln")(x)


@digamma.register(object)
def _digamma(x):
    return _get_special_func_from_scipy("psi")(x)


@trigamma.register(object)
def _trigamma(x):
    return _get_special_func_from_scipy("polygamma")(1, x)


@choose.register(object)
def _choose(n, k):
    return _get_special_func_from_scipy("binom")(n, k)


@factorial.register(object)
def _factorial(x):
    return _get_special_func_from_scipy("factorial")(x)


@gamma.register(object)
def _gamma(x):
    return _get_special_func_from_scipy("gamma")(x)


@lfactorial.register(object)
def _lfactorial(x):
    return np.log(_get_special_func_from_scipy("factorial")(x))


@lchoose.register(object)
def _lchoose(n, k):
    return np.log(_get_special_func_from_scipy("binom")(n, k))


@lbeta.register(object)
def _lbeta(x, y):
    return _get_special_func_from_scipy("betaln")(x, y)


@psigamma.register(object)
def _psigamma(x, deriv):
    return _get_special_func_from_scipy("polygamma")(np.round(deriv), x)
