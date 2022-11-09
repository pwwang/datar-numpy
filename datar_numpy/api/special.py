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


@beta.register(object, backend="numpy")
def _beta(x, y):
    return _get_special_func_from_scipy("beta")(x, y)


@lgamma.register(object, backend="numpy")
def _lgamma(x):
    return _get_special_func_from_scipy("gammaln")(x)


@digamma.register(object, backend="numpy")
def _digamma(x):
    return _get_special_func_from_scipy("psi")(x)


@trigamma.register(object, backend="numpy")
def _trigamma(x):
    return _get_special_func_from_scipy("polygamma")(1, x)


@choose.register(object, backend="numpy")
def _choose(n, k):
    return _get_special_func_from_scipy("binom")(n, k)


@factorial.register(object, backend="numpy")
def _factorial(x):
    return _get_special_func_from_scipy("factorial")(x)


@gamma.register(object, backend="numpy")
def _gamma(x):
    return _get_special_func_from_scipy("gamma")(x)


@lfactorial.register(object, backend="numpy")
def _lfactorial(x):
    return np.log(_get_special_func_from_scipy("factorial")(x))


@lchoose.register(object, backend="numpy")
def _lchoose(n, k):
    return np.log(_get_special_func_from_scipy("binom")(n, k))


@lbeta.register(object, backend="numpy")
def _lbeta(x, y):
    return _get_special_func_from_scipy("betaln")(x, y)


@psigamma.register(object, backend="numpy")
def _psigamma(x, deriv):
    return _get_special_func_from_scipy("polygamma")(np.round(deriv), x)
