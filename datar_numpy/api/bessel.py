from datar.apis.base import (
    bessel_i,
    bessel_j,
    bessel_k,
    bessel_y,
)


def _get_special_func_from_scipy(name):
    """Import bessel functions from scipy on the fly"""
    try:
        from scipy import special
    except ImportError as imperr:  # pragma: no cover
        raise ImportError(
            "`bessel` family requires `scipy` package.\n"
            "Try: pip install -U scipy"
        ) from imperr

    return getattr(special, name)


@bessel_i.register(object)
def _bessel_i(x, nu, expon_scaled: bool = False):
    if nu == 0 and expon_scaled:
        fn = _get_special_func_from_scipy("i0e")
    elif nu == 1 and expon_scaled:
        fn = _get_special_func_from_scipy("i1e")
    elif nu == 0 and not expon_scaled:
        fn = _get_special_func_from_scipy("i0")
    elif nu == 1 and not expon_scaled:
        fn = _get_special_func_from_scipy("i1")
    elif expon_scaled:
        fn = _get_special_func_from_scipy("ive")
    else:
        fn = _get_special_func_from_scipy("iv")

    return fn(nu, x)


@bessel_j.register(object)
def _bessel_j(x, nu):
    if nu == 0:
        fn = _get_special_func_from_scipy("j0")
    elif nu == 1:
        fn = _get_special_func_from_scipy("j1")
    else:
        fn = _get_special_func_from_scipy("jv")

    return fn(nu, x)


@bessel_k.register(object)
def _bessel_k(x, nu, expon_scaled: bool = False):
    if nu == 0 and expon_scaled:
        fn = _get_special_func_from_scipy("k0e")
    elif nu == 1 and expon_scaled:
        fn = _get_special_func_from_scipy("k1e")
    elif nu == 0 and not expon_scaled:
        fn = _get_special_func_from_scipy("k0")
    elif nu == 1 and not expon_scaled:
        fn = _get_special_func_from_scipy("k1")
    elif expon_scaled:
        fn = _get_special_func_from_scipy("kve")
    else:
        fn = _get_special_func_from_scipy("kv")

    return fn(nu, x)


@bessel_y.register(object)
def _bessel_y(x, nu):
    if nu == 0:
        fn = _get_special_func_from_scipy("y0")
    elif nu == 1:
        fn = _get_special_func_from_scipy("y1")
    else:
        fn = _get_special_func_from_scipy("yv")

    return fn(nu, x)

