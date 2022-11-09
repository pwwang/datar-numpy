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


@bessel_i.register(object, backend="numpy")
def _bessel_i(x, nu, expon_scaled: bool = False):
    if nu not in (0, 1):
        fn = "ive" if expon_scaled else "iv"
        return _get_special_func_from_scipy(fn)(nu, x)

    if nu == 0 and expon_scaled:
        fn = "i0e"
    elif nu == 1 and expon_scaled:
        fn = "i1e"
    elif nu == 0 and not expon_scaled:
        fn = "i0"
    elif nu == 1 and not expon_scaled:
        fn = "i1"
    return _get_special_func_from_scipy(fn)(x)


@bessel_j.register(object, backend="numpy")
def _bessel_j(x, nu):
    if nu not in (0, 1):
        return _get_special_func_from_scipy("jv")(nu, x)

    fn = "j0" if nu == 0 else "j1"
    return _get_special_func_from_scipy(fn)(x)


@bessel_k.register(object, backend="numpy")
def _bessel_k(x, nu, expon_scaled: bool = False):
    if nu not in (0, 1):
        fn = "kve" if expon_scaled else "kv"
        return _get_special_func_from_scipy(fn)(nu, x)

    if nu == 0 and expon_scaled:
        fn = "k0e"
    elif nu == 1 and expon_scaled:
        fn = "k1e"
    elif nu == 0 and not expon_scaled:
        fn = "k0"
    elif nu == 1 and not expon_scaled:
        fn = "k1"
    return _get_special_func_from_scipy(fn)(x)


@bessel_y.register(object, backend="numpy")
def _bessel_y(x, nu):
    if nu not in (0, 1):
        return _get_special_func_from_scipy("yv")(nu, x)

    fn = "y0" if nu == 0 else "y1"
    return _get_special_func_from_scipy(fn)(x)
