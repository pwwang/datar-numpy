from __future__ import annotations

import numpy as np
from datar.apis.base import (
    ceiling,
    cov,
    floor,
    mean,
    median,
    pmax,
    pmin,
    sqrt,
    var,
    scale,
    min_,
    max_,
    round_,
    sum_,
    abs_,
    prod,
    sign,
    signif,
    trunc,
    exp,
    log,
    log2,
    log10,
    log1p,
    sd,
    weighted_mean,
    quantile,
    proportions,
)
from ..utils import is_null, make_array, numpy_version


@ceiling.register(object, backend="numpy")
def _ceiling(x):
    return np.ceil(x)


@cov.register(object, backend="numpy")
def _cov(x, y=None, na_rm: bool = False, ddof: int = 1):
    if y is None:
        raise ValueError(
            "In `cov(...)`: `y` must be provided if `x` is a vector"
        )
    return np.cov(x, y, ddof=ddof)[0, 1]


@floor.register(object, backend="numpy")
def _floor(x):
    return np.floor(x)


@mean.register(object, backend="numpy")
def _mean(x, na_rm: bool = False):
    return np.nanmean(x) if na_rm else np.mean(x)


@median.register(object, backend="numpy")
def _median(x, na_rm: bool = False):
    return np.nanmedian(x) if na_rm else np.median(x)


@pmax.register(object, backend="numpy")
def _pmax(x, *more, na_rm: bool = False):
    arrs = np.broadcast_arrays(x, *more)
    return np.nanmax(arrs, axis=0) if na_rm or na_rm else np.max(arrs, axis=0)


@pmin.register(object, backend="numpy")
def _pmin(x, *more, na_rm: bool = False):
    arrs = np.broadcast_arrays(x, *more)
    return np.nanmin(arrs, axis=0) if na_rm or na_rm else np.min(arrs, axis=0)


@sqrt.register(object, backend="numpy")
def _sqrt(x):
    return np.sqrt(x)


@var.register(object, backend="numpy")
def _var(x, na_rm: bool = False, ddof: int = 1):
    return np.nanvar(x, ddof=ddof) if na_rm else np.var(x, ddof=ddof)


@scale.register(object, backend="numpy")
def _scale(x, center=True, scale_=True):
    center_true = center is True
    x = make_array(x)

    # center
    if center is True:
        center = x.mean()

    elif center is not False:
        center = make_array(center)

    if center is not False:
        x = x - center

    # scale
    if scale_ is True:
        if center_true:
            scale_ = np.nanstd(x, ddof=1)
        else:
            scale_ = np.sqrt(np.nansum(x**2) / (len(x) - 1))

    elif scale_ is not False:
        scale_ = make_array(scale_)

    if scale_ is not False:
        x = x / scale_

    return x


@min_.register(object, backend="numpy")
def _min_(x, na_rm: bool = False):
    return np.nanmin(x) if na_rm else np.min(x)


@max_.register(object, backend="numpy")
def _max_(x, na_rm: bool = False):
    return np.nanmax(x) if na_rm else np.max(x)


@round_.register(object, backend="numpy")
def _round_(x, digits: int = 0):
    return np.round(x, digits)


@sum_.register(object, backend="numpy")
def _sum_(x, na_rm: bool = False):
    return np.nansum(x) if na_rm else np.sum(x)


@abs_.register(object, backend="numpy")
def _abs_(x):
    return np.abs(x)


@prod.register(object, backend="numpy")
def _prod(x, na_rm: bool = False):
    return np.nanprod(x) if na_rm else np.prod(x)


@sign.register(object, backend="numpy")
def _sign(x):
    return np.sign(x)


@signif.register(object, backend="numpy")
def _signif(x, digits: int = 6):
    digits = digits - np.ceil(np.log10(np.abs(x)))
    digits = np.broadcast_arrays(0, digits.astype(int))
    digits = np.nanmax(digits, axis=0)
    return np.vectorize(np.round)(x, digits)


@trunc.register(object, backend="numpy")
def _trunc(x):
    return np.trunc(x)


@exp.register(object, backend="numpy")
def _exp(x):
    return np.exp(x)


@log.register(object, backend="numpy")
def _log(x, base: float = np.e):
    return np.log(x) / np.log(base)


@log2.register(object, backend="numpy")
def _log2(x):
    return np.log2(x)


@log10.register(object, backend="numpy")
def _log10(x):
    return np.log10(x)


@log1p.register(object, backend="numpy")
def _log1p(x):
    return np.log1p(x)


@sd.register(object, backend="numpy")
def _sd(x, na_rm: bool = False, ddof: int = 1):
    return np.nanstd(x, ddof=ddof) if na_rm else np.std(x, ddof=ddof)


@weighted_mean.register(object, backend="numpy")
def _weighted_mean(x, w=None, na_rm: bool = False):
    if w is None:
        return np.nanmean(x) if na_rm else np.mean(x)

    if np.nansum(w) == 0:
        return np.nan

    if na_rm:
        x = make_array(x)
        w = make_array(w)
        mask = ~is_null(x)
        x = x[mask]
        w = w[mask]
        if w.size == 0:
            return np.nan

    return np.average(x, weights=w)


@quantile.register(object, backend="numpy")
def _quantile(
    x,
    probs=(0.0, 0.25, 0.5, 0.75, 1.0),
    na_rm: bool = False,
    names: bool = True,  # not supported
    type_: int = 7,
    digits: int | str = 7,  # not supported
):
    methods = {
        1: "inverted_cdf",
        2: "averaged_inverted_cdf",
        3: "closest_observation",
        4: "interpolated_inverted_cdf",
        5: "hazen",
        6: "weibull",
        7: "linear",
        8: "median_unbiased",
        9: "normal_unbiased",
    }
    if numpy_version() < (1, 22):  # pragma: no cover
        kw = {"interpolation": methods.get(type_, type_)}
    else:  # pragma: no cover
        kw = {"method": methods.get(type_, type_)}

    return (
        np.nanquantile(x, probs, **kw)
        if na_rm
        else np.quantile(x, probs, **kw)
    )


@proportions.register(object, backend="numpy")
def _proportions(x, margin=None):
    x = make_array(x)
    return x / np.sum(x)
