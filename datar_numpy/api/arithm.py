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
from ..utils import is_null, make_array


@ceiling.register(object)
def _ceiling(x):
    return np.ceil(x)


@cov.register(object)
def _cov(x, y=None, na_rm: bool = False, ddof: int = 1):
    if y is None:
        raise ValueError(
            "In `cov(...)`: `y` must be provided if `x` is a vector"
        )
    return np.cov(x, y, ddof=ddof)[0, 1]


@floor.register(object)
def _floor(x):
    return np.floor(x)


@mean.register(object)
def _mean(x, na_rm: bool = False):
    return np.nanmean(x) if na_rm else np.mean(x)


@median.register(object)
def _median(x, na_rm: bool = False):
    return np.nanmedian(x) if na_rm else np.median(x)


@pmax.register(object)
def _pmax(x, *more, na_rm: bool = False):
    arrs = np.broadcast_arrays(x, *more)
    return np.nanmax(arrs, axis=0) if na_rm or na_rm else np.max(arrs, axis=0)


@pmin.register(object)
def _pmin(x, *more, na_rm: bool = False):
    arrs = np.broadcast_arrays(x, *more)
    return np.nanmin(arrs, axis=0) if na_rm or na_rm else np.min(arrs, axis=0)


@sqrt.register(object)
def _sqrt(x):
    return np.sqrt(x)


@var.register(object)
def _var(x, na_rm: bool = False, ddof: int = 1):
    return np.nanvar(x, ddof=ddof) if na_rm else np.var(x, ddof=ddof)


@scale.register(object)
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
        scale_ = make_array(scale)

    if scale_ is not False:
        x = x / scale_

    return x


@min_.register(object)
def _min_(x, na_rm: bool = False):
    return np.nanmin(x) if na_rm else np.min(x)


@max_.register(object)
def _max_(x, na_rm: bool = False):
    return np.nanmax(x) if na_rm else np.max(x)


@round_.register(object)
def _round_(x, digits: int = 0):
    return np.round(x, digits)


@sum_.register(object)
def _sum_(x, na_rm: bool = False):
    return np.nansum(x) if na_rm else np.sum(x)


@abs_.register(object)
def _abs_(x):
    return np.abs(x)


@prod.register(object)
def _prod(x, na_rm: bool = False):
    return np.nanprod(x) if na_rm else np.prod(x)


@sign.register(object)
def _sign(x):
    return np.sign(x)


@signif.register(object)
def _signif(x, digits: int = 6):
    return np.around(x, digits - int(np.floor(np.log10(np.abs(x)))) - 1)


@trunc.register(object)
def _trunc(x):
    return np.trunc(x)


@exp.register(object)
def _exp(x):
    return np.exp(x)


@log.register(object)
def _log(x, base: float = np.e):
    return np.log(x) / np.log(base)


@log2.register(object)
def _log2(x):
    return np.log2(x)


@log10.register(object)
def _log10(x):
    return np.log10(x)


@log1p.register(object)
def _log1p(x):
    return np.log1p(x)


@sd.register(object)
def _sd(x, na_rm: bool = False, ddof: int = 1):
    return np.nanstd(x, ddof=ddof) if na_rm else np.std(x, ddof=ddof)


@weighted_mean.register(object)
def _weighted_mean(x, w=None, na_rm: bool = False):
    if w is None:
        return np.nanmean(x) if na_rm else np.mean(x)

    if np.nansum(w) == 0:
        return np.nan

    if na_rm:
        x = make_array(x)
        w = make_array(w)
        mask = ~is_null(x)
        x = x[~mask]
        w = w[~mask]
        if w.size == 0:
            return np.nan

    return np.average(x, weights=w)


@quantile.register(object)
def quantile(
    x,
    probs=(0.0, 0.25, 0.5, 0.75, 1.0),
    na_rm: bool = False,
    names: bool = True,  # not supported
    type_: int = 7,
    digits: int = 7,  # not supported
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
    return (
        np.nanquantile(x, probs, method=methods[type_])
        if na_rm
        else np.quantile(x, probs, method=methods[type_])
    )


@proportions.register(object)
def _proportions(x, margin=None):
    x = make_array(x)
    return x / np.sum(x)