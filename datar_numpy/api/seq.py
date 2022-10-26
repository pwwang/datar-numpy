from __future__ import annotations

import numpy as np

from datar.core.utils import logger
from datar.apis.base import (
    rep,
    c,
    length,
    lengths,
    order,
    sort,
    rank,
    rev,
    sample,
    seq,
    seq_along,
    seq_len,
    match,
)
from ..utils import is_null, make_array, is_scalar
from .asis import _is_type


def _get_rankdata_func_from_scipy():
    """Import rankdata function from scipy on the fly"""
    try:
        from scipy import stats
    except ImportError as imperr:  # pragma: no cover
        raise ImportError(
            "`rank` requires `scipy` package.\n"
            "Try: pip install -U scipy"
        ) from imperr

    return stats.rankdata


@rep.register(object)
def _rep(
    x,
    times=1,
    length=None,
    each=1,
):
    x = make_array(x)
    times = make_array(times)
    length = make_array(length)
    each = make_array(each)
    if times.size == 1:
        times = times[0]
    if length.size >= 1:
        if length.size > 1:
            logger.warning(
                "[datar_numpy] "
                "In rep(...): first element used of 'length' argument"
            )
        length = length[0]
    if each.size == 1:
        each = each[0]

    if not is_scalar(times):
        if times.size != x.size:
            raise ValueError(
                "Invalid times argument, expect length "
                f"{x.size}, got {times.size}"
            )

        if not _is_type(each, int, np.int_) or each != 1:
            raise ValueError(
                "Unexpected each argument when times is an iterable."
            )

    if is_scalar(times) and _is_type(times, int, np.int_):
        x = np.tile(np.repeat(x, each), times)
    else:
        x = np.repeat(x, times)

    if length is None:
        return x

    repeats = length // x.size + 1
    x = np.tile(x, repeats)

    return x[:length]


@c.register(object)
def _c(x, *args):
    args = (x, *args)
    return np.concatenate(
        make_array(xi).flatten()
        for xi in args
    )


@length.register(object)
def _length(x):
    return make_array(x).size


@lengths.register(object)
def _lengths(x) -> np.ndarray[int]:
    return (
        np.array([1], dtype=int)
        if is_scalar(x)
        else np.array([make_array(xi).size for xi in x], dtype=int)
    )


@make_names.register(object)
def _make_names(x, unique: bool = False):
    try:
        from slugify import slugify
    except ImportError as imerr:  # pragma: no cover
        raise ValueError(
            "`make_names()` requires `python-slugify` package.\n"
            "Try: pip install -U slugify"
        ) from imerr

    if is_scalar(names):
        names = [names]

    names = [slugify(name, separator="_", lowercase=False) for name in names]
    names = [f"_{name}" if name[0].isdigit() else name for name in names]
    if unique:
        return repair_names(names, "unique")
    return names


@order.register(object)
def _order(x, decreasing: bool = False, na_last: bool = True):
    na = -np.inf if not na_last or decreasing else np.inf

    x = make_array(x)
    mask = is_null(x)
    x = np.where(mask, na, x)
    out = np.argsort(x)

    if decreasing:
        out = out[::-1]
    return out


@sort.register(object)
def _sort(x, decreasing: bool = False, na_last: bool = True):
    x = make_array(x)
    idx = order(
        x,
        decreasing=decreasing,
        na_last=na_last,
        __ast_fallback="normal",
    )

    return x[idx]


@rank.register(object)
def _rank(x, na_last: bool = True, ties_method: str = "average"):
    if not na_last:
        raise NotImplementedError("na_last=False is not supported yet")
    rankdata = _get_rankdata_func_from_scipy()
    return rankdata(x, method=ties_method)


@rev.register(object)
def _rev(x):
    x = make_array(x)
    return x[::-1]


@sample.register(object)
def _sample(
    x,
    size: int = None,
    replace: bool = False,
    prob: float | np.ndarray[float] = None,
):
    x = make_array(x)
    size = x.size if size is None else int(size)

    return np.random.choice(x, size, replace=replace, p=prob)


@seq.register(object)
def _seq(
    from_=None,
    to=None,
    by=None,
    length_out=None,
    along_with=None,
):
    if along_with is not None:
        return seq_along(along_with, __ast_fallback="normal")

    if not is_scalar(from_):
        return seq_along(from_, __ast_fallback="normal")

    if length_out is not None and from_ is None and to is None:
        return seq_len(length_out, __ast_fallback="normal")

    if from_ is None:
        from_ = 1
    elif to is None:
        from_, to = 1, from_

    if length_out is not None:
        by = (float(to) - float(from_)) / float(length_out)
    elif by is None:
        by = 1 if to > from_ else -1
        length_out = to - from_ + 1 if to > from_ else from_ - to + 1
    else:
        length_out = (to - from_ + 1.1 * by) // by

    return np.array([from_ + n * by for n in range(int(length_out))])


@seq_along.register(object)
def _seq_along(x):
    x = make_array(x)
    return np.arange(x.size) + 1


@seq_len.register(object)
def _seq_len(length_out):
    return np.arange(int(length_out)) + 1


@match.register(object)
def _match(x, table, nomatch=None):
    sorter = np.argsort(table)
    searched = np.searchsorted(table, x, sorter=sorter).ravel()
    out = sorter.take(searched, mode="clip")
    out[~np.isin(x, table)] = nomatch
    return out