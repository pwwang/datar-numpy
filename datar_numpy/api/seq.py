from __future__ import annotations

import numpy as np

from datar.core.utils import logger
from datar.apis.base import (
    rep,
    c_,
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


@rep.register(object, backend="numpy")
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


@c_.register(object, backend="numpy")
def _c(*args):
    return np.concatenate(
        [
            make_array(xi).flatten()
            for xi in args
        ]
    )


@length.register(object, backend="numpy")
def _length(x):
    return make_array(x).shape[0]


@lengths.register(object, backend="numpy")
def _lengths(x) -> np.ndarray[int]:
    return (
        np.array([1], dtype=int)
        if is_scalar(x)
        else np.array([make_array(xi).size for xi in x], dtype=int)
    )


@order.register(object, backend="numpy")
def _order(x, decreasing: bool = False, na_last: bool = True):
    and_ = not na_last and decreasing
    or_ = not na_last or decreasing
    na = -np.inf if or_ and not and_ else np.inf

    x = make_array(x)
    x = np.where(is_null(x), na, x)
    out = np.argsort(x)

    return out[::-1] if decreasing else out


@sort.register(object, backend="numpy")
def _sort(x, decreasing: bool = False, na_last: bool = True):
    x = make_array(x)
    idx = order(
        x,
        decreasing=decreasing,
        na_last=na_last,
        __ast_fallback="normal",
        __backend="numpy",
    )

    return x[idx]


@rank.register(object, backend="numpy")
def _rank(x, na_last: bool = True, ties_method: str = "average"):
    if not na_last:
        raise NotImplementedError("na_last=False is not supported yet")

    try:
        from scipy import stats
    except ImportError as imperr:  # pragma: no cover
        raise ImportError(
            "`rank` requires `scipy` package.\n"
            "Try: pip install -U scipy"
        ) from imperr

    return stats.rankdata(x, method=ties_method)


@rev.register(object, backend="numpy")
def _rev(x):
    x = make_array(x)
    return x[::-1]


@sample.register(object, backend="numpy")
def _sample(
    x,
    size: int = None,
    replace: bool = False,
    prob: float | np.ndarray[float] = None,
):
    x = make_array(x)
    size = x.size if size is None else int(size)

    return np.random.choice(x, size, replace=replace, p=prob)


@seq.register(object, backend="numpy")
def _seq(
    from_,
    to=None,
    by=None,
    length_out=None,
    along_with=None,
):
    if along_with is not None:
        return seq_along(along_with, __backend="numpy", __ast_fallback="normal")

    if not is_scalar(from_):
        return seq_along(from_, __backend="numpy", __ast_fallback="normal")

    if length_out is not None and from_ is None and to is None:
        return seq_len(length_out, __backend="numpy", __ast_fallback="normal")

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


@seq_along.register(object, backend="numpy")
def _seq_along(x):
    x = make_array(x)
    return np.arange(x.size) + 1


@seq_len.register((list, tuple, np.ndarray), backend="numpy")
def _seq_len_obj(length_out):
    if len(length_out) > 1:
        logger.warning(
            "[datar_numpy] In seq_len(...): "
            "first element used of 'length_out' argument"
        )

    length_out = list(length_out)[0]
    return np.arange(length_out) + 1


@seq_len.register((int, np.integer), backend="numpy")
def _seq_len_int(length_out):
    return np.arange(length_out) + 1


@match.register(object, backend="numpy")
def _match(x, table, nomatch=-1):
    sorter = np.argsort(table)
    searched = np.searchsorted(table, x, sorter=sorter).ravel()
    out = sorter.take(searched, mode="clip")
    out[~np.isin(x, table)] = nomatch
    return out
