import numpy as np
from datar.apis.base import (
    all_,
    any_,
    any_na,
    append,
    diff,
    duplicated,
    intersect,
    setdiff,
    setequal,
    unique,
    union,
    head,
    tail,
)

from ..utils import make_array, is_null, is_scalar


@all_.register(object)
def _all_(x):
    return np.all(x)


@any_.register(object)
def _any_(x):
    return np.any(x)


@any_na.register(object)
def _any_na(x):
    return is_null(x) if is_scalar(x) else is_null(x).any()


@append.register(object)
def _append(x, values, after: int = -1):
    x = make_array(x)
    if after is None:
        after = 0
    elif after < 0:
        after += len(x) + 1
    else:
        after += 1
    return np.insert(x, after, values)


@diff.register(object)
def _diff(x, lag: int = 1, differences: int = 1):
    if lag != 1:
        raise ValueError("lag argument not supported")
    x = make_array(x)
    return np.diff(x, n=differences)


@duplicated.register(object)
def _duplicated(x, incomparables=None, from_last: bool = False):
    dups = set()
    out = []
    out_append = out.append
    if incomparables is None:
        incomparables = []

    if from_last:
        x = reversed(x)
    for elem in x:
        if elem in incomparables:
            out_append(False)
        elif elem in dups:
            out_append(True)
        else:
            dups.add(elem)
            out_append(False)
    if from_last:
        out = list(reversed(out))
    return np.array(out, dtype=bool)


@intersect.register(object)
def _intersect(x, y):
    out, idx, _ = np.intersect1d(x, y, return_indices=True)
    return out[np.argsort(idx)]


@setdiff.register(object)
def _setdiff(x, y):
    out = np.setdiff1d(x, y)
    return unique(out, __ast_fallback="normal")


@setequal.register(object)
def _setequal(x, y):
    return np.array_equal(np.unique(x), np.unique(y))


@unique.register(object)
def _unique(x):
    out, idx = np.unique(x, return_index=True)
    return out[np.argsort(idx)]


@union.register(object)
def _union(x, y):
    out = np.concatenate([make_array(x), make_array(y)])
    out, idx = np.unique(out, return_index=True)
    return out[np.argsort(idx)]


@head.register(object)
def _head(x, n: int = 6):
    return make_array(x)[:n]


@tail.register(object)
def _tail(x, n: int = 6):
    return make_array(x)[-n:]