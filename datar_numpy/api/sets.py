import numpy as np
from datar.apis.base import (
    all_,
    any_,
    any_na,
    append,
    outer,
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


@all_.register(object, backend="numpy")
def _all_(x):
    return np.all(x)


@any_.register(object, backend="numpy")
def _any_(x):
    return np.any(x)


@any_na.register(object, backend="numpy")
def _any_na(x):
    return is_null(x) if is_scalar(x) else is_null(x).any()


@append.register(object, backend="numpy")
def _append(x, values, after: int = -1):
    x = make_array(x)
    if after is None:
        after = 0
    elif after < 0:
        after += len(x) + 1
    else:
        after += 1
    return np.insert(x, after, values)


@outer.register(object, backend="numpy")
def _outer(x, y, fun="*"):
    if fun == "*":
        return np.outer(x, y)

    kwargs = {}
    if (
        getattr(fun, "_pipda_functype", None) in ("pipeable", "verb")
    ):  # pragma: no cover
        kwargs["__ast_fallback"] = "normal"
    return np.array([fun(xi, y, **kwargs) for xi in make_array(x)])


@diff.register(object, backend="numpy")
def _diff(x, lag: int = 1, differences: int = 1):
    if lag != 1:
        raise ValueError("lag argument not supported")
    x = make_array(x)
    return np.diff(x, n=differences)


@duplicated.register(object, backend="numpy")
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


@intersect.register(object, backend="numpy")
def _intersect(x, y):
    out, idx, _ = np.intersect1d(x, y, return_indices=True)
    return out[np.argsort(idx)]


@setdiff.register(object, backend="numpy")
def _setdiff(x, y):
    x = make_array(x)
    out = x[~np.in1d(x, y)]
    return unique(out, __backend="numpy", __ast_fallback="normal")


@setequal.register(object, backend="numpy")
def _setequal(x, y):
    return np.array_equal(np.unique(x), np.unique(y))


@unique.register(object, backend="numpy")
def _unique(x):
    out, idx = np.unique(x, return_index=True)
    return out[np.argsort(idx)]


@union.register(object, backend="numpy")
def _union(x, y):
    out = np.concatenate([make_array(x), make_array(y)])
    out, idx = np.unique(out, return_index=True)
    return out[np.argsort(idx)]


@head.register(object, backend="numpy")
def _head(x, n: int = 6):
    return make_array(x)[:n]


@tail.register(object, backend="numpy")
def _tail(x, n: int = 6):
    return make_array(x)[-n:]
