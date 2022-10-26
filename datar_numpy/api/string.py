import re

import numpy as np
from datar.core.utils import logger
from datar.apis.base import (
    grep,
    grepl,
    sub,
    gsub,
    strsplit,
    paste,
    paste0,
    sprintf,
    substr,
    substring,
    startswith,
    endswith,
    strtoi,
    trimws,
    toupper,
    tolower,
    chartr,
    nchar,
    nzchar,
)
from ..utils import is_null, is_scalar, make_array
from .asis import _as_type


def _warn_more_pat_or_rep(pattern, fun, arg="pattern"):
    """Warn when there are more than one pattern or replacement provided"""
    if is_scalar(pattern):
        return pattern
    if len(pattern) == 1:
        return pattern[0]

    logger.warning(
        "[datar_numpy] "
        "In %s(...), argument `%s` has length > 1 and only the "
        "first element will be used",
        fun,
        arg,
    )
    return pattern[0]


def _match(text, pattern, ignore_case, invert, fixed):
    """Do the regex match"""
    if is_null(text):
        return False

    flags = re.IGNORECASE if ignore_case else 0
    if fixed:
        pattern = re.escape(pattern)

    pattern = re.compile(pattern, flags)
    matched = pattern.search(text)
    if invert:
        matched = not bool(matched)
    return bool(matched)


def _sub_(
    pattern,
    replacement,
    x,
    ignore_case=False,
    fixed=False,
    count=1,
    fun="sub",
):
    """Replace a pattern with replacement for elements in x,
    with argument count available
    """
    pattern = _warn_more_pat_or_rep(pattern, fun)
    replacement = _warn_more_pat_or_rep(replacement, fun, "replacement")
    if fixed:
        pattern = re.escape(pattern)

    flags = re.IGNORECASE if ignore_case else 0
    pattern = re.compile(pattern, flags)

    return pattern.sub(repl=replacement, count=count, string=x)


def _paste_(args, sep: str = " ") -> str:
    """Join strings with a separator"""
    return sep.join(args) if len(args) > 0 else sep


def _prepare_nchar(x, type_, keep_na):
    """Prepare arguments for n(z)char"""
    if type_ not in ["chars", "bytes", "width"]:
        raise ValueError(
            f"Invalid type argument, expect 'chars', 'bytes' or 'width', "
            f"got {type_}"
        )
    if keep_na is None:
        keep_na = type != "width"

    return x, keep_na


@np.vectorize
def _nchar_(x, retn, allow_na, keep_na, na_len):
    """Get the size of a scalar string"""
    if is_null(x):
        return np.nan if keep_na else na_len

    if retn == "width":
        try:
            from wcwidth import wcswidth
        except ImportError as imperr:  # pragma: no cover
            raise ImportError(
                "`nchar(x, type='width')` requires `wcwidth` package.\n"
                "Try: pip install -U wcwidth"
            ) from imperr

        return wcswidth(x)
    if retn == "chars":
        return len(x)

    try:
        x = x.encode("utf-8")
    except UnicodeEncodeError:
        if allow_na:
            return np.nan
        raise
    return len(x)


_paste_ = np.vectorize(
    _paste_, [object], excluded={"sep"}, signature="(n)->()"
)
_match = np.vectorize(_match, excluded={"pattern"})
_sub_ = np.vectorize(_sub_, excluded={"pattern", "replacement"})


@grep.register(object)
def _grep(
    pattern,
    x,
    ignore_case=False,
    value=False,
    fixed=False,
    invert=False,
):
    pattern = _warn_more_pat_or_rep(pattern, "grepl")
    matched = _match(
        x,
        pattern,
        ignore_case=ignore_case,
        invert=invert,
        fixed=fixed,
    )
    x = make_array(x)
    return x[matched] if value else np.flatnonzero(matched)


@grepl.register(object)
def _grepl(
    pattern,
    x,
    ignore_case=False,
    fixed=False,
    invert=False,
):
    pattern = _warn_more_pat_or_rep(pattern, "grepl")
    return _match(
        x,
        pattern,
        ignore_case=ignore_case,
        invert=invert,
        fixed=fixed,
    )


@sub.register(object)
def _sub(
    pattern,
    replacement,
    x,
    ignore_case=False,
    fixed=False,
):
    return _sub_(pattern, replacement, x, ignore_case, fixed, 1, "sub")


@gsub.register(object)
def _gsub(
    pattern,
    replacement,
    x,
    ignore_case=False,
    fixed=False,
):
    return _sub_(pattern, replacement, x, ignore_case, fixed, 0, "gsub")


@strsplit.register(object)
def _strsplit(x, split, fixed=False):
    def split_str(string, sep):
        if fixed:
            return string.split(sep)

        sep = re.compile(sep)
        return sep.split(string)

    if is_scalar(x) and is_scalar(split):
        return np.array([split_str(x, split)], dtype=object)
    return np.vectorize(split_str, [object])(x, split)


@paste.register(object)
def _paste(x, *args, sep=" ", collapse=None):
    args = (arg for arg in (x, *args) if is_scalar(arg) or len(arg) > 0)
    pasted = _paste_(np.array(np.broadcast_arrays(*args)).T, sep=sep)
    return pasted if collapse is None else collapse.join(pasted)


@paste0.register(object)
def _paste0(x, *args, collapse=None):
    return paste(x, *args, sep="", collapse=collapse, __ast_fallback="normal")


@sprintf.register(object)
def _sprintf(fmt, *args, **kwargs):
    return np.vectorize(lambda fmt, *args: fmt % args)(
        *np.broadcast_arrays(fmt, *args)
    )


@substr.register(object)
def _substr(x, start, stop):
    return np.vectorize(lambda x, start, stop: x[start:stop])(
        *np.broadcast_arrays(x, start, stop)
    )


@substring.register(object)
def _substring(x, first, last=None):
    return np.vectorize(lambda x, first, last: x[first:last])(
        *np.broadcast_arrays(x, first, last)
    )


@startswith.register(object)
def _startswith(x, prefix):
    return np.char.startswith(x, prefix)


@endswith.register(object)
def _endswith(x, suffix):
    return np.char.endswith(x, suffix)


@strtoi.register(object)
def _strtoi(x, base=0):
    return np.vectorize(int, excluded={"base"})(x, base=base)


@trimws.register(object)
def _trimws(x, which="both", whitespace=r" \t"):
    if which == "both":
        return np.char.strip(x, whitespace)
    if which == "left":
        return np.char.lstrip(x, whitespace)
    if which == "right":
        return np.char.rstrip(x, whitespace)
    raise ValueError("`which` must be one of 'both', 'left', 'right'")


@toupper.register(object)
def _toupper(x):
    return np.char.upper(x)


@tolower.register(object)
def _tolower(x):
    return np.char.lower(x)


@chartr.register(object)
def _chartr(old, new, x):
    old = _warn_more_pat_or_rep(old, "chartr", "old")
    new = _warn_more_pat_or_rep(new, "chartr", "new")
    if len(old) > len(new):
        raise ValueError("'old' is longer than 'new'")

    new = new[: len(old)]
    for oldc, newc in zip(old, new):
        x = np.char.replace(x, oldc, newc)
    return x


@nchar.register(object)
def _nchar(
    x,
    type_: str = "bytes",
    allow_na: bool = True,
    keep_na: bool = False,
    _na_len: int = 2,
):
    x, keep_na = _prepare_nchar(x, type_, keep_na)
    return _nchar_(
        x,
        retn=type_,
        allow_na=allow_na,
        keep_na=keep_na,
        na_len=_na_len,
    )


@nzchar.register(object)
def _nzchar(x, keep_na: bool = False):
    out = _as_type(x, bool, np.bool_)
    if not keep_na:
        return out

    mask = is_null(x)
    if is_scalar(mask):
        return out if not mask else x

    if not mask.any():
        return out

    out = out.astype(object)
    out[mask] = np.nan
    return out