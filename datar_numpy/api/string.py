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
    isnull = is_null(args)
    if isnull.any():
        return np.nan
    return sep.join((str(a) for a in args)) if len(args) > 0 else sep


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

    if isinstance(x, (np.bytes_, np.str_)):
        x = x.item()
    if isinstance(x, bytes):
        return len(x)

    try:
        x = x.encode("utf-8")
    except UnicodeEncodeError:  # pragma: no cover
        if allow_na:
            return np.nan
        raise
    return len(x)


_paste_ = np.vectorize(
    _paste_, [object], excluded={"sep"}, signature="(n)->()"
)
_match = np.vectorize(_match, excluded={"pattern"})
_sub_ = np.vectorize(_sub_, excluded={"pattern", "replacement"})


@grep.register(object, backend="numpy")
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


@grepl.register(object, backend="numpy")
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


@sub.register(object, backend="numpy")
def _sub(
    pattern,
    replacement,
    x,
    ignore_case=False,
    fixed=False,
):
    return _sub_(pattern, replacement, x, ignore_case, fixed, 1, "sub")


@gsub.register(object, backend="numpy")
def _gsub(
    pattern,
    replacement,
    x,
    ignore_case=False,
    fixed=False,
):
    return _sub_(pattern, replacement, x, ignore_case, fixed, 0, "gsub")


@strsplit.register(object, backend="numpy")
def _strsplit(x, split, fixed=False):
    def split_str(string, sep):
        if fixed:
            return string.split(sep)

        sep = re.compile(sep)
        return sep.split(string)

    if is_scalar(x) and is_scalar(split):
        return np.array([split_str(x, split)], dtype=object)
    return np.vectorize(split_str, [object])(x, split)


@paste.register(object, backend="numpy")
def _paste(*args, sep=" ", collapse=None):
    args = (arg for arg in args if is_scalar(arg) or len(arg) > 0)
    pasted = _paste_(np.array(np.broadcast_arrays(*args)).T, sep=sep)
    return pasted if collapse is None else collapse.join(pasted)


@paste0.register(object, backend="numpy")
def _paste0(*args, collapse=None):
    return paste(
        *args,
        sep="",
        collapse=collapse,
        __backend="numpy",
        __ast_fallback="normal",
    )


@sprintf.register(object, backend="numpy")
def _sprintf(fmt, *args):
    return np.vectorize(lambda fmt, *args: fmt % args)(
        *np.broadcast_arrays(fmt, *args)
    )


@substr.register(object, backend="numpy")
def _substr(x, start, stop):
    return np.vectorize(lambda x, start, stop: x[start:stop])(
        *np.broadcast_arrays(x, start, stop)
    )


@substring.register(object, backend="numpy")
def _substring(x, first, last=None):
    return np.vectorize(lambda x, first, last: x[first:last])(
        *np.broadcast_arrays(x, first, last)
    )


@startswith.register(object, backend="numpy")
def _startswith(x, prefix):
    x = make_array(x, dtype=str)
    return np.char.startswith(x, prefix)


@endswith.register(object, backend="numpy")
def _endswith(x, suffix):
    x = make_array(x, dtype=str)
    return np.char.endswith(x, suffix)


@strtoi.register(object, backend="numpy")
def _strtoi(x, base=0):
    return np.vectorize(int, excluded={"base"})(x, base=base)


@trimws.register(object, backend="numpy")
def _trimws(x, which="both", whitespace=r" \t"):
    x = make_array(x, dtype=str)
    if which == "both":
        return np.char.strip(x, whitespace)
    if which == "left":
        return np.char.lstrip(x, whitespace)
    if which == "right":
        return np.char.rstrip(x, whitespace)
    raise ValueError("`which` must be one of 'both', 'left', 'right'")


@toupper.register(object, backend="numpy")
def _toupper(x):
    x = make_array(x, dtype=str)
    return np.char.upper(x)


@tolower.register(object, backend="numpy")
def _tolower(x):
    x = make_array(x, dtype=str)
    return np.char.lower(x)


@chartr.register(object, backend="numpy")
def _chartr(old, new, x):
    x = make_array(x, dtype=str)
    old = _warn_more_pat_or_rep(old, "chartr", "old")
    new = _warn_more_pat_or_rep(new, "chartr", "new")

    new = new[: len(old)]
    for oldc, newc in zip(old, new):
        x = np.char.replace(x, oldc, newc)
    return x


@nchar.register(object, backend="numpy")
def _nchar(
    x,
    type_: str = "width",
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


@nzchar.register(object, backend="numpy")
def _nzchar(x, keep_na: bool = False):
    x = make_array(x)
    mask = is_null(x)
    out = np.where(mask, "", x).astype(str)
    out = np.char.str_len(out)
    out = out.astype(bool)

    if not keep_na:
        return out

    if not mask.any():
        return out

    out = out.astype(object)
    out[mask] = np.nan
    return out
