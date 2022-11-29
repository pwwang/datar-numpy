"""Date time functions"""
import datetime

import numpy as np
from datar.apis.base import as_date


@as_date.register(np.datetime64, backend="numpy")
def _(
    x,
    *,
    format=None,
    try_formats=None,
    optional=False,
    tz=0,
    origin=None,
):
    return as_date(
        x.item(),
        format=format,
        try_formats=try_formats,
        optional=optional,
        tz=tz,
        origin=origin,
        __ast_fallback="normal",
        __backend="numpy",
    )


@as_date.register(datetime.date, backend="numpy")
def _as_date_d(
    x,
    *,
    format=None,
    try_formats=None,
    optional=False,
    tz=0,
    origin=None,
):
    if isinstance(tz, (int, np.integer)):
        tz = datetime.timedelta(hours=int(tz))

    return x + tz


@as_date.register(datetime.datetime, backend="numpy")
def _as_date_dt(
    x,
    *,
    format=None,
    try_formats=None,
    optional=False,
    tz=0,
    origin=None,
):
    if isinstance(tz, (int, np.integer)):
        tz = datetime.timedelta(hours=int(tz))

    return (x + tz).date()


@as_date.register(str, backend="numpy")
def _as_date_str(
    x: str,
    *,
    format=None,
    try_formats=None,
    optional=False,
    tz=0,
    origin=None,
):
    if isinstance(tz, (int, np.integer)):
        tz = datetime.timedelta(hours=int(tz))

    try_formats = try_formats or [
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d %H:%M:%S",
    ]
    if not format:
        format = try_formats
    else:
        format = [format]

    for fmt in format:
        try:
            return (datetime.datetime.strptime(x, fmt) + tz).date()
        except ValueError:
            continue

    if optional:
        return np.nan

    raise ValueError(
        "character string is not in a standard unambiguous format"
    )


@as_date.register((int, np.integer), backend="numpy")
def _as_date_int(
    x,
    *,
    format=None,
    try_formats=None,
    optional=False,
    tz=0,
    origin=None,
):
    if isinstance(tz, (int, np.integer)):
        tz = datetime.timedelta(hours=int(tz))

    if isinstance(origin, str):
        origin = _as_date_str(origin)

    if origin is None:  # pragma: no cover
        origin = datetime.date(1969, 12, 31)

    dt = origin + datetime.timedelta(days=int(x)) + tz

    if isinstance(dt, datetime.datetime):
        return dt.date()
    return dt


@as_date.register((list, tuple, np.ndarray), backend="numpy")
def _as_date_iter(
    x,
    *,
    format=None,
    try_formats=None,
    optional=False,
    tz=0,
    origin=None,
):
    return np.array(
        [
            as_date(
                el,
                format=format,
                try_formats=try_formats,
                optional=optional,
                origin=origin,
                tz=tz,
                __ast_fallback="normal",
                __backend="numpy",
            )
            for el in x
        ],
        dtype=object,
    )
