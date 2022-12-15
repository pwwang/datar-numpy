import numpy as np
from datar.core.plugin import plugin

# For simplug to retrieve the version
from .version import __version__  # noqa: F401
from .utils import flatten_slice, make_array

priority = -1


@plugin.impl
def base_api():
    from .api import (  # noqa: F401
        arithm,
        asis,
        bessel,
        complex,
        constants,
        cum,
        date,
        random,
        seq,
        sets,
        special,
        string,
        trig,
        which,
    )
    return {
        "pi": constants.pi,
        "letters": constants.letters,
        "LETTERS": constants.LETTERS,
        "month_abb": constants.month_abb,
        "month_name": constants.month_name,
        "NaN": constants.NaN,
        "Inf": constants.Inf,
        "NA": constants.NA,
        "NULL": constants.NULL,
    }


@plugin.impl
def get_versions():
    import numpy

    return {
        "datar-numpy": __version__,
        "numpy": numpy.__version__,
    }


@plugin.impl
def c_getitem(item):
    if isinstance(item, slice):
        return flatten_slice(item)

    elif isinstance(item, tuple):
        return np.concatenate(
            [
                flatten_slice(i)
                if isinstance(i, slice)
                else make_array(i)
                for i in item
            ]
        )

    return item
