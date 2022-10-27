from simplug import Simplug

plugin = Simplug("datar")


@plugin.impl
def base_api():
    from .api import (
        arithm,
        asis,
        bessel,
        complex,
        constants,
        cum,
        misc,
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
