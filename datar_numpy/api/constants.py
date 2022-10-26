import numpy as np

from datar.apis.base import (
    letters,
    LETTERS,
    month_abb,
    month_name,
)

pi = np.pi
letters = np.array(letters, dtype="<U1")
LETTERS = np.array(LETTERS, dtype="<U1")
month_abb = np.array(month_abb, dtype="<U3")
month_name = np.array(month_name, dtype="<U9")
NaN = np.nan
Inf = np.inf
NA = np.nan
NULL = None
