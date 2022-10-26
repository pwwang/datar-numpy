import numpy as np
from datar.apis.base import cut


@cut.register(object)
def _cut(
    x,
    breaks,
    labels=None,  # not supported
    include_lowest=False,  # not supported
    right=True,  # not supported
    dig_lab=3,  # not supported
    ordered_result=False,  # not supported
):
    hist, edges = np.histogram(x, bins=breaks)
    out = []
    for i, h in enumerate(hist):
        out.extend([(edges[i], edges[i + 1])] * h)
    return np.array(out, dtype=object)
