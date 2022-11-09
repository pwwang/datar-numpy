
import random as _random

import numpy as np
from datar.apis.base import (
    set_seed,
    rnorm,
    runif,
    rpois,
    rbinom,
    rcauchy,
    rchisq,
    rexp,
)


@set_seed.register(object, backend="numpy")
def _set_seed(seed):
    _random.seed(seed)
    np.random.seed(seed)


@rnorm.register(object, backend="numpy")
def _rnorm(n, mean=0, sd=1):
    return np.random.normal(mean, sd, n)


@runif.register(object, backend="numpy")
def _runif(n, min=0, max=1):
    return np.random.uniform(min, max, n)


@rpois.register(object, backend="numpy")
def _rpois(n, lambda_):
    return np.random.poisson(lambda_, n)


@rbinom.register(object, backend="numpy")
def _rbinom(n, size, prob):
    return np.random.binomial(size, prob, n)


@rcauchy.register(object, backend="numpy")
def _rcauchy(n, location=0, scale=1):
    return np.random.standard_cauchy(n) * scale + location


@rchisq.register(object, backend="numpy")
def _rchisq(n, df):
    return np.random.chisquare(df, n)


@rexp.register(object, backend="numpy")
def _rexp(n, rate=1):
    return np.random.exponential(1 / rate, n)
