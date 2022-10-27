import pytest
import numpy as np
from datar.base import (
    set_seed,
    rnorm,
    runif,
    rpois,
    rbinom,
    rcauchy,
    rchisq,
    rexp,
)
from .utils import assert_equal, assert_iterable_equal


def test_set_seed():
    out0 = rnorm(2)
    set_seed(1)
    out1 = rnorm(2)
    set_seed(1)
    out2 = rnorm(2)
    assert_iterable_equal(out1, out2)
    assert_equal(np.array_equal(out0, out1), False)


def test_rand_generator():
    nums1 = rnorm(2)
    nums2 = runif(2)
    nums3 = rpois(2, 1)
    nums4 = rbinom(2, 10, 0.5)
    nums5 = rcauchy(2)
    nums6 = rchisq(2, 1)
    nums7 = rexp(2, 1)
    assert_equal(len(nums1), 2)
    assert_equal(len(nums2), 2)
    assert_equal(len(nums3), 2)
    assert_equal(len(nums4), 2)
    assert_equal(len(nums5), 2)
    assert_equal(len(nums6), 2)
    assert_equal(len(nums7), 2)
