import pytest
import numpy as np
from datar.base import (
    rep,
    c,
    length,
    lengths,
    order,
    sort,
    rank,
    rev,
    sample,
    seq,
    seq_along,
    seq_len,
    match,
)
from .utils import assert_equal, assert_iterable_equal


def test_rep():
    assert_iterable_equal(rep(1, 3), [1, 1, 1])
    assert_iterable_equal(rep(1, times=3), [1, 1, 1])
    assert_iterable_equal(rep(1, length=3), [1, 1, 1])
    assert_iterable_equal(rep(1, each=3), [1, 1, 1])
    assert_iterable_equal(rep(1, times=3, each=2), [1, 1, 1, 1, 1, 1])
    assert_iterable_equal(rep(1, length=3, each=2), [1, 1, 1])
    assert_iterable_equal(rep([1, 2], times=3), [1, 2, 1, 2, 1, 2])
    assert_iterable_equal(rep([1, 2], length=3), [1, 2, 1])
    assert_iterable_equal(rep([1, 2], each=3), [1, 1, 1, 2, 2, 2])
    assert_iterable_equal(
        rep([1, 2], times=3, each=2), [1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2]
    )
    assert_iterable_equal(rep([1, 2], length=3, each=2), [1, 1, 2])
    assert_iterable_equal(rep([1, 2], times=[1, 2]), [1, 2, 2])


def test_rep_errors(caplog):
    with caplog.at_level("WARNING"):
        rep(1, length=[1, 2])
        assert (
            "In rep(...): first element used of 'length' argument"
            in caplog.text
        )

    with pytest.raises(ValueError):
        rep([1, 2, 3], times=[1, 2])

    with pytest.raises(ValueError):
        rep([1, 2, 3], times=[1, 2, 3], each=2)


def test_c():
    assert_iterable_equal(c(1, 2, 3), [1, 2, 3])
    assert_iterable_equal(c([1, 2], [3, 4]), [1, 2, 3, 4])
    assert_iterable_equal(c([[1, 2], [3, 4]], [5, 6]), [1, 2, 3, 4, 5, 6])
    assert_iterable_equal(c(c(1, 2), 3), [1, 2, 3])


def test_length():
    assert_equal(length(1), 1)
    assert_equal(length([1, 2]), 2)
    assert_equal(length([[1, 2], [3, 4]]), 2)


def test_lengths():
    assert_iterable_equal(lengths(1), [1])
    assert_iterable_equal(lengths([1, 2]), [1, 1])
    assert_iterable_equal(lengths([[1, 2], [3, 4]]), [2, 2])
    assert_iterable_equal(lengths([1, [2, 3], [4, 5, 6]]), [1, 2, 3])


def test_order():
    assert_iterable_equal(order([1, 2, 3]), [0, 1, 2])
    assert_iterable_equal(order([3, 2, 1]), [2, 1, 0])
    assert_iterable_equal(order([3, 2, 1], decreasing=True), [0, 1, 2])
    assert_iterable_equal(order([3, 2, np.nan, 1]), [3, 1, 0, 2])
    assert_iterable_equal(
        order([3, 2, np.nan, 1], na_last=False),
        [2, 3, 1, 0],
    )
    assert_iterable_equal(
        order([3, 2, np.nan, 1], na_last=False, decreasing=True),
        [2, 0, 1, 3],
    )


def test_sort():
    assert_iterable_equal(sort([1, 2, 3]), [1, 2, 3])
    assert_iterable_equal(sort([3, 2, 1]), [1, 2, 3])
    assert_iterable_equal(sort([3, 2, 1], decreasing=True), [3, 2, 1])
    assert_iterable_equal(sort([3, 2, np.nan, 1]), [1, 2, 3, np.nan])
    assert_iterable_equal(
        sort([3, 2, np.nan, 1], na_last=False),
        [np.nan, 1, 2, 3],
    )
    assert_iterable_equal(
        sort([3, 2, np.nan, 1], na_last=False, decreasing=True),
        [np.nan, 3, 2, 1],
    )


def test_rank():
    assert_iterable_equal(rank([1, 2, 2, 3]), [1.0, 2.5, 2.5, 4.0])
    assert_iterable_equal(rank([1, 2, 2, 3], ties_method="dense"), [1, 2, 2, 3])
    assert_iterable_equal(rank([1, 2, 2, 3], ties_method="min"), [1, 2, 2, 4])
    assert_iterable_equal(rank([1, 2, 2, 3], ties_method="max"), [1, 3, 3, 4])

    with pytest.raises(NotImplementedError):
        rank([1, 2, 2, 3], na_last=False)


def test_rev():
    assert_iterable_equal(rev([1, 2, 3]), [3, 2, 1])


def test_sample():
    assert_iterable_equal(sample(1, 3, replace=True), [1, 1, 1])


def test_seq():
    assert_iterable_equal(seq(1, 3), [1, 2, 3])
    assert_iterable_equal(seq(1, 3, 2), [1, 3])
    assert_iterable_equal(seq(None, along_with=[4, 5, 6]), [1, 2, 3])
    assert_iterable_equal(seq([4, 5, 6]), [1, 2, 3])
    assert_iterable_equal(seq(None, length_out=3), [1, 2, 3])
    assert_iterable_equal(seq(None, to=3), [1, 2, 3])
    assert_iterable_equal(seq(3), [1, 2, 3])
    assert_iterable_equal(seq(1, 3, length_out=2), [1, 2])


def test_seq_along():
    assert_iterable_equal(seq_along([4, 5, 6]), [1, 2, 3])


def test_seq_len():
    assert_iterable_equal(seq_len(3), [1, 2, 3])
    # warning, only first element used
    assert_iterable_equal(seq_len([3, 4]), [1, 2, 3])


def test_match():
    assert_iterable_equal(match([1, 2, 3], [2, 3, 4]), [-1, 0, 1])
    assert_iterable_equal(match([1, 2, 3], [2, 3, 4], nomatch=0), [0, 0, 1])
