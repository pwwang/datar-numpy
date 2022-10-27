import pytest
import numpy as np
from datar.base import (
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
from .utils import assert_equal, assert_iterable_equal


def test_grep():
    assert_iterable_equal(grep("b", ["a", "b1", "c"]), [1])
    assert_iterable_equal(grep("b", None), [])
    assert_iterable_equal(grep(["b"], ["a", "b1", "c"]), [1])
    assert_iterable_equal(grep(r"b\d", ["a", "b1", "c"]), [1])
    assert_iterable_equal(grep(r"b\d", ["a", "b1", "c"], fixed=True), [])
    assert_iterable_equal(grep("b", ["a", "b1", "c"], value=True), ["b1"])
    assert_iterable_equal(grep("b", ["a", "b1", "c"], invert=True), [0, 2])
    assert_iterable_equal(grep("b", ["a", "b1", "c"], value=True, invert=True), ["a", "c"])
    assert_iterable_equal(grep("B", ["a", "b1", "c"], ignore_case=True), [1])


def test_grep_pattern_warns(caplog):
    with caplog.at_level("WARNING"):
        grep(["a", "b"], "a")
    assert "pattern" in caplog.text


def test_grepl():
    assert_iterable_equal(grepl("b", ["a", "b1", "c"]), [False, True, False])
    assert_iterable_equal(grepl(r"b\d", ["a", "b1", "c"]), [False, True, False])
    assert_iterable_equal(grepl(r"b\d", ["a", "b1", "c"], fixed=True), [False, False, False])
    assert_iterable_equal(grepl("b", ["a", "b1", "c"], invert=True), [True, False, True])
    assert_iterable_equal(grepl("B", ["a", "b1", "c"], ignore_case=True), [False, True, False])


def test_sub():
    assert_equal(sub("b", "B", "abcb"), "aBcb")
    assert_equal(sub("b", "B", "abcb", fixed=True), "aBcb")
    assert_equal(sub("b", "B", "abcb", ignore_case=True), "aBcb")


def test_gsub():
    assert_equal(gsub("b", "B", "abcb"), "aBcB")
    assert_equal(gsub("b", "B", "abcb", fixed=True), "aBcB")
    assert_equal(gsub("b", "B", "abcb", ignore_case=True), "aBcB")


def test_strsplit():
    assert_iterable_equal(strsplit("a.b.c", r"\.")[0], ["a", "b", "c"])
    assert_iterable_equal(strsplit("a.b.c", ".", fixed=True)[0], ["a", "b", "c"])
    out = strsplit(["a.b.c", "d.e"], ".", fixed=True)
    assert_iterable_equal(out[0], ["a", "b", "c"])
    assert_iterable_equal(out[1], ["d", "e"])


def test_paste():
    assert_equal(paste("a", "b", "c", sep="."), "a.b.c")
    assert_iterable_equal(paste(["a", "b"], ["c", "d"], sep="."), ["a.c", "b.d"])
    assert_equal(paste(["a", "b"], ["c", "d"], sep=".", collapse=","), "a.c,b.d")


def test_paste0():
    assert_equal(paste0("a", "b", "c"), "abc")
    assert_iterable_equal(paste0(["a", "b"], ["c", "d"]), ["ac", "bd"])
    assert_equal(paste0(["a", "b"], ["c", "d"], collapse=","), "ac,bd")


def test_sprintf():
    assert_equal(sprintf("%s-%s", "a", "b"), "a-b")
    assert_iterable_equal(sprintf("%s-%s", ["a", "b"], ["c", "d"]), ["a-c", "b-d"])


def test_substr():
    assert_equal(substr("abc", 1, 2), "b")
    assert_equal(substr("abc", -2, -1), "b")


def test_substring():
    assert_equal(substring("abc", 1), "bc")
    assert_equal(substring("abc", 1, 2), "b")
    assert_equal(substring("abc", -1), "c")
    assert_equal(substring("abc", -2, -1), "b")


def test_startswith():
    assert_equal(startswith("abc", "a"), True)
    assert_iterable_equal(startswith("abc", ["a", "b"]), [True, False])


def test_endswith():
    assert_equal(endswith("abc", "c"), True)
    assert_iterable_equal(endswith("abc", ["a", "c"]), [False, True])


def test_strtoi():
    assert_equal(strtoi("1"), 1)
    assert_iterable_equal(strtoi(["1", "2"]), [1, 2])


def test_trimws():
    assert_equal(trimws(" a b "), "a b")
    assert_equal(trimws(" a b ", "left"), "a b ")
    assert_iterable_equal(trimws([" a ", " b "]), ["a", "b"])
    assert_iterable_equal(trimws([" a ", " b "], "right"), [" a", " b"])
    with pytest.raises(ValueError):
        trimws("a", "nowhich")


def test_toupper():
    assert_equal(toupper("abc"), "ABC")
    assert_iterable_equal(toupper(["a", "b"]), ["A", "B"])


def test_tolower():
    assert_equal(tolower("ABC"), "abc")
    assert_iterable_equal(tolower(["A", "B"]), ["a", "b"])


def test_chartr():
    assert_equal(chartr("a", "b", "abc"), "bbc")
    assert_iterable_equal(chartr("a", "b", ["a", "b"]), ["b", "b"])


def test_nchar():
    assert_equal(nchar("abc"), 3)
    assert_equal(nchar("abc", type_="width"), 3)
    assert_equal(nchar("abc", type_="chars"), 3)
    assert_equal(nchar("abc王", type_="bytes"), 6)
    assert_equal(nchar(b"abc", type_="bytes"), 3)
    assert_equal(nchar(None, keep_na=True), np.nan)
    assert_equal(nchar(None, keep_na=False), 2)
    assert_iterable_equal(nchar(["a", "b"]), [1, 1])
    assert_iterable_equal(nchar(["a", "b"], keep_na=None), [1, 1])
    with pytest.raises(ValueError):
        nchar("abc", type_="badtypes")


def test_nzchar():
    assert_equal(nzchar("abc"), True)
    assert_iterable_equal(nzchar(["a", ""]), [True, False])
    assert_iterable_equal(nzchar(["a", ""], keep_na=True), [True, False])
    assert_iterable_equal(nzchar(["a", "", None], keep_na=True), [True, False, None])
