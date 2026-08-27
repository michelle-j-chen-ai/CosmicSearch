"""Metadata columns must not be capped by a 32-bit Arrow offset.

A `string` array addresses its values with 32-bit offsets, so one array holds at
most 2GiB of characters. The corpus grew past that on `segment_id` and the whole
load failed with `ArrowInvalid: offset overflow while concatenating arrays` --
not a degraded search, no search at all, because one metadata column outgrew its
index width. These cover the two places that concatenate string columns.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.compute as pc
import pytest

import full_corpus as fc


def test_dictionary_codes_encodes_without_combining_the_raw_strings():
    """Per-chunk dictionaries must unify, so codes still index one uniques list."""
    col = pa.chunked_array(
        [
            pa.array(["truck-808", "truck-810", None]),
            pa.array(["truck-999", "truck-808"]),  # a value the first chunk lacks
        ]
    )
    codes, uniques = fc._dictionary_codes(col)
    assert codes.dtype == "int32"
    rebuilt = [uniques[c] if c >= 0 else None for c in codes]
    assert rebuilt == col.to_pylist()


def test_dictionary_codes_marks_nulls_as_minus_one():
    codes, uniques = fc._dictionary_codes(pa.chunked_array([pa.array([None, "a"])]))
    assert codes[0] == -1
    assert uniques == ["a"]


@pytest.mark.parametrize("value_type", [pa.string(), pa.large_string()])
def test_segment_predicates_work_against_a_large_string_column(value_type):
    """`segment_id` is held as large_string; the filters compare it against plain
    string value sets and scalars, which must not silently stop matching."""
    seg = (
        pa.chunked_array([pa.array(["seg-a", "seg-b"]), pa.array(["seg-c", None])])
        .cast(pa.large_string())
        .combine_chunks()
    )
    assert seg.type == pa.large_string()

    hit = pc.is_in(seg, value_set=pa.array(["seg-b", "seg-c"], type=value_type))
    assert pc.fill_null(hit, False).to_pylist() == [False, True, True, False]

    eq = pc.equal(seg, pa.scalar("seg-c", type=value_type))
    assert pc.fill_null(eq, False).to_pylist() == [False, False, True, False]

    assert seg.take(pa.array([0, 2])).to_pylist() == ["seg-a", "seg-c"]
