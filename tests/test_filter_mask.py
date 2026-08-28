"""An empty filter must match nothing, not everything.

`_merge_filters` intersects a request's filters with a downsample's. When the two
are disjoint the intersection is an empty set — a real answer, meaning no row
satisfies both. Testing that set for truthiness read it as "no filter given" and
ranked the entire corpus, presented to the user as a filtered result.
"""

from __future__ import annotations

import numpy as np
import pytest

import web_server as ws


class _Corpus:
    """Just enough of FullCorpus for filter_mask."""

    num_rows = 4

    def __init__(self):
        self._meta = {
            "vehicle_uniques": ["truck-1", "truck-2"],
            "vehicle": np.array([0, 0, 1, 1]),
            "run_uuid_uniques": ["run-a", "run-b"],
            "run_uuid": np.array([0, 1, 0, 1]),
            "chunk_start_unix": np.array([10, 20, 30, 40], dtype=np.int64),
            "segment_id": None,
        }

    filter_mask = __import__("full_corpus").FullCorpus.filter_mask
    segment_mask = __import__("full_corpus").FullCorpus.segment_mask


def test_no_filters_returns_none():
    assert _Corpus().filter_mask() is None


def test_empty_vehicle_set_matches_nothing():
    mask = _Corpus().filter_mask(vehicles=set())
    assert mask is not None and not mask.any()


def test_empty_run_uuid_set_matches_nothing():
    mask = _Corpus().filter_mask(run_uuids=set())
    assert mask is not None and not mask.any()


def test_populated_filter_still_selects():
    mask = _Corpus().filter_mask(vehicles={"truck-2"})
    assert mask.tolist() == [False, False, True, True]


@pytest.mark.parametrize(
    "base, extra, expected_key",
    [
        ({"run_uuids": {"a"}}, {"run_uuids": {"b"}}, set()),          # disjoint
        ({"run_uuids": {"a", "b"}}, {"run_uuids": {"b"}}, {"b"}),      # overlap
        ({"run_uuids": None}, {"run_uuids": {"b"}}, {"b"}),            # absent base
    ],
)
def test_merge_filters_keeps_an_empty_intersection(base, extra, expected_key):
    assert ws._merge_filters(base, extra)["run_uuids"] == expected_key


def test_disjoint_merge_then_mask_selects_nothing():
    """The end-to-end shape of the bug: two filters that cannot both hold."""
    merged = ws._merge_filters({"run_uuids": {"run-a"}}, {"run_uuids": {"run-b"}})
    mask = _Corpus().filter_mask(**merged)
    assert mask is not None, "an impossible filter must not read as no filter"
    assert not mask.any()
