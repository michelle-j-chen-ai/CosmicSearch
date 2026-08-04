"""Offline unit tests for the black-dwarf consolidation tool's pure helpers.

The S3 streaming/append path needs live object storage and is exercised
operationally; these cover the URI/marker-key logic that resumability and the
no-clobber rail depend on.
"""

from __future__ import annotations

import pytest

from tools.consolidate_black_dwarf import dest_bucket_key, marker_key

SRC_ROOT = "michelle/nls_search/black-dwarf/embeddings/cars/"
DEST_KEY = "michelle/nls_search/black-dwarf/table/video_embeddings.lance"


def test_dest_bucket_key_splits_and_strips_trailing_slash():
    assert dest_bucket_key("s3://bkt/a/b/c/") == ("bkt", "a/b/c")
    assert dest_bucket_key("s3://bkt/a/b/c") == ("bkt", "a/b/c")


def test_dest_bucket_key_rejects_non_s3():
    with pytest.raises(ValueError):
        dest_bucket_key("gs://bkt/a")


def test_marker_key_is_shard_rank_tagged_under_dest():
    src = f"s3://bkt/{SRC_ROOT}shard_00/rank=00000/video_embeddings.lance"
    assert (
        marker_key(DEST_KEY, src, SRC_ROOT)
        == f"{DEST_KEY}/_ingest_markers/shard_00_rank=00000.done"
    )


def test_marker_key_is_unique_per_shard_and_rank():
    uris = [
        f"s3://bkt/{SRC_ROOT}shard_00/rank=00000/video_embeddings.lance",
        f"s3://bkt/{SRC_ROOT}shard_00/rank=00001/video_embeddings.lance",
        f"s3://bkt/{SRC_ROOT}shard_11/rank=00000/video_embeddings.lance",
    ]
    keys = [marker_key(DEST_KEY, u, SRC_ROOT) for u in uris]
    assert len(set(keys)) == len(keys)


def test_marker_key_handles_src_root_without_trailing_slash():
    src = f"s3://bkt/{SRC_ROOT}shard_03/rank=00002/video_embeddings.lance"
    assert marker_key(DEST_KEY, src, SRC_ROOT.rstrip("/")) == marker_key(
        DEST_KEY, src, SRC_ROOT
    )
