"""Offline checks for tools/build_test_corpora: synthetic local shards, no S3.

Run from the repo root:
    python -m pytest tests/test_build_test_corpora.py
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import conftest
import lance
import lance_writer
import numpy as np
import pyarrow as pa
import pytest
import search_engine
import threshold_search as ts

from bench_common import pca_basis

MODEL_DIM = conftest.MODEL_DIM  # 768


# ---------------------------------------------------------------------------
# Synthetic legacy-shard builder (local — Task 4 will centralize this in
# bench_e2e.py and repoint this import).
# ---------------------------------------------------------------------------
def _make_legacy_shard(
    dir_: Path, n: int, seed: int, week: str, *, with_vehicle: bool = False
) -> str:
    """Write a synthetic legacy-format video_embeddings.lance.

    Legacy shards carry ``vector`` (768-d fp32 FSL) plus ``chunk_id``,
    ``run_uuid``, ``chunk_start_unix``, ``source_media_uri``, and optionally
    ``metadata_json`` (a JSON string column). Returns the dataset URI.
    """
    rng = np.random.default_rng(seed)
    dir_.mkdir(parents=True, exist_ok=True)
    vectors = rng.standard_normal((n, MODEL_DIM)).astype("float32")
    chunk_ids = [f"{week}-chunk-{i}" for i in range(n)]
    run_uuids = [f"run-{week}"] * n
    chunk_starts = np.arange(
        1_700_000_000, 1_700_000_000 + n, dtype="int64"
    )
    source_uris = [f"s3://bucket/{week}/{i}.mp4" for i in range(n)]

    cols: dict[str, object] = {
        "vector": pa.FixedSizeListArray.from_arrays(
            pa.array(vectors.reshape(-1)), MODEL_DIM
        ),
        "chunk_id": pa.array(chunk_ids),
        "run_uuid": pa.array(run_uuids),
        "chunk_start_unix": pa.array(chunk_starts),
        "source_media_uri": pa.array(source_uris),
    }
    if with_vehicle:
        metadata_json = [
            json.dumps({"vehicle": f"veh-{week}"}) for _ in range(n)
        ]
    else:
        metadata_json = [json.dumps({}) for _ in range(n)]
    cols["metadata_json"] = pa.array(metadata_json)

    uri = str(dir_ / "video_embeddings.lance")
    lance.write_dataset(pa.table(cols), uri, mode="create")
    return uri


# ---------------------------------------------------------------------------
# Shard namedtuple for tests (mirrors the builder's internal representation)
# ---------------------------------------------------------------------------
from collections import namedtuple

_TestShard = namedtuple(
    "_TestShard", ["prefix_name", "rank", "uri", "dataset"]
)


def _shard(prefix_name: str, rank: int, uri: str) -> _TestShard:
    return _TestShard(prefix_name, rank, uri, lance.dataset(uri))


def _make_shards(
    tmp: Path, specs: list[tuple[str, int, int, int, bool]]
) -> list[_TestShard]:
    """Create synthetic shards from specs.

    Each spec: (prefix_name, rank, n_rows, seed, with_vehicle).
    """
    shards = []
    for prefix_name, rank, n, seed, with_vehicle in specs:
        rank_dir = tmp / prefix_name / f"rank={rank:05d}"
        uri = _make_legacy_shard(rank_dir, n, seed, prefix_name, with_vehicle=with_vehicle)
        shards.append(_shard(prefix_name, rank, uri))
    return shards


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_selection_takes_mce113_then_recent_weeks_until_target() -> None:
    """1 mce113 shard (100 rows) + week1..week3 shards (100 each); fraction
    targeting 250 rows -> mce113 + week1 + week2 selected, week3 not."""
    import build_test_corpora as btc

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        shards = _make_shards(
            tmp_path,
            [
                ("mce113", 0, 100, 0, False),
                ("week1_20250101", 0, 100, 1, True),
                ("week2_20241225", 0, 100, 2, True),
                ("week3_20241218", 0, 100, 3, True),
            ],
        )
        total = btc.count_prod_rows(shards)
        assert total == 400
        target = int(total * 0.625)  # 250 rows
        selected = btc.select_shards(shards, target)
        selected_prefixes = [s.prefix_name for s in selected]
        assert "mce113" in selected_prefixes
        assert "week1_20250101" in selected_prefixes
        assert "week2_20241225" in selected_prefixes
        assert "week3_20241218" not in selected_prefixes
        # mce113 comes first
        assert selected_prefixes[0] == "mce113"


def test_dedup_on_chunk_id_keeps_first() -> None:
    """Overlapping chunk_ids between two shards -> converted corpus has unique
    chunk_ids and row count == distinct count."""
    import build_test_corpora as btc

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        # Two shards with overlapping chunk_ids: both have "dup-0".."dup-4",
        # second shard adds 3 unique ones.
        rank_dir_1 = tmp_path / "mce113" / "rank=00000"
        _make_legacy_shard(rank_dir_1, 5, 0, "dup", with_vehicle=False)

        # Second shard with overlapping chunk_ids
        rank_dir_2 = tmp_path / "week1_20250101" / "rank=00000"
        rng = np.random.default_rng(1)
        n2 = 8
        # First 5 overlap with shard 1, last 3 are unique
        chunk_ids = [f"dup-chunk-{i}" for i in range(5)] + [
            f"uniq-chunk-{i}" for i in range(3)
        ]
        rank_dir_2.mkdir(parents=True, exist_ok=True)
        cols = {
            "vector": pa.FixedSizeListArray.from_arrays(
                pa.array(rng.standard_normal(n2 * MODEL_DIM).astype("float32")),
                MODEL_DIM,
            ),
            "chunk_id": pa.array(chunk_ids),
            "run_uuid": pa.array(["run-w1"] * n2),
            "chunk_start_unix": pa.array(
                np.arange(1_700_000_000, 1_700_000_000 + n2, dtype="int64")
            ),
            "source_media_uri": pa.array(
                [f"s3://b/w1/{i}.mp4" for i in range(n2)]
            ),
            "metadata_json": pa.array([json.dumps({})] * n2),
        }
        uri2 = str(rank_dir_2 / "video_embeddings.lance")
        lance.write_dataset(pa.table(cols), uri2, mode="create")

        shards = [
            _shard("mce113", 0, str(rank_dir_1 / "video_embeddings.lance")),
            _shard("week1_20250101", 0, uri2),
        ]

        # Build the threshold corpus locally
        dest = tmp_path / "dest"
        btc.build_threshold_corpus(shards, dest, fraction=1.0)

        ds = lance.dataset(str(dest / "threshold_prod_slice" / "corpus.lance"))
        seg_ids = ds.to_table(columns=["segment_id"], scan_in_order=True)
        seg_list = seg_ids.column("segment_id").to_pylist()
        assert len(seg_list) == len(set(seg_list)), "duplicate segment_ids found"
        # 5 unique from shard1 + 3 unique from shard2 = 8
        assert len(seg_list) == 8


def test_vehicle_derivation() -> None:
    """mce113 path -> 'mce113'; weekly + metadata_json vehicle -> that value;
    weekly + no vehicle key -> None."""
    import build_test_corpora as btc

    # mce113 prefix -> "mce113"
    assert btc.derive_vehicle("mce113_20250101", "{}") == "mce113"
    assert btc.derive_vehicle("mce113", '{"vehicle": "ignored"}') == "mce113"

    # weekly prefix with vehicle in metadata_json
    assert btc.derive_vehicle("week1_20250101", '{"vehicle": "truck-808"}') == "truck-808"

    # weekly prefix without vehicle key
    assert btc.derive_vehicle("week2_20241225", "{}") is None
    assert btc.derive_vehicle("week3_20241218", '{"other": "val"}') is None


def test_converted_corpus_is_exact_threshold_and_searchable() -> None:
    """End-to-end local build: assert lance_writer.is_exact_threshold_dataset(ds)
    and a ThresholdCorpus search with vehicle filter returns only mce113 rows."""
    import build_test_corpora as btc

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        shards = _make_shards(
            tmp_path,
            [
                ("mce113", 0, 30, 0, False),
                ("week1_20250101", 0, 20, 1, True),
            ],
        )
        dest = tmp_path / "dest"
        btc.build_threshold_corpus(shards, dest, fraction=1.0)

        corpus_uri = str(dest / "threshold_prod_slice" / "corpus.lance")
        ds = lance.dataset(corpus_uri)
        assert lance_writer.is_exact_threshold_dataset(ds), (
            "converted corpus must be an exact-threshold dataset"
        )

        corpus = ts.ThresholdCorpus(ds)
        # A query in the PCA row space
        pca, _ = lance_writer.read_pca_metadata(ds)
        rng = np.random.default_rng(42)
        latent = rng.standard_normal(conftest.D)
        latent /= np.linalg.norm(latent)
        query = (pca.astype(np.float64).T @ latent).astype("float32")

        # Use a very low tau to get all rows
        hits_all = corpus.threshold_search(query, tau=-1e9)
        assert len(hits_all) == 50, f"expected 50 rows, got {len(hits_all)}"

        # Filter to mce113 vehicle only
        hits_mce = corpus.threshold_search(query, tau=-1e9, vehicle="mce113")
        for h in hits_mce:
            assert h.vehicle == "mce113", f"non-mce113 row in filtered results: {h.vehicle}"
        assert len(hits_mce) == 30, f"expected 30 mce113 rows, got {len(hits_mce)}"


def test_master_copy_loads_via_search_engine_with_vehicle_column() -> None:
    """Local build of the master copy on synthetic shards; assert
    search_engine._load_corpus_lance loads it (rank-shard layout, extra vehicle
    column tolerated), row count matches, and the corpus's vehicle column is
    populated for mce113 rows in the Lance dataset itself."""
    import build_test_corpora as btc

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        shards = _make_shards(
            tmp_path,
            [
                ("mce113", 0, 15, 0, False),
                ("week1_20250101", 0, 10, 1, True),
            ],
        )
        dest = tmp_path / "dest"
        btc.build_master_copy(shards, dest)

        master_dir = dest / "master_prod_slice"
        rank_dirs = sorted(
            d for d in master_dir.iterdir() if d.is_dir() and d.name.startswith("rank=")
        )
        assert len(rank_dirs) == 2, f"expected 2 rank dirs, got {len(rank_dirs)}"

        # Prove _load_corpus_lance loads the master copy (the rank-shard path
        # search_engine.load_corpus uses for non-.lance URIs).
        corpus = search_engine._load_corpus_lance(
            master_dir, "test://master_prod_slice", "float32"
        )
        assert corpus.num_rows == 25, f"expected 25 rows, got {corpus.num_rows}"

        # The vehicle column is present and populated in the Lance datasets
        # themselves (load_corpus doesn't surface it via _load_corpus_lance, but
        # the column MUST be there for downstream consumers / threshold path).
        for rank_dir in rank_dirs:
            ds_uri = rank_dir / "video_embeddings.lance"
            ds = lance.dataset(str(ds_uri))
            assert "vehicle" in ds.schema.names, (
                f"vehicle column missing from {rank_dir.name}"
            )
            vehicle_col = ds.to_table(columns=["vehicle"]).column("vehicle").to_pylist()
            for v in vehicle_col:
                assert v is not None, f"NULL vehicle in {rank_dir.name}"


def test_dest_prefix_under_sibogeng_is_refused() -> None:
    """The builder hard-refuses --dest-prefix under sibogeng/ (the prod
    namespace): calling the entry fn with such a dest raises BEFORE any S3
    client is constructed."""
    import build_test_corpora as btc

    with pytest.raises(AssertionError, match="sibogeng"):
        btc.run_build(
            source_prefix="s3://some-bucket/source/",
            dest_prefix="s3://some-bucket/sibogeng/dest/",
        )
