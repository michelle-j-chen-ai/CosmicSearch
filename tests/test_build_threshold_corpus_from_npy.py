"""Offline checks for tools/build_threshold_corpus_from_npy: synthetic npy corpus, no S3.

Run from the repo root:
    python -m pytest tests/test_build_threshold_corpus_from_npy.py
"""

from __future__ import annotations

from pathlib import Path

import build_threshold_corpus_from_npy as builder
import lance
import lance_writer
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

MODEL_DIM = builder.MODEL_DIM  # 768
D = 256


def _unit_rows(n: int, seed: int = 0, rank: int | None = None) -> np.ndarray:
    """`n` row-L2-normalized fp32 vectors, the shape a real corpus has.

    With `rank`, the rows are drawn from a random `rank`-dimensional subspace of
    the 768-d space. Real embeddings are effectively low-rank, which is what lets
    a PCA-256 projection preserve the cosine the app ranks by; full-rank Gaussian
    rows have no such structure and lose most of their energy under truncation.
    """
    rng = np.random.default_rng(seed)
    if rank is None:
        x = rng.standard_normal((n, MODEL_DIM)).astype("float32")
    else:
        basis, _ = np.linalg.qr(rng.standard_normal((MODEL_DIM, rank)))
        x = (rng.standard_normal((n, rank)) @ basis.T).astype("float32")
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype("float32")


def _write_fast_corpus(
    source_dir: Path,
    n: int,
    *,
    seed: int = 0,
    with_vehicle: bool = False,
    rank: int | None = None,
) -> np.ndarray:
    """Write an `embeddings.npy` + `metadata.parquet` pair; return the vectors."""
    source_dir.mkdir(parents=True, exist_ok=True)
    vectors = _unit_rows(n, seed, rank=rank)
    np.save(source_dir / builder.NPY_MATRIX_FILE, vectors)

    rng = np.random.default_rng(seed + 1)
    # Deliberately unsorted so the writer's (chunk_start_unix, vehicle) sort is
    # exercised rather than accidentally satisfied by the input.
    chunk_start = rng.integers(1_772_000_000, 1_772_100_000, size=n).astype("int64")
    columns = {
        "chunk_id": [f"run-{i % 7}#t{chunk_start[i]}" for i in range(n)],
        "run_uuid": [f"run-{i % 7}" for i in range(n)],
        "chunk_start_unix": chunk_start,
        "chunk_end_unix": chunk_start + 8,
        "source_media_uri": [f"s3://bucket/clip-{i}.mp4" for i in range(n)],
        "segment_id": [
            f"2026-02-27_10-05-15_truck-{800 + i % 3}"
            f"-1772187920785547000-1772187950785546999"
            for i in range(n)
        ],
    }
    if with_vehicle:
        columns["vehicle"] = [f"preset-{i % 2}" for i in range(n)]
    pq.write_table(pa.table(columns), source_dir / lance_writer.METADATA_FILE)
    return vectors


# ---------------------------------------------------------------------------
# vehicle_from_segment_id
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "segment_id,expected",
    [
        # Frontier: leading dashed date, vehicle last, trailing ns timestamps.
        (
            "2026-02-27_10-05-15_truck-809-1772187920785547000-1772187950785546999",
            "truck-809",
        ),
        # Neuron: vehicle first.
        ("mce113_2026-06-04_run-1780547400000000000", "mce113"),
        ("", None),
        (None, None),
    ],
)
def test_vehicle_from_segment_id(segment_id, expected):
    assert builder.vehicle_from_segment_id(segment_id) == expected


# ---------------------------------------------------------------------------
# assert_rows_normalized
# ---------------------------------------------------------------------------
def test_assert_rows_normalized_accepts_unit_rows():
    assert builder.assert_rows_normalized(_unit_rows(500)) < builder._NORM_TOLERANCE


def test_assert_rows_normalized_rejects_unnormalized_rows():
    # Scoring and the fitted quantization scales both assume unit rows, so a
    # scaled corpus must fail loudly rather than silently shift the score scale.
    with pytest.raises(ValueError, match="not L2-normalized"):
        builder.assert_rows_normalized(_unit_rows(500) * 2.0)


# ---------------------------------------------------------------------------
# build_metadata
# ---------------------------------------------------------------------------
def test_build_metadata_derives_vehicle_when_absent(tmp_path):
    _write_fast_corpus(tmp_path, 32)
    meta = builder.build_metadata(tmp_path / lance_writer.METADATA_FILE, 32)
    assert set(meta.column("vehicle").to_pylist()) == {
        "truck-800",
        "truck-801",
        "truck-802",
    }


def test_build_metadata_keeps_existing_vehicle_column(tmp_path):
    _write_fast_corpus(tmp_path, 16, with_vehicle=True)
    meta = builder.build_metadata(tmp_path / lance_writer.METADATA_FILE, 16)
    assert set(meta.column("vehicle").to_pylist()) == {"preset-0", "preset-1"}


def test_build_metadata_rejects_row_count_mismatch(tmp_path):
    _write_fast_corpus(tmp_path, 16)
    with pytest.raises(ValueError, match="does not match"):
        builder.build_metadata(tmp_path / lance_writer.METADATA_FILE, 17)


# ---------------------------------------------------------------------------
# project_and_quantize
# ---------------------------------------------------------------------------
def test_project_and_quantize_blocking_matches_single_shot(monkeypatch):
    vectors = _unit_rows(700, seed=3)
    pca = builder.pca_basis(vectors)
    monkeypatch.setattr(builder, "_PROJECT_BLOCK_ROWS", 97)  # forces ragged blocks
    projected, corpus_i8, scale = builder.project_and_quantize(vectors, pca)
    # Not bit-exact: a 97-row matmul and a 700-row one sum in different orders,
    # so BLAS returns results that differ in the last float32 ulps.
    assert projected == pytest.approx((vectors @ pca.T).astype("float32"), abs=1e-6)
    assert corpus_i8.dtype == np.int8
    # Dequantization error is bounded by half a quantization step per dim.
    dequant = corpus_i8.astype(np.float32) * (scale / np.float32(127.0))
    assert np.all(np.abs(projected - dequant) <= scale / 127.0 / 2 + 1e-6)


# ---------------------------------------------------------------------------
# build (end to end)
# ---------------------------------------------------------------------------
def test_build_produces_an_exact_threshold_dataset(tmp_path):
    vectors = _write_fast_corpus(tmp_path / "source", 900, seed=5)
    corpus_uri = builder.build(tmp_path / "source", tmp_path / "out")
    ds = lance.dataset(corpus_uri)

    assert ds.count_rows() == 900
    # The reference threshold corpus's exact column set (lance_writer drops the
    # source's chunk_id / source_media_uri).
    assert ds.schema.names == [
        "run_uuid",
        "chunk_start_unix",
        "chunk_end_unix",
        "segment_id",
        "vehicle",
        lance_writer.EMBEDDING_I8_COLUMN,
        lance_writer.VECTOR_FP_COLUMN,
    ]
    assert lance_writer.is_exact_threshold_dataset(ds)

    pca, scale = lance_writer.read_pca_metadata(ds)
    assert pca.shape == (D, MODEL_DIM)
    assert scale.shape == (D,)

    table = ds.to_table(scan_in_order=True)
    assert table.column("vehicle").null_count == 0
    assert table.column("chunk_end_unix").null_count == 0


def test_build_stores_the_true_pre_quantization_projection(tmp_path):
    _write_fast_corpus(tmp_path / "source", 600, seed=7)
    corpus_uri = builder.build(tmp_path / "source", tmp_path / "out")
    ds = lance.dataset(corpus_uri)
    _pca, scale = lance_writer.read_pca_metadata(ds)

    table = ds.to_table(
        columns=[lance_writer.EMBEDDING_I8_COLUMN, lance_writer.VECTOR_FP_COLUMN],
        scan_in_order=True,
    )

    def matrix(name):
        col = table.column(name).combine_chunks()
        return col.flatten().to_numpy(zero_copy_only=False).reshape(
            -1, col.type.list_size
        )

    fp = matrix(lance_writer.VECTOR_FP_COLUMN)
    dequant = matrix(lance_writer.EMBEDDING_I8_COLUMN).astype(np.float32) * (
        scale / np.float32(127.0)
    )
    # A genuine pre-quantization column carries information the int8 screen does
    # not; equality would mean the writer fell back to dequant(int8).
    assert not np.array_equal(fp, dequant)
    assert np.abs(fp - dequant).max() <= (scale / 127.0 / 2).max() + 1e-6


def test_build_preserves_cosine_scores_through_the_basis(tmp_path):
    vectors = _write_fast_corpus(tmp_path / "source", 800, seed=11, rank=D)
    corpus_uri = builder.build(tmp_path / "source", tmp_path / "out")
    ds = lance.dataset(corpus_uri)
    pca, _scale = lance_writer.read_pca_metadata(ds)

    meta = pq.read_table(tmp_path / "source" / lance_writer.METADATA_FILE)
    row_by_segment_start = {
        (s, int(c)): i
        for i, (s, c) in enumerate(
            zip(meta.column("segment_id").to_pylist(),
                meta.column("chunk_start_unix").to_pylist())
        )
    }

    table = ds.to_table(
        columns=["segment_id", "chunk_start_unix", lance_writer.VECTOR_FP_COLUMN],
        scan_in_order=True,
    )
    col = table.column(lance_writer.VECTOR_FP_COLUMN).combine_chunks()
    fp = col.flatten().to_numpy(zero_copy_only=False).reshape(-1, col.type.list_size)

    rng = np.random.default_rng(13)
    query = rng.standard_normal(MODEL_DIM).astype("float32")
    query /= np.linalg.norm(query)
    query_pca = (pca @ query).astype(np.float32)

    # The corpus is rank-<=256 in the fitted basis, so projected scores reproduce
    # the 768-d cosine the app ranks by.
    segs = table.column("segment_id").to_pylist()
    starts = table.column("chunk_start_unix").to_pylist()
    for row in range(0, len(segs), 97):
        source_row = row_by_segment_start[(segs[row], int(starts[row]))]
        exact = float(vectors[source_row] @ query)
        projected = float(fp[row] @ query_pca)
        assert projected == pytest.approx(exact, abs=2e-3)
