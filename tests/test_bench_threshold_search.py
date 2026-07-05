"""Hermetic checks for the PoC benchmark harness: no S3, no network, no model.

The harness's value is that its zero-false-negative check is a REAL test (the
oracle is a genuine pre-quantization fp32 signal, not a dequant(int8) copy of
the screen). These tests lock that in: run_benchmark must report zero false
negatives AND a screen-vs-oracle gap that is real quantization error (orders
of magnitude above fp rounding), covered by eps.

Run from the repo root:
    python -m pytest tests/test_bench_threshold_search.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import bench_threshold_search as bench
import lance
import lance_writer
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

_D = 256
_MODEL_DIM = 768


def _orthonormal_rows(rng: np.random.Generator, d: int, model_dim: int) -> np.ndarray:
    q, _ = np.linalg.qr(rng.standard_normal((model_dim, d)))
    return q[:, :d].T.astype(np.float32)


def _build_genuine_fp32_dataset(tmp_path: Path, n: int, seed: int) -> lance.LanceDataset:
    """A v2.1 dataset whose vector_fp is a GENUINE pre-quantization fp32 signal
    (via pca_projection_fp32.npy), so the screen (int8) and the oracle
    (vector_fp) differ by real quantization error -- not a dequant(int8) copy."""
    rng = np.random.default_rng(seed)
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()

    pca = _orthonormal_rows(rng, _D, _MODEL_DIM)
    true_fp32 = (rng.standard_normal((n, _D)) * rng.uniform(0.5, 3.0, size=_D)).astype("float32")
    scale = np.abs(true_fp32).max(axis=0).astype("float32")
    corpus_i8 = np.round(true_fp32 * (127.0 / scale)).clip(-127, 127).astype("int8")

    np.save(artifact_dir / lance_writer.PCA_FILE, pca)
    np.save(artifact_dir / lance_writer.SCALE_FILE, scale)
    np.save(artifact_dir / lance_writer.CORPUS_INT8_FILE, corpus_i8)
    np.save(artifact_dir / lance_writer.PRE_QUANT_FP32_FILE, true_fp32)
    chunk_start = rng.integers(1_700_000_000, 1_700_100_000, size=n).astype("int64")
    pq.write_table(
        pa.table(
            {
                "run_uuid": [f"run-{i % 5}" for i in range(n)],
                "chunk_start_unix": chunk_start,
                "segment_id": [f"run-{i % 5}#t{i}" for i in range(n)],
            }
        ),
        artifact_dir / lance_writer.METADATA_FILE,
    )
    return lance_writer.build_dataset(artifact_dir, str(tmp_path / "out.lance"))


def test_run_benchmark_proves_zero_false_negatives_against_genuine_fp32() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        ds = _build_genuine_fp32_dataset(Path(tmp), n=6_000, seed=3)
        result = bench.run_benchmark(ds, repeats=3, tau_percentile=99.0, seed=7)

        assert result.zero_false_negatives, result
        assert result.false_negatives == 0
        assert result.extras == 0
        assert result.matches > 0, "test tau should produce a non-empty match set"
        # The gap the eps bound must cover is REAL int8 quantization error, not
        # fp rounding: if the oracle were a dequant(int8) copy of the screen
        # (the circular bug), this would collapse to ~1e-7.
        assert result.max_int8_gap > 1e-4, result.max_int8_gap
        assert result.eps_covers_int8_gap, (result.max_int8_gap, result.eps)
        # Timing fields are populated (non-negative, warm ordering holds).
        assert result.cold_hydrate_s >= 0
        assert result.warm_query_s_min <= result.warm_query_s_p50 <= result.warm_query_s_p95
        assert result.e2e_s > 0


def test_build_int8_artifact_emits_genuine_pre_quant_projection() -> None:
    # build_int8_artifact needs scikit-learn (lazy import); skip if absent so
    # the default suite stays sklearn-free.
    import pytest

    pytest.importorskip("sklearn")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        rng = np.random.default_rng(11)
        n = 2_000
        # Rank-~256 unit-norm embeddings, like the real corpus.
        basis, _ = np.linalg.qr(rng.standard_normal((_MODEL_DIM, _D)))
        emb = (rng.standard_normal((n, _D)) @ basis.T).astype("float32")
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)
        meta = pa.table(
            {
                "run_uuid": [f"r{i % 7}" for i in range(n)],
                "chunk_start_unix": np.arange(n, dtype="int64"),
                "chunk_id": [f"r{i % 7}#t{i}" for i in range(n)],
            }
        )

        artifact_dir = bench.build_int8_artifact(
            emb, meta, tmp_path / "art", sample_rows=None, seed=1
        )
        for name in (
            lance_writer.PCA_FILE,
            lance_writer.SCALE_FILE,
            lance_writer.CORPUS_INT8_FILE,
            lance_writer.METADATA_FILE,
            lance_writer.PRE_QUANT_FP32_FILE,
        ):
            assert (artifact_dir / name).exists(), name

        # The built dataset's vector_fp must be the SVD projection (genuine
        # fp32), i.e. NOT equal to the dequantized int8 -- that is exactly what
        # makes the harness's zero-FN check non-circular.
        ds = lance_writer.build_dataset(artifact_dir, str(tmp_path / "out.lance"))
        vfp = bench.ts._fixed_size_list_matrix(
            ds.to_table(columns=[lance_writer.VECTOR_FP_COLUMN], scan_in_order=True),
            lance_writer.VECTOR_FP_COLUMN,
            np.float64,
        )
        i8 = bench.ts._fixed_size_list_matrix(
            ds.to_table(columns=[lance_writer.EMBEDDING_I8_COLUMN], scan_in_order=True),
            lance_writer.EMBEDDING_I8_COLUMN,
            np.float64,
        )
        _pca, scale = lance_writer.read_pca_metadata(ds)
        dequant = i8 * (scale.astype(np.float64) / 127.0)
        assert np.abs(vfp - dequant).max() > 1e-4, (
            "vector_fp collapsed to dequant(int8) -- pre-quant projection not stored"
        )

        result = bench.run_benchmark(ds, repeats=2, tau_percentile=90.0, seed=2)
        assert result.zero_false_negatives, result


def test_sample_rows_downsamples_deterministically() -> None:
    import pytest

    pytest.importorskip("sklearn")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        rng = np.random.default_rng(5)
        n = 3_000
        emb = rng.standard_normal((n, _D)).astype("float32")
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)
        meta = pa.table(
            {
                "run_uuid": ["r"] * n,
                "chunk_start_unix": np.arange(n, dtype="int64"),
                "chunk_id": [f"c{i}" for i in range(n)],
            }
        )
        art = bench.build_int8_artifact(emb, meta, tmp_path / "a", sample_rows=500, seed=9)
        corpus_i8 = np.load(art / lance_writer.CORPUS_INT8_FILE)
        assert corpus_i8.shape[0] == 500
        assert pq.read_table(art / lance_writer.METADATA_FILE).num_rows == 500


def test_upload_directory_walks_all_files() -> None:
    import oci_s3

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "ds.lance"
        (root / "data").mkdir(parents=True)
        (root / "data" / "0.lance").write_bytes(b"x")
        (root / "_versions").mkdir()
        (root / "_versions" / "1.manifest").write_bytes(b"y")

        calls: list[tuple[str, str, str]] = []

        class _StubClient:
            def upload_file(self, local, bucket, key):
                calls.append((local, bucket, key))

        count = oci_s3.upload_directory(root, "s3://bkt/prefix/ds.lance", _StubClient())
        assert count == 2
        keys = sorted(k for _, _, k in calls)
        assert keys == [
            "prefix/ds.lance/_versions/1.manifest",
            "prefix/ds.lance/data/0.lance",
        ], keys
        assert all(bucket == "bkt" for _, bucket, _ in calls)
