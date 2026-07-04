"""End-to-end check for search_engine's Lance 2.1 dispatch.

Builds a real Lance 2.1 exact-threshold dataset (via lance_writer.build_dataset,
same synthetic-artifact fixture as test_threshold_search.py), points
search_engine at it by monkeypatching local_cache.ensure_corpus_local (no
network / no S3 / no model download), and checks:

  1. load_threshold_corpus opens a real 2.1 dataset as a ThresholdCorpus, and
     its .threshold_search(...) matches a brute-force fp32 oracle exactly.
  2. load_corpus -- the shared entry point web_server.py/app.py use, whose
     callers assume a full Corpus (rank_top_k/score_corpus/.matrix/.time_span)
     -- raises a clear error for a v2.1 dataset instead of silently returning
     a ThresholdCorpus those callers cannot use (see search_engine.py's
     load_corpus docstring for why they are split).
  3. A legacy (non-2.1) `.lance` dataset does NOT false-trigger that error --
     it still goes through the existing _load_corpus_lance_dataset path and
     returns a plain Corpus.
  4. The npy path (_load_corpus_npy -> rank_top_k) is unaffected by this
     change -- no behavior regression on the existing top-k path.

Run from the repo root:
    python -m pytest tests/test_load_corpus_threshold.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import lance
import lance_writer
import local_cache
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import search_engine
import threshold_search as ts

_D = 256
_MODEL_DIM = 768


def _write_synthetic_artifacts(tmp_dir: Path, n: int, seed: int = 0) -> Path:
    """Same fixture shape as test_threshold_search.py / test_lance_writer.py."""
    rng = np.random.default_rng(seed)
    artifact_dir = tmp_dir / "artifact"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    pca = rng.standard_normal((_D, _MODEL_DIM)).astype("float32")
    scale = rng.uniform(0.01, 0.5, size=_D).astype("float32")
    corpus_i8 = rng.integers(-127, 128, size=(n, _D), dtype=np.int8)

    np.save(artifact_dir / lance_writer.PCA_FILE, pca)
    np.save(artifact_dir / lance_writer.SCALE_FILE, scale)
    np.save(artifact_dir / lance_writer.CORPUS_INT8_FILE, corpus_i8)

    chunk_start = rng.integers(1_700_000_000, 1_700_100_000, size=n).astype("int64")
    vehicles = rng.choice(["veh_a", "veh_b", "veh_c"], size=n)
    table = pa.table(
        {
            "run_uuid": [f"run-{i % 5}" for i in range(n)],
            "chunk_start_unix": chunk_start,
            "chunk_end_unix": chunk_start + 5,
            "segment_id": [f"seg-{i}" for i in range(n)],
            "vehicle": vehicles,
        }
    )
    pq.write_table(table, artifact_dir / lance_writer.METADATA_FILE)
    return artifact_dir


def _query(seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    q = rng.standard_normal(_MODEL_DIM).astype("float32")
    return q / np.linalg.norm(q)


def _brute_force_oracle(
    ds: lance.LanceDataset, query: np.ndarray, tau: float
) -> set[int]:
    pca, _scale = lance_writer.read_pca_metadata(ds)
    query_pca = ts._project_query(query, pca).astype(np.float64)
    table = ds.to_table(columns=[lance_writer.VECTOR_FP_COLUMN])
    fp = ts._fixed_size_list_matrix(table, lance_writer.VECTOR_FP_COLUMN, np.float64)
    scores = fp @ query_pca
    return set(np.nonzero(scores >= tau)[0].tolist())


def _patch_ensure_corpus_local(local_dir: Path):
    """Monkeypatch search_engine.local_cache.ensure_corpus_local for one call.

    Returns a context manager-free (manual restore) pair: call `.restore()`
    when done. Avoids any real S3/local_cache download machinery -- load_corpus
    only needs a local directory, which the test already built directly.
    """
    original = local_cache.ensure_corpus_local
    local_cache.ensure_corpus_local = lambda embeddings_uri, client: local_dir

    class _Restorer:
        def restore(self) -> None:
            local_cache.ensure_corpus_local = original

    return _Restorer()


def test_load_threshold_corpus_opens_v21_dataset() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        artifact_dir = _write_synthetic_artifacts(tmp_path, n=5_000, seed=0)
        out_dir = tmp_path / "corpus.lance"
        lance_writer.build_dataset(artifact_dir, str(out_dir))

        restorer = _patch_ensure_corpus_local(out_dir)
        try:
            corpus = search_engine.load_threshold_corpus("s3://bucket/corpus.lance")
        finally:
            restorer.restore()

        assert isinstance(corpus, ts.ThresholdCorpus), (
            f"expected a ThresholdCorpus for a real 2.1 dataset, got {type(corpus)!r}"
        )

        ds = lance.dataset(str(out_dir))
        query = _query()
        # tau at the median exact score -> a mix of ABOVE/BAND/BELOW rows.
        pca, _scale = lance_writer.read_pca_metadata(ds)
        query_pca = ts._project_query(query, pca).astype(np.float64)
        table = ds.to_table(columns=[lance_writer.VECTOR_FP_COLUMN])
        fp = ts._fixed_size_list_matrix(
            table, lance_writer.VECTOR_FP_COLUMN, np.float64
        )
        all_scores = fp @ query_pca
        tau = float(np.percentile(all_scores, 90.0))

        hits = corpus.threshold_search(query, tau)
        got_ids = {h.row_id for h in hits}
        expected_ids = _brute_force_oracle(ds, query, tau)
        assert got_ids == expected_ids, (
            f"row-id mismatch: missing={expected_ids - got_ids} "
            f"extra={got_ids - expected_ids}"
        )
        for h in hits:
            assert h.score >= tau
            assert h.segment_id, "ThresholdHit should carry resolvable metadata"


def test_load_corpus_raises_on_v21_dataset() -> None:
    """load_corpus (the shared Corpus-returning entrypoint) must refuse a v2.1
    exact-threshold dataset with a clear error rather than silently handing
    web_server.py/app.py an object that breaks on their first .matrix/
    .time_span()/score_corpus() call."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        artifact_dir = _write_synthetic_artifacts(tmp_path, n=500, seed=1)
        out_dir = tmp_path / "corpus.lance"
        lance_writer.build_dataset(artifact_dir, str(out_dir))

        restorer = _patch_ensure_corpus_local(out_dir)
        try:
            import pytest

            with pytest.raises(ValueError, match="load_threshold_corpus"):
                search_engine.load_corpus("s3://bucket/corpus.lance", "float16")
        finally:
            restorer.restore()


def test_load_corpus_does_not_false_trigger_on_legacy_lance_dataset() -> None:
    """A plain (pre-2.1) `.lance` dataset must still use _load_corpus_lance_dataset."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        n = 20
        rng = np.random.default_rng(2)
        table = pa.table(
            {
                "chunk_id": [f"run-0#t{1_700_000_000 + i}" for i in range(n)],
                "run_uuid": ["run-0"] * n,
                "chunk_start_unix": np.arange(1_700_000_000, 1_700_000_000 + n, dtype="int64"),
                "source_media_uri": [f"s3://bucket/{i}.mp4" for i in range(n)],
                "segment_id": [f"seg-{i}" for i in range(n)],
                "vector": pa.FixedSizeListArray.from_arrays(
                    pa.array(
                        rng.standard_normal(n * _MODEL_DIM).astype("float32")
                    ),
                    _MODEL_DIM,
                ),
            }
        )
        out_dir = tmp_path / "legacy.lance"
        lance.write_dataset(table, str(out_dir), mode="create")
        ds = lance.dataset(str(out_dir))
        assert not lance_writer.is_v21_dataset(ds), (
            "fixture bug: legacy dataset should not look like a 2.1 dataset"
        )

        restorer = _patch_ensure_corpus_local(out_dir)
        try:
            corpus = search_engine.load_corpus("s3://bucket/legacy.lance", "float16")
        finally:
            restorer.restore()

        assert isinstance(corpus, search_engine.Corpus), (
            f"legacy .lance dataset must dispatch to the existing plain Corpus "
            f"path, got {type(corpus)!r}"
        )
        assert corpus.num_rows == n
        assert corpus.matrix.shape == (n, _MODEL_DIM)


def test_rank_top_k_unaffected_on_npy_path() -> None:
    """The npy dispatch branch (and rank_top_k) still works after the new branch."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        n = 50
        rng = np.random.default_rng(3)
        matrix = rng.standard_normal((n, _MODEL_DIM)).astype("float32")
        matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
        np.save(tmp_path / local_cache.NPY_MATRIX_FILE, matrix)
        meta = pa.table(
            {
                "chunk_id": [f"run-0#t{1_700_000_000 + i}" for i in range(n)],
                "run_uuid": ["run-0"] * n,
                "chunk_start_unix": np.arange(
                    1_700_000_000, 1_700_000_000 + n, dtype="int64"
                ),
                "source_media_uri": [f"s3://bucket/{i}.mp4" for i in range(n)],
                "segment_id": [f"seg-{i}" for i in range(n)],
            }
        )
        pq.write_table(meta, tmp_path / local_cache.NPY_METADATA_FILE)

        restorer = _patch_ensure_corpus_local(tmp_path)
        try:
            corpus = search_engine.load_corpus("s3://bucket/npy-corpus", "float32")
        finally:
            restorer.restore()

        assert isinstance(corpus, search_engine.Corpus)
        query = matrix[7].copy()
        hits = search_engine.rank_top_k(query, corpus, top_k=5)
        assert hits[0].index == 7
        assert hits[0].rank == 1
        assert all(
            hits[i].score >= hits[i + 1].score for i in range(len(hits) - 1)
        )
