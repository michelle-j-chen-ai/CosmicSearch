"""Offline checks for lance_writer: synthetic artifacts, no network, no model.

Run from this directory:
    python lance_writer_test.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import lance
import lance_writer
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

_D = 256
_MODEL_DIM = 768


def _write_synthetic_artifacts(tmp_dir: Path, n: int, seed: int = 0) -> Path:
    """Write a gpu_corpus-style int8 PCA artifact under `tmp_dir` and return it."""
    rng = np.random.default_rng(seed)
    artifact_dir = tmp_dir / "artifact"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    pca = rng.standard_normal((_D, _MODEL_DIM)).astype("float32")
    scale = rng.uniform(0.01, 0.5, size=_D).astype("float32")
    corpus_i8 = rng.integers(-127, 128, size=(n, _D), dtype=np.int8)

    np.save(artifact_dir / lance_writer.PCA_FILE, pca)
    np.save(artifact_dir / lance_writer.SCALE_FILE, scale)
    np.save(artifact_dir / lance_writer.CORPUS_INT8_FILE, corpus_i8)

    # Deliberately unsorted chunk_start_unix / vehicle so the writer's sort is
    # exercised (not accidentally already-sorted input).
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


def test_data_storage_version_is_2_1() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        artifact_dir = _write_synthetic_artifacts(tmp_path, n=500)
        ds = lance_writer.build_dataset(artifact_dir, str(tmp_path / "out.lance"))
        assert ds.data_storage_version == "2.1", ds.data_storage_version


def test_vector_columns_are_fixed_size_list_of_correct_width() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        artifact_dir = _write_synthetic_artifacts(tmp_path, n=200)
        ds = lance_writer.build_dataset(artifact_dir, str(tmp_path / "out.lance"))
        schema = ds.schema
        i8_field = schema.field(lance_writer.EMBEDDING_I8_COLUMN)
        fp_field = schema.field(lance_writer.VECTOR_FP_COLUMN)
        assert str(i8_field.type.value_type) == "int8", i8_field.type
        assert i8_field.type.list_size == _D, i8_field.type
        assert str(fp_field.type.value_type) == "float", fp_field.type
        assert fp_field.type.list_size == _D, fp_field.type


def test_embedding_i8_roundtrips_input() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        artifact_dir = _write_synthetic_artifacts(tmp_path, n=300)
        input_i8 = np.load(artifact_dir / lance_writer.CORPUS_INT8_FILE)
        input_starts = pq.read_table(artifact_dir / lance_writer.METADATA_FILE).column(
            "chunk_start_unix"
        ).to_numpy(zero_copy_only=False)

        ds = lance_writer.build_dataset(artifact_dir, str(tmp_path / "out.lance"))
        out_table = ds.to_table(columns=[lance_writer.EMBEDDING_I8_COLUMN, "chunk_start_unix"])
        out_i8 = np.stack(out_table.column(lance_writer.EMBEDDING_I8_COLUMN).to_pylist())
        out_starts = out_table.column("chunk_start_unix").to_numpy(zero_copy_only=False)

        # Match written rows back to input rows by chunk_start_unix + row content,
        # since the writer physically reorders rows.
        order = np.argsort(input_starts, kind="stable")
        expected = input_i8[order]
        assert out_i8.shape == expected.shape, (out_i8.shape, expected.shape)
        np.testing.assert_array_equal(out_starts, input_starts[order])
        np.testing.assert_array_equal(out_i8.astype(np.int8), expected)


def test_rows_ordered_by_chunk_start_then_vehicle() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        artifact_dir = _write_synthetic_artifacts(tmp_path, n=400)
        ds = lance_writer.build_dataset(artifact_dir, str(tmp_path / "out.lance"))
        out_table = ds.to_table(columns=["chunk_start_unix", "vehicle"])
        starts = out_table.column("chunk_start_unix").to_numpy(zero_copy_only=False)
        vehicles = out_table.column("vehicle").to_pylist()

        keys = list(zip(starts.tolist(), vehicles))
        assert keys == sorted(keys), "rows are not sorted by (chunk_start_unix, vehicle)"


def test_scalar_indices_exist() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        artifact_dir = _write_synthetic_artifacts(tmp_path, n=200)
        ds = lance_writer.build_dataset(artifact_dir, str(tmp_path / "out.lance"))
        indices = ds.list_indices()
        by_column = {tuple(i["fields"]): i["type"] for i in indices}
        assert by_column[("chunk_start_unix",)] == "BTree", by_column
        assert by_column[("segment_id",)] == "BTree", by_column
        assert by_column[("vehicle",)] == "Bitmap", by_column


def test_compact_files_is_noop_on_already_1m_row_fragments() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        # Small dataset -> a single fragment, already well under the 1M target,
        # so compaction should not touch it.
        artifact_dir = _write_synthetic_artifacts(tmp_path, n=100)
        ds = lance_writer.build_dataset(artifact_dir, str(tmp_path / "out.lance"))
        metrics = ds.optimize.compact_files(target_rows_per_fragment=1_000_000)
        assert metrics.fragments_removed == 0, metrics
        assert metrics.files_added == 0, metrics


def test_builder_runs_from_synthetic_artifacts_with_no_real_corpus() -> None:
    # The whole builder + read path with no dependency on real corpus data.
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        artifact_dir = _write_synthetic_artifacts(tmp_path, n=64)
        out_uri = str(tmp_path / "out.lance")
        ds = lance_writer.build_dataset(artifact_dir, out_uri)
        assert ds.count_rows() == 64
        assert lance_writer.is_v21_dataset(ds)
        reopened = lance.dataset(out_uri)
        assert lance_writer.is_v21_dataset(reopened)


def test_pca_metadata_roundtrips_through_schema() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        artifact_dir = _write_synthetic_artifacts(tmp_path, n=50)
        pca_in = np.load(artifact_dir / lance_writer.PCA_FILE)
        scale_in = np.load(artifact_dir / lance_writer.SCALE_FILE)
        ds = lance_writer.build_dataset(artifact_dir, str(tmp_path / "out.lance"))
        pca_out, scale_out = lance_writer.read_pca_metadata(ds)
        np.testing.assert_array_equal(pca_out, pca_in)
        np.testing.assert_array_equal(scale_out, scale_in)


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL {name}: {exc}")
    raise SystemExit(1 if failures else 0)
