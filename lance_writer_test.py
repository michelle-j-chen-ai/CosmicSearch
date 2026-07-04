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
        input_segment_ids = (
            pq.read_table(artifact_dir / lance_writer.METADATA_FILE)
            .column("segment_id")
            .to_pylist()
        )
        # segment_id is generated as f"seg-{i}" (unique per row) by
        # _write_synthetic_artifacts, so it is a reliable row identifier even
        # when chunk_start_unix/vehicle sort keys collide across rows.
        row_by_segment_id = {seg: i for i, seg in enumerate(input_segment_ids)}
        assert len(row_by_segment_id) == len(input_segment_ids), "segment_id not unique"

        ds = lance_writer.build_dataset(artifact_dir, str(tmp_path / "out.lance"))
        out_table = ds.to_table(columns=[lance_writer.EMBEDDING_I8_COLUMN, "segment_id"])
        out_i8 = np.stack(out_table.column(lance_writer.EMBEDDING_I8_COLUMN).to_pylist())
        out_segment_ids = out_table.column("segment_id").to_pylist()

        # Match each written row back to its input row by segment_id, not by
        # re-deriving the writer's sort order -- this stays correct regardless
        # of ties in the (chunk_start_unix, vehicle) sort keys.
        expected_order = [row_by_segment_id[seg] for seg in out_segment_ids]
        expected = input_i8[expected_order]
        assert out_i8.shape == expected.shape, (out_i8.shape, expected.shape)
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


def test_compact_files_merges_small_fragments_then_is_noop() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        artifact_dir = _write_synthetic_artifacts(tmp_path, n=100)
        out_uri = str(tmp_path / "out.lance")

        # Write with a small max_rows_per_file so the 100 synthetic rows land
        # in multiple small fragments -- build_dataset's own compact_files
        # call would immediately merge these, so write directly via
        # build_table + lance.write_dataset to observe the pre-compaction
        # fragment layout.
        table = lance_writer.build_table(artifact_dir)
        lance.write_dataset(
            table,
            out_uri,
            mode="create",
            data_storage_version=lance_writer.DATA_STORAGE_VERSION,
            max_rows_per_file=10,
        )
        ds = lance.dataset(out_uri)
        fragments_before = len(ds.get_fragments())
        assert fragments_before > 1, (
            "expected multiple fragments before compaction, got "
            f"{fragments_before}"
        )

        # Real compaction: fragments should actually merge toward one
        # 1M-row-target fragment, not just report success trivially.
        metrics = ds.optimize.compact_files(target_rows_per_fragment=1_000_000)
        ds = lance.dataset(out_uri)
        fragments_after = len(ds.get_fragments())
        assert metrics.fragments_removed == fragments_before, metrics
        assert fragments_after == 1, (fragments_after, metrics)
        assert ds.count_rows() == 100, ds.count_rows()

        # Now a second compaction call on the now-single, already-at-target
        # fragment is a genuine no-op.
        noop_metrics = ds.optimize.compact_files(target_rows_per_fragment=1_000_000)
        assert noop_metrics.fragments_removed == 0, noop_metrics
        assert noop_metrics.files_added == 0, noop_metrics


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
