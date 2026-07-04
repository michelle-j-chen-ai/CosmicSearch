"""Offline checks for lance_writer: synthetic artifacts, no network, no model.

Run from the repo root:
    python -m pytest tests/test_lance_writer.py
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
        by_column = {tuple(i.field_names): i.type_url for i in ds.describe_indices()}
        assert by_column[("chunk_start_unix",)] == "/lance.table.BTreeIndexDetails", by_column
        assert by_column[("segment_id",)] == "/lance.table.BTreeIndexDetails", by_column
        assert by_column[("vehicle",)] == "/lance.table.BitmapIndexDetails", by_column


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


def test_vector_fp_falls_back_to_dequantized_int8_without_pre_quant_source() -> None:
    # Today's production artifact contract has no pre-quantization fp32
    # source (see module docstring); vector_fp must equal dequant(int8)
    # exactly in that case, not silently drift or randomly differ.
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        artifact_dir = _write_synthetic_artifacts(tmp_path, n=300)
        assert not (artifact_dir / lance_writer.PRE_QUANT_FP32_FILE).exists()
        ds = lance_writer.build_dataset(artifact_dir, str(tmp_path / "out.lance"))
        corpus_i8 = np.load(artifact_dir / lance_writer.CORPUS_INT8_FILE)
        scale = np.load(artifact_dir / lance_writer.SCALE_FILE)
        out_table = ds.to_table(columns=[lance_writer.VECTOR_FP_COLUMN, "segment_id"])
        vector_fp = np.stack(out_table.column(lance_writer.VECTOR_FP_COLUMN).to_pylist())
        # Match rows back by segment_id (writer sorts rows; see
        # test_embedding_i8_roundtrips_input for why positional comparison is unsafe).
        input_segment_ids = (
            pq.read_table(artifact_dir / lance_writer.METADATA_FILE)
            .column("segment_id")
            .to_pylist()
        )
        row_by_segment_id = {seg: i for i, seg in enumerate(input_segment_ids)}
        expected_order = [row_by_segment_id[seg] for seg in out_table.column("segment_id").to_pylist()]
        expected = corpus_i8[expected_order].astype("float32") * (scale.astype("float32") / np.float32(127.0))
        np.testing.assert_array_equal(vector_fp, expected)


def test_vector_fp_uses_pre_quant_fp32_when_provided() -> None:
    # When the artifact provides the true pre-quantization projection, it must
    # be stored verbatim as vector_fp, NOT overwritten by dequant(int8) --
    # this is what makes the re-rank an actual independent fp32 reference
    # instead of a restatement of the int8 screen.
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        artifact_dir = _write_synthetic_artifacts(tmp_path, n=150, seed=5)
        n = 150
        rng = np.random.default_rng(99)
        true_fp32 = rng.standard_normal((n, _D)).astype("float32")
        np.save(artifact_dir / lance_writer.PRE_QUANT_FP32_FILE, true_fp32)

        corpus_i8 = np.load(artifact_dir / lance_writer.CORPUS_INT8_FILE)
        scale = np.load(artifact_dir / lance_writer.SCALE_FILE)
        dequant = corpus_i8.astype("float32") * (scale.astype("float32") / np.float32(127.0))
        # Sanity: the true source really does differ from what dequant would
        # have produced, so this test cannot pass by coincidence.
        assert not np.allclose(true_fp32, dequant)

        ds = lance_writer.build_dataset(artifact_dir, str(tmp_path / "out.lance"))
        out_table = ds.to_table(columns=[lance_writer.VECTOR_FP_COLUMN, "segment_id"])
        vector_fp = np.stack(out_table.column(lance_writer.VECTOR_FP_COLUMN).to_pylist())
        input_segment_ids = (
            pq.read_table(artifact_dir / lance_writer.METADATA_FILE)
            .column("segment_id")
            .to_pylist()
        )
        row_by_segment_id = {seg: i for i, seg in enumerate(input_segment_ids)}
        expected_order = [row_by_segment_id[seg] for seg in out_table.column("segment_id").to_pylist()]
        np.testing.assert_array_equal(vector_fp, true_fp32[expected_order])


def test_is_v21_dataset_false_without_pca_metadata() -> None:
    # A dataset with the right columns/version but stripped/absent PCA schema
    # metadata must NOT look like a valid exact-threshold dataset -- otherwise
    # read_pca_metadata raises an opaque KeyError deep inside a caller that
    # trusted is_v21_dataset's True as a green light.
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        n = 20
        table = pa.table(
            {
                "run_uuid": ["r"] * n,
                "chunk_start_unix": np.arange(n, dtype="int64"),
                "chunk_end_unix": np.arange(n, dtype="int64"),
                "segment_id": [f"s{i}" for i in range(n)],
                "vehicle": ["v"] * n,
                lance_writer.EMBEDDING_I8_COLUMN: pa.FixedSizeListArray.from_arrays(
                    pa.array(np.zeros(n * _D, dtype="int8")), _D
                ),
                lance_writer.VECTOR_FP_COLUMN: pa.FixedSizeListArray.from_arrays(
                    pa.array(np.zeros(n * _D, dtype="float32")), _D
                ),
            }
        )
        out_uri = str(tmp_path / "no_meta.lance")
        lance.write_dataset(
            table, out_uri, mode="create", data_storage_version=lance_writer.DATA_STORAGE_VERSION
        )
        ds = lance.dataset(out_uri)
        assert not lance_writer.is_v21_dataset(ds), (
            "dataset has the right columns/version but no PCA schema metadata "
            "-- is_v21_dataset must not report it as valid"
        )
        import pytest

        with pytest.raises(ValueError, match="schema metadata is missing"):
            lance_writer.read_pca_metadata(ds)


def test_required_string_columns_cast_regardless_of_source_encoding() -> None:
    # run_uuid/segment_id must land in the written schema as plain string
    # regardless of whether the source metadata.parquet encoded them as
    # dictionary<string> -- otherwise the written dataset's schema (and any
    # index built on segment_id) varies with an incidental encoding choice.
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        artifact_dir = tmp_path / "artifact"
        artifact_dir.mkdir()
        n = 40
        rng = np.random.default_rng(3)
        np.save(artifact_dir / lance_writer.PCA_FILE, rng.standard_normal((_D, _MODEL_DIM)).astype("float32"))
        np.save(artifact_dir / lance_writer.SCALE_FILE, rng.uniform(0.01, 0.5, size=_D).astype("float32"))
        np.save(artifact_dir / lance_writer.CORPUS_INT8_FILE, rng.integers(-127, 128, size=(n, _D), dtype=np.int8))
        chunk_start = rng.integers(1_700_000_000, 1_700_100_000, size=n).astype("int64")
        table = pa.table(
            {
                "run_uuid": pa.array([f"run-{i % 3}" for i in range(n)]).dictionary_encode(),
                "chunk_start_unix": chunk_start,
                "chunk_end_unix": chunk_start + 5,
                "segment_id": pa.array([f"seg-{i}" for i in range(n)]).dictionary_encode(),
                "vehicle": ["veh_a"] * n,
            }
        )
        pq.write_table(table, artifact_dir / lance_writer.METADATA_FILE)

        ds = lance_writer.build_dataset(artifact_dir, str(tmp_path / "out.lance"))
        schema = ds.schema
        assert schema.field("run_uuid").type == pa.string(), schema.field("run_uuid").type
        assert schema.field("segment_id").type == pa.string(), schema.field("segment_id").type


def test_build_dataset_raises_clear_error_on_existing_uri() -> None:
    import pytest

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        artifact_dir = _write_synthetic_artifacts(tmp_path, n=30)
        out_uri = str(tmp_path / "out.lance")
        lance_writer.build_dataset(artifact_dir, out_uri)
        with pytest.raises(FileExistsError, match="already exists"):
            lance_writer.build_dataset(artifact_dir, out_uri)
