"""Offline checks for lance_writer: synthetic artifacts, no network, no model.

Run from the repo root:
    python -m pytest tests/test_lance_writer.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import conftest
import lance
import lance_writer
import numpy as np
import pyarrow as pa
import pytest
import threshold_search as ts


def test_written_dataset_has_the_layout_the_scan_depends_on() -> None:
    """Storage version, FSL vector columns, physical sort, and scalar indices.

    These are one test because they are one contract: the scan reads
    `embedding_i8` and `take()`s `vector_fp`, which needs both columns
    fixed-width at a full-zip storage version, and prefilter pruning needs
    the rows physically sorted with the indices present.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        ds = conftest.build_corpus(tmp_path, n=500)

        assert ds.data_storage_version == "2.2", ds.data_storage_version

        for name, value_type in (
            (lance_writer.EMBEDDING_I8_COLUMN, "int8"),
            (lance_writer.VECTOR_FP_COLUMN, "float"),
        ):
            field = ds.schema.field(name)
            assert str(field.type.value_type) == value_type, field.type
            assert field.type.list_size == conftest.D, field.type

        table = ds.to_table(columns=["chunk_start_unix", "vehicle"])
        keys = list(zip(table.column("chunk_start_unix").to_pylist(),
                        table.column("vehicle").to_pylist()))
        assert keys == sorted(keys), "rows must be written physically sorted"

        indexed = {
            (tuple(idx.field_names), idx.index_type) for idx in ds.describe_indices()
        }
        assert indexed == {
            (("chunk_start_unix",), "BTree"),
            (("segment_id",), "BTree"),
            (("vehicle",), "Bitmap"),
        }, indexed


def test_embedding_i8_and_pca_metadata_roundtrip() -> None:
    # The screen column and the basis needed to project a query must come back
    # byte-identical, modulo the writer's physical row sort.
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        artifact_dir = conftest.write_artifact(tmp_path / "artifact", n=300)
        ds = lance_writer.build_dataset(artifact_dir, str(tmp_path / "out.lance"))

        pca_out, scale_out = lance_writer.read_pca_metadata(ds)
        np.testing.assert_array_equal(pca_out, np.load(artifact_dir / lance_writer.PCA_FILE))
        np.testing.assert_array_equal(scale_out, np.load(artifact_dir / lance_writer.SCALE_FILE))

        written = ts._fixed_size_list_matrix(
            ds.to_table(columns=[lance_writer.EMBEDDING_I8_COLUMN], scan_in_order=True),
            lance_writer.EMBEDDING_I8_COLUMN,
            np.int8,
        )
        source = np.load(artifact_dir / lance_writer.CORPUS_INT8_FILE)
        assert written.shape == source.shape
        np.testing.assert_array_equal(np.sort(written, axis=0), np.sort(source, axis=0))


@pytest.mark.parametrize("pre_quant", [False, True])
def test_vector_fp_is_pre_quant_when_supplied_else_dequantized_int8(pre_quant: bool) -> None:
    """`vector_fp` is the true pre-quantization projection only when the
    artifact ships one; otherwise it is `int8 * scale / 127`, which carries no
    information beyond the screen column (see lance_writer's module docstring).
    Getting this backwards would silently weaken the re-rank to a no-op.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        artifact_dir = conftest.write_artifact(
            tmp_path / "artifact", n=200, pre_quant_fp32=pre_quant
        )
        ds = lance_writer.build_dataset(artifact_dir, str(tmp_path / "out.lance"))

        table = ds.to_table(
            columns=[lance_writer.EMBEDDING_I8_COLUMN, lance_writer.VECTOR_FP_COLUMN],
            scan_in_order=True,
        )
        vector_fp = ts._fixed_size_list_matrix(
            table, lance_writer.VECTOR_FP_COLUMN, np.float32
        )
        corpus_i8 = ts._fixed_size_list_matrix(
            table, lance_writer.EMBEDDING_I8_COLUMN, np.int8
        )
        scale = np.load(artifact_dir / lance_writer.SCALE_FILE)
        dequantized = corpus_i8.astype(np.float32) * (scale / np.float32(127.0))

        if pre_quant:
            assert not np.allclose(vector_fp, dequantized), (
                "vector_fp equals dequant(int8) even though a genuine "
                "pre-quantization projection was supplied"
            )
        else:
            np.testing.assert_allclose(vector_fp, dequantized, rtol=0, atol=0)


@pytest.mark.parametrize(
    "version, with_metadata, expected",
    [
        (lance_writer.DATA_STORAGE_VERSION, True, True),
        # Written before the writer's version moved -- must still read.
        (lance_writer.MIN_DATA_STORAGE_VERSION, True, True),
        # Pre-full-zip: take() is no longer 1 IOP/row, so the cost model breaks.
        ("2.0", True, False),
        # Right columns, no basis: read_pca_metadata would raise deep inside a
        # caller that trusted the gate as a green light.
        (lance_writer.DATA_STORAGE_VERSION, False, False),
    ],
)
def test_is_exact_threshold_dataset_gate(version: str, with_metadata: bool, expected: bool) -> None:
    n = 20
    table = pa.table(
        {
            "run_uuid": ["r"] * n,
            "chunk_start_unix": np.arange(n, dtype="int64"),
            "chunk_end_unix": np.arange(n, dtype="int64"),
            "segment_id": [f"s{i}" for i in range(n)],
            "vehicle": ["v"] * n,
            lance_writer.EMBEDDING_I8_COLUMN: pa.FixedSizeListArray.from_arrays(
                pa.array(np.zeros(n * conftest.D, dtype="int8")), conftest.D
            ),
            lance_writer.VECTOR_FP_COLUMN: pa.FixedSizeListArray.from_arrays(
                pa.array(np.zeros(n * conftest.D, dtype="float32")), conftest.D
            ),
        }
    )
    if with_metadata:
        table = table.replace_schema_metadata(
            {
                lance_writer.META_KEY_PCA_COMPONENTS: lance_writer._encode_array(
                    np.zeros((conftest.D, conftest.MODEL_DIM), dtype="float32")
                ),
                lance_writer.META_KEY_QUANT_SCALES: lance_writer._encode_array(
                    np.ones(conftest.D, dtype="float32")
                ),
            }
        )
    with tempfile.TemporaryDirectory() as tmp:
        uri = str(Path(tmp) / "gate.lance")
        lance.write_dataset(table, uri, mode="create", data_storage_version=version)
        assert lance_writer.is_exact_threshold_dataset(lance.dataset(uri)) is expected


def test_build_dataset_refuses_to_write_over_an_existing_uri() -> None:
    # There is no append/repair path, so a silent overwrite would be a data
    # loss bug and a half-built dataset must be deleted rather than reused.
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        artifact_dir = conftest.write_artifact(tmp_path / "artifact", n=50)
        out_uri = str(tmp_path / "out.lance")
        lance_writer.build_dataset(artifact_dir, out_uri)
        with pytest.raises(FileExistsError, match="already exists"):
            lance_writer.build_dataset(artifact_dir, out_uri)
