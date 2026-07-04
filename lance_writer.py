"""Lance 2.1 dataset writer for the exact-threshold corpus layout.

Converts an existing `gpu_corpus`-style int8 PCA artifact (`pca_components.npy`
+ `quant_scales.npy` + `corpus_int8.npy` + `metadata.parquet`, see gpu_corpus.py
for the artifact contract) into a single Lance dataset with two fixed-width
vector columns:

  embedding_i8  FSL<int8,256>    the resident screen column (gpu_corpus'
                                 `_score_i8` numba kernel scans this directly)
  vector_fp     FSL<float32,256> exact re-rank column, take()-only

`vector_fp` is the int8 dequantized back to fp32 (`int8 * scale / 127`) -- this
builder does not re-fit SVD or have access to a pre-quantization fp32 corpus,
it only converts the artifact that already exists.

Both vector columns are >=128 bytes/value (256 * 4 = 1024B, 256 * 1 = 256B),
so under `data_storage_version="2.1"` they get Lance's full-zip encoding,
which makes a `take()` a single IOP per row -- the property the re-rank path
depends on.

Rows are written physically sorted by (chunk_start_unix, vehicle) so that
Lance's per-fragment min/max stats let date/vehicle prefilters skip whole
fragments. The PCA basis and per-dim quantization scales travel with the
dataset as schema metadata (base64-encoded .npy bytes) so a reader can
recover them without a side-channel file.
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import lance
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# Artifact filenames, matching gpu_corpus.py.
PCA_FILE = "pca_components.npy"
SCALE_FILE = "quant_scales.npy"
CORPUS_INT8_FILE = "corpus_int8.npy"
METADATA_FILE = "metadata.parquet"

DATA_STORAGE_VERSION = "2.1"
EMBEDDING_I8_COLUMN = "embedding_i8"
VECTOR_FP_COLUMN = "vector_fp"

# Candidate vehicle-id column names in the source metadata, first present wins
# (mirrors gpu_corpus.load_gpu_corpus / search_engine._VEHICLE_COLUMNS).
_VEHICLE_COLUMNS = ("vehicle", "vehicle_name", "vehicle_id")

# Schema-metadata keys carrying the PCA basis + scales alongside the dataset.
META_KEY_PCA_COMPONENTS = b"nls.pca_components"
META_KEY_QUANT_SCALES = b"nls.quant_scales"

_DEFAULT_TARGET_ROWS_PER_FRAGMENT = 1_000_000
_DEFAULT_MAX_ROWS_PER_FILE = 1_048_576


def _encode_array(arr: np.ndarray) -> bytes:
    """Serialize an ndarray to bytes (base64 of its .npy encoding)."""
    buf = io.BytesIO()
    np.save(buf, arr)
    return base64.b64encode(buf.getvalue())


def decode_array(encoded: bytes) -> np.ndarray:
    """Inverse of `_encode_array`; used by readers to recover the PCA basis."""
    return np.load(io.BytesIO(base64.b64decode(encoded)))


def read_pca_metadata(ds: lance.LanceDataset) -> tuple[np.ndarray, np.ndarray]:
    """Return (pca_components, quant_scales) stored in `ds`'s schema metadata."""
    meta = ds.schema.metadata or {}
    return (
        decode_array(meta[META_KEY_PCA_COMPONENTS]),
        decode_array(meta[META_KEY_QUANT_SCALES]),
    )


def _fixed_size_list(values: np.ndarray, item_type: pa.DataType) -> pa.Array:
    """Build a FixedSizeListArray of width `values.shape[1]` from a (N, D) array."""
    n, d = values.shape
    flat = pa.array(values.reshape(-1), type=item_type)
    return pa.FixedSizeListArray.from_arrays(flat, d)


def _read_metadata_table(metadata_path: Path, n_rows: int) -> pa.Table:
    """Read the artifact's scalar metadata, filling optional columns as null."""
    meta = pq.read_table(metadata_path)
    if meta.num_rows != n_rows:
        raise ValueError(
            f"metadata row count ({meta.num_rows}) does not match corpus row "
            f"count ({n_rows})"
        )
    cols = set(meta.column_names)

    def required(name: str) -> pa.Array:
        if name not in cols:
            raise ValueError(f"metadata.parquet missing required column {name!r}")
        return meta.column(name).combine_chunks()

    def optional(name: str, pa_type: pa.DataType) -> pa.Array:
        if name in cols:
            return meta.column(name).combine_chunks()
        return pa.nulls(n_rows, type=pa_type)

    veh_name = next((c for c in _VEHICLE_COLUMNS if c in cols), None)
    vehicle = meta.column(veh_name).combine_chunks() if veh_name else pa.nulls(
        n_rows, type=pa.string()
    )

    return pa.table(
        {
            "run_uuid": required("run_uuid"),
            "chunk_start_unix": required("chunk_start_unix").cast(pa.int64()),
            "chunk_end_unix": optional("chunk_end_unix", pa.int64()).cast(pa.int64()),
            "segment_id": required("segment_id"),
            "vehicle": vehicle.cast(pa.string()),
        }
    )


def build_table(artifact_dir: Path) -> pa.Table:
    """Assemble the full Arrow table (vectors + scalars), sorted for writing.

    Does not write anything; separated from `build_dataset` so a caller (or a
    test) can inspect the pre-write table.
    """
    artifact_dir = Path(artifact_dir)
    pca = np.load(artifact_dir / PCA_FILE).astype("float32")  # (D, 768)
    scale = np.load(artifact_dir / SCALE_FILE).astype("float32")  # (D,)
    corpus_i8 = np.load(artifact_dir / CORPUS_INT8_FILE)  # (N, D) int8
    if corpus_i8.dtype != np.int8:
        raise ValueError(f"{CORPUS_INT8_FILE} must be int8, got {corpus_i8.dtype}")
    n, d = corpus_i8.shape
    if pca.shape[0] != d or scale.shape[0] != d:
        raise ValueError(
            f"PCA dim mismatch: corpus_int8 D={d}, pca_components D={pca.shape[0]}, "
            f"quant_scales D={scale.shape[0]}"
        )

    vector_fp = corpus_i8.astype("float32") * (scale / 127.0)  # (N, D) dequantized

    scalars = _read_metadata_table(artifact_dir / METADATA_FILE, n)

    table = scalars.append_column(
        EMBEDDING_I8_COLUMN, _fixed_size_list(corpus_i8, pa.int8())
    ).append_column(VECTOR_FP_COLUMN, _fixed_size_list(vector_fp, pa.float32()))

    # Physical sort by (chunk_start_unix, vehicle) so per-fragment min/max stats
    # let date/vehicle prefilters skip whole fragments at read time.
    table = table.sort_by([("chunk_start_unix", "ascending"), ("vehicle", "ascending")])

    schema_metadata = dict(table.schema.metadata or {})
    schema_metadata[META_KEY_PCA_COMPONENTS] = _encode_array(pca)
    schema_metadata[META_KEY_QUANT_SCALES] = _encode_array(scale)
    return table.replace_schema_metadata(schema_metadata)


def build_dataset(
    artifact_dir: Path,
    out_uri: str,
    *,
    target_rows_per_fragment: int = _DEFAULT_TARGET_ROWS_PER_FRAGMENT,
    max_rows_per_file: int = _DEFAULT_MAX_ROWS_PER_FILE,
) -> lance.LanceDataset:
    """Build the Lance 2.1 dataset at `out_uri` from artifacts in `artifact_dir`.

    Writes, compacts to `target_rows_per_fragment`-row fragments, and creates
    the BTREE(chunk_start_unix) / BTREE(segment_id) / BITMAP(vehicle) scalar
    indices. Returns the resulting dataset handle.
    """
    table = build_table(artifact_dir)
    lance.write_dataset(
        table,
        out_uri,
        mode="create",
        data_storage_version=DATA_STORAGE_VERSION,
        max_rows_per_file=max_rows_per_file,
    )
    ds = lance.dataset(out_uri)
    ds.optimize.compact_files(target_rows_per_fragment=target_rows_per_fragment)
    ds.create_scalar_index("chunk_start_unix", "BTREE")
    ds.create_scalar_index("segment_id", "BTREE")
    ds.create_scalar_index("vehicle", "BITMAP")
    return lance.dataset(out_uri)


def is_v21_dataset(ds: lance.LanceDataset) -> bool:
    """True if `ds` is an exact-threshold Lance 2.1 dataset (vs legacy .lance)."""
    names = set(ds.schema.names)
    return (
        ds.data_storage_version == DATA_STORAGE_VERSION
        and EMBEDDING_I8_COLUMN in names
        and VECTOR_FP_COLUMN in names
    )
