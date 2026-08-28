"""Convert a fast-format npy corpus into a consolidated Lance corpus.

The consolidated corpus (``vlm/corpus/video_embeddings.lance``) is what
`full_corpus` browses and what the Flyte embedding pipeline upserts into. Its
schema is defined in core-stack (`vlm/schemas/consolidated_corpus_schema.py`),
so this script does not restate it: it copies the schema, the schema metadata
and the per-model field metadata from a REFERENCE table the pipeline itself
produced. A run therefore cannot drift from what `corpus_store.upsert_corpus`
would write, and the result stays a valid merge target for later incremental
runs (`merge_insert` on ``chunk_id``).

Vectors come from the fast corpus's ``embeddings.npy``; the PCA-256 basis and
int8 scales are read from the reference table's vector-column metadata, so the
serving columns land in the same space the app's thresholds were calibrated in.

USAGE::

    PYTHONPATH=. python tools/build_consolidated_corpus_from_npy.py \\
        --source-dir /path/to/fast_corpus \\
        --reference-uri s3://bucket/prefix/reference.lance \\
        --out-dir /path/to/build

Writes locally only -- build, inspect, then upload.
"""

from __future__ import annotations

import argparse
import base64
import io
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import lance
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from local_cache import NPY_MATRIX_FILE

# Rows per written batch. Bounds transient memory: one batch holds a
# (rows, 768) fp32 slice plus its (rows, 256) projection.
_BATCH_ROWS = 200_000
# The consolidated corpus is written at Lance storage version 2.0 (matching the
# production tables); do not let this resolve to the running pylance default.
_DATA_STORAGE_VERSION = "2.0"
_MAX_ROWS_PER_FILE = 1_048_576

_PCA_KEY = b"nls.pca_components"
_SCALE_KEY = b"nls.quant_scales"
_DASHED_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_TRAILING_NS = re.compile(r"(-\d{15,})+$")


def vehicle_from_segment_id(segment_id: str | None) -> str | None:
    """Derive the vehicle from neuron and frontier segment-id conventions."""
    if not segment_id:
        return None
    core = _TRAILING_NS.sub("", segment_id)
    parts = core.split("_")
    if _DASHED_DATE.match(parts[0]):
        return parts[-1]
    return parts[0]


def dt_from_unix(chunk_start_unix: np.ndarray) -> list[str]:
    """The UTC ``yyyy-MM-dd`` partition each chunk belongs to."""
    return [
        datetime.fromtimestamp(int(ts), tz=timezone.utc).strftime("%Y-%m-%d")
        for ts in chunk_start_unix
    ]


def decode_basis(field: pa.Field) -> tuple[np.ndarray, np.ndarray]:
    """Read ``(pca, scale)`` from a model vector column's field metadata."""
    md = field.metadata or {}
    missing = [k.decode() for k in (_PCA_KEY, _SCALE_KEY) if k not in md]
    if missing:
        raise ValueError(
            f"reference column {field.name!r} is missing {missing}; it carries no "
            "serving basis, so the converted rows could not be scored against it"
        )
    load = lambda b: np.load(io.BytesIO(base64.b64decode(b)))  # noqa: E731
    return load(md[_PCA_KEY]).astype("float32"), load(md[_SCALE_KEY]).astype("float32")


def model_columns(schema: pa.Schema) -> tuple[str, str, str]:
    """The ``(vector, embedding_i8, vector_fp)`` column names for the one model."""
    vectors = [
        f.name
        for f in schema
        if f.name.startswith("vector_") and not f.name.startswith("vector_fp_")
    ]
    if len(vectors) != 1:
        raise ValueError(
            f"expected exactly one model vector column, found {vectors}; this "
            "converter writes a single model's columns"
        )
    model = vectors[0][len("vector_") :]
    return vectors[0], f"embedding_i8_{model}", f"vector_fp_{model}"


def scalar_columns(
    meta: pa.Table, n_rows: int, created_at: float
) -> dict[str, pa.Array]:
    """Build every non-vector column, deriving the ones the source lacks."""
    have = set(meta.column_names)

    def col(name: str, typ: pa.DataType) -> pa.Array:
        if name in have:
            return meta.column(name).combine_chunks().cast(typ)
        return pa.nulls(n_rows, type=typ)

    starts = meta.column("chunk_start_unix").combine_chunks().to_numpy()
    segment_ids = meta.column("segment_id").to_pylist() if "segment_id" in have else []
    return {
        "chunk_id": col("chunk_id", pa.string()),
        "run_uuid": col("run_uuid", pa.string()),
        "chunk_start_unix": col("chunk_start_unix", pa.int64()),
        "chunk_end_unix": col("chunk_end_unix", pa.int64()),
        "dt": pa.array(dt_from_unix(starts), type=pa.string()),
        "source_media_uri": col("source_media_uri", pa.string()),
        "source_media_sha256": col("source_media_sha256", pa.string()),
        "segment_id": col("segment_id", pa.string()),
        "dx_internal_id": col("dx_internal_id", pa.int64()),
        "dx_segment_id": col("dx_segment_id", pa.string()),
        "vehicle": pa.array(
            [vehicle_from_segment_id(s) for s in segment_ids] or [None] * n_rows,
            type=pa.string(),
        ),
        "created_at_unix_s": pa.array([created_at] * n_rows, type=pa.float64()),
        "metadata_json": col("metadata_json", pa.string()),
    }


def _fsl(values: np.ndarray, item_type: pa.DataType) -> pa.Array:
    flat = pa.array(values.reshape(-1), type=item_type)
    return pa.FixedSizeListArray.from_arrays(flat, values.shape[1])


def batches(
    embeddings: np.ndarray,
    scalars: dict[str, pa.Array],
    schema: pa.Schema,
    pca: np.ndarray,
    scale: np.ndarray,
    cols: tuple[str, str, str],
):
    """Yield RecordBatches matching ``schema``, projecting vectors per batch."""
    vec_col, i8_col, fp_col = cols
    n = embeddings.shape[0]
    for start in range(0, n, _BATCH_ROWS):
        stop = min(start + _BATCH_ROWS, n)
        raw = np.asarray(embeddings[start:stop], dtype=np.float32)
        projected = (raw @ pca.T).astype("float32")
        i8 = np.clip(np.round(projected * 127.0 / scale), -127, 127).astype(np.int8)
        data = {name: arr[start:stop] for name, arr in scalars.items()}
        data[vec_col] = _fsl(raw, pa.float32())
        data[i8_col] = _fsl(i8, pa.int8())
        data[fp_col] = _fsl(projected, pa.float32())
        yield pa.RecordBatch.from_arrays(
            [data[f.name] for f in schema], schema=schema
        )
        print(f"  {stop:,}/{n:,} rows", flush=True)


def build(source_dir: Path, reference_uri: str, out_dir: Path, storage_options: dict):
    reference = lance.dataset(reference_uri, storage_options=storage_options)
    schema = reference.schema
    cols = model_columns(schema)
    pca, scale = decode_basis(schema.field(cols[0]))
    print(f"reference: {reference_uri}")
    print(f"  model columns: {cols}")
    print(f"  basis: pca {pca.shape}, scale {scale.shape}")

    embeddings = np.load(source_dir / NPY_MATRIX_FILE, mmap_mode="r")
    meta = pq.read_table(source_dir / "metadata.parquet")
    if meta.num_rows != embeddings.shape[0]:
        raise ValueError(
            f"metadata rows ({meta.num_rows}) != embedding rows "
            f"({embeddings.shape[0]})"
        )
    n = embeddings.shape[0]
    print(f"source: {n:,} rows x {embeddings.shape[1]} dims")

    scalars = scalar_columns(meta, n, time.time())
    out_uri = str(out_dir / "video_embeddings.lance")
    out_dir.mkdir(parents=True, exist_ok=True)
    reader = pa.RecordBatchReader.from_batches(
        schema, batches(embeddings, scalars, schema, pca, scale, cols)
    )
    lance.write_dataset(
        reader,
        out_uri,
        mode="create",
        data_storage_version=_DATA_STORAGE_VERSION,
        max_rows_per_file=_MAX_ROWS_PER_FILE,
    )
    return out_uri


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-dir", required=True, type=Path)
    p.add_argument("--reference-uri", required=True)
    p.add_argument("--out-dir", required=True, type=Path)
    args = p.parse_args(argv)

    import configparser, os  # noqa: PLC0415

    c = configparser.ConfigParser()
    c.read(os.path.expanduser("~/.aws/credentials"))
    so = {
        "aws_access_key_id": c["oci"]["aws_access_key_id"],
        "aws_secret_access_key": c["oci"]["aws_secret_access_key"],
        "aws_endpoint": "https://idskhu5vqvtl.compat.objectstorage."
        "us-phoenix-1.oraclecloud.com",
        "aws_region": "us-phoenix-1",
        "virtual_hosted_style_request": "false",
    }
    uri = build(args.source_dir, args.reference_uri, args.out_dir, so)
    print(f"done: {uri}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
