"""Offline: add ``segment_id`` / ``chunk_end_unix`` / ``dx_internal_id`` to a
sharded NLS embedding corpus, joined from an already-enriched corpus's metadata.

The offline segment-scan corpus (``black-dwarf/embeddings/cars/``) carries only
``chunk_id`` / ``run_uuid`` / ``chunk_start_unix``. It has no ``segment_id``, so a
scan cannot key its output by segment -- which is the leading explanation for a
scan reporting SUCCEEDED with 0 qualifying segments while ~9% of clips score above
tau: every match is dropped when its segment cannot be resolved.

The fields already exist. ``white-dwarf-int8/metadata.parquet`` holds 47.8M rows
of ``chunk_id, run_uuid, chunk_start_unix, source_media_uri, segment_id,
chunk_end_unix, dx_internal_id, dx_segment_id, vehicle`` -- a superset of the
black-dwarf corpus, keyed by the SAME ``chunk_id`` (``{run_uuid}#t{unix}``).
Measured: 50,000 sampled black-dwarf chunk_ids all matched (100%). So this is a
metadata join, not a rebuild -- the embeddings are a different model, but the
per-clip identity is shared, and Lance ``add_columns`` never touches the vectors.

Why a join and not the model's own build: black-dwarf and white-dwarf are
different ENCODERS over the SAME clips. ``chunk_id`` identifies the clip, so
clip-level metadata transfers; nothing model-specific is copied.

Pipeline
--------
1. Load the metadata parquet into a ``chunk_id -> (fields...)`` map (held in RAM;
   ~20GB for 47.8M rows, so this needs a large-memory host).
2. Per shard, ``add_columns`` the requested fields, reading only ``chunk_id``.
   Chunks absent from the map get nulls rather than failing the shard.

Idempotent: a shard that already has ALL requested columns is skipped unless
``--overwrite``. Shards are independent Lance datasets, so a partial run is safe
to resume and per-shard failures are reported for retry.

Usage
-----
    # coverage first, no writes
    python scripts/backfill_metadata_columns.py \
        --corpus s3://.../black-dwarf/embeddings/cars/ \
        --metadata-parquet s3://.../white-dwarf-int8/metadata.parquet \
        --dry-run

    # then write
    python scripts/backfill_metadata_columns.py \
        --corpus s3://.../black-dwarf/embeddings/cars/ \
        --metadata-parquet s3://.../white-dwarf-int8/metadata.parquet \
        --workers 16

Auth: the standard AWS_* env the app already uses (see oci_s3).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from concurrent import futures

import lance
import pyarrow as pa
import pyarrow.fs as pafs
import pyarrow.parquet as pq

import oci_s3

LOGGER = logging.getLogger(__name__)

KEY_COLUMN = "chunk_id"
DEFAULT_COLUMNS = ["segment_id", "chunk_end_unix", "dx_internal_id"]
DATASET_NAME = "video_embeddings.lance"
# Arrow type per joinable field, so a null lands as a typed null (a string column
# of Nones would otherwise infer as null-typed and break the schema merge).
FIELD_TYPES = {
    "segment_id": pa.string(),
    "dx_segment_id": pa.string(),
    "vehicle": pa.string(),
    "chunk_end_unix": pa.int64(),
    "dx_internal_id": pa.int64(),
}


def _child_prefixes(client: object, bucket: str, prefix: str) -> list[str]:
    """Immediate child prefixes of `prefix` (one delimited LIST, not a full walk)."""
    out: list[str] = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter="/"):
        for entry in page.get("CommonPrefixes", []) or []:
            out.append(entry["Prefix"])
    return sorted(out)


def _shard_datasets(corpus_uri: str) -> list[str]:
    """Every ``shard_NN/rank=NNNNN/video_embeddings.lance`` URI under `corpus_uri`.

    Two delimited LISTs rather than a recursive walk: a Lance dataset holds
    thousands of transaction files, so listing objects is millions of keys.
    """
    bucket, prefix = oci_s3.parse_s3_uri(corpus_uri.rstrip("/") + "/")
    client = oci_s3.s3_client()
    uris: list[str] = []
    for shard in _child_prefixes(client, bucket, prefix):
        for rank in _child_prefixes(client, bucket, shard):
            if not rank[len(shard) :].startswith("rank="):
                continue
            uris.append(f"s3://{bucket}/{rank.rstrip('/')}/{DATASET_NAME}")
    return sorted(uris)


def _open_parquet(uri: str) -> pq.ParquetFile:
    """ParquetFile for a local path or an s3:// URI on the OCI endpoint."""
    if not uri.startswith("s3://"):
        return pq.ParquetFile(uri)
    bucket, key = oci_s3.parse_s3_uri(uri)
    s3 = pafs.S3FileSystem(
        endpoint_override=os.environ.get("AWS_ENDPOINT_URL_S3"),
        region=os.environ.get("AWS_REGION", "us-phoenix-1"),
        access_key=os.environ.get("AWS_ACCESS_KEY_ID"),
        secret_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
    )
    return pq.ParquetFile(s3.open_input_file(f"{bucket}/{key}"))


def _load_map(uri: str, columns: list[str]) -> dict[str, tuple]:
    """`chunk_id -> tuple(columns)` from the enriched metadata parquet.

    Streamed in batches so the parquet is never fully materialized as Arrow on
    top of the dict; the dict itself is the memory cost (~20GB at 47.8M rows).
    """
    pf = _open_parquet(uri)
    mapping: dict[str, tuple] = {}
    for batch in pf.iter_batches(batch_size=500_000, columns=[KEY_COLUMN] + columns):
        keys = batch.column(KEY_COLUMN).to_pylist()
        cols = [batch.column(c).to_pylist() for c in columns]
        for i, k in enumerate(keys):
            if k is not None:
                mapping[k] = tuple(c[i] for c in cols)
        if len(mapping) % 5_000_000 < 500_000:
            LOGGER.info("  loaded %d metadata rows", len(mapping))
    LOGGER.info("metadata map: %d chunk_ids, fields %s", len(mapping), columns)
    return mapping


def _coverage(uri: str, mapping: dict, storage_options: dict) -> tuple[int, int]:
    """(rows, rows_present_in_map) for one shard. No writes."""
    ds = lance.dataset(uri, storage_options=storage_options)
    rows = hit = 0
    for batch in ds.to_batches(columns=[KEY_COLUMN]):
        vals = batch.column(KEY_COLUMN).to_pylist()
        rows += len(vals)
        hit += sum(1 for k in vals if k in mapping)
    return rows, hit


def _add_columns(
    uri: str,
    mapping: dict,
    columns: list[str],
    storage_options: dict,
    overwrite: bool,
) -> tuple[int, int]:
    """Add `columns` to one shard. Returns (rows, rows_matched_in_map)."""
    ds = lance.dataset(uri, storage_options=storage_options)
    present = [c for c in columns if c in ds.schema.names]
    if present and not overwrite:
        if len(present) == len(columns):
            LOGGER.info("%s: all columns present, skipping", uri)
            return ds.count_rows(), -1
        LOGGER.warning("%s: has %s already; rerun with --overwrite", uri, present)
        return ds.count_rows(), -1
    if present:
        ds.drop_columns(present)
        ds = lance.dataset(uri, storage_options=storage_options)

    matched = 0
    nulls = tuple(None for _ in columns)

    def _map_batch(batch: pa.RecordBatch) -> pa.RecordBatch:
        nonlocal matched
        keys = batch.column(KEY_COLUMN).to_pylist()
        rows = [mapping.get(k, nulls) for k in keys]
        matched += sum(1 for k in keys if k in mapping)
        arrays = [
            pa.array([r[i] for r in rows], type=FIELD_TYPES[c])
            for i, c in enumerate(columns)
        ]
        return pa.RecordBatch.from_arrays(arrays, names=columns)

    # read_columns keeps the scan to chunk_id: the 768-d vectors are never read
    # and never rewritten, so this is a metadata-scale write over a ~5GB shard.
    ds.add_columns(_map_batch, read_columns=[KEY_COLUMN])
    total = ds.count_rows()
    LOGGER.info("%s: %d rows, %d matched", uri, total, matched)
    return total, matched


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, help="corpus prefix (s3:// or local)")
    parser.add_argument(
        "--metadata-parquet", required=True, help="enriched metadata parquet"
    )
    parser.add_argument(
        "--columns",
        default=",".join(DEFAULT_COLUMNS),
        help=f"comma-separated fields to add (default: {','.join(DEFAULT_COLUMNS)})",
    )
    parser.add_argument("--overwrite", action="store_true", help="replace existing")
    parser.add_argument("--dry-run", action="store_true", help="report coverage only")
    parser.add_argument("--workers", type=int, default=16, help="shards in parallel")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    columns = [c.strip() for c in args.columns.split(",") if c.strip()]
    unknown = [c for c in columns if c not in FIELD_TYPES]
    if unknown:
        LOGGER.error("unknown column(s) %s; known: %s", unknown, sorted(FIELD_TYPES))
        return 2

    storage_options = oci_s3.lance_storage_options()
    uris = _shard_datasets(args.corpus)
    if not uris:
        LOGGER.error("no %s datasets under %s", DATASET_NAME, args.corpus)
        return 1
    LOGGER.info("found %d shard datasets", len(uris))

    mapping = _load_map(args.metadata_parquet, columns)

    if args.dry_run:
        total = matched = 0
        with futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            pending = [
                pool.submit(_coverage, u, mapping, storage_options) for u in uris
            ]
            for done, fut in enumerate(futures.as_completed(pending), start=1):
                rows, hit = fut.result()
                total += rows
                matched += hit
                LOGGER.info(
                    "%d/%d shards | %d/%d rows covered", done, len(uris), matched, total
                )
        pct = (100.0 * matched / total) if total else 0.0
        LOGGER.info("dry run: %d/%d rows would be filled (%.1f%%)", matched, total, pct)
        return 0

    grand_total = grand_matched = 0
    failures: list[str] = []
    with futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = {
            pool.submit(
                _add_columns, u, mapping, columns, storage_options, args.overwrite
            ): u
            for u in uris
        }
        for fut in futures.as_completed(pending):
            uri = pending[fut]
            try:
                rows, matched = fut.result()
            except Exception as exc:  # noqa: BLE001 -- report and continue
                LOGGER.error("%s FAILED: %s", uri, exc)
                failures.append(uri)
                continue
            grand_total += rows
            if matched >= 0:
                grand_matched += matched
    LOGGER.info(
        "done: %d rows, %d matched, %d shard(s) failed",
        grand_total, grand_matched, len(failures),
    )
    for uri in failures:
        LOGGER.error("retry: %s", uri)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
