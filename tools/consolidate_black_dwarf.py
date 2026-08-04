"""Consolidate the rank-sharded black-dwarf embedding corpus into ONE Lance table.

The production black-dwarf corpus is 94 independent Lance datasets
(``shard_NN/rank=NNNNN/video_embeddings.lance``). Independent datasets cannot be
upserted as a unit, so this rewrites every row into a single Lance dataset whose
logical key is ``chunk_id`` -- the table an ingestion pipeline can ``merge_insert``
new embeddings into while the app serves it.

This is a pure passthrough: the 19 source columns and the fp32 ``vector[768]`` are
copied unchanged. The exact-threshold layout (``vehicle``/``segment_id``
enrichment, PCA-256 + int8, physical sort, scalar indexes) is layered on top of
this table afterwards, not here.

Mechanics:
  * Each source shard is read as a streaming ``RecordBatchReader`` and handed to
    ``write_dataset`` -- peak memory is one batch, never a whole shard.
  * The first write to an empty destination is ``mode="create"``; every later
    write appends. Each append is one dataset version.
  * ``data_storage_version=2.2`` (random ``take()`` is ~1 IOP/row) and
    ``enable_stable_row_ids=True`` (row ids survive append/compaction, which the
    upsert-while-serving position-safety relies on).
  * Resumable: a durable ``_ingest_markers/<shard>.done`` object is written per
    completed shard, so a re-run skips finished shards. Safe after a transient
    failure or an expired credential.
  * On completion it verifies ``rows == distinct(chunk_id)`` -- the merge key must
    be unique for upserts to be well defined.

USAGE::

    python tools/consolidate_black_dwarf.py \\
        --dest s3://bucket/michelle/nls_search/black-dwarf/table/video_embeddings.lance

    # resume the same command after an interruption; finished shards are skipped

Credentials come from the standard ``AWS_*`` environment (for OCI object storage,
export them and set the region/endpoint below via the same env the app uses).
"""

from __future__ import annotations

import argparse
import os
import time

import boto3
import lance
import pyarrow as pa
import pyarrow.compute as pc

BUCKET = "neuron-prod-data-intelligence-exploratory"
SRC_ROOT = "michelle/nls_search/black-dwarf/embeddings/cars/"
ENDPOINT = "https://idskhu5vqvtl.compat.objectstorage.us-phoenix-1.oraclecloud.com"
REGION = "us-phoenix-1"

MERGE_KEY = "chunk_id"
BATCH_ROWS = 32_768
MAX_ROWS_PER_FILE = 1_000_000
STORAGE_VERSION = "2.2"


def storage_options() -> dict:
    """Lance object-store options from the standard AWS_* environment."""
    so = {
        "access_key_id": os.environ["AWS_ACCESS_KEY_ID"],
        "secret_access_key": os.environ["AWS_SECRET_ACCESS_KEY"],
        "aws_region": REGION,
        "aws_endpoint": ENDPOINT,
        "virtual_hosted_style_request": "false",
    }
    if os.environ.get("AWS_SESSION_TOKEN"):
        so["aws_session_token"] = os.environ["AWS_SESSION_TOKEN"]
    return so


def s3_client():
    return boto3.Session().client("s3", region_name=REGION, endpoint_url=ENDPOINT)


def dest_bucket_key(dest_uri: str) -> tuple[str, str]:
    """Split an ``s3://bucket/key`` URI into ``(bucket, key)`` (no trailing slash)."""
    if not dest_uri.startswith("s3://"):
        raise ValueError(f"expected s3:// URI, got {dest_uri!r}")
    b, _, k = dest_uri[len("s3://"):].partition("/")
    return b, k.rstrip("/")


def marker_key(dest_key: str, src_uri: str, src_root: str) -> str:
    """Durable per-shard completion marker key under the destination prefix.

    ``.../cars/shard_00/rank=00000/video_embeddings.lance`` (src_root ``.../cars/``)
    -> ``<dest_key>/_ingest_markers/shard_00_rank=00000.done``.
    """
    _, src_key = dest_bucket_key(src_uri)
    root = src_root.strip("/") + "/"
    rel = src_key[len(root):] if src_key.startswith(root) else src_key
    tag = rel.replace("/video_embeddings.lance", "").replace("/", "_")
    return f"{dest_key}/_ingest_markers/{tag}.done"


def list_rank_datasets(client, bucket: str, src_root: str) -> list[str]:
    """Every ``shard_NN/rank=NNNNN/video_embeddings.lance`` URI, sorted."""
    p = client.get_paginator("list_objects_v2")

    def prefixes(pfx: str) -> list[str]:
        out: list[str] = []
        for page in p.paginate(Bucket=bucket, Prefix=pfx, Delimiter="/"):
            out += [e["Prefix"] for e in page.get("CommonPrefixes", [])]
        return out

    shards = sorted(prefixes(src_root))
    ranks = [r for sh in shards for r in sorted(prefixes(sh))]
    return [f"s3://{bucket}/{r}video_embeddings.lance" for r in ranks]


def convert(src_uris: list[str], dest_uri: str, src_root: str, so: dict) -> int:
    """Stream each source dataset into ``dest_uri``; return total rows written."""
    client = s3_client()
    dest_b, dest_k = dest_bucket_key(dest_uri)

    def done(uri: str) -> bool:
        r = client.list_objects_v2(
            Bucket=dest_b, Prefix=marker_key(dest_k, uri, src_root), MaxKeys=1
        )
        return bool(r.get("KeyCount"))

    dest_exists = bool(
        client.list_objects_v2(Bucket=dest_b, Prefix=dest_k + "/", MaxKeys=1).get(
            "KeyCount"
        )
    )
    total = 0
    for i, uri in enumerate(src_uris):
        tag = uri.rsplit("/", 3)
        label = "/".join(tag[-3:-1])
        if done(uri):
            total += lance.dataset(uri, storage_options=so).count_rows()
            print(f"[{i + 1}/{len(src_uris)}] skip   (already ingested) {label}", flush=True)
            continue
        t0 = time.time()
        src = lance.dataset(uri, storage_options=so)
        reader = src.scanner(batch_size=BATCH_ROWS, scan_in_order=True).to_reader()
        mode = "append" if dest_exists else "create"
        lance.write_dataset(
            reader,
            dest_uri,
            mode=mode,
            storage_options=so,
            data_storage_version=STORAGE_VERSION,
            enable_stable_row_ids=True,
            max_rows_per_file=MAX_ROWS_PER_FILE,
        )
        dest_exists = True
        client.put_object(Bucket=dest_b, Key=marker_key(dest_k, uri, src_root), Body=b"")
        n = src.count_rows()
        total += n
        print(
            f"[{i + 1}/{len(src_uris)}] {mode:6s} +{n:>9,} rows "
            f"(cum {total:>11,}) in {time.time() - t0:5.1f}s  {label}",
            flush=True,
        )
    return total


def verify(dest_uri: str, so: dict) -> None:
    """Assert the merge key is unique: rows == distinct(chunk_id)."""
    ds = lance.dataset(dest_uri, storage_options=so)
    n = ds.count_rows()
    ck = ds.scanner(columns=[MERGE_KEY]).to_table().column(MERGE_KEY)
    distinct = pc.count_distinct(ck).as_py()
    status = "OK (unique key)" if distinct == n else "WARNING: duplicate keys"
    print(f"verify: rows={n:,}  distinct {MERGE_KEY}={distinct:,}  {status}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dest", required=True, help="destination .lance URI")
    ap.add_argument("--src-root", default=SRC_ROOT, help="source prefix under the bucket")
    ap.add_argument("--limit", type=int, default=0, help="only first N shards (0=all)")
    args = ap.parse_args()

    if "sibogeng/" in args.dest:
        raise SystemExit("refusing to write under sibogeng/ (prod source namespace)")

    so = storage_options()
    uris = list_rank_datasets(s3_client(), BUCKET, args.src_root)
    print(f"found {len(uris)} source rank datasets", flush=True)
    if args.limit:
        uris = uris[: args.limit]
        print(f"limiting to first {len(uris)} shards", flush=True)

    t0 = time.time()
    total = convert(uris, args.dest, args.src_root, so)
    print(f"\nwrote {total:,} rows to {args.dest} in {time.time() - t0:.1f}s", flush=True)
    verify(args.dest, so)


if __name__ == "__main__":
    main()
