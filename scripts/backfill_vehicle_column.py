"""Offline: add a ``vehicle`` column to a sharded NLS embedding corpus.

The offline segment-scan corpus (``NLS_SCAN_EMBEDDINGS_URI``, currently
``black-dwarf/embeddings/cars/``) carries ``run_uuid`` but NOT ``vehicle``, while
the app's resident browse corpus does. A vehicle filter therefore matches nothing
in the scan and the workload finishes SUCCEEDED with 0 qualifying segments -- an
empty result indistinguishable from a genuine one. This script closes that gap by
writing the missing column onto the corpus itself, so the scan filters vehicles
exactly like the app does.

``vehicle`` is a property of the DRIVE, not the clip, so no re-embedding is
needed: a ``run_uuid -> vehicle_name`` map is enough to fill all ~34M clip rows.
Measured on shard_00/rank=00000: 497,494 rows over 18,208 distinct runs (~27
clips per run), so the corpus-wide map is on the order of 1M runs -- large for a
filter payload, trivial as a column. Lance ``add_columns`` writes ONLY the new
column; the 768-d vectors are never read or rewritten.

Pipeline
--------
1. ``--dump-runs``: scan every shard's ``run_uuid`` column and write the distinct
   values to a parquet. This is the input list for step 2.
2. Resolve ``run_uuid -> vehicle_name`` in Trino (see ``TRINO_SQL`` below) and
   save the result as a parquet with columns ``run_uuid``, ``vehicle``.
3. Default mode: for each shard, ``add_columns`` a ``vehicle`` column mapping each
   row's ``run_uuid`` through that map. Rows whose run is unmapped get null --
   matching ``search_engine.vehicle_mask``, where a null never matches a filter.

Idempotent: a shard that already has a ``vehicle`` column is skipped unless
``--overwrite`` is passed, so a partial run is safe to resume. Shards are
independent Lance datasets, so a failure mid-way leaves earlier shards valid.

Usage
-----
    # 1. distinct runs in the corpus -> parquet
    python scripts/backfill_vehicle_column.py \
        --corpus s3://.../black-dwarf/embeddings/cars/ \
        --dump-runs ./corpus_runs.parquet

    # 2. (Trino) run TRINO_SQL against those ids -> ./vehicle_map.parquet

    # 3. write the column
    python scripts/backfill_vehicle_column.py \
        --corpus s3://.../black-dwarf/embeddings/cars/ \
        --vehicle-map ./vehicle_map.parquet [--dry-run]

Auth: the standard AWS_* env the app already uses (see oci_s3).
"""

from __future__ import annotations

import argparse
import logging
import sys
from concurrent import futures

import lance
import pyarrow as pa
import pyarrow.parquet as pq

import oci_s3

LOGGER = logging.getLogger(__name__)

VEHICLE_COLUMN = "vehicle"
RUN_COLUMN = "run_uuid"
# Each shard holds one dataset at shard_NN/rank=NNNNN/video_embeddings.lance.
DATASET_NAME = "video_embeddings.lance"

# Resolve the corpus's runs to their vehicle. ``duration_in_seconds_view.uuid`` is
# the run uuid and ``vehicle_name`` the fleet vehicle id (e.g. "mce113"), the same
# id space the app's vehicle filter uses.
TRINO_SQL = """
SELECT DISTINCT
  CAST(d.uuid AS VARCHAR) AS run_uuid,
  d.vehicle_name          AS vehicle
FROM ursa_log_management.public.duration_in_seconds_view d
WHERE d.vehicle_name IS NOT NULL
"""


def _child_prefixes(client: object, bucket: str, prefix: str) -> list[str]:
    """Immediate child prefixes of `prefix` (one delimited LIST, not a full walk)."""
    out: list[str] = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter="/"):
        for entry in page.get("CommonPrefixes", []) or []:
            out.append(entry["Prefix"])
    return sorted(out)


def _shard_datasets(corpus_uri: str, storage_options: dict) -> list[str]:
    """Every ``shard_NN/rank=NNNNN/video_embeddings.lance`` URI under `corpus_uri`.

    Walks the two prefix levels with Delimiter="/" rather than listing objects: a
    Lance dataset holds thousands of transaction files, so a full recursive list
    over a 12-shard corpus is millions of keys and minutes of paging.
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


def _shard_runs(uri: str, storage_options: dict) -> set[str]:
    """Distinct run_uuid in one shard (reads only that column)."""
    ds = lance.dataset(uri, storage_options=storage_options)
    runs: set[str] = set()
    for batch in ds.to_batches(columns=[RUN_COLUMN]):
        runs.update(batch.column(RUN_COLUMN).to_pylist())
    return runs


def _dump_runs(
    uris: list[str], storage_options: dict, out_path: str, workers: int
) -> int:
    """Write the distinct run_uuid across all shards to `out_path` (parquet).

    ~44s per shard sequentially (measured), so 94 shards is over an hour; the
    reads are independent and I/O bound, hence the pool.
    """
    runs: set[str] = set()
    with futures.ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(_shard_runs, u, storage_options): u for u in uris}
        for done, fut in enumerate(futures.as_completed(pending), start=1):
            runs.update(fut.result())
            LOGGER.info("%d/%d shards -> %d distinct runs", done, len(uris), len(runs))
    runs.discard(None)
    table = pa.table({RUN_COLUMN: sorted(runs)})
    pq.write_table(table, out_path)
    LOGGER.info("wrote %d distinct run_uuid to %s", table.num_rows, out_path)
    return table.num_rows


def _load_vehicle_map(path: str) -> dict[str, str]:
    """`run_uuid -> vehicle` from the Trino output parquet."""
    table = pq.read_table(path, columns=[RUN_COLUMN, VEHICLE_COLUMN])
    mapping = {
        r: v
        for r, v in zip(
            table.column(RUN_COLUMN).to_pylist(),
            table.column(VEHICLE_COLUMN).to_pylist(),
        )
        if r and v
    }
    LOGGER.info("vehicle map: %d runs", len(mapping))
    return mapping


def _add_vehicle(
    uri: str, mapping: dict[str, str], storage_options: dict, overwrite: bool
) -> tuple[int, int]:
    """Add `vehicle` to one shard. Returns (rows, rows_with_a_vehicle)."""
    ds = lance.dataset(uri, storage_options=storage_options)
    if VEHICLE_COLUMN in ds.schema.names and not overwrite:
        LOGGER.info("%s: vehicle column already present, skipping", uri)
        return ds.count_rows(), -1
    if VEHICLE_COLUMN in ds.schema.names:
        ds.drop_columns([VEHICLE_COLUMN])
        ds = lance.dataset(uri, storage_options=storage_options)

    filled = 0

    def _map_batch(batch: pa.RecordBatch) -> pa.RecordBatch:
        nonlocal filled
        vehicles = [mapping.get(r) for r in batch.column(RUN_COLUMN).to_pylist()]
        filled += sum(1 for v in vehicles if v is not None)
        return pa.RecordBatch.from_arrays(
            [pa.array(vehicles, type=pa.string())], names=[VEHICLE_COLUMN]
        )

    # read_columns keeps the scan to run_uuid: the vector columns are never read
    # and never rewritten, so this is a metadata-scale write over a ~5GB shard.
    ds.add_columns(_map_batch, read_columns=[RUN_COLUMN])
    total = ds.count_rows()
    LOGGER.info("%s: %d rows, %d with a vehicle", uri, total, filled)
    return total, filled


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, help="corpus prefix (s3:// or local)")
    parser.add_argument("--dump-runs", help="write distinct run_uuid here and exit")
    parser.add_argument("--vehicle-map", help="parquet with run_uuid,vehicle")
    parser.add_argument(
        "--overwrite", action="store_true", help="replace an existing vehicle column"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="report coverage without writing"
    )
    parser.add_argument(
        "--workers", type=int, default=8, help="shards to process concurrently"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    storage_options = oci_s3.lance_storage_options()
    uris = _shard_datasets(args.corpus, storage_options)
    if not uris:
        LOGGER.error("no %s datasets under %s", DATASET_NAME, args.corpus)
        return 1
    LOGGER.info("found %d shard datasets", len(uris))

    if args.dump_runs:
        _dump_runs(uris, storage_options, args.dump_runs, args.workers)
        return 0

    if not args.vehicle_map:
        LOGGER.error("--vehicle-map is required unless --dump-runs is given")
        return 2
    mapping = _load_vehicle_map(args.vehicle_map)

    if args.dry_run:
        # Coverage check only: how many rows WOULD get a vehicle. Parallel for the
        # same reason the write is -- 94 shards at ~44s each is over an hour serial.
        def _coverage(uri: str) -> tuple[int, int]:
            ds = lance.dataset(uri, storage_options=storage_options)
            rows = hit = 0
            for batch in ds.to_batches(columns=[RUN_COLUMN]):
                vals = batch.column(RUN_COLUMN).to_pylist()
                rows += len(vals)
                hit += sum(1 for r in vals if mapping.get(r))
            return rows, hit

        total = matched = 0
        with futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            for done, fut in enumerate(
                futures.as_completed([pool.submit(_coverage, u) for u in uris]), 1
            ):
                rows, hit = fut.result()
                total += rows
                matched += hit
                LOGGER.info("%d/%d shards | %d/%d rows covered", done, len(uris), matched, total)
        pct = (100.0 * matched / total) if total else 0.0
        LOGGER.info("dry run: %d/%d rows would get a vehicle (%.1f%%)", matched, total, pct)
        return 0

    # The shards are independent Lance datasets and the work is object-store I/O
    # bound, so run them concurrently; one failing shard leaves the rest valid.
    grand_total = grand_filled = 0
    failures: list[str] = []
    with futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = {
            pool.submit(
                _add_vehicle, uri, mapping, storage_options, args.overwrite
            ): uri
            for uri in uris
        }
        for fut in futures.as_completed(pending):
            uri = pending[fut]
            try:
                rows, filled = fut.result()
            except Exception as exc:  # noqa: BLE001 -- report and continue
                LOGGER.error("%s FAILED: %s", uri, exc)
                failures.append(uri)
                continue
            grand_total += rows
            if filled >= 0:
                grand_filled += filled
    LOGGER.info(
        "done: %d rows, %d with a vehicle, %d shard(s) failed",
        grand_total, grand_filled, len(failures),
    )
    for uri in failures:
        LOGGER.error("retry: %s", uri)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
