"""Build two testing corpora from a ~10% slice of the production corpus.

Derives a *master copy* (legacy Lance shards re-written with a ``vehicle``
column + sequential rank renumbering) and a *threshold corpus* (the exact-
threshold int8 PCA Lance dataset the threshold search path scores against)
from a selectable subset of the production embedding shards.

USAGE (dry-run, for human review before any real build)::

    python tools/build_test_corpora.py \\
        --source-prefix s3://bucket/prod/embeddings/ \\
        --dest-prefix   s3://bucket/test-slice/ \\
        --dry-run

Full local build (no S3 writes)::

    python tools/build_test_corpora.py \\
        --source-prefix s3://bucket/prod/embeddings/ \\
        --local-dest /tmp/test_corpora

Safety rails:
  - The source prefix is opened **read-only** (list + read, never write).
  - The builder hard-refuses ``--dest-prefix`` under ``sibogeng/`` (the prod
    namespace) — enforced BEFORE any S3 client is constructed.
  - ``--local-dest`` writes both corpora to a local dir instead of S3.

S3 access is isolated in ``_list_shard_uris`` and ``_open_shard`` so everything
else is testable locally without an S3 client.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import namedtuple
from pathlib import Path
from urllib.parse import urlparse

import lance
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import lance_writer
from bench_common import MODEL_DIM, D, int8_quantize, pca_basis

# ---------------------------------------------------------------------------
# Shard representation
# ---------------------------------------------------------------------------
ShardInfo = namedtuple("ShardInfo", ["prefix_name", "rank", "uri", "dataset"])


# ---------------------------------------------------------------------------
# Vehicle derivation (pure function, unit-tested standalone)
# ---------------------------------------------------------------------------
def derive_vehicle(prefix_name: str, metadata_json: str | None) -> str | None:
    """Derive the vehicle id for a shard row from its batch prefix + metadata.

    - ``mce113*`` prefixes -> ``"mce113"`` (the curated batch).
    - Weekly prefixes (``weekN_...``) -> parse ``metadata_json`` (a JSON string)
      for a ``vehicle`` key; ``None`` when the key is absent or the JSON is
      empty/invalid.
    """
    if prefix_name.startswith("mce113"):
        return "mce113"
    if metadata_json:
        try:
            obj = json.loads(metadata_json)
        except (json.JSONDecodeError, TypeError):
            return None
        if isinstance(obj, dict):
            v = obj.get("vehicle")
            return str(v) if v is not None else None
    return None


# ---------------------------------------------------------------------------
# Count + select (operate on ShardInfo lists, no S3)
# ---------------------------------------------------------------------------
def count_prod_rows(shards: list[ShardInfo]) -> int:
    """Sum ``count_rows()`` across all shards (metadata-only reads)."""
    return sum(s.dataset.count_rows() for s in shards)


def _week_number(prefix_name: str) -> int | None:
    """Extract the week ordinal from a ``weekN_...`` prefix; None for non-weekly."""
    m = re.match(r"^week(\d+)_", prefix_name)
    return int(m.group(1)) if m else None


def select_shards(shards: list[ShardInfo], target_rows: int) -> list[ShardInfo]:
    """Select shards to reach ``target_rows``, whole-shard granularity.

    Selection order:
      1. All ``mce113/rank=*`` shards first.
      2. Then weekly shards in week order (week1 first = most recent first,
         parsed from the ``weekN_`` prefix, ascending N).
      3. ``mce113_missing37`` is NOT selected (dedup ambiguity); logged as skipped.

    Whole shards are accumulated until the target is reached. The last shard
    that crosses the target IS included (we take whole shards, not partial).
    """
    mce113_shards = []
    weekly_shards = []
    skipped = []
    for s in shards:
        if s.prefix_name.startswith("mce113_missing"):
            skipped.append(s)
            continue
        if s.prefix_name.startswith("mce113"):
            mce113_shards.append(s)
        elif _week_number(s.prefix_name) is not None:
            weekly_shards.append(s)
        else:
            # Unknown prefix type — treat like weekly (append after weekly sort)
            weekly_shards.append(s)

    weekly_shards.sort(key=lambda s: (_week_number(s.prefix_name) or 9999, s.prefix_name))

    selected = list(mce113_shards)
    accumulated = count_prod_rows(selected)
    for s in weekly_shards:
        if accumulated >= target_rows:
            break
        selected.append(s)
        accumulated += s.dataset.count_rows()

    for s in skipped:
        print(f"  skipped (dedup ambiguity): {s.prefix_name}/rank={s.rank:05d}")

    return selected


# ---------------------------------------------------------------------------
# Master copy (write-as-new Lance with vehicle column)
# ---------------------------------------------------------------------------
def build_master_copy(shards: list[ShardInfo], dest_dir: Path) -> Path:
    """Write the master copy: each shard re-written as a NEW Lance dataset with
    a ``vehicle`` column appended and sequential rank renumbering.

    Output layout::

        <dest_dir>/master_prod_slice/rank=NNNNN/video_embeddings.lance/

    Returns the master_prod_slice directory path.
    """
    out_root = Path(dest_dir) / "master_prod_slice"
    out_root.mkdir(parents=True, exist_ok=True)

    for new_rank, shard in enumerate(shards):
        rank_dir = out_root / f"rank={new_rank:05d}"
        rank_dir.mkdir(parents=True, exist_ok=True)
        out_uri = str(rank_dir / "video_embeddings.lance")

        table = shard.dataset.to_table()
        n = table.num_rows

        # Derive vehicle for each row from the shard's prefix + per-row metadata.
        vehicles: list[str | None] = []
        has_meta_json = "metadata_json" in table.column_names
        if has_meta_json:
            meta_col = table.column("metadata_json").to_pylist()
        else:
            meta_col = [None] * n
        for i in range(n):
            vehicles.append(derive_vehicle(shard.prefix_name, meta_col[i]))

        table = table.append_column(
            "vehicle",
            pa.array(vehicles, type=pa.string()),
        )
        lance.write_dataset(table, out_uri, mode="create")

    return out_root


# ---------------------------------------------------------------------------
# Threshold corpus (exact-threshold int8 PCA Lance dataset)
# ---------------------------------------------------------------------------
def build_threshold_corpus(
    shards: list[ShardInfo], dest_dir: Path, *, fraction: float = 0.10
) -> Path:
    """Build the exact-threshold corpus from the selected shards.

    Streams the selected tables, concatenates ``vector`` (768 fp32) + metadata,
    dedups on ``chunk_id`` (keeping first occurrence), fits PCA-256 via the
    uncentered Gram method, projects + int8-quantizes, and writes the artifact
    set + Lance dataset via ``lance_writer.build_dataset``.

    Output: ``<dest_dir>/threshold_prod_slice/corpus.lance``

    Returns the threshold_prod_slice directory path.
    """
    dest_dir = Path(dest_dir)
    out_dir = dest_dir / "threshold_prod_slice"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Concatenate vectors + metadata across shards, dedup on chunk_id.
    vectors_list: list[np.ndarray] = []
    chunk_ids: list[str] = []
    run_uuids: list[str] = []
    chunk_starts: list[int] = []
    vehicles: list[str | None] = []
    seen: set[str] = set()

    for shard in shards:
        table = shard.dataset.to_table()
        n = table.num_rows
        if n == 0:
            continue

        # Extract vector column (768 fp32 FSL).
        from interval_core import _vector_column_name, _vectors_from_arrow

        vec_col = _vector_column_name(table)
        vecs = _vectors_from_arrow(table, vec_col)

        cid_col = table.column("chunk_id").to_pylist()
        ru_col = table.column("run_uuid").to_pylist()
        cs_col = table.column("chunk_start_unix").to_pylist()
        has_meta_json = "metadata_json" in table.column_names
        if has_meta_json:
            meta_col = table.column("metadata_json").to_pylist()
        else:
            meta_col = [None] * n

        for i in range(n):
            cid = cid_col[i]
            if cid in seen:
                continue
            seen.add(cid)
            vectors_list.append(vecs[i])
            chunk_ids.append(cid)
            run_uuids.append(ru_col[i])
            chunk_starts.append(int(cs_col[i]))
            vehicles.append(derive_vehicle(shard.prefix_name, meta_col[i]))

    if not vectors_list:
        raise ValueError("no rows selected after dedup — empty corpus")

    embeddings = np.ascontiguousarray(np.stack(vectors_list).astype("float32"))
    n_rows = embeddings.shape[0]

    # 2. Fit PCA-256 via the uncentered Gram method.
    pca = pca_basis(embeddings)

    # 3. Project + int8 quantize.
    projected = (embeddings @ pca.T).astype("float32")
    corpus_i8, scale = int8_quantize(projected)

    # 4. Write artifacts.
    artifact_dir = out_dir / "artifact"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    np.save(artifact_dir / lance_writer.PCA_FILE, pca)
    np.save(artifact_dir / lance_writer.SCALE_FILE, scale)
    np.save(artifact_dir / lance_writer.CORPUS_INT8_FILE, corpus_i8)
    np.save(artifact_dir / lance_writer.PRE_QUANT_FP32_FILE, projected)

    metadata = pa.table(
        {
            "run_uuid": pa.array(run_uuids, type=pa.string()),
            "chunk_start_unix": pa.array(chunk_starts, type=pa.int64()),
            "segment_id": pa.array(chunk_ids, type=pa.string()),
            "vehicle": pa.array(vehicles, type=pa.string()),
        }
    )
    pq.write_table(metadata, artifact_dir / lance_writer.METADATA_FILE)

    # 5. Build the Lance dataset.
    corpus_uri = str(out_dir / "corpus.lance")
    lance_writer.build_dataset(artifact_dir, corpus_uri)

    return out_dir


# ---------------------------------------------------------------------------
# S3 access (isolated — everything else is local-testable)
# ---------------------------------------------------------------------------
def _list_shard_uris(source_prefix: str) -> list[str]:
    """List ``<source-prefix>/<batch>/rank=NNNNN/video_embeddings.lance/`` URIs.

    Isolated so all other logic is testable locally. Uses the OCI S3 client.
    """
    import oci_s3

    client = oci_s3.s3_client()
    bucket, key_prefix = oci_s3.parse_s3_uri(source_prefix.rstrip("/") + "/")
    paginator = client.get_paginator("list_objects_v2")

    shard_uris: list[str] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=key_prefix, Delimiter="/"):
        for entry in page.get("CommonPrefixes", []) or []:
            batch_prefix = entry.get("Prefix", "")
            # List rank= dirs under each batch prefix
            for rank_page in paginator.paginate(
                Bucket=bucket, Prefix=batch_prefix, Delimiter="/"
            ):
                for rank_entry in rank_page.get("CommonPrefixes", []) or []:
                    rank_prefix = rank_entry.get("Prefix", "")
                    tail = rank_prefix[len(batch_prefix) :].rstrip("/")
                    if tail.startswith("rank="):
                        shard_uris.append(f"s3://{bucket}/{rank_prefix}video_embeddings.lance")
    return sorted(shard_uris)


def _open_shard(uri: str) -> lance.LanceDataset:
    """Open a Lance dataset at ``uri`` (S3 or local). Isolated for testability."""
    if uri.startswith("s3://"):
        import oci_s3

        opts = oci_s3.lance_storage_options()
        return lance.dataset(uri, storage_options=opts)
    return lance.dataset(uri)


def _parse_shard_info(uri: str) -> ShardInfo:
    """Parse a shard URI into a ShardInfo (prefix_name, rank, uri, dataset).

    Expected URI shape:
        .../<batch_prefix>/rank=NNNNN/video_embeddings.lance
    or a local path like:
        <tmp>/<batch_prefix>/rank=NNNNN/video_embeddings.lance
    """
    # Extract the batch prefix (the dir two levels above video_embeddings.lance).
    parts = uri.replace("s3://", "").replace("\\", "/").rstrip("/").split("/")
    # parts[-1] = "video_embeddings.lance", parts[-2] = "rank=NNNNN", parts[-3] = batch
    if len(parts) < 3:
        raise ValueError(f"unexpected shard URI format: {uri}")
    rank_str = parts[-2]
    prefix_name = parts[-3]
    m = re.match(r"^rank=(\d+)$", rank_str)
    if not m:
        raise ValueError(f"cannot parse rank from {rank_str!r} in {uri}")
    rank = int(m.group(1))
    dataset = _open_shard(uri)
    return ShardInfo(prefix_name=prefix_name, rank=rank, uri=uri, dataset=dataset)


# ---------------------------------------------------------------------------
# Safety rail: refuse dest under sibogeng/
# ---------------------------------------------------------------------------
def _assert_dest_safe(dest_prefix: str) -> None:
    """Hard-refuse a dest prefix under ``sibogeng/`` BEFORE any S3 client is built.

    Checks both the bucket name and the key path SEGMENTS for ``sibogeng``, so
    neither ``s3://sibogeng/...`` nor ``s3://bucket/sibogeng/...`` nor the
    no-trailing-slash form ``s3://bucket/sibogeng`` is allowed (the builder
    appends ``/master_prod_slice/...``, so a bare segment still lands under
    the prod namespace).
    """
    parsed = urlparse(dest_prefix)
    segments = parsed.path.strip("/").split("/")
    if "sibogeng" in parsed.netloc or "sibogeng" in segments:
        raise ValueError(
            f"refusing to write dest under sibogeng/ (prod namespace): "
            f"{dest_prefix!r}"
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def run_build(
    source_prefix: str,
    dest_prefix: str,
    *,
    fraction: float = 0.10,
    dry_run: bool = False,
    local_dest: str | None = None,
) -> None:
    """Top-level build entry point.

    ``dest_prefix`` is checked against the sibogeng safety rail BEFORE any S3
    client is constructed. ``local_dest`` overrides the S3 dest for local
    end-to-end testing.
    """
    _assert_dest_safe(dest_prefix)

    print(f"source: {source_prefix}")
    print(f"dest:   {local_dest or dest_prefix}")
    print(f"fraction: {fraction:.1%}")

    # 1. Enumerate + count
    shard_uris = _list_shard_uris(source_prefix)
    if not shard_uris:
        print("no shards found under source prefix")
        return
    shards = [_parse_shard_info(u) for u in shard_uris]

    total = count_prod_rows(shards)
    target = int(total * fraction)
    print(f"\ntotal rows: {total:,}")
    print(f"target rows ({fraction:.0%}): {target:,}")

    # 2. Select
    selected = select_shards(shards, target)
    print(f"\nselected {len(selected)} shard(s):")
    for s in selected:
        print(f"  {s.prefix_name}/rank={s.rank:05d} -> {s.dataset.count_rows():,} rows")

    if dry_run:
        print("\n--dry-run: stopping before any writes.")
        return

    # 3. Build
    dest = Path(local_dest) if local_dest else None
    if dest is None:
        # S3 dest — not supported in offline tests; the real path writes to S3.
        raise NotImplementedError(
            "S3 dest writes use lance.write_dataset with storage_options; "
            "use --local-dest for offline testing."
        )

    print("\nbuilding master copy...")
    build_master_copy(selected, dest)
    print(f"  -> {dest / 'master_prod_slice'}")

    print("\nbuilding threshold corpus...")
    build_threshold_corpus(selected, dest, fraction=fraction)
    print(f"  -> {dest / 'threshold_prod_slice' / 'corpus.lance'}")

    print("\ndone.")


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--source-prefix", required=True, help="S3 prefix of prod embeddings")
    p.add_argument("--dest-prefix", required=True, help="S3 prefix for output corpora")
    p.add_argument("--fraction", type=float, default=0.10, help="target fraction of prod rows")
    p.add_argument("--dry-run", action="store_true", help="count + select only, no writes")
    p.add_argument("--local-dest", default=None, help="local dir for output (no S3 writes)")
    args = p.parse_args()

    run_build(
        source_prefix=args.source_prefix,
        dest_prefix=args.dest_prefix,
        fraction=args.fraction,
        dry_run=args.dry_run,
        local_dest=args.local_dest,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
