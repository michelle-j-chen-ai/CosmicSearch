"""Build the exact-threshold black-dwarf corpus (full corpus, streaming).

This is ``build_test_corpora.build_threshold_corpus`` generalized from an
in-RAM ~10% slice to the full black-dwarf corpus. It reuses the same math --
``bench_common.pca_basis`` (uncentered-Gram PCA-256), ``bench_common.int8_quantize``
(per-dim symmetric int8), and ``lance_writer.build_dataset`` (the exact-threshold
Lance layout: ``embedding_i8`` FSL<int8,256> + ``vector_fp`` FSL<float32,256>,
physically sorted by ``(chunk_start_unix, vehicle)``, with BTREE indexes on
``chunk_start_unix``/``segment_id``/``run_uuid`` and a BITMAP index on ``vehicle``).

Two changes are required at full scale, because the slice builder holds every
vector in RAM (``np.stack`` of ~105 GB at 34.4M rows):

  * **Sample-fit basis.** The PCA subspace is stable (the corpus is ~rank-256),
    so the basis is fit on a bounded sample held in RAM, exactly as the slice
    builder fits on a slice.
  * **Streaming project + exact scale.** One pass over every shard projects each
    batch through the fixed basis into an on-disk memmap, tracking the true
    per-dimension max-abs so the int8 scale covers the whole corpus (no clipping,
    so the eps bound stays valid). The projection memmap is then quantized in
    blocks. Peak RAM is one batch, not the corpus.

``segment_id`` is the per-row key (= ``chunk_id``, the mini-segment id), matching
the slice builder. ``vehicle`` is not present in the black-dwarf source (neither
the content-hash shard path nor ``metadata_json`` carries it), so it is left null
here; recovering it is an Ursa ``run_uuid`` join, a follow-up.

NOTE on scale: ``lance_writer.build_dataset`` assembles and sorts the whole table
in RAM (``vector_fp`` alone is ~35 GB at 34.4M rows). Validate on a slice on a
normal box; the full corpus needs a high-memory node (or a chunked-write variant
of build_dataset).

USAGE::

    # slice validation to a local dataset
    python tools/build_black_dwarf_threshold.py \\
        --dest /tmp/bd_threshold_slice/corpus.lance \\
        --workdir /home/me/bd_build --limit-shards 3

    # full build (high-memory node)
    python tools/build_black_dwarf_threshold.py \\
        --dest s3://.../black-dwarf/threshold_table/corpus.lance \\
        --workdir /data/bd_build
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import boto3
import lance
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import bench_common
import lance_writer

BUCKET = "neuron-prod-data-intelligence-exploratory"
SRC_ROOT = "michelle/nls_search/black-dwarf/embeddings/cars/"
ENDPOINT = "https://idskhu5vqvtl.compat.objectstorage.us-phoenix-1.oraclecloud.com"
REGION = "us-phoenix-1"

MODEL_DIM = bench_common.MODEL_DIM  # 768
D = bench_common.D  # 256
BATCH_ROWS = 32_768
QUANT_BLOCK = 1_000_000


def storage_options() -> dict:
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


def list_rank_datasets(src_root: str) -> list[str]:
    c = s3_client()
    p = c.get_paginator("list_objects_v2")

    def prefixes(pfx: str) -> list[str]:
        out: list[str] = []
        for page in p.paginate(Bucket=BUCKET, Prefix=pfx, Delimiter="/"):
            out += [e["Prefix"] for e in page.get("CommonPrefixes", [])]
        return out

    shards = sorted(prefixes(src_root))
    ranks = [r for sh in shards for r in sorted(prefixes(sh))]
    return [f"s3://{BUCKET}/{r}video_embeddings.lance" for r in ranks]


def _vectors(batch: pa.RecordBatch) -> np.ndarray:
    """(n, 768) fp32 from the FixedSizeList<float>[768] ``vector`` column."""
    fsl = batch.column("vector")
    flat = fsl.values.to_numpy(zero_copy_only=False).astype("float32", copy=False)
    return flat.reshape(len(batch), MODEL_DIM)


def fit_basis(uris: list[str], so: dict, sample_rows: int) -> np.ndarray:
    """Fit the PCA-256 basis on a bounded sample (subspace is corpus-stable)."""
    buf: list[np.ndarray] = []
    got = 0
    for uri in uris:
        ds = lance.dataset(uri, storage_options=so)
        for batch in ds.scanner(columns=["vector"], batch_size=BATCH_ROWS).to_batches():
            buf.append(_vectors(batch))
            got += len(batch)
            if got >= sample_rows:
                break
        if got >= sample_rows:
            break
    sample = np.ascontiguousarray(np.concatenate(buf)[:sample_rows])
    print(f"fitting PCA-256 basis on {sample.shape[0]:,} sampled rows", flush=True)
    return bench_common.pca_basis(sample)  # (256, 768)


def project_and_collect(uris, so, basis, N, workdir):
    """Stream every row: project -> proj memmap, track scale, collect metadata."""
    proj_path = workdir / lance_writer.PRE_QUANT_FP32_FILE
    proj = np.lib.format.open_memmap(
        proj_path, mode="w+", dtype="float32", shape=(N, D)
    )
    maxabs = np.zeros(D, dtype="float32")
    chunk_ids = np.empty(N, dtype=object)
    run_uuids = np.empty(N, dtype=object)
    chunk_starts = np.empty(N, dtype="int64")
    row = 0
    for i, uri in enumerate(uris):
        t0 = time.time()
        ds = lance.dataset(uri, storage_options=so)
        for batch in ds.scanner(
            columns=["vector", "chunk_id", "run_uuid", "chunk_start_unix"],
            batch_size=BATCH_ROWS,
        ).to_batches():
            n = len(batch)
            p = _vectors(batch) @ basis.T  # (n, 256)
            proj[row:row + n] = p
            np.maximum(maxabs, np.abs(p).max(axis=0), out=maxabs)
            chunk_ids[row:row + n] = batch.column("chunk_id").to_pylist()
            run_uuids[row:row + n] = batch.column("run_uuid").to_pylist()
            chunk_starts[row:row + n] = batch.column("chunk_start_unix").to_numpy()
            row += n
        print(f"[{i + 1}/{len(uris)}] projected cum {row:,} in {time.time() - t0:.1f}s",
              flush=True)
    if row != N:
        raise ValueError(f"row mismatch: streamed {row}, expected {N}")
    proj.flush()
    scale = np.where(maxabs == 0, np.float32(1.0), maxabs).astype("float32")
    return proj, scale, chunk_ids, run_uuids, chunk_starts


def write_artifacts(workdir, basis, scale, proj, chunk_ids, run_uuids, chunk_starts):
    """Write the artifact set build_dataset consumes."""
    N = proj.shape[0]
    np.save(workdir / lance_writer.PCA_FILE, basis)
    np.save(workdir / lance_writer.SCALE_FILE, scale)
    # Quantize the projection in blocks into an int8 memmap (matches int8_quantize).
    corpus_i8 = np.lib.format.open_memmap(
        workdir / lance_writer.CORPUS_INT8_FILE, mode="w+", dtype="int8", shape=(N, D)
    )
    inv = (127.0 / scale).astype("float32")
    for s in range(0, N, QUANT_BLOCK):
        e = min(s + QUANT_BLOCK, N)
        corpus_i8[s:e] = np.clip(np.round(proj[s:e] * inv), -127, 127).astype("int8")
    corpus_i8.flush()
    metadata = pa.table({
        "run_uuid": pa.array(run_uuids, type=pa.string()),
        "chunk_start_unix": pa.array(chunk_starts, type=pa.int64()),
        "segment_id": pa.array(chunk_ids, type=pa.string()),  # = chunk_id (mini-seg id)
    })
    pq.write_table(metadata, workdir / lance_writer.METADATA_FILE)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dest", required=True, help="output .lance URI (local or s3://)")
    ap.add_argument("--workdir", required=True, help="disk dir for artifacts/memmaps")
    ap.add_argument("--src-root", default=SRC_ROOT)
    ap.add_argument("--sample-rows", type=int, default=6_000_000, help="PCA-fit sample")
    ap.add_argument("--limit-shards", type=int, default=0, help="first N shards (0=all)")
    args = ap.parse_args()

    if "sibogeng/" in args.dest:
        raise SystemExit("refusing to write under sibogeng/ (prod source namespace)")

    so = storage_options()
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    uris = list_rank_datasets(args.src_root)
    if args.limit_shards:
        uris = uris[: args.limit_shards]
    N = sum(lance.dataset(u, storage_options=so).count_rows() for u in uris)
    print(f"{len(uris)} shards, {N:,} rows total", flush=True)

    t0 = time.time()
    basis = fit_basis(uris, so, min(args.sample_rows, N))
    proj, scale, cids, ruuids, starts = project_and_collect(uris, so, basis, N, workdir)
    write_artifacts(workdir, basis, scale, proj, cids, ruuids, starts)
    print(f"artifacts written in {time.time() - t0:.1f}s; building dataset...", flush=True)

    # build_dataset writes/compacts/indexes a LOCAL dataset (no storage_options).
    # For an s3:// dest, build locally then upload the dataset directory.
    if args.dest.startswith("s3://"):
        local = str(workdir / "_dataset" / "corpus.lance")
        lance_writer.build_dataset(workdir, local)
        print(f"built local dataset {local}; uploading to {args.dest}...", flush=True)
        _upload_dir(local, args.dest)
    else:
        lance_writer.build_dataset(workdir, args.dest)
    print(f"threshold dataset built at {args.dest} in {time.time() - t0:.1f}s", flush=True)


def _upload_dir(local_dir: str, dest_uri: str) -> None:
    """Sync a local Lance dataset directory up to an s3:// dest (OCI endpoint)."""
    import subprocess

    env = {
        **os.environ,
        "AWS_ENDPOINT_URL_S3": ENDPOINT,
        "AWS_DEFAULT_REGION": REGION,
        "AWS_REQUEST_CHECKSUM_CALCULATION": "WHEN_REQUIRED",
    }
    subprocess.run(
        ["aws", "s3", "sync", local_dir.rstrip("/"), dest_uri.rstrip("/")],
        env=env, check=True,
    )


if __name__ == "__main__":
    main()
