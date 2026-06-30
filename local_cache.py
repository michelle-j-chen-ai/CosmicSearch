"""Disk cache for downloaded Lance corpora and model snapshots.

A user supplies a Lance embeddings URI at query time; we download its shards to
a local cache directory keyed by the URI, guarded by a file lock so concurrent
requests (and other Cloud Run instances sharing a GCS-fuse mount) download it at
most once. A ".complete" marker makes the cache hit cheap and crash-safe: a
partial download leaves no marker and is re-fetched.

Cache root resolution:
- NLS_CACHE_ROOT env wins if set.
- else the GCS-fuse mount when present (shared across instances, persistent --
  download paid once ever). Apps Platform V2 mounts it at /mnt/data; the older
  convention is /gcs.
- else /tmp/nls_cache (local dev; ephemeral).
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import logging
import os
import re
from collections.abc import Iterator
from pathlib import Path

import oci_s3

LOGGER = logging.getLogger(__name__)

_COMPLETE_MARKER = ".nls_download_complete"


def cache_root() -> Path:
    override = os.environ.get("NLS_CACHE_ROOT", "").strip()
    if override:
        return Path(override)
    # Apps Platform V2 mounts the GCS-fuse volume at /mnt/data; /gcs is the
    # older convention. Either is shared across instances and persistent.
    for mount in (Path("/mnt/data"), Path("/gcs")):
        if mount.is_dir():
            return mount / "nls_cache"
    return Path("/tmp/nls_cache")


def _uri_key(uri: str) -> str:
    """Filesystem-safe, stable key for a URI: readable tail + content hash."""
    normalized = uri.strip().rstrip("/")
    digest = hashlib.sha256(normalized.encode()).hexdigest()[:16]
    tail = normalized.rsplit("/", 1)[-1]
    safe_tail = re.sub(r"[^A-Za-z0-9_.-]", "_", tail)[:40]
    return f"{safe_tail}-{digest}"


@contextlib.contextmanager
def _file_lock(lock_path: Path) -> Iterator[None]:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(lock_path, "w")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(handle, fcntl.LOCK_UN)
        handle.close()


# Files of the fast corpus format (see search_engine._load_corpus_npy).
NPY_MATRIX_FILE = "embeddings.npy"
NPY_METADATA_FILE = "metadata.parquet"

# Files of the int8 PCA artifact (see gpu_corpus.py): a quantized, PCA-reduced
# corpus for very large embedding sets (~48M rows) that can't fit as fp32.
GPU_INT8_FILE = "corpus_int8.npy"
GPU_ARTIFACT_FILES = (
    "corpus_int8.npy",
    "pca_components.npy",
    "quant_scales.npy",
    "metadata.parquet",
)


def ensure_corpus_local(embeddings_uri: str, client: object) -> Path:
    """Return a local dir holding the corpus, downloading once.

    Two on-disk corpus formats are supported, detected by what lives under the
    URI:
    - fast: a top-level `embeddings.npy` (+ `metadata.parquet`) -- downloaded as
      two files and loaded via a contiguous matrix read in seconds;
    - lance: `rank=NNNNN/` Lance shards -- downloaded shard by shard.

    The mapping URI -> local dir is deterministic (`_uri_key`), so any process
    pointed at the same cache root reuses an existing download instead of
    re-fetching.
    """
    dest = cache_root() / "corpus" / _uri_key(embeddings_uri)
    marker = dest / _COMPLETE_MARKER
    if marker.exists():
        LOGGER.info("corpus cache hit: %s -> %s", embeddings_uri, dest)
        return dest
    with _file_lock(dest.parent / f"{dest.name}.lock"):
        if marker.exists():  # another worker finished while we waited
            return dest
        base = embeddings_uri.rstrip("/") + "/"
        if oci_s3.object_exists(base + GPU_INT8_FILE, client):
            # int8 PCA artifact: four flat files (corpus_int8 + pca + scales +
            # metadata). Loaded by gpu_corpus.load_gpu_corpus.
            LOGGER.info(
                "corpus cache miss (gpu int8): downloading %s -> %s", base, dest
            )
            for name in GPU_ARTIFACT_FILES:
                oci_s3.download_object(base + name, dest / name, client)
        elif oci_s3.object_exists(base + NPY_MATRIX_FILE, client):
            LOGGER.info("corpus cache miss (npy): downloading %s -> %s", base, dest)
            for name in (NPY_MATRIX_FILE, NPY_METADATA_FILE):
                oci_s3.download_object(base + name, dest / name, client)
        elif embeddings_uri.rstrip("/").endswith(".lance"):
            # A single direct Lance dataset (e.g. .../chunks.lance): download the
            # whole dataset prefix (data/, _versions/, ...) into dest.
            LOGGER.info(
                "corpus cache miss (lance dataset): downloading %s -> %s", base, dest
            )
            oci_s3.download_s3_prefix(base, dest, client)
        else:
            LOGGER.info("corpus cache miss (lance): downloading %s -> %s", base, dest)
            rank_table_uris = oci_s3.discover_rank_tables(embeddings_uri, client)
            if not rank_table_uris:
                raise FileNotFoundError(
                    f"no {NPY_MATRIX_FILE} or rank=NNNNN/ Lance shards under "
                    f"{embeddings_uri}"
                )
            for rank_uri in rank_table_uris:
                _, key = oci_s3.parse_s3_uri(rank_uri)
                rank_name = key.rstrip("/").rsplit("/", 1)[-1]
                oci_s3.download_s3_prefix(
                    rank_uri.rstrip("/") + "/", dest / rank_name, client
                )
        marker.write_text(embeddings_uri)
    return dest


def ensure_model_local(model_artifact_uri: str, client: object) -> Path:
    """Return a local dir holding the merged model snapshot, downloading once."""
    dest = cache_root() / "models" / _uri_key(model_artifact_uri)
    marker = dest / _COMPLETE_MARKER
    if marker.exists():
        LOGGER.info("model cache hit: %s -> %s", model_artifact_uri, dest)
        return dest
    with _file_lock(dest.parent / f"{dest.name}.lock"):
        if marker.exists():
            return dest
        LOGGER.info("model cache miss: downloading %s -> %s", model_artifact_uri, dest)
        oci_s3.download_s3_prefix(model_artifact_uri.rstrip("/") + "/", dest, client)
        if not (dest / "config.json").exists():
            raise FileNotFoundError(
                f"downloaded model snapshot missing config.json: {model_artifact_uri}"
            )
        marker.write_text(model_artifact_uri)
    return dest
