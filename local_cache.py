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
