"""FastAPI backend for the VLM natural-language video search UI.

Thin API over the existing engine: it reuses ``search_engine`` (encode / score /
rank / window), ``oci_s3`` (presigned MP4 URLs), ``dora_client`` (Data Explorer
segment sets) and ``config`` unchanged, and serves a hand-built frontend from
``web/``. The model + corpus load once at startup (minutes) and stay resident,
exactly like the Streamlit ``cache_resource`` path.

Run locally:
    uvicorn web_server:app --host 127.0.0.1 --port 8501
"""

from __future__ import annotations

import csv
import dataclasses
import base64
import binascii
import datetime as dt
import gzip
import hashlib
import html
import io
import json
import logging
import os
import re
import shutil
import tempfile
import threading
import time
import urllib.parse
import uuid
from pathlib import Path

# Cap CPU threads BEFORE numpy/torch/BLAS import. On Cloud Run the container sees the
# HOST core count (dozens), so OpenMP/MKL/OpenBLAS and torch would each spawn far more
# threads than the allocated vCPUs -> catastrophic oversubscription thrash (observed:
# a single image encode took 60-97s and a corpus scoring 250s). Pin every thread pool
# to the allocated vCPU count (NLS_NUM_THREADS, default 8 = the -full/base cpu limit).
# setdefault so an explicit deploy env still wins; must run before the first numpy/torch
# import (search_engine, below) for the BLAS backends to honor it.
_NLS_THREADS = os.environ.get("NLS_NUM_THREADS", "8")
for _tv in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_tv, _NLS_THREADS)

import analytics
import botocore.exceptions
import db
import dora_client
import local_cache
import nls_launcher
import oci_s3
import search_engine
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    RedirectResponse,
    Response,
)
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import numpy as np
from config import AppConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger(__name__)

HERE = os.path.dirname(os.path.abspath(__file__))
_CORPUS_ERRORS = (
    ValueError,
    FileNotFoundError,
    OSError,
    botocore.exceptions.BotoCoreError,
    botocore.exceptions.ClientError,
)

app = FastAPI(title="VLM Video Search")
_state: dict = {"model_ready": False, "ready": False, "load_error": None}


@app.middleware("http")
async def _no_cache_static(request: Request, call_next):
    """Tell browsers never to serve a stale app.js/style.css: after a deploy the
    frontend must match the backend, or requests fail with confusing 422s."""
    response = await call_next(request)
    path = request.url.path
    if path.startswith("/static/") or path in ("/", "/index.html"):
        response.headers["Cache-Control"] = "no-cache, must-revalidate"
    return response


@app.on_event("startup")
def _load_engine() -> None:
    """Bind the port immediately; load the heavy engine in the background.

    The encoder (~5 GB) + default corpus (up to 1M x 768) take minutes to pull
    and load -- far longer than Cloud Run's startup probe. Doing that work in the
    startup event would block uvicorn from accepting connections, so the probe
    fails and the revision never goes live. Instead we set up the cheap state
    here and warm the model/corpus on a daemon thread; endpoints return 503 until
    ``_state["ready"]``.
    """
    import torch as _torch  # local: confirm the thread pin took effect (logged once)
    LOGGER.info(
        "thread config: torch=%d OMP=%s MKL=%s (NLS_NUM_THREADS=%s) -- capped to "
        "allocated vCPUs to avoid Cloud Run oversubscription",
        _torch.get_num_threads(), os.environ.get("OMP_NUM_THREADS"),
        os.environ.get("MKL_NUM_THREADS"), os.environ.get("NLS_NUM_THREADS", "8"),
    )
    cfg = AppConfig.from_env()
    _state["cfg"] = cfg
    _state["s3"] = oci_s3.s3_client()
    # One corpus, named once. Kept in _state because several endpoints stamp it
    # onto persisted rows (exported vectors, threshold policies) as the space
    # those numbers belong to.
    _state["active_uri"] = full_corpus_module().DEFAULT_CORPUS_TABLE_URI
    threading.Thread(target=_warm_engine, name="engine-warmup", daemon=True).start()
    # Pre-warm the DORA gRPC channel + machine token now (independent of the
    # model), so the first segment-set selection doesn't pay channel/auth setup.
    threading.Thread(
        target=dora_client.prewarm, name="dora-prewarm", daemon=True
    ).start()


def _warm_engine() -> None:
    """Load the model, then pre-warm the default corpus (runs once, off the
    request path).

    Readiness is gated on the MODEL only -- that is all that is needed to encode
    queries and to load any corpus on demand. The default corpus (which can be
    the large 1M set, slow to pull on a cold start) is pre-warmed afterward as a
    convenience, but must NOT block switching to a smaller corpus.
    """
    try:
        cfg = _state["cfg"]
        LOGGER.info("loading encoder %s ...", cfg.model_artifact_uri or "base")
        _state["processor"], _state["model"] = search_engine.load_model(
            cfg.model_artifact_uri, cfg.device
        )
        _state["model_uri"] = cfg.model_artifact_uri  # active encoder (swappable)
        _state["model_ready"] = True  # endpoints unblock here
        LOGGER.info("model ready")
        db.init_schema()  # best-effort; logs + continues if exp-db is unreachable
        # There is one corpus, so start its load here rather than
        # waiting for someone to search: the read and decode take minutes, and
        # paying that on a user's first query reads as a hang. Runs on its own
        # thread and never gates readiness -- browse works throughout.
        try:
            _full_corpus_begin_load()
            LOGGER.info("full corpus load started in the background")
        except Exception as exc:  # noqa: BLE001 -- warm-up must not wedge startup
            LOGGER.warning("full corpus pre-warm could not start: %s", exc)
        _state["ready"] = True
    except Exception as exc:  # noqa: BLE001 -- warmup must record failure, not vanish
        _state["load_error"] = str(exc)
        LOGGER.exception("engine warmup failed: %s", exc)


def _require_ready() -> None:
    """Guard endpoints that need the model; 503 only while the model is loading.

    Gated on the model rather than the default corpus, so a user can switch to
    (and search) a smaller corpus without waiting for the large default to warm.
    """
    if _state.get("model_ready"):
        return
    if _state.get("load_error"):
        raise HTTPException(503, f"engine failed to load: {_state['load_error']}")
    raise HTTPException(503, "model is still loading; retry in a moment")


@app.get("/healthz")
def healthz() -> dict:
    """Liveness/readiness: the port is up immediately; ``ready`` flips when the
    model + corpus finish loading in the background."""
    return {
        "status": "ok",
        "model_ready": bool(_state.get("model_ready")),
        "ready": bool(_state.get("ready")),  # model + default corpus pre-warmed
        "load_error": _state.get("load_error"),
    }


def _is_frontier() -> bool:
    """True on the frontier/trucking deployment (derived from runtime env: Cloud Run service
    name / DORA hostname). Both deployments run the same image."""
    svc = os.getenv("K_SERVICE", "")
    host = os.getenv("URSA_SDK_GRPC_HOSTNAME", "")
    return "trucking" in svc.lower() or "frontier" in host.lower()


def _offline_scan_enabled() -> bool:
    """Whether the offline (Lilypad) full-corpus segment scan is offered.

    Off by default: `/api/full_export` selects and writes over the same 34M rows
    in-process, so a launch buys a wait and a second copy of the corpus. It also
    scanned a DIFFERENT (staler) table than the app searched, which is how an
    export could disagree with the results that produced it.

    Kept behind ``NLS_OFFLINE_SCAN=on`` rather than deleted, for the one case the
    online path refuses: a threshold so permissive that the match set exceeds
    ``_FULL_EXPORT_MAX_ROWS``.
    """
    return os.getenv("NLS_OFFLINE_SCAN", "").strip().lower() in ("1", "on", "true", "yes")


@app.get("/api/platform")
def platform() -> dict:
    """Which deployment this is (cars vs trucking) -- both run the same image, so
    it's derived from runtime env (Cloud Run service name / DORA hostname). Drives
    the top-left platform tag + whether the offline scan is offered. Not gated on
    model readiness, so the tag shows immediately during warmup."""
    is_trucks = _is_frontier()
    return {
        "name": "trucks" if is_trucks else "cars",
        "label": "TRUCKING" if is_trucks else "CARS",
        # Frontier holds the whole dataset in CPU memory, so the offline scan is hidden there.
        "offline_scan": _offline_scan_enabled(),
    }


# Serializes corpus loads so concurrent callers never each kick off their own
# multi-GB np.load of the SAME corpus. Without this, during the minutes-long load
# of a large corpus the background prewarm AND every request-path _get_corpus call
# (e.g. each /api/corpus page poll) start a separate load, stacking several 7GB
# copies in memory and OOM-killing the container (signal 9) even at 32GiB. With a
# per-URI lock only ONE load runs; the rest block briefly then get the cached one.
_CORPUS_LOCKS_GUARD = threading.Lock()
_CORPUS_LOCKS: dict[str, threading.Lock] = {}


# --- helpers ----------------------------------------------------------------
def _date_bounds(from_date: str | None, to_date: str | None, corpus):
    """Translate ISO dates to [start, end) unix seconds, open when at corpus edge."""
    lo, hi = corpus.time_span()
    lo_d = dt.datetime.fromtimestamp(lo, tz=dt.timezone.utc).date()
    hi_d = dt.datetime.fromtimestamp(hi, tz=dt.timezone.utc).date()
    f = dt.date.fromisoformat(from_date) if from_date else lo_d
    t = dt.date.fromisoformat(to_date) if to_date else hi_d
    start = dt.datetime.combine(f, dt.time.min, tzinfo=dt.timezone.utc)
    end = dt.datetime.combine(t, dt.time.min, tzinfo=dt.timezone.utc) + dt.timedelta(
        days=1
    )
    start_unix = None if f <= lo_d else int(start.timestamp())
    end_unix = None if t >= hi_d else int(end.timestamp())
    return start_unix, end_unix


# Segment-set ids are fetched in the BACKGROUND: a large Data Explorer set can be
# millions of ids (thousands of gRPC pages, ~minutes), and we must never block a
# search request on it. _segment_ids(uuid) returns the cached frozenset if ready,
# else kicks off a one-shot background load and returns None ("pending"); the
# search then runs unfiltered and the UI is told to retry when the set is ready.
_SEG: dict[str, dict] = {}  # uuid -> {status, ids, count, err}
_SEG_LOCK = threading.Lock()

# Cached boolean corpus-mask per (corpus uri, segment-set uuid). Computing the
# mask is O(corpus) set-membership; caching it makes every later search / page /
# funnel an O(corpus) AND instead of re-deriving membership from the (possibly
# multi-million-id) set each time. A mask is num_rows bools (~100 KB), so a few
# are cheap to keep resident.
_SEG_MASK: dict[tuple[str, str], object] = {}
_SEG_MASK_LOCK = threading.Lock()


def _segment_mask(uri: str, uuid: str | None, corpus, allowed_ids):
    """Cached corpus-aligned boolean mask for a segment set, or None if no set /
    not yet loaded. Computed once per (uri, uuid); membership is immutable."""
    if not uuid or allowed_ids is None:
        return None
    key = (uri, uuid)
    with _SEG_MASK_LOCK:
        m = _SEG_MASK.get(key)
    if m is not None and len(m) == corpus.num_rows:
        return m
    m = search_engine.segment_mask(corpus, allowed_ids)
    with _SEG_MASK_LOCK:
        _SEG_MASK[key] = m
    return m


def _seg_cache_path(uuid: str):
    """On-disk location for a fetched segment-id set (gzipped, one id per line).

    Keyed by dataset uuid alone: a DORA dataset version is immutable, so the
    membership never changes once the version is published. Caching here turns
    the multi-minute gRPC pull into a one-time cost that also survives restarts.
    """
    return local_cache.cache_root() / "segment_sets" / f"{uuid}.txt.gz"


def _read_seg_cache(uuid: str) -> frozenset[str] | None:
    path = _seg_cache_path(uuid)
    if not path.exists():
        return None
    try:
        with gzip.open(path, "rt") as fh:
            ids = frozenset(line.rstrip("\n") for line in fh if line.rstrip("\n"))
    except (OSError, gzip.BadGzipFile, EOFError) as exc:
        # Corrupt/partial cache file: treat as a miss and re-fetch fresh.
        LOGGER.warning("segment-set disk cache unreadable for %s: %s", uuid, exc)
        return None
    LOGGER.info("segment-set %s loaded from disk cache (%d ids)", uuid, len(ids))
    return ids


def _write_seg_cache(uuid: str, ids: frozenset[str]) -> None:
    path = _seg_cache_path(uuid)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        with gzip.open(tmp, "wt") as fh:
            fh.write("\n".join(ids))
        tmp.replace(path)  # atomic publish so readers never see a partial file
    except OSError as exc:  # disk cache is best-effort; never fail the load
        LOGGER.warning("could not write segment-set disk cache for %s: %s", uuid, exc)


def _segment_ids(uuid: str) -> frozenset[str] | None:
    with _SEG_LOCK:
        rec = _SEG.get(uuid)
        if rec and rec["status"] == "done":
            return rec["ids"]
        if rec and rec["status"] == "loading":
            return None
        _SEG[uuid] = {"status": "loading", "ids": None, "count": 0, "err": None}

    def _progress(n: int) -> None:
        with _SEG_LOCK:
            cur = _SEG.get(uuid)
            if cur and cur["status"] == "loading":
                cur["count"] = n

    def _load() -> None:
        try:
            ids = _read_seg_cache(uuid)
            if ids is None:
                ids = dora_client.fetch_segment_ids(uuid, progress=_progress)
                _write_seg_cache(uuid, ids)
            rec = {"status": "done", "ids": ids, "count": len(ids), "err": None}
        except Exception as exc:  # noqa: BLE001 -- background loader must never die silently
            LOGGER.warning("segment-set load failed for %s: %s", uuid, exc)
            rec = {"status": "error", "ids": None, "count": 0, "err": str(exc)}
        with _SEG_LOCK:
            _SEG[uuid] = rec
        LOGGER.info("segment-set %s -> %s (%d ids)", uuid, rec["status"], rec["count"])

    threading.Thread(target=_load, name=f"segload-{uuid[:8]}", daemon=True).start()
    return None


def _resolve_segment_filter(seg_uuid: str | None):
    """Return (allowed_ids_or_None, pending: bool, count). pending=True means the
    set is still loading in the background, so the search runs unfiltered."""
    if not seg_uuid:
        return None, False, 0
    ids = _segment_ids(seg_uuid)
    if ids is None:
        with _SEG_LOCK:
            count = _SEG.get(seg_uuid, {}).get("count", 0)
        return None, True, count
    return ids, False, len(ids)


# Set cardinality of a bitmap-path segment set, cached for status/UI display
# (the cached mask only knows matched-corpus-rows, not the full set size).
_SEG_BM_COUNT: dict[str, int] = {}


def _resolve_segment_mask(uri: str, seg_uuid: str | None, corpus):
    """Corpus-aligned segment-set mask: ``(mask_or_None, pending, count)``.

    Fast path -- when the corpus carries ``dx_internal_id``: fetch the set's
    roaring bitmap in ONE ``DescribeDataSet(include_bitmap)`` call and AND it
    against the corpus. No pagination, never pending; ``count`` is the set
    cardinality. The mask is cached per ``(uri, uuid)`` like the legacy path.

    Fallback -- older corpora without ``dx_internal_id``: the external_id
    pagination path (background load, may report ``pending``).
    """
    if not seg_uuid:
        return None, False, 0
    if corpus.has_internal_ids():
        key = (uri, seg_uuid)
        with _SEG_MASK_LOCK:
            m = _SEG_MASK.get(key)
        if m is not None and len(m) == corpus.num_rows:
            return m, False, _SEG_BM_COUNT.get(seg_uuid, int(m.sum()))
        try:
            set_bm = dora_client.fetch_segment_bitmap(seg_uuid)
        except dora_client.DoraUnavailable as exc:
            LOGGER.warning(
                "bitmap segment filter unavailable for %s: %s", seg_uuid, exc
            )
            with _SEG_LOCK:
                _SEG[seg_uuid] = {
                    "status": "error",
                    "ids": None,
                    "count": 0,
                    "err": str(exc),
                }
            return None, False, 0
        m = search_engine.segment_mask_from_bitmap(corpus, set_bm)
        count = len(set_bm)
        with _SEG_MASK_LOCK:
            _SEG_MASK[key] = m
        _SEG_BM_COUNT[seg_uuid] = count
        # Report through the same _SEG record the status/prefetch endpoints read.
        with _SEG_LOCK:
            _SEG[seg_uuid] = {
                "status": "done",
                "ids": None,
                "count": count,
                "err": None,
            }
        return m, False, count
    # Fallback: external_id pagination path.
    allowed, pending, count = _resolve_segment_filter(seg_uuid)
    mask = _segment_mask(uri, seg_uuid, corpus, allowed)
    return mask, pending, count


# --- Lance/parquet downsample dataset --------------------------------------
# A second downsample source (alongside the Data Explorer segment set): the user
# points at an arbitrary Lance or parquet dataset whose ``segment_id`` column is
# the set of segments to keep. We read that column once (immutable per path),
# then build a corpus-aligned mask with the same ``segment_mask`` primitive and
# AND it with the segment-set mask. So a chunk survives only if its segment is in
# BOTH filters -- "only the mini-segments that are left".
_LANCE_FILTER_IDS: dict[str, tuple[str, frozenset[str]]] = {}
_LANCE_FILTER_LOCK = threading.Lock()


# Downsample-dataset cache lives on LOCAL disk, NOT the gcs-fuse mount. A lance is
# a directory; finalizing the download with an atomic directory rename
# (tmp.replace(dir)) fans out into many FUSE ops over gcs-fuse and, under the fd
# pressure of the resident multi-GB corpus, fails with EMFILE ("Too many open
# files"). Local /tmp makes the rename a single syscall. The datasets are tiny
# (KBs) and immutable per path, so re-downloading per cold start is negligible.
_FILTER_CACHE_BASE = Path(tempfile.gettempdir()) / "nls_filter_sets"


def _lance_filter_cache_dir(uri: str) -> Path:
    h = hashlib.sha1(uri.encode("utf-8")).hexdigest()[:16]
    return _FILTER_CACHE_BASE / h


def _lance_filter_ids(uri: str) -> tuple[str, frozenset[str]]:
    """``(key_column, ids)`` for a downsample dataset at ``uri`` (lance or parquet).

    Downloaded + read once per path and cached in memory (the dataset is
    immutable for a given URI). Reads from the SAME object store the corpus loads
    from (the OCI S3-compat client) -- a path on a bucket the app cannot reach
    would otherwise stall on retry backoff, so we use a low-retry client that
    fails fast and surfaces the error to the UI.
    """
    with _LANCE_FILTER_LOCK:
        cached = _LANCE_FILTER_IDS.get(uri)
    if cached is not None:
        return cached
    local_dir = _lance_filter_cache_dir(uri)
    if not local_dir.exists():
        tmp = local_dir.with_name(local_dir.name + ".tmp")
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
        oci_s3.download_s3_prefix(
            uri.rstrip("/") + "/", tmp, oci_s3.s3_client(fast_fail=True)
        )
        local_dir.parent.mkdir(parents=True, exist_ok=True)
        tmp.replace(local_dir)
    is_lance = uri.rstrip("/").endswith(".lance")
    key, ids = search_engine.read_filter_ids(local_dir, is_lance)
    with _LANCE_FILTER_LOCK:
        _LANCE_FILTER_IDS[uri] = (key, ids)
    LOGGER.info("lance downsample %s -> %d %ss", uri, len(ids), key)
    return key, ids


# Downsample ids up to this count travel inline in the workload config (gRPC-friendly;
# a segment_id is ~60 chars). Larger sets are written to a single-column parquet next to
# the scan output and passed BY REFERENCE (filter_ids_uri / segment_set_ids_uri) -- the
# worker reads them in one GET, so there is NO upper bound on the downsample size.
_SCAN_INLINE_IDS_MAX = 10_000
# Upper bound on how many segments a scan may auto-register as a DORA segment set. Beyond
# this the set is corpus-sized (a symptom of too-low a threshold) and unusable, and the
# DORA CreateDataSet call would hang -- so we skip + annotate instead of registering.
_SEGSET_MAX_SEGMENTS = 300_000


def _ids_by_reference(ids: list[str], out_root: str, basename: str) -> str:
    """Write a downsample id set as a one-column parquet under the scan's output root
    and return its URI. Pass-by-reference keeps arbitrarily large id sets out of the
    workload config (inline they bloat the submit payload past gRPC limits)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    buf = io.BytesIO()
    pq.write_table(pa.table({"segment_id": pa.array(ids, type=pa.string())}), buf)
    uri = f"{out_root.rstrip('/')}/{basename}.parquet"
    oci_s3.put_bytes(uri, buf.getvalue(), oci_s3.s3_client(), "application/octet-stream")
    LOGGER.info("scan downsample: wrote %d ids by reference to %s", len(ids), uri)
    return uri


def _scan_downsample_ids(uri: str) -> tuple[str, list[str]]:
    """``(corpus_key, sorted ids)`` for the OFFLINE-SCAN downsample dataset at ``uri``.

    The app (which already powers the in-app lance filter) resolves the membership here;
    the launch path ships the ids inline when small, else by parquet reference (no size
    cap). Uses the segment_id-first priority (no dx) so the worker can match the
    segment_id string space."""
    local_dir = _lance_filter_cache_dir(uri)
    if not local_dir.exists():
        tmp = local_dir.with_name(local_dir.name + ".tmp")
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
        oci_s3.download_s3_prefix(uri.rstrip("/") + "/", tmp, oci_s3.s3_client(fast_fail=True))
        local_dir.parent.mkdir(parents=True, exist_ok=True)
        tmp.replace(local_dir)
    is_lance = uri.rstrip("/").endswith(".lance")
    key, ids = search_engine.read_filter_ids(
        local_dir, is_lance, key_columns=search_engine._SCAN_FILTER_KEY_COLUMNS
    )
    if not ids:
        raise ValueError("downsample dataset has no usable segment ids")
    LOGGER.info("scan downsample %s -> %d %s ids", uri, len(ids), key)
    return key, sorted(ids)


_SCAN_MANIFEST_S3 = None  # cached fast_fail OCI client for manifest reads


def _read_scan_manifest(lance_uri: str) -> dict | None:
    """The result manifest.json a completed scan writes next to its Lance output;
    None if absent/unreadable. Best-effort (the scan may still be running, or an old
    image may not have written one)."""
    lance_uri = (lance_uri or "").rstrip("/")
    if not lance_uri.endswith("/segments.lance"):
        return None
    manifest_uri = lance_uri[: -len("/segments.lance")] + "/manifest.json"
    # MUST use the fast_fail client (2 attempts, short timeouts): a missing/unreadable
    # manifest has to fail in seconds, not retry for minutes on the polled /api/scans
    # path. Cache it module-level so the first poll doesn't rebuild it per scan.
    global _SCAN_MANIFEST_S3
    if _SCAN_MANIFEST_S3 is None:
        _SCAN_MANIFEST_S3 = oci_s3.s3_client(fast_fail=True)
    client = _SCAN_MANIFEST_S3
    try:
        bucket, key = oci_s3.parse_s3_uri(manifest_uri)
        body = client.get_object(Bucket=bucket, Key=key)["Body"].read()
        return json.loads(body)
    except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError, ValueError, OSError) as exc:
        LOGGER.info("scan manifest unavailable (%s): %s", manifest_uri, exc)
        return None


def _scan_counts_from_manifest(manifest: dict) -> dict:
    """Compact result counts for the Recent-scans panel: total segments/clips and the
    per-tag breakdown. Prefers distinct-segment counts (Data-Explorer semantics); older
    manifests only carry interval counts, so fall back to those and flag which it is."""
    per_tag_segments = manifest.get("segments_per_tag") or {}
    per_tag = per_tag_segments or manifest.get("intervals_per_tag") or {}
    return {
        "num_segments": manifest.get("num_segments"),
        "num_clips_scanned": manifest.get("num_clips_scanned"),
        "per_tag": per_tag,
        "per_tag_is_segments": bool(per_tag_segments),
    }


def _scan_counts(execution_id: str, status: str, lance_uri: str, cached) -> dict | None:
    """Result counts for a scan row: return the cached copy, else (once the scan has
    SUCCEEDED) read them from the manifest ONCE and cache the result.

    Caching is the point: this runs on the polled /api/scans path, so a manifest must be
    read at most once per scan. A missing/unreadable manifest is cached as an empty {}
    sentinel so we never re-hit OCI for it on subsequent polls (an old scan's manifest
    won't appear later). ``cached is not None`` distinguishes "already checked, nothing"
    ({}) from "never checked" (None)."""
    if cached is not None:
        return cached or None
    if "SUCCEEDED" not in (status or "").upper() or not lance_uri:
        return None
    manifest = _read_scan_manifest(lance_uri)
    counts = _scan_counts_from_manifest(manifest) if manifest else {}
    db.set_scan_counts(execution_id, counts)
    return counts or None


def _dx_bitmap_from_segment_ids(corpus, seg_ids):
    """Roaring bitmap of the ``dx_internal_id``s of corpus rows whose ``segment_id``
    is in ``seg_ids`` -- i.e. convert a segment-id downsample set to the dx-internal
    bitmask the corpus already carries, so the mask is an int bitmap intersection."""
    from pyroaring import BitMap

    bm = BitMap()
    dxs = corpus.dx_internal_id or []
    for s, d in zip(corpus.segment_id, dxs):
        if d is not None and s in seg_ids:
            bm.add(int(d))
    return bm


def _lance_filter_mask(corpus_uri: str, lance_uri: str, corpus, key: str, ids):
    """Cached corpus-aligned membership mask for a downsample dataset.

    Prefers the dx_internal_id roaring-bitmap intersection (fast int membership):
    directly when the dataset carries ``dx_internal_id``, else by converting a
    ``segment_id`` set to dx ids via the corpus when it has them. Falls back to
    string membership on ``segment_id``/``run_uuid`` for corpora without dx ids."""
    cache_key = (corpus_uri, f"lance::{key}::{lance_uri}")
    with _SEG_MASK_LOCK:
        m = _SEG_MASK.get(cache_key)
    if m is not None and len(m) == corpus.num_rows:
        return m
    if key == "dx_internal_id" and corpus.has_internal_ids():
        from pyroaring import BitMap

        m = search_engine.segment_mask_from_bitmap(corpus, BitMap(int(x) for x in ids))
    elif key == "segment_id" and corpus.has_internal_ids():
        m = search_engine.segment_mask_from_bitmap(
            corpus, _dx_bitmap_from_segment_ids(corpus, ids)
        )
    else:
        values = (
            corpus.segment_id
            if key in ("segment_id", "dx_internal_id")
            else corpus.run_uuid
        )
        m = search_engine.value_mask(values, ids)
    with _SEG_MASK_LOCK:
        _SEG_MASK[cache_key] = m
    return m


def _parse_vehicles(vehicle: str | None) -> frozenset[str]:
    """Parse the vehicle filter input (comma/whitespace-separated ids) into a set.

    Empty/None -> empty set (filter inert). e.g. "truck-808, truck-810" or a
    newline-separated list of car vehicle_names."""
    if not vehicle:
        return frozenset()
    return frozenset(v.strip() for v in re.split(r"[,\s]+", vehicle.strip()) if v.strip())


def _parse_drive_ids(drive_id: str | None) -> frozenset[str]:
    """Parse the drive-id (run_uuid) filter input (comma/whitespace-separated)
    into a set. Empty/None -> empty set (filter inert)."""
    if not drive_id:
        return frozenset()
    return frozenset(
        d.strip() for d in re.split(r"[,\s]+", drive_id.strip()) if d.strip()
    )


def _resident_corpus():
    """The full corpus if it is already loaded, else None (no load triggered)."""
    with _FULL_LOCK:
        return _FULL["corpus"]


def _current_user(request: Request) -> str:
    """Authenticated email from the IAP header; 'local' off-platform."""
    for key in ("x-goog-authenticated-user-email", "x-goog-authenticated-user-id"):
        raw = request.headers.get(key)
        if raw:
            return raw.split(":", 1)[-1]
    return "local"


def _ns(sec: int | None) -> int | None:
    """Epoch seconds -> nanoseconds (chunk bounds are whole seconds, so lossless)."""
    return int(sec) * 1_000_000_000 if sec is not None else None


def _ns_f(sec: float | None) -> int | None:
    """Fractional epoch seconds -> nanoseconds (interpolated interval bounds)."""
    return int(round(float(sec) * 1_000_000_000)) if sec is not None else None


def _utc(sec: int | None) -> str:
    """Epoch seconds -> 'YYYY-MM-DD HH:MM:SS UTC' (empty string when None)."""
    if sec is None:
        return ""
    return dt.datetime.fromtimestamp(int(sec), tz=dt.timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )


def _hit_dict(h) -> dict:
    """One hit as the UI grid expects it.

    Takes either a `search_engine.RankedHit` or a `full_corpus.Hit`: the two
    agree on every field but the row address, and they number different things
    -- `index` is a resident-corpus position, `row` a full-corpus one. Each hit
    carries whichever it has, so a mark made on it can be resolved against the
    corpus it actually came from.
    """
    return {
        "rank": h.rank,
        "score": round(float(h.score), 4),
        "chunk_id": h.chunk_id,
        "run_uuid": h.run_uuid,
        "segment_id": h.segment_id,
        "start_timestamp_ns": _ns(h.chunk_start_unix),
        "end_timestamp_ns": _ns(h.chunk_end_unix),
        "start_utc": _utc(h.chunk_start_unix),
        "end_utc": _utc(h.chunk_end_unix),
        "source_media_uri": h.source_media_uri,
        "index": getattr(h, "index", -1),
        "row": getattr(h, "row", None),
    }


# --- request models ---------------------------------------------------------
class SearchRequest(BaseModel):
    query: str
    page: int = 0
    page_size: int = 24
    start_rank: int | None = None  # 1-based; overrides page when set
    start_score: float | None = None  # start at first clip scoring <= this
    from_date: str | None = None
    to_date: str | None = None
    segment_set_uuid: str | None = None
    filter_lance_uri: str | None = None
    # Vehicle-id filter: comma/space-separated ids matched against the corpus
    # vehicle column (inert if the corpus has none). AND'd with the other filters.
    vehicle: str | None = None
    # Drive-id (run_uuid) filter: comma/space-separated; keeps only those drives.
    drive_id: str | None = None
    embeddings_uri: str | None = None


class Mark(BaseModel):
    chunk_id: str
    segment_id: str = ""
    mark: str  # "up" | "down"
    index: int | None = None  # resident-corpus row index, for refine
    # Full-corpus row position. Distinct from `index` on purpose: the two number
    # different tables, and a full-corpus row reused as a resident index would
    # land on an unrelated clip whenever it happens to fall in range.
    row: int | None = None
    rank: int | None = None
    score: float | None = None


class RefineRequest(BaseModel):
    query: str  # original text query (label + optional text anchor)
    marks: list[Mark] = []
    page: int = 0
    page_size: int = 24
    start_rank: int | None = None
    start_score: float | None = None
    from_date: str | None = None
    to_date: str | None = None
    segment_set_uuid: str | None = None
    filter_lance_uri: str | None = None
    # Vehicle-id filter: comma/space-separated ids matched against the corpus
    # vehicle column (inert if the corpus has none). AND'd with the other filters.
    vehicle: str | None = None
    # Drive-id (run_uuid) filter: comma/space-separated; keeps only those drives.
    drive_id: str | None = None
    embeddings_uri: str | None = None
    negative_weight: float = 0.5
    text_weight: float = 0.0


class ExportRequest(BaseModel):
    query: str
    k: int = 100
    tag: str = ""
    from_date: str | None = None
    to_date: str | None = None
    segment_set_uuid: str | None = None
    filter_lance_uri: str | None = None
    segment_set_name: str | None = None
    filter_lance_uri: str | None = None
    # Vehicle-id filter: comma/space-separated ids matched against the corpus
    # vehicle column (inert if the corpus has none). AND'd with the other filters.
    vehicle: str | None = None
    # Drive-id (run_uuid) filter: comma/space-separated; keeps only those drives.
    drive_id: str | None = None
    embeddings_uri: str | None = None
    marks: list[Mark] = []
    # When true, collapse the ranking to one row per segment_id -- the best (highest-scoring)
    # clip per 30s source segment -- before taking the top-k. Inert when the corpus has no
    # segment_id (clips without one are kept as-is, never merged).
    dedupe_segment: bool = False
    # When true, also register the exported top-k segments as a DORA segment set
    # ("dataset"), named after the export file. Distinct from segment_set_uuid/
    # segment_set_name above, which are the input FILTER set.
    create_segment_set: bool = False
    # Interval mode: instead of one row per fixed 8s mini-segment, project the
    # per-clip scores onto the 4s stride grid, threshold, merge contiguous cells
    # per drive, and export variable-length interpolated intervals (CSV+parquet
    # only; segment-set registration is not yet supported for intervals).
    interval: bool = False
    # "k": threshold = the k-th largest grid-cell score (uses ``k`` above).
    # "score": threshold = ``interval_score``.
    interval_mode: str = "k"
    interval_score: float | None = None


class ThresholdSearchRequest(BaseModel):
    """Fit a per-tag score cutoff from labeled 👍/👎 marks, and return a batch of
    UNLABELED boundary clips to keep labeling (active learning). Carries the same
    query + filters as the other search endpoints so the scores it tunes against
    are exactly the ones the app will select with."""

    query: str
    from_date: str | None = None
    to_date: str | None = None
    segment_set_uuid: str | None = None
    filter_lance_uri: str | None = None
    vehicle: str | None = None
    drive_id: str | None = None
    embeddings_uri: str | None = None
    marks: list[Mark] = []
    # "f1" (max F-beta) | "youden" (max TPR-FPR) | "precision" (max recall s.t.
    # precision >= min_precision). See search_engine.fit_threshold.
    objective: str = "f1"
    beta: float = 1.0
    min_precision: float = 0.9
    # >0 holds out a fraction of labels so the reported metrics aren't optimistic.
    val_fraction: float = 0.0
    # How many boundary clips to hand back for the next labeling round.
    sample_size: int = 12
    band: float = 0.08


class VectorSearchRequest(BaseModel):
    """Rank by a pre-computed query vector -- used to RESUME a saved search."""

    vector: list[float]
    query: str = ""  # the NL text the vector came from (label only)
    page: int = 0
    page_size: int = 24
    start_rank: int | None = None
    start_score: float | None = None
    from_date: str | None = None
    to_date: str | None = None
    segment_set_uuid: str | None = None
    filter_lance_uri: str | None = None
    # Vehicle-id filter: comma/space-separated ids matched against the corpus
    # vehicle column (inert if the corpus has none). AND'd with the other filters.
    vehicle: str | None = None
    # Drive-id (run_uuid) filter: comma/space-separated; keeps only those drives.
    drive_id: str | None = None
    embeddings_uri: str | None = None


class WindowSearchRequest(BaseModel):
    """Search by an example video window already embedded in the corpus.

    The user names a drive (``run_uuid``) or 30s source segment (``segment_id``)
    and an optional [start_ns, end_ns] window; the matching pre-embedded chunks
    are mean-pooled into the query vector. Carries the same date / segment-set /
    vehicle / drive filters as the other search endpoints.
    """

    run_uuid: str = ""
    segment_id: str = ""
    start_ns: int = 0  # window start, unix nanoseconds (0 = open on this side)
    end_ns: int = 0  # window end, unix nanoseconds (0 = open on this side)
    query: str = ""  # optional label for search history
    page: int = 0
    page_size: int = 24
    start_rank: int | None = None
    start_score: float | None = None
    from_date: str | None = None
    to_date: str | None = None
    segment_set_uuid: str | None = None
    filter_lance_uri: str | None = None
    vehicle: str | None = None
    drive_id: str | None = None
    embeddings_uri: str | None = None


class ConfigQuery(BaseModel):
    query: str
    k: int = 100
    # Per-query cosine cutoff. >0 => keep clips with score >= threshold (capped at k as a
    # safety max); 0/unset => pure top-k. Lets Download CSV select by similarity OR by rank.
    threshold: float = 0.0


class ConfigExportRequest(BaseModel):
    """Batch export: each query contributes its own top-k, concatenated into one
    CSV/parquet -- the same artifact flow as a single Export, run N times."""

    queries: list[ConfigQuery] = []
    dedupe: bool = True
    # When true, collapse to one row per segment_id (best-scoring clip per segment) after the
    # per-clip dedup. Inert when the corpus lacks segment_id; clips without one are kept.
    dedupe_segment: bool = False
    tag: str = ""
    from_date: str | None = None
    to_date: str | None = None
    segment_set_uuid: str | None = None
    filter_lance_uri: str | None = None
    segment_set_name: str | None = None
    filter_lance_uri: str | None = None
    # Vehicle-id filter: comma/space-separated ids matched against the corpus
    # vehicle column (inert if the corpus has none). AND'd with the other filters.
    vehicle: str | None = None
    # Drive-id (run_uuid) filter: comma/space-separated; keeps only those drives.
    drive_id: str | None = None
    embeddings_uri: str | None = None
    # Register the exported (deduped) segments as a DORA segment set, named after
    # the export file. See ExportRequest.create_segment_set.
    create_segment_set: bool = False


class CurateRow(BaseModel):
    """One kept row from a curate-preview selection (the client sends back exactly the
    rows the user chose to export). Mirrors the export CSV/parquet columns."""

    query: str = ""
    rank: int = 0
    score: float = 0.0
    segment_id: str = ""
    chunk_id: str = ""
    run_uuid: str = ""
    start_timestamp_ns: int | None = None
    end_timestamp_ns: int | None = None
    source_media_uri: str = ""


class CurateExportRequest(BaseModel):
    """Export an explicit, user-curated set of rows (from the Curate-from-config
    preview) -- the same CSV/parquet/segment-set artifacts as ``/api/export_config``,
    but built from exactly these rows rather than re-running the queries."""

    rows: list[CurateRow] = []
    create_segment_set: bool = False
    embeddings_uri: str | None = None
    segment_set_uuid: str | None = None
    filter_lance_uri: str | None = None
    from_date: str | None = None
    to_date: str | None = None


# --- API --------------------------------------------------------------------
@app.get("/api/corpus")
def corpus_info(uri: str | None = None) -> dict:
    """Info for THE corpus. There is only one, and it cannot be switched.

    `uri` used to load and activate an arbitrary Lance table. Pointing a run at
    the wrong table silently produces plausible results -- a query scored against
    another model's embeddings still returns a ranked list -- so the corpus is
    pinned to `full_corpus.DEFAULT_CORPUS_TABLE_URI` and the parameter is
    rejected rather than ignored.
    """
    if uri and uri.strip() and uri.strip() != full_corpus_module().DEFAULT_CORPUS_TABLE_URI:
        raise HTTPException(
            400,
            "the corpus is fixed to "
            f"{full_corpus_module().DEFAULT_CORPUS_TABLE_URI} and cannot be switched",
        )
    cfg = _state["cfg"]
    c = _require_full_corpus()
    lo, hi = c.time_span()
    return {
        "num_rows": c.num_rows,
        "dim": c.dim,
        "matrix_dtype": str(c.corpus_i8.dtype),
        "span_lo_date": dt.datetime.fromtimestamp(lo, tz=dt.timezone.utc)
        .date()
        .isoformat(),
        "span_hi_date": dt.datetime.fromtimestamp(hi, tz=dt.timezone.utc)
        .date()
        .isoformat(),
        "has_segment_id": c.has_segment_id(),
        "embeddings_uri": c.corpus_uri,
        "corpus_version": c.dataset_version,
        # The ACTIVE (possibly runtime-swapped) encoder, not just the configured
        # default -- so the pill reflects /api/model switches. Friendly name
        # (e.g. "white-dwarf") via _MODEL_LABELS; raw URI kept in model_uri.
        "model": _model_label(_model_label_uri()) or "base Cosmos-Embed1-448p",
        "model_uri": _model_label_uri(),
        "device": cfg.device,
    }


def _model_label_uri() -> str:
    """The active encoder URI (empty == base model)."""
    return _state.get("model_uri", _state["cfg"].model_artifact_uri) or ""


def _activate_model(uri: str) -> None:
    """Load `uri` (s3 merged snapshot, or '' for the base model) and swap it in as
    the resident encoder. Invalidates the memoized scores since the query/corpus
    joint space changes with the model."""
    proc, model = search_engine.load_model(uri, _state["cfg"].device)
    _state["processor"], _state["model"] = proc, model
    _state["model_uri"] = uri
    _state["last"] = None  # cached scores/order/vec are model-specific
    LOGGER.info("active encoder switched to %s", uri or "base")


@app.get("/api/model")
def model_info(uri: str | None = None) -> dict:
    """Get the active encoder, or (when ``uri`` is given) swap to a different
    checkpoint at runtime so embedding search can be tested across models.

    NOTE: this must match the model the corpus was embedded with, or query and
    corpus vectors won't share a space -- pair a model swap with the matching
    corpus. Loading a ~5GB snapshot blocks while it downloads + loads."""
    _require_ready()
    if uri is not None:
        u = uri.strip()
        if u != _model_label_uri():
            try:
                _activate_model(u)
            except _CORPUS_ERRORS as exc:
                raise HTTPException(400, f"could not load model: {exc}")
    cur = _model_label_uri()
    return {"model_uri": cur, "label": _model_label(cur) or "base Cosmos-Embed1-448p"}


def _window(corpus, scores, order, page, page_size, start_rank, start_score, t0, label):
    """Shared result-window builder for /api/search and /api/refine.

    Resolves the page start from (in priority order) start_score, start_rank, or
    page, so the frontend's "jump to rank" / "jump to similarity" controls and
    Prev/Next all route through one path.
    """
    total = int(order.size)
    if total == 0:
        return {
            "total": 0,
            "page": 0,
            "page_size": page_size,
            "hits": [],
            "score_hi": None,
            "score_lo": None,
            "label": label,
            "elapsed_ms": round((time.time() - t0) * 1000, 1),
        }
    if start_score is not None:
        start = search_engine.start_index_for_score(scores, order, start_score)
    elif start_rank is not None:
        start = int(start_rank) - 1
    else:
        start = max(0, page) * page_size
    start = max(0, min(start, total - 1))
    hits = search_engine.hits_from_order(corpus, scores, order, start, page_size)
    return {
        "total": total,
        "page": start // page_size,
        "page_size": page_size,
        "elapsed_ms": round((time.time() - t0) * 1000, 1),
        "score_hi": round(float(scores[order[0]]), 4),
        "score_lo": round(float(scores[order[-1]]), 4),
        "label": label,
        "hits": [_hit_dict(h) for h in hits],
    }


@app.post("/api/search")
def search(req: SearchRequest, request: Request) -> dict:
    """Retired. This ranked the 2M resident corpus, which no longer exists.

    Kept as an explicit 410 rather than deleted: a 404 tells an external caller
    nothing, and quietly serving the full corpus from here would change the
    result set under a client that never asked for it.
    """
    raise _gone("/api/full_search")


@app.post("/api/refine")
def refine(req: RefineRequest, request: Request) -> dict:
    """Retired. This ranked the 2M resident corpus, which no longer exists.

    Kept as an explicit 410 rather than deleted: a 404 tells an external caller
    nothing, and quietly serving the full corpus from here would change the
    result set under a client that never asked for it.
    """
    raise _gone("/api/full_refine")


@app.get("/api/search_session/{session_id}")
def search_session(session_id: int) -> dict:
    """The stored query + search vector + filters for a past export, by row id.

    Backs "Resume" on the Search-history page: the frontend fetches this, then
    re-ranks via /api/search_by_vector with the same vector -- so a teammate can
    pick up exactly where a previous search left off.
    """
    s = db.get_session(session_id)
    if not s:
        raise HTTPException(404, "search session not found")

    def _json_list(key: str) -> list:
        raw = s.get(key) or ""
        try:
            val = json.loads(raw) if raw and raw != "null" else []
        except json.JSONDecodeError:
            return []
        return val if isinstance(val, list) else []

    return {
        "id": s["id"],
        "tag": s.get("tag") or "",
        "query": s.get("query") or "",
        "embeddings_uri": s.get("embeddings_uri") or "",
        "segment_set_uuid": s.get("segment_set_uuid") or "",
        "segment_set_name": s.get("segment_set_name") or "",
        "from_date": s.get("date_from"),
        "to_date": s.get("date_to"),
        # Lance-downsample + vehicle + drive-id filters, so Resume restores the full set.
        "filter_lance_uri": s.get("filter_lance_uri") or "",
        "vehicle": s.get("vehicle") or "",
        "drive_id": s.get("drive_id") or "",
        "vector": _json_list("search_vector_json"),
        # The 👍/👎 marks saved with this search, so Resume can restore them.
        "thumbs_up": _json_list("thumbs_up_json"),
        "thumbs_down": _json_list("thumbs_down_json"),
    }


@app.post("/api/search_by_vector")
def search_by_vector(req: VectorSearchRequest) -> dict:
    """Retired. This ranked the 2M resident corpus, which no longer exists.

    Kept as an explicit 410 rather than deleted: a 404 tells an external caller
    nothing, and quietly serving the full corpus from here would change the
    result set under a client that never asked for it.
    """
    raise _gone("/api/full_search_by_vector")


@app.post("/api/search_by_upload")
def search_by_upload(req: UploadEncodeRequest) -> dict:
    """Encode a dragged-and-dropped image or short video into the joint space and
    return its query vector. The client then ranks with it via /api/search_by_vector,
    so paging, refine, save, and offline-scan export all work on the uploaded vector
    unchanged -- an uploaded example is just another query vector (video-to-video
    retrieval, the corpus's own modality). Returns {vector, dim, label, n_frames}."""
    _require_ready()
    if not _state.get("model_ready"):
        raise HTTPException(503, "model still loading; try again in a moment")
    raw = list(req.frames_b64) if req.frames_b64 else ([req.image_b64] if req.image_b64 else [])
    if not raw:
        raise HTTPException(400, "no frames provided")
    if len(raw) > _UPLOAD_MAX_FRAMES:
        raise HTTPException(413, f"too many frames ({len(raw)} > {_UPLOAD_MAX_FRAMES})")
    frames: list[bytes] = []
    for f in raw:
        b64 = f.split(",", 1)[-1]  # tolerate a data-URL prefix
        try:
            data = base64.b64decode(b64, validate=False)
        except (ValueError, binascii.Error):
            raise HTTPException(400, "invalid base64 frame data")
        if not data:
            raise HTTPException(400, "empty frame")
        if len(data) > _UPLOAD_MAX_FRAME_BYTES:
            raise HTTPException(413, "a frame is too large (max 8MB each)")
        frames.append(data)
    try:
        vec = search_engine.encode_frames_list(
            frames, _state["processor"], _state["model"], _state["cfg"].device
        )
    except ModuleNotFoundError as exc:  # Pillow absent in the serving image
        raise HTTPException(501, f"image decoding unavailable on this deployment: {exc}")
    except Exception as exc:  # noqa: BLE001 -- surface any decode/encode failure cleanly
        raise HTTPException(400, f"could not encode upload: {type(exc).__name__}: {exc}")
    LOGGER.info("encoded upload %r (%d frames) -> %d-d query vector",
                req.filename, len(frames), len(vec))
    return {"vector": [float(x) for x in vec], "dim": len(vec),
            "label": (req.filename or "uploaded example"), "n_frames": len(frames)}


@app.post("/api/search_by_window")
def search_by_window(req: WindowSearchRequest) -> dict:
    """Search by an example video window (query-by-example over the whole corpus).

    Resolve the mini-segment chunks already embedded for the given drive/segment
    and time window, mean-pool their ORIGINAL embeddings into a query vector,
    then rank the full corpus. The pooling reads the matched clips' 768-d vectors
    from S3 -- a handful of rows -- because the resident screen is quantized and
    PCA-reduced and cannot reconstruct a usable query direction.
    """
    if not req.run_uuid.strip() and not req.segment_id.strip():
        raise HTTPException(400, "provide a run_uuid or segment_id for the query window")
    if not _state.get("model_ready"):
        raise HTTPException(503, "model still loading")
    corpus = _require_full_corpus()
    # UI works in unix nanoseconds (like the result timestamps); the corpus
    # chunk bounds are unix seconds. 0 leaves that side of the window open.
    start_s = int(req.start_ns) // 1_000_000_000 if req.start_ns else 0
    end_s = int(req.end_ns) // 1_000_000_000 if req.end_ns else 0
    try:
        rows = corpus.window_rows(
            run_uuid=req.run_uuid, segment_id=req.segment_id,
            start_unix=start_s, end_unix=end_s,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    if rows.size == 0:
        key = req.run_uuid.strip() or req.segment_id.strip()
        raise HTTPException(404, f"no embedded clips for {key!r} in that window")
    try:
        mat = corpus.vectors_for(rows[:_WINDOW_MAX_POOL])
    except RuntimeError as exc:
        raise HTTPException(409, str(exc))
    pooled = mat.mean(axis=0)
    norm = float(np.linalg.norm(pooled))
    if norm == 0.0:
        raise HTTPException(422, "the window's embeddings cancelled to a zero vector")
    vec = (pooled / norm).astype(np.float32)

    out = _full_rank(corpus, vec, req, label=req.run_uuid.strip() or req.segment_id.strip())
    # Filmstrip preview of the clips the query was pooled from.
    preview = corpus.to_arrow(rows[:12], np.zeros(min(12, rows.size))).to_pydict()
    out["query_clips"] = [
        {
            "chunk_id": preview["chunk_id"][i],
            "run_uuid": preview["run_uuid"][i],
            "segment_id": preview["segment_id"][i],
            "start_timestamp_ns": preview["start_timestamp_ns"][i],
            "end_timestamp_ns": preview["end_timestamp_ns"][i],
            "source_media_uri": preview["source_media_uri"][i],
        }
        for i in range(len(preview["chunk_id"]))
    ]
    out["query_clip_count"] = int(rows.size)
    return out


@app.get("/api/segment_set_prefetch")
def segment_set_prefetch(uuid: str) -> dict:
    """Start the background id-load for a segment set without running a search.

    The frontend calls this the moment a set is selected, so the (cached) DORA
    pull overlaps the user composing their query -- by search time the ids are
    usually already resident, so the first filtered search needs no re-run.

    When the resident corpus carries ``dx_internal_id``, this resolves the set
    via the one-call roaring-bitmap path (synchronous, ~sub-second) instead of
    paginating external_ids; either way the result is reported through ``_SEG``.
    """
    corpus = _resident_corpus()
    if corpus is not None and corpus.has_internal_ids():
        _full_segment_ids(uuid)
    else:
        _segment_ids(uuid)  # idempotent: starts the loader only if not cached/loading
    with _SEG_LOCK:
        rec = _SEG.get(uuid) or {"status": "loading", "count": 0, "err": None}
    return {
        "status": rec["status"],
        "count": rec["count"],
        "ready": rec["status"] == "done",
        "error": rec.get("err"),
    }


@app.get("/api/segment_set_status")
def segment_set_status(uuid: str) -> dict:
    """Background-load status of a segment set's ids (ready / loading / error)."""
    with _SEG_LOCK:
        rec = _SEG.get(uuid) or {"status": "idle", "count": 0, "err": None}
    return {
        "status": rec["status"],
        "count": rec["count"],
        "ready": rec["status"] == "done",
        "error": rec.get("err"),
    }


@app.get("/api/segment_sets")
def segment_sets(name_filter: str = "") -> list[dict]:
    if not name_filter.strip():
        return []
    try:
        sets = dora_client.list_segment_sets(name_filter.strip())
    except dora_client.DoraUnavailable as exc:
        raise HTTPException(503, str(exc))
    return [
        {
            "name": s.name,
            "version": s.version,
            "uuid": s.dataset_uuid,
            "num_segments": s.num_segments,
            "label": s.label(),
        }
        for s in sets[:200]
    ]


@app.get("/api/video")
def video(uri: str) -> RedirectResponse:
    try:
        url = oci_s3.presign_get(uri, _state["s3"], _state["cfg"].presign_ttl_s)
    except (ValueError, botocore.exceptions.ClientError) as exc:
        raise HTTPException(404, f"video unavailable: {exc}")
    return RedirectResponse(url)


def _export_base(label: str) -> str:
    """A single unique stem shared by an export's CSV download, parquet object, and
    (optionally) its DORA segment set, so all three carry the same name.

    ``label`` is the user tag (single export) or a fixed prefix like
    ``config_export``; sanitized + stamped + a short random suffix for uniqueness.
    """
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", label).strip("_") or "export"
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{safe}_{stamp}_{uuid.uuid4().hex[:8]}"


def _create_export_segment_set(
    seg_ids, name: str, *, provenance: dict
) -> tuple[str, str]:
    """Register the exported segments as a DORA dataset. Returns ``(label, error)``.

    ``seg_ids`` is any iterable of segment_id (DORA external_id) -- deduped here.
    Best-effort: a DORA failure is reported via the error string (and logged), never
    raised, so the CSV download is unaffected.
    """
    ids = sorted({s for s in seg_ids if s})
    if not ids:
        return "", "no segments to register"
    # Same ceiling the scan path enforces. Beyond it the set is corpus-sized (a
    # symptom of too low a threshold) and the DORA CreateDataSet call hangs, so
    # report it instead of registering. Only the scan path checked this; an
    # in-app export large enough to hit it would have hung the request.
    if len(ids) > _SEGSET_MAX_SEGMENTS:
        return "", (
            f"not registered: {len(ids):,} segments exceeds "
            f"{_SEGSET_MAX_SEGMENTS:,} cap (raise the threshold or lower k)"
        )
    meta = {k: v for k, v in provenance.items() if v not in (None, "", [])}
    meta["num_segments"] = len(ids)
    try:
        uuid_str, version = dora_client.create_dataset(name, ids, custom_metadata=meta)
    except Exception as exc:  # noqa: BLE001 -- optional add-on; must never fail the
        # export/CSV download. Any failure (DORA unreachable, proto/SDK surprise) is
        # reported back to the UI via the error string instead of 500-ing the export.
        LOGGER.warning(
            "export segment-set create failed (%s): %s", type(exc).__name__, exc,
            exc_info=True,
        )
        return "", f"{type(exc).__name__}: {exc}"
    return f"{uuid_str} v{version} ({len(ids)} segs)", ""


def _write_export_parquet(hits: list, tag: str, name: str | None = None) -> str:
    """Write the exported top-k rows as a parquet under the OCI export prefix.

    Returns the s3:// URI written, or "" if export is disabled or the upload
    fails -- best-effort, so a failed parquet write never blocks the CSV
    download. The columns mirror the CSV exactly. When ``name`` is given the object
    stem is exactly ``name`` (so it matches the CSV download + segment-set name);
    otherwise a name is derived from ``tag``.
    """
    prefix = _state["cfg"].export_s3_prefix.strip()
    if not prefix:
        return ""
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.table(
        {
            "rank": [h.rank for h in hits],
            "score": [float(h.score) for h in hits],
            "segment_id": [h.segment_id for h in hits],
            "chunk_id": [h.chunk_id for h in hits],
            "run_uuid": [h.run_uuid for h in hits],
            "start_timestamp_ns": [_ns(h.chunk_start_unix) for h in hits],
            "end_timestamp_ns": [_ns(h.chunk_end_unix) for h in hits],
            "source_media_uri": [h.source_media_uri for h in hits],
            "tag": [tag for _ in hits],
        }
    )
    sink = io.BytesIO()
    pq.write_table(table, sink)

    stem = name or _export_base(tag)
    key = f"{prefix.rstrip('/')}/{stem}.parquet"
    try:
        oci_s3.put_bytes(key, sink.getvalue(), _state["s3"], "application/octet-stream")
        LOGGER.info("export parquet written: %s (%d rows)", key, len(hits))
        return key
    except (
        ValueError,
        botocore.exceptions.BotoCoreError,
        botocore.exceptions.ClientError,
    ) as exc:
        LOGGER.warning("export parquet write failed (%s): %s", type(exc).__name__, exc)
        return ""


def _interval_rows(intervals: list, corpus, query: str) -> list[dict]:
    """Flatten ScoredIntervals into export dicts, pulling the peak clip's
    chunk_id / segment_id / source_media_uri from the corpus for preview."""
    rows = []
    for rank, iv in enumerate(intervals, start=1):
        i = iv.peak_index
        rows.append(
            {
                "rank": rank,
                "peak_score": round(float(iv.peak_score), 6),
                "mean_score": round(float(iv.mean_score), 6),
                "run_uuid": iv.run_uuid,
                "start_timestamp_ns": _ns_f(iv.start_unix),
                "end_timestamp_ns": _ns_f(iv.end_unix),
                "duration_s": round(float(iv.end_unix - iv.start_unix), 3),
                "num_chunks": int(iv.num_cells),
                "segment_id": corpus.segment_id[i] if i >= 0 else "",
                "chunk_id": corpus.chunk_id[i] if i >= 0 else "",
                "source_media_uri": corpus.source_media_uri[i] if i >= 0 else "",
                "tag": query,
            }
        )
    return rows


_INTERVAL_COLS = [
    "rank",
    "peak_score",
    "mean_score",
    "run_uuid",
    "start_timestamp_ns",
    "end_timestamp_ns",
    "duration_s",
    "num_chunks",
    "segment_id",
    "chunk_id",
    "source_media_uri",
    "tag",
]


def _interval_csv(rows: list[dict]) -> str:
    """CSV for interval export rows (one variable-length interval per row)."""
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(_INTERVAL_COLS)
    for r in rows:
        w.writerow([r[c] for c in _INTERVAL_COLS])
    return buf.getvalue()


def _write_interval_export_parquet(rows: list[dict], name: str) -> str:
    """Write interval export rows as a parquet under the OCI export prefix.

    Best-effort (returns "" on disabled/failure), mirroring _write_export_parquet;
    the object stem is exactly ``name`` so CSV + parquet share one name.
    """
    prefix = _state["cfg"].export_s3_prefix.strip()
    if not prefix:
        return ""
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.table({c: [r[c] for r in rows] for c in _INTERVAL_COLS})
    sink = io.BytesIO()
    pq.write_table(table, sink)
    key = f"{prefix.rstrip('/')}/{name}.parquet"
    try:
        oci_s3.put_bytes(key, sink.getvalue(), _state["s3"], "application/octet-stream")
        LOGGER.info("interval export parquet written: %s (%d rows)", key, len(rows))
        return key
    except (
        ValueError,
        botocore.exceptions.BotoCoreError,
        botocore.exceptions.ClientError,
    ) as exc:
        LOGGER.warning(
            "interval parquet write failed (%s): %s", type(exc).__name__, exc
        )
        return ""


class SaveVectorRequest(BaseModel):
    # Persist the current (refined) search vector under a tag, for reuse on the Curate
    # page. vector omitted -> use the CURRENT vector cached in _state["last"]["vec"]
    # (the relevance-feedback-refined one), which a plain text re-encode can't recover.
    tag: str
    query: str = ""
    vector: list[float] | None = None
    # The active filter set, persisted with the vector so Resume restores the EXACT
    # search context (date range / segment set / lance downsample / vehicle / drive).
    from_date: str | None = None
    to_date: str | None = None
    segment_set_uuid: str | None = None
    filter_lance_uri: str | None = None
    segment_set_name: str | None = None
    filter_lance_uri: str | None = None
    vehicle: str | None = None
    drive_id: str | None = None
    # The 👍/👎 marks labeled during this search, persisted so Resume restores the exact
    # feedback state (each: {chunk_id, segment_id, index, rank, score}). Empty preserves
    # any previously-saved marks (the DB upsert only overwrites when non-empty).
    thumbs_up: list[dict] = []
    thumbs_down: list[dict] = []
    # Export defaults remembered with the vector: top-k and the per-tag cosine threshold
    # (0 = top-k). The Export table pre-fills these for the tag.
    k: int = 0
    threshold: float = 0.0


class SegmentScanRequest(BaseModel):
    # Tags to scan for (the Export page sends its assembled query lines). Each tag's
    # vector is resolved from the DB (db.vectors_for_tags, reusing the saved/refined
    # vector); a tag with no stored vector is encoded with the active model and saved.
    # thresholds is the PER-TAG cosine cutoff ({tag: float}); a tag missing from it
    # falls back to ``default_threshold``. (No single global scan threshold.)
    tags: list[str]
    thresholds: dict[str, float] = {}
    default_threshold: float = 0.3
    from_date: str | None = None
    to_date: str | None = None
    # Active filter set: forwarded into the workflow inputs (nls_launcher) AND persisted on
    # the scan_jobs record. NB: the offline worker currently applies only the date window;
    # segment-set/lance/vehicle/drive ride along in the workflow config but are not yet
    # enforced by the scan (that needs a worker change). Preview/Download honor all of them.
    segment_set_uuid: str | None = None
    filter_lance_uri: str | None = None
    segment_set_name: str | None = None
    filter_lance_uri: str | None = None
    vehicle: str | None = None
    drive_id: str | None = None
    # When true, register the scan's qualifying segments as a DORA segment set once it
    # completes (browsable in Data Explorer). Opt-in, like the export's create_segment_set.
    create_segment_set: bool = False
    # Output is always keyed by segment_id. merge_intervals=True (default) merges contiguous
    # above-threshold clips into variable-length spans per segment; False emits one best
    # (highest-scoring) clip per segment (no interval merge).
    merge_intervals: bool = True
    # Top-K retrieval: when set, return the K highest-scoring distinct segments per tag
    # (ranked by best-clip score) instead of everything above a threshold. Requires a
    # downsample/segment-set scope (the worker resolves it to member chunks and ranks
    # within it); rejected otherwise. null => ordinary threshold scan.
    top_k: int | None = None


@app.post("/api/save_vector")
def save_vector(req: SaveVectorRequest, request: Request) -> dict:
    """Persist the current/refined search vector under a tag (search page "Save vector").

    Stores it in exp-db keyed by tag so the Curate page can launch a scan over it later
    without re-encoding (refined vectors are not recoverable from the query text alone)."""
    tag = req.tag.strip()
    if not tag:
        raise HTTPException(400, "tag is required")
    vec = req.vector
    if not vec:
        last = _state.get("last", {}).get("vec")
        vec = last.tolist() if last is not None else None
    if not vec:
        raise HTTPException(400, "run a search or refine first to define a vector to save")
    cfg = _state["cfg"]
    uri = _state["active_uri"]
    saved = db.insert_export(
        {
            "user_email": _current_user(request),
            "query": req.query.strip() or tag,
            "tag": tag,
            # Remember the chosen export defaults (k + cosine threshold) with the vector; the
            # conservative upsert keeps a prior threshold when this save sends 0/none.
            "k": int(req.k) or 0,
            "threshold": float(req.threshold) if req.threshold else None,
            "num_results": 0,
            "model_uri": cfg.model_artifact_uri,
            "embeddings_uri": uri,
            # Persist the full active filter set so Resume restores the exact context.
            "date_from": req.from_date,
            "date_to": req.to_date,
            "segment_set_uuid": req.segment_set_uuid,
            "segment_set_name": req.segment_set_name,
            "filter_lance_uri": req.filter_lance_uri,
            "vehicle": req.vehicle,
            "drive_id": req.drive_id,
            # The labeled 👍/👎 marks, so Resume restores the feedback state.
            "thumbs_up": req.thumbs_up,
            "thumbs_down": req.thumbs_down,
            "search_vector": [float(x) for x in vec],
            "parquet_uri": "",
        },
        upsert_by_tag=True,
    )
    if not saved:
        raise HTTPException(502, "could not save vector (exp-db unavailable)")
    LOGGER.info("saved search vector under tag %r (dim %d)", tag, len(vec))
    return {"tag": tag, "dim": len(vec)}


def _scan_idem_key(
    req: "SegmentScanRequest", tags: list[str], model_uri: str, scan_uri: str,
    client_key: str = "",
) -> str:
    """Stable dedup key for a segment-scan launch.

    A client-supplied Idempotency-Key wins (lets a Spark job coalesce its own
    retries); otherwise hash the canonicalized, output-determining request fields
    so two identical scans share one workload. Excludes purely cosmetic fields
    (segment_set_name) that don't change the produced Lance."""
    if client_key.strip():
        return hashlib.sha256(("ck:" + client_key.strip()).encode()).hexdigest()
    canon = json.dumps(
        {
            "tags": sorted(tags),
            "thresholds": {k: float(v) for k, v in sorted((req.thresholds or {}).items())},
            "default_threshold": float(req.default_threshold),
            "model_uri": model_uri or "",
            "scan_embeddings_uri": scan_uri or "",
            "filter_lance_uri": req.filter_lance_uri or "",
            "from": req.from_date or "",
            "to": req.to_date or "",
            "segment_set_uuid": req.segment_set_uuid or "",
            "vehicle": req.vehicle or "",
            "drive_id": req.drive_id or "",
            "merge_intervals": bool(req.merge_intervals),
            "register_segset": bool(req.create_segment_set),
            "top_k": int(req.top_k) if req.top_k else 0,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canon.encode()).hexdigest()


class _SingleFlightCall:
    __slots__ = ("done", "result", "exc")

    def __init__(self) -> None:
        self.done = threading.Event()
        self.result = None
        self.exc: BaseException | None = None


_SF_LOCK = threading.Lock()
_SF_CALLS: dict[str, _SingleFlightCall] = {}


def _scan_single_flight(key: str, fn):
    """Coalesce concurrent same-key calls on THIS instance to one execution of ``fn``.

    The first caller (leader) runs ``fn``; the rest wait and share its result or
    exception. This spares the DB a thundering herd of one transaction per executor;
    cross-instance dedup is still enforced by ``db.launch_or_get``. The key is freed
    as soon as the leader finishes, so a later (non-concurrent) identical request
    re-leads and hits the DB dedup (returns the persisted workload id)."""
    with _SF_LOCK:
        call = _SF_CALLS.get(key)
        leader = call is None
        if leader:
            call = _SF_CALLS[key] = _SingleFlightCall()
    if not leader:
        call.done.wait()
        if call.exc is not None:
            raise call.exc
        return call.result
    try:
        call.result = fn()
        return call.result
    except BaseException as exc:  # propagate identically to all waiters
        call.exc = exc
        raise
    finally:
        call.done.set()
        with _SF_LOCK:
            _SF_CALLS.pop(key, None)


@app.post("/api/launch_segment_scan")
def launch_segment_scan(req: SegmentScanRequest, request: Request) -> dict:
    """Launch the per-segment multi-tag scan over an arbitrary set of tags (the Curate
    page's assembled queries). Each tag's vector is reused from exp-db when present
    (db.vectors_for_tags, scoped to the active model) and otherwise encoded + persisted,
    then the full corpus is scored into a per-segment Lance table (one interval column per
    tag). Returns the Lilypad workload id."""
    tags: list[str] = []
    for t in req.tags:
        t = t.strip()
        if t and t not in tags:
            tags.append(t)
    if not tags:
        raise HTTPException(400, "provide at least one non-empty tag")
    # Top-K is only tractable/meaningful scoped to a segment set or lance downsample (the
    # worker resolves that scope to member chunks and ranks within it). Reject otherwise.
    if req.top_k is not None and req.top_k <= 0:
        raise HTTPException(400, "top_k must be a positive integer")
    if req.top_k and not (req.segment_set_uuid or req.filter_lance_uri):
        raise HTTPException(400, "top_k requires a segment set or lance downsample scope")
    if not _offline_scan_enabled():
        raise HTTPException(
            403,
            "the offline scan is retired; /api/full_export selects and writes over "
            "the whole corpus in-process. Set NLS_OFFLINE_SCAN=on only for a match "
            "set too large for it.",
        )
    if not nls_launcher.available():
        raise HTTPException(503, "scan launch unavailable (lilypad SDK / machine creds missing)")
    cfg = _state["cfg"]
    corpus = _require_full_corpus()
    uri = corpus.corpus_uri

    # Server-side launch dedup. Identical concurrent requests -- e.g. a Spark stage
    # firing the same scan from every executor -- must coalesce to ONE Lilypad
    # workload. Prefer a client-supplied Idempotency-Key header (so a job's retries
    # across stages coalesce too), else hash the output-determining request fields.
    # A deterministic scan_id from the key keeps the output Lance path stable.
    client_key = request.headers.get("Idempotency-Key", "")
    idem = _scan_idem_key(req, tags, cfg.model_artifact_uri, cfg.scan_embeddings_uri, client_key)
    scan_id = idem[:32]
    output_dir = f"s3://{cfg.scan_output_bucket}/{cfg.scan_output_prefix.rstrip('/')}"
    user_email = _current_user(request)

    def _launch_and_record() -> tuple[dict, dict]:
        """The ONE real launch: resolve each tag's vector + the downsample id-set and
        submit the workload. Runs once per idem_key (in-process single-flight + DB
        advisory lock); duplicate requests never reach here -- they get the cached id."""
        stored = db.vectors_for_tags(tags, cfg.model_artifact_uri)
        search_vectors: dict[str, list[float]] = {}
        encoded: list[str] = []
        for tag in tags:
            vec = stored.get(tag)
            if vec is not None and len(vec) == corpus.dim:
                search_vectors[tag] = [float(x) for x in vec]
                continue
            if not _state.get("model_ready"):
                raise HTTPException(503, f"model still loading; cannot encode new tag {tag!r}")
            v = search_engine.encode_query(
                tag, _state["processor"], _state["model"], cfg.device
            ).tolist()
            search_vectors[tag] = v
            encoded.append(tag)
            # Persist the freshly-encoded vector under its tag so the next launch reuses
            # it. Upserts on the unique tag key (refreshes the tag's one history row).
            db.insert_export(
                {
                    "user_email": user_email,
                    "query": tag,
                    "tag": tag,
                    "k": 0,
                    "num_results": 0,
                    "model_uri": cfg.model_artifact_uri,
                    "embeddings_uri": uri,
                    "search_vector": v,
                    "parquet_uri": "",
                }
            )
        # Per-tag cosine cutoff: each tag uses its own threshold (default for any unset).
        thresholds = {
            tag: float(req.thresholds.get(tag, req.default_threshold)) for tag in tags
        }
        # Remember each tag's threshold so the Export table pre-fills it next time (like k).
        db.set_tag_thresholds(thresholds)
        # Resolve the lance downsample HERE (the app reads the user's dataset; the
        # worker only ever GETs by exact key). Small id sets travel inline in the
        # workload config; larger ones are written as a parquet next to the scan
        # output and passed by reference -- NO size cap either way.
        filter_key, filter_ids = "", []
        if req.filter_lance_uri:
            try:
                filter_key, filter_ids = _scan_downsample_ids(req.filter_lance_uri.strip())
            except (ValueError, botocore.exceptions.ClientError) as exc:
                raise HTTPException(400, f"could not read downsample dataset: {exc}")
        # Resolve the DX segment set to its external segment_ids so the worker can restrict
        # output to the set (the scan corpus has no dx column; matching is by segment_id).
        seg_set_ids: list[str] = []
        if req.segment_set_uuid:
            try:
                seg_set_ids = sorted(dora_client.fetch_segment_ids(req.segment_set_uuid))
            except dora_client.DoraUnavailable as exc:
                raise HTTPException(400, f"could not resolve segment set: {exc}")
        filter_ids_uri, seg_set_ids_uri = "", ""
        scan_out_root = f"{output_dir}/{scan_id}"
        if len(filter_ids) > _SCAN_INLINE_IDS_MAX:
            filter_ids_uri, filter_ids = (
                _ids_by_reference(filter_ids, scan_out_root, "filter_ids"), []
            )
        if len(seg_set_ids) > _SCAN_INLINE_IDS_MAX:
            seg_set_ids_uri, seg_set_ids = (
                _ids_by_reference(seg_set_ids, scan_out_root, "segment_set_ids"), []
            )
        try:
            res = nls_launcher.launch_segment_scan(
                search_vectors=search_vectors,
                thresholds=thresholds,
                embeddings_uri=cfg.scan_embeddings_uri,
                output_dir=output_dir,
                scan_id=scan_id,
                start_date=req.from_date or cfg.scan_default_from_date,
                end_date=req.to_date or "",
                segment_set_uuid=req.segment_set_uuid or "",
                segment_set_name=req.segment_set_name or "",
                filter_lance_uri=req.filter_lance_uri or "",
                filter_key=filter_key,
                filter_segment_ids=filter_ids,
                filter_ids_uri=filter_ids_uri,
                segment_set_ids=seg_set_ids,
                segment_set_ids_uri=seg_set_ids_uri,
                vehicle=req.vehicle or "",
                drive_id=req.drive_id or "",
                merge_intervals=bool(req.merge_intervals),
                top_k=req.top_k,
            )
        except nls_launcher.LauncherUnavailable as exc:
            # Logged as well as returned: the HTTPException body reaches the
            # browser but never the service logs, so a launch failure was only
            # diagnosable by asking the user to read their screen.
            LOGGER.error("segment scan launch failed: %s", exc, exc_info=True)
            raise HTTPException(502, f"could not launch segment scan: {exc}")
        LOGGER.info(
            "launched per-segment scan %s for %d tags (%d reused, %d encoded)",
            res.get("execution_id"), len(tags), len(tags) - len(encoded), len(encoded),
        )
        # Segment-set name (used iff create_segment_set): readable, scan-unique.
        segset_name = (
            f"nls-scan-{tags[0]}-{scan_id[:8]}" if len(tags) == 1
            else f"nls-scan-{scan_id[:8]}"
        )
        record = {
            "execution_id": res.get("execution_id"),
            "user_email": user_email,
            "tags": tags,
            "thresholds": thresholds,
            "output_dir": output_dir,
            "lance_uri": res.get("lance_uri") or "",
            "console_url": res.get("url") or "",
            "status": "LAUNCHED",
            "register_segset": bool(req.create_segment_set),
            "segset_name": segset_name,
            # Full filter set the scan was launched with (date is applied by the worker
            # today; the rest are recorded for provenance / future enforcement).
            "filters": {
                "from_date": req.from_date or cfg.scan_default_from_date,
                "to_date": req.to_date or "",
                "segment_set_uuid": req.segment_set_uuid or "",
                "segment_set_name": req.segment_set_name or "",
                "filter_lance_uri": req.filter_lance_uri or "",
                "vehicle": req.vehicle or "",
                "drive_id": req.drive_id or "",
                "merge_intervals": bool(req.merge_intervals),
                # Top-K vs threshold mode, so Recent scans can show which ranking the
                # scan actually ran (the per-tag thresholds are recorded either way, but
                # in Top-K mode the worker ignores them and ranks within the scope).
                "top_k": int(req.top_k) if req.top_k else None,
            },
        }
        res["tags"] = tags
        res["encoded"] = encoded
        res["thresholds"] = thresholds
        return res, record

    # In-process single-flight collapses the up-to-80 concurrent same-key requests on
    # THIS instance to one db.launch_or_get; db.launch_or_get is the cross-instance
    # authority (advisory lock on the key, then insert keyed by idem_key).
    res, deduplicated = _scan_single_flight(
        idem, lambda: db.launch_or_get(idem, _launch_and_record)
    )
    if deduplicated:
        # A dedup hit must point at a LIVE (or succeeded) workload. The stored status
        # can lag (it only advances when something polls), so check the live phase:
        # if the existing execution actually FAILED, release its key and relaunch --
        # otherwise an identical retry of a failed scan returns the dead workload
        # (and a lance_uri that will never exist).
        try:
            live = nls_launcher.scan_status(res.get("execution_id") or "")
        except nls_launcher.LauncherUnavailable:
            live = {}
        if live.get("done") and (live.get("phase") or "") != "SUCCEEDED":
            db.update_scan_job(
                res.get("execution_id") or "", live.get("phase") or "", live.get("error") or ""
            )
            db.release_scan_idem(idem)
            LOGGER.info(
                "dedup hit on FAILED workload %s (idem=%s); relaunching fresh",
                res.get("execution_id"), scan_id,
            )
            res, deduplicated = db.launch_or_get(idem, _launch_and_record)
    res["deduplicated"] = deduplicated
    res.setdefault("workload_id", res.get("execution_id"))
    # A dedup hit returns only the existing workload id; the output Lance path is
    # deterministic (output_dir/scan_id/segments.lance -- scan_id derives from the
    # idem key), so reconstruct it here. This keeps EVERY response (fresh or deduped)
    # carrying lance_uri, so a blocking caller can read the same output Lance either way.
    if not res.get("lance_uri"):
        res["lance_uri"] = f"{output_dir}/{scan_id}/segments.lance"
    if deduplicated:
        LOGGER.info(
            "deduplicated scan launch (idem=%s) -> existing workload %s",
            scan_id, res.get("execution_id"),
        )
    return res


def _maybe_register_scan_segset(execution: str, *, done: bool, phase: str) -> dict:
    """If a COMPLETED scan was flagged to register a DORA segment set and hasn't yet,
    read its output Lance ``segment_id``s and register the set (exactly once). Returns
    ``{segset_uuid, segset_label}`` (may be empty). Best-effort -- any failure is recorded
    on the job and never raised into the caller (the scan + its status are unaffected)."""
    if not (done and phase == "SUCCEEDED"):
        return {}
    job = db.get_scan_job(execution)
    if not job or not job.get("register_segset"):
        return {}
    if job.get("segset_uuid"):  # already registered
        return {"segset_uuid": job["segset_uuid"], "segset_label": job.get("segset_label") or ""}
    lance_uri = job.get("lance_uri") or ""
    name = job.get("segset_name") or f"nls-scan-{execution}"
    if not lance_uri:
        return {}
    # Cheap pre-check: the manifest already knows the segment count, so a corpus-sized
    # result is skipped WITHOUT reading its (multi-million-row) lance at all.
    manifest = _read_scan_manifest(lance_uri) or {}
    n_manifest = manifest.get("num_segments")
    if isinstance(n_manifest, int) and n_manifest > _SEGSET_MAX_SEGMENTS:
        note = f"not registered: {n_manifest:,} segments exceeds {_SEGSET_MAX_SEGMENTS:,} cap (raise the tag's threshold)"
        db.set_scan_segset(execution, "", note)
        LOGGER.warning("scan segset: %s -> %s", execution, note)
        return {"segset_label": note}
    try:
        import lance

        tbl = lance.dataset(
            lance_uri, storage_options=oci_s3.lance_storage_options()
        ).to_table(columns=["segment_id"])
        seg_ids = sorted({s for s in tbl.column("segment_id").to_pylist() if s})
    except Exception as exc:  # noqa: BLE001 -- best-effort add-on
        LOGGER.warning(
            "scan segset: reading %s failed (%s): %s", lance_uri, type(exc).__name__, exc
        )
        return {"segset_label": f"read failed: {type(exc).__name__}"}
    if not seg_ids:
        db.set_scan_segset(execution, "", "no qualifying segments")
        return {"segset_label": "no qualifying segments"}
    # Guardrail: a corpus-sized result (e.g. a tag whose threshold is far too low) would
    # try to push millions of ids into DORA CreateDataSet -- which hangs and, since the
    # refresher is single-flight, jams every other pending registration behind it. Such a
    # set is unusable anyway; skip it, record why, and stop it re-attempting on every poll.
    if len(seg_ids) > _SEGSET_MAX_SEGMENTS:
        note = f"not registered: {len(seg_ids):,} segments exceeds {_SEGSET_MAX_SEGMENTS:,} cap (raise the tag's threshold)"
        db.set_scan_segset(execution, "", note)
        LOGGER.warning("scan segset: %s -> %s", execution, note)
        return {"segset_label": note}
    label, err = _create_export_segment_set(
        seg_ids,
        name,
        provenance={
            "source": "nls-segment-scan",
            "execution_id": execution,
            "tags": job.get("tags") or [],
        },
    )
    if err:
        return {"segset_label": f"register failed: {err}"}
    uuid_str = label.split()[0] if label else ""
    db.set_scan_segset(execution, uuid_str, label)
    LOGGER.info("scan %s registered DORA segment set %s (%s)", execution, uuid_str, name)
    return {"segset_uuid": uuid_str, "segset_label": label}


@app.get("/api/scan_status")
def scan_status(execution: str) -> dict:
    """Phase of a launched scan workload (for the UI to poll); {done, phase, error}.
    Also refreshes the stored scan_jobs status, and -- when the scan has SUCCEEDED and was
    flagged to register a DORA segment set -- registers it once (best-effort)."""
    if not nls_launcher.available():
        raise HTTPException(503, "scan status unavailable")
    try:
        status = nls_launcher.scan_status(execution)
    except nls_launcher.LauncherUnavailable as exc:
        raise HTTPException(502, f"could not fetch scan status: {exc}")
    db.update_scan_job(execution, status.get("phase") or "", status.get("error") or "")
    segset = _maybe_register_scan_segset(
        execution, done=bool(status.get("done")), phase=status.get("phase") or ""
    )
    status.update(segset)
    return status


def _scan_status_terminal(status: str) -> bool:
    """True iff a scan's stored status is a final phase (no more polling needed).
    A launched scan progresses LAUNCHED -> RUNNING -> SUCCEEDED/FAILED/ABORTED; only
    the last group is terminal (Lilypad reports failure as EXPERIMENT_FAILED)."""
    up = (status or "").upper()
    return any(t in up for t in ("SUCCEEDED", "FAILED", "ABORTED", "COMPLETED"))


# Single-flight background refresher for the scans list. All the slow external work --
# Lilypad status for in-flight jobs, manifest-count back-fill (OCI), and pending DORA
# segment-set registration (reads the whole output lance; can take minutes per row) --
# runs HERE, off the request path, and persists its results to the scan_jobs rows. The
# GET endpoint is then always a pure DB read that just kicks this and reports whether a
# refresh is in flight, so the panel converges to live truth without ever blocking.
_SCAN_REFRESH_LOCK = threading.Lock()
_SCAN_REFRESH = {"running": False, "last": 0.0}
_SCAN_REFRESH_MIN_INTERVAL_S = 15.0


def _row_needs_refresh(row: dict) -> bool:
    """True if a scan row has anything a background refresh could advance: a non-final
    status, missing result counts, or a pending segment-set registration."""
    # In-app exports are already final when written: there is no workload to poll,
    # no manifest.json beside the artifact, and the artifact is a parquet rather
    # than a segments.lance. Feeding one to this worker made it retry a segment-set
    # registration against interval rows on every poll, forever.
    if (row.get("source") or "lilypad") == "app":
        return False
    status = (row.get("status") or "").upper()
    if not _scan_status_terminal(status):
        return True
    if "SUCCEEDED" not in status:
        return False
    if row.get("counts") is None and row.get("lance_uri"):
        return True
    # Segment-set registration is settled once it has an outcome -- a uuid (registered) OR
    # a recorded label (skipped as too large / no qualifying segments). Only a row with
    # neither still needs a registration attempt; this stops a skipped set from re-reading
    # its (huge) lance on every poll.
    return (
        bool(row.get("register_segset"))
        and not (row.get("segset_uuid") or "")
        and not (row.get("segset_label") or "")
    )


def _kick_scan_refresh(rows: list[dict], force: bool = False) -> bool:
    """Start the background refresh for ``rows`` unless one is already running (or ran
    very recently, unless forced). Returns True iff a refresh is now in flight."""
    stale = [r for r in rows if _row_needs_refresh(r)]
    if not stale:
        return False
    now = time.time()
    with _SCAN_REFRESH_LOCK:
        if _SCAN_REFRESH["running"]:
            return True
        if not force and now - _SCAN_REFRESH["last"] < _SCAN_REFRESH_MIN_INTERVAL_S:
            return False
        _SCAN_REFRESH["running"] = True
    threading.Thread(
        target=_refresh_scan_rows, args=(stale,), daemon=True, name="scan-refresh"
    ).start()
    return True


def _refresh_scan_rows(rows: list[dict]) -> None:
    """Background worker: bring each stale scan row up to live truth and persist it.

    Three passes, fast-to-slow, so the quick wins land first: (1) ALL statuses (bounded
    gRPC, ~seconds total) -- the panel's repoll picks these up in real time; (2) result
    counts (one bounded OCI manifest read per scan, cached); (3) pending segment-set
    registrations LAST (each reads the scan's whole output lance -- minutes per row) so
    they can never delay a status update. Failures are logged and skipped so one bad row
    never blocks the rest."""
    launcher_up = nls_launcher.available()
    rows = [r for r in rows if r.get("execution_id")]
    try:
        # Pass 1: statuses (fast, bounded).
        n_status = 0
        for row in rows:
            status = row.get("status") or ""
            if not launcher_up or _scan_status_terminal(status):
                continue
            try:
                live = nls_launcher.scan_status(row["execution_id"], timeout=8.0)
                phase = (live.get("phase") or "").strip()
                if phase and phase != status:
                    db.update_scan_job(row["execution_id"], phase, live.get("error") or "")
                    row["status"] = phase
                    n_status += 1
            except nls_launcher.LauncherUnavailable as exc:
                LOGGER.info("scan refresh: %s status unavailable: %s", row["execution_id"], exc)
        LOGGER.info("scan refresh: %d/%d statuses advanced", n_status, len(rows))
        # Pass 2: result counts (bounded manifest read, cached incl. negative sentinel).
        for row in rows:
            if (
                "SUCCEEDED" in (row.get("status") or "").upper()
                and row.get("counts") is None
                and row.get("lance_uri")
            ):
                _scan_counts(row["execution_id"], row["status"], row["lance_uri"], None)
        # Pass 3: pending segment-set registrations (slow; strictly last).
        for row in rows:
            if (
                "SUCCEEDED" in (row.get("status") or "").upper()
                and row.get("register_segset")
                and not (row.get("segset_uuid") or "")
            ):
                LOGGER.info("scan refresh: registering segment set for %s", row["execution_id"])
                _maybe_register_scan_segset(row["execution_id"], done=True, phase="SUCCEEDED")
    finally:
        with _SCAN_REFRESH_LOCK:
            _SCAN_REFRESH["running"] = False
            _SCAN_REFRESH["last"] = time.time()


@app.get("/api/scans")
def scans(limit: int = 50, live: bool = False) -> dict:
    """Recent launched per-segment scans (newest first) for the Export tab's panel; each
    entry: execution_id, status, tags, thresholds, counts, console_url, output lance,
    segment-set info, timestamps.

    Always a PURE DB read -- the response never blocks on Lilypad/DORA/OCI. Stale rows
    (in-flight status, missing counts, pending segment-set) are refreshed by a throttled
    single-flight background worker that this endpoint kicks; ``refreshing`` in the
    response tells the client to re-poll shortly to pick up the refreshed rows.
    ``live=1`` (the panel's Reload button) forces the kick past the throttle."""
    rows = db.list_scan_jobs(limit=limit)
    refreshing = _kick_scan_refresh(rows, force=bool(live))
    jobs = [
        {
            "execution_id": row.get("execution_id"),
            "status": row.get("status") or "",
            "error": row.get("error") or "",
            "tags": row.get("tags") or [],
            "thresholds": row.get("thresholds") or {},
            "counts": row.get("counts") or None,
            "output_dir": row.get("output_dir") or "",
            "lance_uri": row.get("lance_uri") or "",
            "console_url": row.get("console_url") or "",
            # "lilypad" (offline launch) vs "app" (in-app full-corpus export).
            "source": row.get("source") or "lilypad",
            "register_segset": bool(row.get("register_segset")),
            "segset_uuid": row.get("segset_uuid") or "",
            "segset_label": row.get("segset_label") or "",
            # The filter set the scan was launched with (date/segment-set/lance/vehicle/
            # drive), so the Recent scans panel can show exactly what scope it ran over.
            "filters": row.get("filters") or {},
            "created_at": _fmt_ts(row.get("created_at")),
            "updated_at": _fmt_ts(row.get("updated_at")),
        }
        for row in rows
    ]
    return {"jobs": jobs, "refreshing": refreshing}


@app.get("/api/export_file")
def export_file(uri: str) -> RedirectResponse:
    """Presigned GET for an export parquet, restricted to the export prefix."""
    prefix = _state["cfg"].export_s3_prefix.strip().rstrip("/")
    if not prefix or not uri.startswith(prefix + "/"):
        raise HTTPException(404, "not an export artifact")
    try:
        url = oci_s3.presign_get(uri, _state["s3"], _state["cfg"].presign_ttl_s)
    except (ValueError, botocore.exceptions.ClientError) as exc:
        raise HTTPException(404, f"parquet unavailable: {exc}")
    return RedirectResponse(url)


def _score_histogram(scores: np.ndarray, tau: float | None, bins: int = 50) -> dict | None:
    """Histogram of per-clip similarity over the whole corpus, optionally marking
    a threshold ``tau``. Returns None when there are no finite scores."""
    finite = scores[np.isfinite(scores)]
    if not finite.size:
        return None
    counts, edges = np.histogram(finite, bins=bins)
    return {
        "edges": [round(float(e), 4) for e in edges.tolist()],
        "counts": [int(c) for c in counts.tolist()],
        "total": int(finite.size),
        "min": round(float(finite.min()), 4),
        "max": round(float(finite.max()), 4),
        "mean": round(float(finite.mean()), 4),
        "tau": (round(float(tau), 6) if tau is not None else None),
        "above_tau": (int(np.count_nonzero(finite >= tau)) if tau is not None else None),
    }


@app.post("/api/score_distribution")
def score_distribution(req: ExportRequest, request: Request) -> dict:
    """Similarity-score histogram across the entire corpus for the current query,
    INDEPENDENT of interval preview/export. Marks the interval threshold tau the
    current mode/k/cutoff would produce, so it doubles as a threshold picker."""
    if not _state.get("model_ready"):
        raise HTTPException(503, "model still loading")
    corpus = _require_full_corpus()
    vec = search_engine.encode_query(
        req.query.strip(), _state["processor"], _state["model"], _state["cfg"].device
    )
    scores, err = corpus.score(vec)
    mask = corpus.filter_mask(**_full_filters(req))
    allowed = mask if mask is not None else np.ones(corpus.num_rows, dtype=bool)
    mode = "score" if req.interval_mode == "score" else "k"
    if mode == "score":
        tau = float(req.interval_score or 0.0)
    else:
        # k-th best score among the rows that passed the filters -- the cutoff a
        # top-k export would use, which is what makes this a threshold picker.
        sub = scores[allowed]
        k = max(1, min(int(req.k), int(sub.size)))
        tau = float(np.partition(sub, sub.size - k)[sub.size - k]) if sub.size else 0.0
    dist = _score_histogram(scores, tau)
    if dist is None:
        raise HTTPException(400, "no finite scores to summarize")
    dist["mode"] = mode
    dist["score_kind"] = "bounded_approx"
    dist["score_error_bound"] = round(float(err), 4)
    dist["num_rows_searched"] = corpus.num_rows
    dist["candidates"] = int(allowed.sum())
    return dist


# --- learned threshold policy: cached weights per embedding space -----------
# Serving predicts the suggested tau with a dot product over cached weights (no
# fit on the request path). Fitting happens in a background thread as episodes
# accumulate, or via POST /api/refit_policy. Falls back to the heuristic when no
# policy is fitted yet (< ~20 labeled tunes for the corpus).
_policy_cache: dict[str, tuple[dict | None, float]] = {}
_policy_lock = threading.Lock()
_POLICY_TTL_S = 300.0
_policy_episode_count = 0


def _load_policy(uri: str) -> dict | None:
    now = time.time()
    with _policy_lock:
        ent = _policy_cache.get(uri or "")
        if ent and (now - ent[1]) < _POLICY_TTL_S:
            return ent[0]
    pol = db.get_threshold_policy(uri or "")
    with _policy_lock:
        _policy_cache[uri or ""] = (pol, now)
    return pol


def _suggested_tau(uri: str, stats: dict) -> float:
    """Learned policy prediction when one is fitted for this corpus, else heuristic."""
    pol = _load_policy(uri)
    if pol:
        return search_engine.predict_threshold(stats, pol)
    return search_engine.heuristic_threshold(stats)


def _refit_policy(uri: str) -> dict | None:
    """Fit + persist the ridge policy for one embedding space from its episodes."""
    eps = [e for e in db.threshold_episodes() if (e.get("embeddings_uri") or "") == (uri or "")]
    pol = search_engine.fit_threshold_policy(eps)
    if pol:
        pol["embeddings_uri"] = uri or ""
        db.upsert_threshold_policy(pol)
        with _policy_lock:
            _policy_cache[uri or ""] = (pol, time.time())
    return pol


def _maybe_refit_policy_async(uri: str) -> None:
    """Every few logged episodes, refit in the background (off the request path)."""
    global _policy_episode_count
    _policy_episode_count += 1
    if _policy_episode_count % 5 != 0:
        return
    threading.Thread(target=_refit_policy, args=(uri,), daemon=True).start()


@app.post("/api/refit_policy")
def refit_policy(request: Request) -> dict:
    """Fit + persist the learned threshold policy for the active corpus now.
    Returns the fit summary (or a note when there aren't enough episodes yet)."""
    _require_ready()
    uri = _state["active_uri"]
    pol = _refit_policy(uri)
    if not pol:
        return {"fitted": False, "embeddings_uri": uri,
                "note": "Not enough labeled episodes yet for this corpus (need ~20). Heuristic stays live."}
    return {"fitted": True, "embeddings_uri": uri, "feature_names": pol["feature_names"],
            "weights": pol["weights"], "n_episodes": pol["n_episodes"],
            "mae_policy": pol["mae_policy"], "mae_heuristic": pol["mae_heuristic"]}


@app.post("/api/threshold_search")
def threshold_search(req: ThresholdSearchRequest, request: Request) -> dict:
    """Retired. This ranked the 2M resident corpus, which no longer exists.

    Kept as an explicit 410 rather than deleted: a 404 tells an external caller
    nothing, and quietly serving the full corpus from here would change the
    result set under a client that never asked for it.
    """
    raise _gone("/api/full_threshold")


@app.post("/api/export")
def export(req: ExportRequest, request: Request) -> Response:
    """Retired. This ranked the 2M resident corpus, which no longer exists.

    Kept as an explicit 410 rather than deleted: a 404 tells an external caller
    nothing, and quietly serving the full corpus from here would change the
    result set under a client that never asked for it.
    """
    raise _gone("/api/full_export")


@app.post("/api/export_config")
def export_config(req: ConfigExportRequest, request: Request) -> Response:
    """Batch export: run several text queries, each top-k WITHIN the active filters,
    concatenate (optionally dedupe by chunk_id -- highest score wins), and return one
    CSV + write one parquet to S3 -- the same artifact flow as ``/api/export``.

    Per query, the search vector is reused from exp-db when a prior export stored it
    for this exact ``(query, corpus, model)``; otherwise it is encoded and persisted
    so the next run reuses it (and the query shows up in Search history).
    """
    if not _state.get("model_ready"):
        raise HTTPException(503, "model still loading")
    queries = [q for q in req.queries if q.query.strip()]
    if not queries:
        raise HTTPException(400, "no queries in config")
    corpus = _require_full_corpus()
    uri = corpus.corpus_uri
    # One filter set (date + vehicle + drive + segment set + lance) for every query.
    filters = _full_filters(req)
    model_uri = _state["cfg"].model_artifact_uri

    rows: list = []  # (query, Hit) in config order, per-query rank 1..k
    new_encodes = 0
    saved_any = False
    for cq in queries:
        q = cq.query.strip()
        vec, reused = _query_vector(q, uri, model_uri, corpus)
        # Threshold when given (clips scoring >= it, capped at k as a safety max),
        # else pure top-k. Selecting top-k first and trimming is the same set: the
        # ranking is score-descending, so the above-threshold clips are its prefix.
        hits, _cand = corpus.search(vec, limit=max(1, int(cq.k)), **filters)
        if cq.threshold and cq.threshold > 0:
            hits = [h for h in hits if float(h.score) >= float(cq.threshold)]
        LOGGER.info(
            "config export: %r -> %d hits (k=%d threshold=%s, vector %s)",
            q, len(hits), int(cq.k), cq.threshold or "-", "reused" if reused else "encoded",
        )
        rows.extend((q, h) for h in hits)
        if not reused:
            # New query -> save it under tag = the query so it joins Search history and is
            # reused next run. Upserts on the unique tag key (refreshes the tag's one row).
            # Reused vectors are read-only (no write here).
            new_encodes += 1
            saved_any = (
                db.insert_export(
                    {
                        "user_email": _current_user(request),
                        "query": q,
                        "tag": q,
                        "k": int(cq.k),
                        "threshold": float(cq.threshold) if cq.threshold else None,
                        "num_results": len(hits),
                        "model_uri": model_uri,
                        "embeddings_uri": uri,
                        "segment_set_uuid": req.segment_set_uuid,
                        "segment_set_name": req.segment_set_name,
                        "date_from": req.from_date,
                        "date_to": req.to_date,
                        "filter_lance_uri": req.filter_lance_uri,
                        "vehicle": req.vehicle,
                        "drive_id": req.drive_id,
                        "thumbs_up": [],
                        "thumbs_down": [],
                        "search_vector": vec.tolist(),
                        "parquet_uri": "",
                    }
                )
                or saved_any
            )

    if req.dedupe:
        best: dict[str, tuple] = {}
        for q, h in rows:
            cur = best.get(h.chunk_id)
            if cur is None or h.score > cur[1].score:
                best[h.chunk_id] = (q, h)
        seen: set[str] = set()
        deduped: list = []
        for _q, h in rows:
            if h.chunk_id in seen:
                continue
            seen.add(h.chunk_id)
            deduped.append(best[h.chunk_id])  # highest-scoring occurrence
        rows = deduped

    if req.dedupe_segment:
        # Collapse to the best (highest-scoring) clip per segment_id, keeping rows that have
        # no segment_id (can't be merged). First occurrence preserves the existing order.
        best_seg: dict[str, tuple] = {}
        for q, h in rows:
            sid = (h.segment_id or "").strip()
            if not sid:
                continue
            cur = best_seg.get(sid)
            if cur is None or h.score > cur[1].score:
                best_seg[sid] = (q, h)
        seen_seg: set[str] = set()
        deduped_seg: list = []
        for q, h in rows:
            sid = (h.segment_id or "").strip()
            if not sid:
                deduped_seg.append((q, h))  # no segment_id -> keep as-is
                continue
            if sid in seen_seg:
                continue
            seen_seg.add(sid)
            deduped_seg.append(best_seg[sid])  # highest-scoring clip for this segment
        rows = deduped_seg

    csv_text = _config_csv(rows)

    base = _export_base("config_export")
    parquet_uri = _write_config_export_parquet(rows, name=base)

    segset_label, segset_error = "", ""
    if req.create_segment_set:
        segset_label, segset_error = _create_export_segment_set(
            (h.segment_id for _, h in rows),
            base,
            provenance={
                "source": "nls_search_app",
                "queries": sorted({q for q, _ in rows}),
                "embeddings_uri": uri,
                "model_uri": model_uri,
                "segment_set_uuid": req.segment_set_uuid,
                "filter_lance_uri": req.filter_lance_uri,
                "date_from": req.from_date,
                "date_to": req.to_date,
            },
        )

    # "saved" == every query's vector is in exp-db: any new insert succeeded, or
    # there were no new encodes (all reused -> already persisted).
    saved = saved_any or new_encodes == 0
    return Response(
        content=csv_text,
        media_type="text/csv",
        headers={
            "Content-Disposition": f'attachment; filename="{base}.csv"',
            "X-NLS-Saved": "1" if saved else "0",
            "X-NLS-Parquet": parquet_uri,
            "X-NLS-Export-Name": base,
            "X-NLS-Segset": segset_label,
            "X-NLS-Segset-Error": segset_error,
        },
    )


@app.post("/api/curate_preview")
def curate_preview(req: ConfigExportRequest, request: Request) -> dict:
    """Run the config queries and return the per-query top-k as JSON for review -- no
    CSV/parquet/segment-set writes. The pure search half of ``/api/export_config``;
    the client applies the dedupe toggle + select/deselect live, then commits via
    ``/api/curate_export``.

    New-query vectors are persisted (insert-if-absent, tag=query) exactly as
    export_config does, so previewed queries are reusable and show in Search history.
    """
    if not _state.get("model_ready"):
        raise HTTPException(503, "model still loading")
    queries = [q for q in req.queries if q.query.strip()]
    if not queries:
        raise HTTPException(400, "no queries in config")
    corpus = _require_full_corpus()
    uri = corpus.corpus_uri
    filters = _full_filters(req)
    model_uri = _state["cfg"].model_artifact_uri

    per_query: list[dict] = []
    for cq in queries:
        q = cq.query.strip()
        vec, reused = _query_vector(q, uri, model_uri, corpus)
        hits, _cand = corpus.search(vec, limit=max(1, int(cq.k)), **filters)
        per_query.append(
            {
                "query": q,
                "k": int(cq.k),
                "num_hits": len(hits),
                "hits": [_hit_dict(h) for h in hits],
            }
        )
        if not reused:
            # Mirror export_config: persist new queries once (read-only reuse next
            # time), so a preview run also seeds Search history.
            db.insert_export(
                {
                    "user_email": _current_user(request),
                    "query": q,
                    "tag": q,
                    "k": int(cq.k),
                    "num_results": len(hits),
                    "model_uri": model_uri,
                    "embeddings_uri": uri,
                    "segment_set_uuid": req.segment_set_uuid,
                    "segment_set_name": req.segment_set_name,
                    "date_from": req.from_date,
                    "date_to": req.to_date,
                    "thumbs_up": [],
                    "thumbs_down": [],
                    "search_vector": vec.tolist(),
                    "parquet_uri": "",
                }
            )
    return {"per_query": per_query, "model_uri": model_uri}


@app.post("/api/curate_export")
def curate_export(req: CurateExportRequest, request: Request) -> Response:
    """Export an explicit, user-curated set of rows (CSV + parquet + optional DX
    segment set), built from exactly ``req.rows`` -- the rows the user kept in the
    Curate-from-config preview. Same artifacts/headers as ``/api/export_config``."""
    _require_ready()
    if not req.rows:
        raise HTTPException(400, "no rows selected to export")
    uri = _state["active_uri"]

    def _sec(ns: int | None) -> int | None:
        return int(ns) // 1_000_000_000 if ns is not None else None

    # Rebuild (query, RankedHit) tuples so the shared CSV/parquet writers apply
    # unchanged. index is export-irrelevant (set 0).
    rows = [
        (
            r.query,
            search_engine.RankedHit(
                rank=int(r.rank),
                index=0,
                chunk_id=r.chunk_id,
                run_uuid=r.run_uuid,
                chunk_start_unix=_sec(r.start_timestamp_ns),
                source_media_uri=r.source_media_uri,
                segment_id=r.segment_id,
                score=float(r.score),
                chunk_end_unix=_sec(r.end_timestamp_ns),
            ),
        )
        for r in req.rows
    ]

    csv_text = _config_csv(rows)
    base = _export_base("curate")
    parquet_uri = _write_config_export_parquet(rows, name=base)

    segset_label, segset_error = "", ""
    if req.create_segment_set:
        segset_label, segset_error = _create_export_segment_set(
            (h.segment_id for _, h in rows),
            base,
            provenance={
                "source": "nls_search_app",
                "queries": sorted({q for q, _ in rows}),
                "embeddings_uri": uri,
                "model_uri": _state["cfg"].model_artifact_uri,
                "segment_set_uuid": req.segment_set_uuid,
                "filter_lance_uri": req.filter_lance_uri,
                "date_from": req.from_date,
                "date_to": req.to_date,
            },
        )

    return Response(
        content=csv_text,
        media_type="text/csv",
        headers={
            "Content-Disposition": f'attachment; filename="{base}.csv"',
            "X-NLS-Parquet": parquet_uri,
            "X-NLS-Export-Name": base,
            "X-NLS-Segset": segset_label,
            "X-NLS-Segset-Error": segset_error,
        },
    )


app.mount("/static", StaticFiles(directory=os.path.join(HERE, "web")), name="static")


@app.get("/")
def index(request: Request) -> FileResponse:
    # Record a visit for usage analytics, but only for real (IAP-authenticated)
    # users -- skip health-probe / unauthenticated hits so the counts are real.
    user = _current_user(request)
    if user and user != "local":
        analytics.record_visit(user, uuid.uuid4().hex, time.time())
    return FileResponse(os.path.join(HERE, "web", "index.html"))


@app.get("/curate")
def curate_page() -> FileResponse:
    """Serve the SPA shell so a refresh / deep-link at /curate boots into the
    Export view (history picker + preview/download + scan; client router shows it)."""
    return FileResponse(os.path.join(HERE, "web", "index.html"))


def _fmt_ts(value) -> str:
    """Render a timestamp/datetime cell compactly (UTC)."""
    if value is None:
        return ""
    try:
        return value.strftime("%Y-%m-%d %H:%M")  # datetime from the driver
    except AttributeError:
        return html.escape(str(value))


def _parquet_cell(uri) -> str:
    """Render the parquet path as a link to its presigned download (or '-')."""
    s = "" if uri is None else str(uri).strip()
    if not s:
        return "<span class=muted>&mdash;</span>"
    href = "/api/export_file?uri=" + urllib.parse.quote(s, safe="")
    esc = html.escape(s)
    return f'<a class=mono href="{href}" title="{esc}">{esc}</a>'


# Shared dark-theme styling + top nav for the server-rendered pages (/analytics,
# /tags). Kept as plain strings (not f-strings) so CSS braces need no escaping.
_PAGE_CSS = """<style>
  :root { color-scheme: dark; }
  body { background:#0f1115; color:#e6e8ec; font:14px/1.5 'IBM Plex Sans',system-ui,sans-serif; margin:0; padding:0 32px 40px; }
  a { color:#c8ff4d; }
  h1 { font-size:22px; margin:0 0 4px; }
  h2 { font-size:15px; text-transform:uppercase; letter-spacing:.08em; color:#9aa0a6; margin:32px 0 10px; }
  .sub { color:#9aa0a6; margin:0 0 8px; }
  .warn { color:#ffd166; }
  table { border-collapse:collapse; width:100%; font-size:13px; table-layout:fixed; }
  th,td { text-align:left; padding:7px 10px; border-bottom:1px solid #232733; vertical-align:top;
    overflow-wrap:anywhere; word-break:break-word; }
  th { color:#9aa0a6; font-weight:600; position:sticky; top:0; background:#0f1115; }
  tr:hover td { background:#161a22; }
  td.num { text-align:right; font-variant-numeric:tabular-nums; }
  td.mono,.mono { font-family:'IBM Plex Mono',ui-monospace,monospace; color:#9aa0a6; font-size:12px;
    word-break:break-all; }
  td.q { width:22%; min-width:200px; }
  td.tag,.tagname { color:#c8ff4d; font-weight:600; }
  td.name { font-weight:600; }
  td.empty { color:#9aa0a6; font-style:italic; }
  .muted { color:#5f6672; }
  .cards { display:flex; gap:16px; margin:8px 0 4px; flex-wrap:wrap; }
  .card { background:#161a22; border:1px solid #232733; border-radius:10px; padding:14px 18px; min-width:130px; }
  .card .n { font-size:26px; font-weight:700; color:#c8ff4d; font-variant-numeric:tabular-nums; }
  .card .l { font-size:12px; color:#9aa0a6; text-transform:uppercase; letter-spacing:.06em; }
  .half { display:inline-block; vertical-align:top; width:48%; min-width:320px; margin-right:1%; }
  details summary { cursor:pointer; color:#c8ff4d; }
  pre.blob { max-width:480px; max-height:160px; overflow:auto; white-space:pre-wrap;
    word-break:break-all; background:#0b0d11; border:1px solid #232733; border-radius:6px;
    padding:8px; font-size:11px; color:#9aa0a6; margin:6px 0 0; }
  nav.topnav { position:sticky; top:0; background:#0f1115; border-bottom:1px solid #232733;
    margin:0 -32px 18px; padding:0 32px; display:flex; gap:8px; z-index:5; }
  nav.topnav a { text-decoration:none; color:#fff; font-weight:700; font-size:16px;
    padding:16px 18px; border-bottom:3px solid transparent; }
  nav.topnav a:hover { border-bottom-color:#3a4150; }
  nav.topnav a.active { border-bottom-color:#c8ff4d; }
  a.dlbtn { display:inline-block; margin:6px 0 0; font-size:11px; font-weight:600;
    color:#0f1115; background:#c8ff4d; border-radius:6px; padding:3px 9px; text-decoration:none; }
  a.dlbtn:hover { background:#d8ff70; }
  a.resume { display:inline-block; font-size:12px; font-weight:700; color:#0f1115;
    background:#c8ff4d; border-radius:7px; padding:5px 11px; text-decoration:none; white-space:nowrap; }
  a.resume:hover { background:#d8ff70; }
  .resume.off { display:inline-block; font-size:12px; color:#5f6672; background:#1b1f27;
    border-radius:7px; padding:5px 11px; }
</style>"""


def _nav(active: str) -> str:
    items = [
        ("/", "Natural-Language Video Search", "search"),
        ("/tags", "Search history", "tags"),
    ]
    links = "".join(
        f'<a href="{href}" class="{"active" if key == active else ""}">{label}</a>'
        for href, label, key in items
    )
    return f"<nav class=topnav>{links}</nav>"


# Scoped styles so the Search-history table renders correctly when injected into
# the search page's #history-view (client-side tab swap); the standalone /tags
# page styles it via _PAGE_CSS instead.
_HISTORY_FRAG_CSS = """<style>
#history-view h1 { font-size:22px; margin:8px 0 4px; }
#history-view .sub { color:#9aa0a6; margin:0 0 8px; font-size:13px; }
#history-view h2 { font-size:13px; text-transform:uppercase; letter-spacing:.08em; color:#9aa0a6; margin:18px 0 10px; }
#history-view table { border-collapse:collapse; width:100%; font-size:13px; table-layout:fixed; }
#history-view th,#history-view td { text-align:left; padding:7px 10px; border-bottom:1px solid #232733; vertical-align:top; overflow-wrap:anywhere; word-break:break-word; }
#history-view th { color:#9aa0a6; font-weight:600; }
#history-view td.num { text-align:right; font-variant-numeric:tabular-nums; }
#history-view td.mono,#history-view .mono { font-family:'IBM Plex Mono',monospace; color:#9aa0a6; font-size:12px; word-break:break-all; }
#history-view td.q { width:22%; min-width:200px; }
#history-view td.tagname { color:#c8ff4d; font-weight:600; }
#history-view td.empty { color:#9aa0a6; font-style:italic; }
#history-view .muted { color:#5f6672; }
#history-view details summary { cursor:pointer; color:#c8ff4d; }
#history-view pre.blob { max-width:480px; max-height:160px; overflow:auto; white-space:pre-wrap; word-break:break-all; background:#0b0d11; border:1px solid #232733; border-radius:6px; padding:8px; font-size:11px; color:#9aa0a6; margin:6px 0 0; }
#history-view a.dlbtn { display:inline-block; margin:6px 0 0; font-size:11px; font-weight:600; color:#0f1115; background:#c8ff4d; border-radius:6px; padding:3px 9px; text-decoration:none; }
#history-view a.resume { display:inline-block; font-size:12px; font-weight:700; color:#0f1115; background:#c8ff4d; border-radius:7px; padding:5px 11px; text-decoration:none; white-space:nowrap; }
#history-view a.resume:hover { background:#d8ff70; }
#history-view .resume.off { display:inline-block; font-size:12px; color:#5f6672; background:#1b1f27; border-radius:7px; padding:5px 11px; }
</style>"""


# Friendly display names for known model checkpoints, shown in the "VLM / model"
# column in place of the raw checkpoint directory name.
_MODEL_LABELS = {
    "v3_lr_5e5-ckpt-6549": "red dwarf",
    "rd_4g_siglip_bias7-final": "white-dwarf",
    "maxsim-mainfull-ckpt14500": "black dwarf",
}


def _model_label(uri) -> str:
    """Friendly model name for the VLM/model column (falls back to the URI tail)."""
    tail = ("" if uri is None else str(uri).rstrip("/")).rsplit("/", 1)[-1]
    return _MODEL_LABELS.get(tail, tail)


def _search_history_inner() -> str:
    """The Search-history heading + table (no page chrome) -- shared by the full
    /tags page and the /api/search_history fragment used for client-side tab
    switching, so the markup never diverges."""
    tags = db.tags_catalog()

    def _esc(v) -> str:
        return html.escape("" if v is None else str(v))

    def _short_uri(u) -> str:
        """Trim a long s3 model/corpus URI to its identifying tail."""
        s = "" if u is None else str(u).rstrip("/")
        return s.rsplit("/", 1)[-1] or s

    rows = ""
    for t in tags:
        sv = t.get("search_vector_json") or ""
        if sv and sv not in ("[]", "null"):
            # Download the raw 768-d query vector as JSON, no round-trip needed.
            fname = (
                re.sub(r"[^A-Za-z0-9._-]+", "_", t.get("tag") or "").strip("_")
                or "search"
            ) + "_vector.json"
            href = "data:application/json;charset=utf-8," + urllib.parse.quote(sv)
            vec_html = (
                f"<details><summary>{_esc(t['vec_dim'])}-d</summary>"
                f'<a class=dlbtn download="{_esc(fname)}" href="{href}">&#8595; download .json</a>'
                f"<pre class=blob>{_esc(sv)}</pre></details>"
            )
        else:
            vec_html = "<span class=muted>—</span>"
        has_vec = bool(sv and sv not in ("[]", "null"))
        sid = t.get("id")
        if has_vec and sid is not None:
            resume_html = (
                f'<a class=resume data-id="{_esc(sid)}" href="/?resume={_esc(sid)}"'
                ' title="Reopen this search on the video-search page with the same '
                'query + search vector">&#8635; Resume</a>'
            )
        else:
            resume_html = '<span class="resume off">Resume</span>'
        rows += (
            f"<tr><td>{resume_html}</td>"
            f"<td class=tagname>{_esc(t['tag']) or '<span class=muted>(untagged)</span>'}</td>"
            f"<td class=q>{_esc(t['query'])}</td>"
            f"<td>{vec_html}</td>"
            f"<td class=num>{_esc(t['k'])}</td>"
            f"<td class=mono title=\"{_esc(t['model_uri'])}\">{_esc(_model_label(t['model_uri']))}</td>"
            f"<td class=mono>{_esc(t['embeddings_uri'])}</td>"
            f"<td>{_esc(t['segment_set_name'])}</td>"
            f"<td class=num>{_esc(t['num_results'])}</td>"
            f"<td>{_parquet_cell(t.get('parquet_uri'))}</td></tr>"
        )
    if not rows:
        rows = (
            "<tr><td colspan=10 class=empty>No searches yet — run a search, "
            "set a tag, and Download to register one.</td></tr>"
        )

    return f"""  <h1>Search history</h1>
  <p class=sub>Every Download, newest first — its tag (if set), the query + search
     vector that produced it, the top-k cutoff, the model / embedding version
     used, and the parquet written to S3.</p>

  <h2>Searches ({len(tags)})</h2>
  <table>
    <thead><tr>
      <th>Resume</th><th>Tag</th><th>NL query</th><th>Search vector</th><th>k</th>
      <th>VLM / model</th><th>Corpus / embeddings</th><th>Segment set</th>
      <th>Results</th><th>Parquet</th>
    </tr></thead>
    <tbody>{rows}</tbody>
  </table>"""


@app.get("/tags", response_class=HTMLResponse)
def tags_page() -> HTMLResponse:
    """Standalone Search-history page (direct hits). The search page loads the
    same content client-side via /api/search_history without a full reload."""
    body = f"""<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width, initial-scale=1">
<title>NLS Search — Search history</title>
{_PAGE_CSS}</head><body>
  {_nav("tags")}
{_search_history_inner()}
</body></html>"""
    return HTMLResponse(body)


@app.get("/api/search_history", response_class=HTMLResponse)
def search_history_fragment() -> HTMLResponse:
    """Search-history table as a self-contained HTML fragment (scoped styles), for
    the search page's client-side tab switch -- no full reload, no re-init."""
    return HTMLResponse(_HISTORY_FRAG_CSS + _search_history_inner())


@app.get("/api/tags_catalog")
def tags_catalog_json() -> dict:
    """Search history as JSON, backing the Export tab's history picker. Keyed on TAG
    (the export-log primary identifier): rows WITHOUT a tag are skipped, and rows are
    deduped to the newest per tag. So a user picks prior searches by tag (with a per-row
    k + threshold) instead of remembering them. Best-effort: ``{"entries": []}``."""
    seen: set[str] = set()
    entries: list[dict] = []
    for row in db.tags_catalog(limit=500):
        tag = (row.get("tag") or "").strip()
        if not tag or tag in seen:
            continue
        seen.add(tag)
        entries.append(
            {
                "id": row.get("id"),
                "tag": tag,
                "query": (row.get("query") or "").strip(),
                "k": int(row.get("k") or 0) or 50,
                # Last-used per-tag scan threshold (0 when never set -> the table uses its default).
                "threshold": float(row.get("threshold") or 0) or 0.0,
                "num_results": int(row.get("num_results") or 0),
                "model_uri": row.get("model_uri") or "",
                # Global friendly model name (e.g. "white-dwarf") from _MODEL_LABELS, so the
                # picker matches the corpus/model pills instead of showing the raw URI tail.
                "model_label": _model_label(row.get("model_uri") or ""),
                "embeddings_uri": row.get("embeddings_uri") or "",
                "vec_dim": int(row.get("vec_dim") or 0),
                # The active filter set saved alongside the vector (so the picker can show
                # exactly what scope produced it, and Resume/scan reuse the same filters).
                "filters": {
                    "from_date": row.get("date_from") or "",
                    "to_date": row.get("date_to") or "",
                    "segment_set_uuid": row.get("segment_set_uuid") or "",
                    "segment_set_name": row.get("segment_set_name") or "",
                    "filter_lance_uri": row.get("filter_lance_uri") or "",
                    "vehicle": row.get("vehicle") or "",
                    "drive_id": row.get("drive_id") or "",
                },
            }
        )
    return {"entries": entries}


@app.get("/analytics", response_class=HTMLResponse)
def analytics_page(limit: int = 200) -> HTMLResponse:
    """Server-rendered view of the export log + the segment sets used in exports.

    Reads straight from exp-db (best-effort): every Download writes an
    ``export_log`` row, and this page is how anyone with access reviews them and
    the distinct Data Explorer segment sets that have been saved against.
    """
    exports = db.recent_exports(limit)
    seg_sets = db.segment_sets_used()
    visits = analytics.load_visits()  # platform usage (one record per page visit)
    searches = analytics.load_searches()  # one record per text search (query kept)

    def _esc(v) -> str:
        return html.escape("" if v is None else str(v))

    def _fmt_unix(ts) -> str:
        try:
            return dt.datetime.fromtimestamp(float(ts), tz=dt.timezone.utc).strftime(
                "%Y-%m-%d %H:%M"
            )
        except (TypeError, ValueError):
            return ""

    # ---- Platform usage ----
    total_visits = len(visits)
    unique_users = sorted({v.get("user") for v in visits if v.get("user")})
    visit_rows = (
        "".join(
            f"<tr><td>{_fmt_unix(v.get('ts_unix'))}</td><td>{_esc(v.get('user'))}</td></tr>"
            for v in visits[:25]
        )
        or "<tr><td colspan=2 class=empty>No visits recorded yet.</td></tr>"
    )
    users_rows = (
        "".join(
            f"<tr><td>{_esc(u)}</td>"
            f"<td class=num>{sum(1 for v in visits if v.get('user') == u)}</td></tr>"
            for u in unique_users
        )
        or "<tr><td colspan=2 class=empty>—</td></tr>"
    )

    # ---- Searches (what people search for) ----
    total_searches = len(searches)
    query_counts: dict[str, int] = {}
    query_last: dict[str, float] = {}
    for s in searches:
        q = (s.get("query") or "").strip()
        if not q:
            continue
        query_counts[q] = query_counts.get(q, 0) + 1
        query_last[q] = max(query_last.get(q, 0.0), float(s.get("ts_unix") or 0))
    top_queries = sorted(query_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    top_query_rows = (
        "".join(
            f"<tr><td class=tag>{_esc(q)}</td><td class=num>{fmtint(n)}</td>"
            f"<td>{_fmt_unix(query_last.get(q))}</td></tr>"
            for q, n in top_queries[:40]
        )
        or "<tr><td colspan=3 class=empty>No searches recorded yet.</td></tr>"
    )
    recent_search_rows = (
        "".join(
            f"<tr><td>{_fmt_unix(s.get('ts_unix'))}</td>"
            f"<td>{_esc(s.get('user'))}</td>"
            f"<td class=tag>{_esc(s.get('query'))}</td>"
            f"<td class=num>{'' if s.get('num_results') is None else fmtint(s.get('num_results'))}</td></tr>"
            for s in searches[:40]
        )
        or "<tr><td colspan=4 class=empty>No searches recorded yet.</td></tr>"
    )

    seg_rows = (
        "".join(
            f"<tr><td class=name>{_esc(s['segment_set_name'])}</td>"
            f"<td class=mono>{_esc(s['segment_set_uuid'])}</td>"
            f"<td class=num>{_esc(s['times_used'])}</td>"
            f"<td class=num>{_esc(s['total_results'])}</td>"
            f"<td>{_fmt_ts(s['last_used'])}</td></tr>"
            for s in seg_sets
        )
        or "<tr><td colspan=5 class=empty>No segment sets used in any export yet.</td></tr>"
    )

    def _vector_cell(e) -> str:
        """Collapsible cell showing the persisted search vector + 👍/👎 marks."""
        parts = []
        sv = e.get("search_vector_json") or ""
        if sv and sv not in ("[]", "null"):
            parts.append(
                f"<details><summary>{_esc(e['vec_dim'])}-d vector</summary>"
                f"<pre class=blob>{_esc(sv)}</pre></details>"
            )
        else:
            parts.append("<span class=muted>(no vector)</span>")
        if e["num_up"] or e["num_down"]:
            parts.append(
                f"<details><summary>{_esc(e['num_up'])}&#128077; "
                f"{_esc(e['num_down'])}&#128078; marks</summary>"
                f"<pre class=blob>up: {_esc(e.get('thumbs_up_json') or '[]')}\n"
                f"down: {_esc(e.get('thumbs_down_json') or '[]')}</pre></details>"
            )
        return "".join(parts)

    export_rows = (
        "".join(
            f"<tr><td>{_fmt_ts(e['created_at'])}</td>"
            f"<td>{_esc(e['user_email'])}</td>"
            f"<td class=q>{_esc(e['query'])}</td>"
            f"<td class=tag>{_esc(e['tag'])}</td>"
            f"<td class=num>{_esc(e['k'])}</td>"
            f"<td class=num>{_esc(e['num_results'])}</td>"
            f"<td>{_esc(e['segment_set_name'])}</td>"
            f"<td>{_vector_cell(e)}</td></tr>"
            for e in exports
        )
        or "<tr><td colspan=8 class=empty>No exports recorded yet — hit Download on a search to create one.</td></tr>"
    )

    # ---- Threshold-tuning feedback (👍/👎) ----
    # Every threshold-sweep fit logs an episode with the running 👍/👎 tally; this is
    # the durable record of the labeling done in the refine+threshold steps. (Marks
    # also ride along on saved searches -- shown per-row in "Recent exports" above.)
    episodes = db.threshold_episodes(limit=2000)
    # Distinct human judgments (one per query+segment+user). NOT sum(n_pos) over
    # episodes: each episode logs a session's *running* tally, so summing them
    # re-counts the same thumbs ~17x (that was the inflated 22k/13k). The ledger
    # (feedback_marks) is one row per judgment, so its counts are the true totals.
    fb = db.feedback_totals()
    fb_total_up = fb["up"]
    fb_total_down = fb["down"]
    fb_queries = fb["queries"]
    fb_tags = sorted({e.get("tag") for e in episodes if e.get("tag")})
    fb_rows = (
        "".join(
            f"<tr><td>{_fmt_ts(e.get('created_at'))}</td>"
            f"<td class=tag>{_esc(e.get('tag'))}</td>"
            f"<td class=num>{_esc(e.get('n_pos'))}&#128077;</td>"
            f"<td class=num>{_esc(e.get('n_neg'))}&#128078;</td>"
            f"<td class=num>{('' if e.get('fit_tau') is None else format(float(e['fit_tau']), '.3f'))}</td>"
            f"<td class=num>{('' if e.get('f1') is None else format(float(e['f1']), '.2f'))}</td></tr>"
            for e in episodes[:40]
        )
        or "<tr><td colspan=6 class=empty>No threshold-tuning feedback recorded yet — "
        "label 👍/👎 during a threshold sweep to create the first.</td></tr>"
    )

    unavailable = (
        ""
        if (exports or seg_sets or episodes)
        else "<p class=warn>exp-db returned no rows (it may be empty, or unreachable — "
        "saves are best-effort). Run a search and Download to create the first row.</p>"
    )

    body = f"""<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width, initial-scale=1">
<title>NLS Search — Analytics</title>
{_PAGE_CSS}</head><body>
  {_nav("analytics")}
  <h1>NLS Search · Analytics</h1>
  <p class=sub>Export log + usage from <span class=mono>{_esc(db.SCHEMA_NAME)}.export_log</span>
     and visit records</p>
  {unavailable}

  <h2>Platform usage</h2>
  <div class=cards>
    <div class=card><div class=n>{fmtint(total_visits)}</div><div class=l>total visits</div></div>
    <div class=card><div class=n>{fmtint(len(unique_users))}</div><div class=l>unique users</div></div>
    <div class=card><div class=n>{fmtint(total_searches)}</div><div class=l>searches ({fmtint(len(query_counts))} unique)</div></div>
    <div class=card><div class=n>{fmtint(len(exports))}</div><div class=l>exports</div></div>
    <div class=card><div class=n>{fmtint(fb_total_up)} &#128077; / {fmtint(fb_total_down)} &#128078;</div><div class=l>distinct feedback labels</div></div>
  </div>
  <div class=half>
    <h2>Users ({len(unique_users)})</h2>
    <table><thead><tr><th>User</th><th>Visits</th></tr></thead><tbody>{users_rows}</tbody></table>
  </div>
  <div class=half>
    <h2>Recent visits</h2>
    <table><thead><tr><th>When (UTC)</th><th>User</th></tr></thead><tbody>{visit_rows}</tbody></table>
  </div>

  <h2>Top searches ({fmtint(len(query_counts))} unique queries, {fmtint(total_searches)} total)</h2>
  <p class=sub>What people search for (every text search by a signed-in user; query text is recorded).</p>
  <div class=half>
    <h2>Most searched</h2>
    <table><thead><tr><th>Query</th><th>Count</th><th>Last (UTC)</th></tr></thead><tbody>{top_query_rows}</tbody></table>
  </div>
  <div class=half>
    <h2>Recent searches</h2>
    <table><thead><tr><th>When (UTC)</th><th>User</th><th>Query</th><th>Results</th></tr></thead><tbody>{recent_search_rows}</tbody></table>
  </div>

  <h2>Segment sets used in exports ({len(seg_sets)})</h2>
  <table>
    <thead><tr><th>Name</th><th>UUID</th><th>Times used</th><th>&Sigma; results</th><th>Last used (UTC)</th></tr></thead>
    <tbody>{seg_rows}</tbody>
  </table>

  <h2>Recent exports ({len(exports)})</h2>
  <table>
    <thead><tr><th>When (UTC)</th><th>User</th><th>Query</th><th>Tag</th><th>k</th>
      <th>Results</th><th>Segment set</th><th>Search vector &amp; marks</th></tr></thead>
    <tbody>{export_rows}</tbody>
  </table>

  <h2>Feedback labels ({fmtint(fb_total_up)} &#128077; / {fmtint(fb_total_down)} &#128078; distinct, over {fmtint(fb_queries)} queries)</h2>
  <p class=sub>Each 👍/👎 you label during the refine + threshold-sweep steps is recorded here
     (from <span class=mono>{_esc(db.SCHEMA_NAME)}.threshold_episodes</span>).</p>
  <table>
    <thead><tr><th>When (UTC)</th><th>Tag</th><th>👍</th><th>👎</th><th>fit τ</th><th>F1</th></tr></thead>
    <tbody>{fb_rows}</tbody>
  </table>
</body></html>"""
    return HTMLResponse(body)


def fmtint(n) -> str:
    try:
        return f"{int(n):,}"
    except (TypeError, ValueError):
        return str(n)


# ===================== full-corpus (on-demand) search =======================
# The resident browse corpus is a few million clips. This path searches the
# WHOLE corpus (34.4M) by holding its int8/PCA-256 screen in memory instead of
# fp32 vectors: 8.8GB rather than the ~105GB the same rows would cost as fp32,
# swept in ~180ms on this instance's 8 cores.
#
# Loaded ON DEMAND, never at startup: the read+decode costs ~2 minutes and
# ~12GB, which nobody should pay unless they ask for full-corpus search. The
# load runs on a background thread so the request that triggers it returns
# immediately and the UI can poll `/api/full_corpus_status`.
_FULL_LOCK = threading.Lock()
_FULL: dict = {"corpus": None, "status": "idle", "error": "", "started": 0.0}


def _full_load_worker() -> None:
    import full_corpus

    try:
        corpus = full_corpus.load()
        with _FULL_LOCK:
            _FULL["corpus"] = corpus
            _FULL["status"] = "ready"
            _FULL["error"] = ""
        LOGGER.info("full corpus ready: %d rows", corpus.num_rows)
    except Exception as exc:  # noqa: BLE001 -- surfaced through the status endpoint
        LOGGER.exception("full corpus load failed")
        with _FULL_LOCK:
            _FULL["corpus"] = None
            _FULL["status"] = "error"
            _FULL["error"] = f"{type(exc).__name__}: {exc}"


def _full_corpus_begin_load() -> str:
    """Start the background load if it is not already running/ready."""
    with _FULL_LOCK:
        if _FULL["status"] in ("loading", "ready"):
            return _FULL["status"]
        _FULL["status"] = "loading"
        _FULL["error"] = ""
        _FULL["started"] = time.time()
    threading.Thread(
        target=_full_load_worker, name="full-corpus-load", daemon=True
    ).start()
    return "loading"


# How deep full-corpus paging goes. Selection cost grows with depth, and a
# ranked list is not a useful way to read past a few thousand rows -- narrow the
# filters or use an export instead.
_FULL_MAX_DEPTH = 5000

# Clips mean-pooled into a query-by-example vector. A long drive can match tens
# of thousands of chunks, and pooling all of them is both a large S3 read and a
# worse query -- the direction washes out. The window is the user's selection;
# this only bounds the read.
_WINDOW_MAX_POOL = 512


def _full_refresh_worker() -> None:
    """Drop the resident corpus, then load the current one.

    Dropping FIRST is not a style choice: the process already holds ~24GB (model,
    browse matrix, full corpus) against a 32Gi ceiling, so building a replacement
    beside the old one needs ~37GB and is killed part-way. The cost is that
    full-corpus search is unavailable for the length of the reload; browse is
    unaffected, and searches in that window get the same 503-and-poll they get
    before the first load.
    """
    import gc

    with _FULL_LOCK:
        previous = _FULL["corpus"]
        _FULL["corpus"] = None
        _FULL["status"] = "loading"
        _FULL["error"] = ""
        _FULL["started"] = time.time()
    del previous
    with _FULL_SEG_LOCK:
        _FULL_SEG_IDS.clear()
    gc.collect()
    _full_load_worker()


@app.post("/api/full_corpus_refresh")
def full_corpus_refresh() -> dict:
    """Reload the full corpus from the current Lance, picking up new rows.

    Intended for a scheduler: the corpus grows daily and a resident copy is a
    snapshot. Returns immediately; the reload runs on a background thread and is
    observable through /api/full_corpus_status.

    A refresh already in flight is not restarted -- a second trigger would drop a
    half-built corpus and start over, which is strictly worse than letting the
    first finish.
    """
    with _FULL_LOCK:
        if _FULL["status"] == "loading":
            return {"status": "loading", "note": "refresh already in progress"}
    threading.Thread(
        target=_full_refresh_worker, name="full-corpus-refresh", daemon=True
    ).start()
    return {"status": "refreshing"}


class FullRescoreRequest(BaseModel):
    query: str
    rows: list[int]


@app.post("/api/full_rescore")
def full_rescore(req: FullRescoreRequest) -> dict:
    """Exact 768-d cosine for rows already returned by /api/full_search.

    The search answers from a quantized, PCA-reduced screen, so its scores carry
    a +/-eps bound and are not comparable to thresholds calibrated on the float
    corpus. This reads the original embeddings for the rows on screen and returns
    the real cosine.

    Separate from the search because it is one S3 round trip with a ~1.65s floor
    whatever the row count: folding it in would turn a 190ms search into nearly
    two seconds for every caller, including those who never look at the score.
    """
    if not req.rows:
        raise HTTPException(400, "rows must not be empty")
    if len(req.rows) > 1000:
        raise HTTPException(400, "at most 1000 rows per rescore")
    if not _state.get("model_ready"):
        raise HTTPException(503, "model still loading")
    with _FULL_LOCK:
        corpus = _FULL["corpus"]
    if corpus is None:
        raise HTTPException(503, "full corpus is not resident; run a search first")
    vec = search_engine.encode_query(
        (req.query or "").strip(), _state["processor"], _state["model"],
        _state["cfg"].device,
    )
    t0 = time.perf_counter()
    try:
        exact = corpus.rescore(req.rows, vec)
    except RuntimeError as exc:
        # Raised when a row no longer resolves to the clip the search reported,
        # i.e. the snapshot moved. Better a 409 than a confident wrong number.
        raise HTTPException(409, str(exc))
    took_ms = round((time.perf_counter() - t0) * 1000, 1)
    LOGGER.info("full rescore: %d rows in %.1fms", len(exact), took_ms)
    return {
        "took_ms": took_ms,
        "score_kind": "exact",
        "source_column": full_corpus_module().vector_full_column(),
        "scores": [{"row": r, "score": round(sc, 4)} for r, sc in exact.items()],
    }


def full_corpus_module():
    import full_corpus

    return full_corpus


class FullExportRequest(BaseModel):
    """Export from the FULL corpus, in-process. Replaces the offline Lilypad scan.

    Selection is `k` (bounded by construction) or `threshold` (unbounded, so
    `max_rows` caps it). `exact` re-scores the selected rows against the original
    768-d embeddings; it costs one S3 round trip per 100k rows and is refused
    when projected past `_FULL_EXACT_BUDGET_S` rather than silently downgraded.
    """

    query: str
    tag: str = ""
    k: int | None = None
    threshold: float | None = None
    max_rows: int = 300_000
    exact: bool = False
    interval: bool = False
    dedupe_segment: bool = False
    create_segment_set: bool = False
    from_date: str | None = None
    to_date: str | None = None
    vehicle: str | None = None
    drive_id: str | None = None
    segment_set_uuid: str | None = None
    filter_lance_uri: str | None = None


class FullThresholdRequest(BaseModel):
    """Fit a cutoff for a full-corpus query from labeled marks.

    Separate from `/api/threshold_search` because that one addresses marks by
    resident-corpus row index, and full-corpus hits have no resident index --
    they carry `row`, a position in the 34M table. Feeding those to the resident
    endpoint silently dropped every label (they fail its bounds check), which
    left every tag on the label-free mean+3*std heuristic without saying so.
    """

    query: str
    marks: list[Mark] = []
    objective: str = "f1"
    beta: float = 1.0
    min_precision: float = 0.9
    val_fraction: float = 0.0
    sample_size: int = 12
    band: float = 0.02
    from_date: str | None = None
    to_date: str | None = None
    vehicle: str | None = None
    drive_id: str | None = None
    segment_set_uuid: str | None = None
    filter_lance_uri: str | None = None


class FullSearchRequest(BaseModel):
    query: str
    limit: int = 50
    page: int = 0
    from_date: str | None = None
    to_date: str | None = None
    # Same comma/space-separated forms the resident search accepts. Unlike the
    # resident path, MULTIPLE vehicles are honoured here.
    vehicle: str | None = None
    drive_id: str | None = None
    segment_set_uuid: str | None = None
    filter_lance_uri: str | None = None


# Segment-set membership resolved against the FULL corpus, cached per set uuid.
# The resident corpus is pinned to one dataset version for its lifetime, so a
# set's mask cannot go stale under it; the cache is cleared when it reloads.
_FULL_SEG_IDS: dict[str, frozenset] = {}
_FULL_SEG_LOCK = threading.Lock()


def _full_segment_ids(seg_uuid: str | None) -> frozenset | None:
    """The segment_ids of a Data Explorer set, for filtering the full corpus.

    Uses the external_id (segment_id) path rather than the dx_internal_id bitmap
    the resident corpus prefers: dx_internal_id is populated on only ~0.3% of the
    consolidated corpus, so a bitmap AND would drop nearly every row and look
    exactly like a set that legitimately matched nothing.

    Raises rather than returning None on failure. A filter that silently fails to
    apply returns a confident result set over the whole corpus, which is worse
    than an error because nothing in the answer looks wrong.
    """
    if not seg_uuid:
        return None
    with _FULL_SEG_LOCK:
        cached = _FULL_SEG_IDS.get(seg_uuid)
    if cached is not None:
        return cached
    allowed, pending, _count = _resolve_segment_filter(seg_uuid)
    if pending:
        raise HTTPException(503, "segment set is still loading; retry in a moment")
    if allowed is None:
        raise HTTPException(400, f"could not resolve segment set {seg_uuid}")
    ids = frozenset(allowed)
    with _FULL_SEG_LOCK:
        _FULL_SEG_IDS[seg_uuid] = ids
    LOGGER.info("full-corpus segment set %s: %d ids", seg_uuid, len(ids))
    return ids


def _full_lance_filter(lance_uri: str | None) -> dict:
    """Filter kwargs for a lance/parquet downsample dataset, for the full corpus.

    The dataset names its own key column. `segment_id` and `run_uuid` both have
    a resident counterpart here; `dx_internal_id` does not usefully -- it is
    populated on ~0.3% of this corpus, so intersecting on it would drop nearly
    every row and be indistinguishable from a downsample that legitimately
    matched almost nothing. That case errors instead.
    """
    if not lance_uri or not lance_uri.strip():
        return {}
    try:
        key, ids = _lance_filter_ids(lance_uri.strip())
    except _CORPUS_ERRORS as exc:
        raise HTTPException(400, f"could not read downsample dataset: {exc}")
    if key == "segment_id":
        return {"segment_ids": frozenset(ids)}
    if key == "run_uuid":
        return {"run_uuids": set(ids)}
    raise HTTPException(
        400,
        f"downsample dataset is keyed by {key!r}, which the full corpus cannot "
        "filter on; provide one keyed by segment_id or run_uuid",
    )


def _merge_filters(base: dict, extra: dict) -> dict:
    """AND a downsample's filters into the request's own (intersection per key)."""
    out = dict(base)
    for key, val in extra.items():
        cur = out.get(key)
        out[key] = val if not cur else (set(cur) & set(val))
    return out


def _full_filters(req) -> dict:
    """Filter kwargs for `full_corpus` from any request carrying the standard
    filter fields. One definition so no endpoint can quietly support a different
    subset than the others."""
    corpus = _require_full_corpus()
    start_unix, end_unix = _date_bounds(
        getattr(req, "from_date", None), getattr(req, "to_date", None), corpus
    )
    return _merge_filters(
        {
            "vehicles": set(_parse_vehicles(getattr(req, "vehicle", None))) or None,
            "run_uuids": set(_parse_drive_ids(getattr(req, "drive_id", None))) or None,
            "date_range": (start_unix, end_unix) if (start_unix or end_unix) else None,
            "segment_ids": _full_segment_ids(getattr(req, "segment_set_uuid", None)),
        },
        _full_lance_filter(getattr(req, "filter_lance_uri", None)),
    )


def _gone(replacement: str) -> "HTTPException":
    """410 for an endpoint that ranked the retired 2M corpus.

    Deleting these outright would give an external caller a 404 and no idea why;
    returning a plausible answer from a different corpus would be worse. The
    replacement is named so a caller can move.
    """
    return HTTPException(
        410,
        f"this endpoint ranked the retired 2M corpus; use {replacement}, which "
        "covers the whole table",
    )


def _query_vector(query: str, uri: str, model_uri: str, corpus):
    """``(vector, reused)``: reuse a stored search vector for this query from exp-db
    when present and dimensionally valid, else encode fresh. The lookup is by TAG
    (config export persists each query under tag=query) scoped to this corpus+model,
    so a prior run's vector is reused read-only. Backs "Export from config" reuse."""
    stored = db.find_vector_by_tag(query, uri, model_uri)
    if stored is not None:
        vec = np.asarray(stored, dtype=np.float32)
        if vec.ndim == 1 and vec.shape[0] == corpus.dim:
            return vec, True
        LOGGER.info(
            "stored vector for %r has dim %d != corpus %d; re-encoding",
            query,
            int(vec.shape[0]) if vec.ndim == 1 else -1,
            corpus.dim,
        )
    vec = search_engine.encode_query(
        query, _state["processor"], _state["model"], _state["cfg"].device
    )
    return vec, False


def _config_csv(rows: list) -> str:
    """CSV text for multi-query rows -- a list of ``(query, RankedHit)``. The query is
    written into the ``tag`` column (no separate query column). Shared by
    ``/api/export_config`` and ``/api/curate_export`` so their output is identical."""
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(
        [
            "rank",
            "score",
            "segment_id",
            "chunk_id",
            "run_uuid",
            "start_timestamp_ns",
            "end_timestamp_ns",
            "source_media_uri",
            "tag",
        ]
    )
    for q, h in rows:
        w.writerow(
            [
                h.rank,
                f"{h.score:.6f}",
                h.segment_id,
                h.chunk_id,
                h.run_uuid,
                _ns(h.chunk_start_unix),
                _ns(h.chunk_end_unix),
                h.source_media_uri,
                q,
            ]
        )
    return buf.getvalue()


def _write_config_export_parquet(rows: list, name: str | None = None) -> str:
    """Write multi-query config-export rows as parquet under the export prefix.

    ``rows`` is a list of ``(query, RankedHit)``; the row's query is written into the
    ``tag`` column (no separate query column). Mirrors ``_write_export_parquet``;
    best-effort (returns "" on failure). When ``name`` is given the object stem is
    exactly ``name`` (matching the CSV download + segment-set name)."""
    prefix = _state["cfg"].export_s3_prefix.strip()
    if not prefix:
        return ""
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.table(
        {
            "rank": [h.rank for _, h in rows],
            "score": [float(h.score) for _, h in rows],
            "segment_id": [h.segment_id for _, h in rows],
            "chunk_id": [h.chunk_id for _, h in rows],
            "run_uuid": [h.run_uuid for _, h in rows],
            "start_timestamp_ns": [_ns(h.chunk_start_unix) for _, h in rows],
            "end_timestamp_ns": [_ns(h.chunk_end_unix) for _, h in rows],
            "source_media_uri": [h.source_media_uri for _, h in rows],
            "tag": [q for q, _ in rows],
        }
    )
    sink = io.BytesIO()
    pq.write_table(table, sink)

    stem = name or _export_base("config_export")
    key = f"{prefix.rstrip('/')}/{stem}.parquet"
    try:
        oci_s3.put_bytes(key, sink.getvalue(), _state["s3"], "application/octet-stream")
        LOGGER.info("config export parquet written: %s (%d rows)", key, len(rows))
        return key
    except (
        ValueError,
        botocore.exceptions.BotoCoreError,
        botocore.exceptions.ClientError,
    ) as exc:
        LOGGER.warning(
            "config export parquet write failed (%s): %s", type(exc).__name__, exc
        )
        return ""


# An export holds its selected rows, their metadata columns and (when exact) a
# slab of fetched vectors, in a process that already sits near its memory
# ceiling with the model and the 8.8GB screen resident. Two at once is what
# OOM-kills the container, so exports are serialized rather than queued
# per-request.
_FULL_EXPORT_LOCK = threading.Lock()

# Exact re-scoring is bounded by TIME, not row count. Memory is not the
# constraint -- exact_scores fetches in chunks and frees each one, so peak stays
# flat whatever the total -- and the old 300,000-row cap was inherited from the
# DORA segment-set limit, which is a different system's problem. What actually
# breaks is the Cloud Run request timeout (600s): overrun it and the export dies
# mid-write, losing both the artifact and the caller's CSV.
#
# 300s leaves the other half of the budget for the scan, the interval merge, the
# parquet write and the upload. It is also how long the export lock is held, so
# every concurrent export gets a 429 for that whole window.
_FULL_EXACT_BUDGET_S = float(os.getenv("NLS_EXACT_BUDGET_S", "300"))

# Measured: 3.17s per 10,000 rows of vector_black_dwarf via take(). Used only to
# refuse up front; the fetch also checks the real clock between chunks, because a
# slow day makes this constant optimistic and a mid-write timeout is worse than
# an early refusal.
_EXACT_MS_PER_ROW = 0.317

# Hard ceiling on a threshold export regardless of what the caller asks for.
# A permissive tau matches a tenth of the corpus (a mean+3*std cutoff selected
# 3.4M rows in practice), and the row count -- not the scan -- is what decides
# how much memory this endpoint uses.
_FULL_EXPORT_MAX_ROWS = 2_000_000


def _full_export_selection(corpus, req: "FullExportRequest", vec) -> object:
    """Resolve a FullExportRequest to a `full_corpus.Selection`."""
    start_unix, end_unix = _date_bounds(req.from_date, req.to_date, corpus)
    filters = {
        "vehicles": set(_parse_vehicles(req.vehicle)) or None,
        "run_uuids": set(_parse_drive_ids(req.drive_id)) or None,
        "date_range": (start_unix, end_unix) if (start_unix or end_unix) else None,
        "segment_ids": _full_segment_ids(req.segment_set_uuid),
    }
    filters = _merge_filters(filters, _full_lance_filter(req.filter_lance_uri))
    if req.k is not None:
        return corpus.select(vec, k=int(req.k), **filters)
    return corpus.select(
        vec,
        tau=float(req.threshold),
        max_rows=min(int(req.max_rows or 0) or _FULL_EXPORT_MAX_ROWS, _FULL_EXPORT_MAX_ROWS),
        **filters,
    )


class RetrieveRequest(BaseModel):
    """The one retrieval interface: a query, a cutoff, and how to return it.

    `k` and `threshold` are the same operation -- a top-k HAS a threshold, the
    k-th score -- so they share one cascade rather than two implementations.
    `output` decides serialization only: a page of JSON for browsing, a CSV
    attachment (plus parquet, segment set and a Recent-scans row) for export.
    """

    # Query: text, or a pre-computed vector (resume / query-by-example).
    query: str = ""
    vector: list[float] | None = None

    # Cutoff -- exactly one.
    k: int | None = None
    threshold: float | None = None

    # Cascade control. "auto" resolves the eps band through layer 2 when it is
    # usable; "never" answers from the resident screen alone; "always" refuses
    # rather than silently returning bounded scores.
    refine: str = "auto"
    # Layer 3: replace scores with the true 768-d cosine. A separate question
    # from membership, and a separate S3 read.
    exact: bool = False

    output: str = "hits"          # "hits" (JSON page) | "csv" (file + artifacts)
    page: int = 0
    limit: int = 50
    max_rows: int = 0             # threshold-mode ceiling; 0 = the global one

    # Export-only options, ignored for output="hits".
    tag: str = ""
    interval: bool = False
    dedupe_segment: bool = False
    create_segment_set: bool = False

    from_date: str | None = None
    to_date: str | None = None
    vehicle: str | None = None
    drive_id: str | None = None
    segment_set_uuid: str | None = None
    filter_lance_uri: str | None = None
    marks: list[Mark] = []        # downvoted rows are excluded from results


@app.post("/api/retrieve")
def retrieve(req: RetrieveRequest, request: Request):
    """Search and export, over one selection.

    These were two endpoints over two implementations of the same search, and
    every divergence between them shipped as a bug in one path and not the
    other. Now there is one `full_corpus.select` underneath and this endpoint
    chooses only how to serialize it.
    """
    if (req.k is None) == (req.threshold is None):
        raise HTTPException(400, "pass exactly one of k or threshold")
    if req.output not in ("hits", "csv"):
        raise HTTPException(400, "output must be 'hits' or 'csv'")
    corpus = _require_full_corpus()
    vec = _retrieve_vector(req, corpus)
    exclude = [
        int(m.row) for m in req.marks
        if m.mark == "down" and m.row is not None and 0 <= int(m.row) < corpus.num_rows
    ]

    if req.output == "csv":
        # Export keeps its own lock, artifact writes and Recent-scans row; only
        # the selection is shared.
        return _retrieve_export(req, request, corpus, vec, exclude)
    return _retrieve_hits(req, corpus, vec, exclude)


def _retrieve_vector(req: "RetrieveRequest", corpus) -> np.ndarray:
    """The query direction: a supplied vector, else the encoded text."""
    if req.vector:
        vec = np.asarray(req.vector, dtype=np.float32)
        if vec.ndim != 1 or vec.shape[0] != corpus.dim:
            raise HTTPException(
                400, f"vector dim {vec.shape[0]} != corpus dim {corpus.dim}"
            )
        return vec
    if not req.query.strip():
        raise HTTPException(400, "provide a query or a vector")
    if not _state.get("model_ready"):
        raise HTTPException(503, "model still loading")
    return search_engine.encode_query(
        req.query.strip(), _state["processor"], _state["model"], _state["cfg"].device
    )


def _retrieve_selection(req: "RetrieveRequest", corpus, vec, exclude, *, k=None):
    """Run the one search. `k` overrides the request's cutoff for paging."""
    try:
        return corpus.select(
            vec,
            k=k if k is not None else req.k,
            tau=None if k is not None else req.threshold,
            max_rows=min(
                int(req.max_rows or 0) or _FULL_EXPORT_MAX_ROWS, _FULL_EXPORT_MAX_ROWS
            ),
            refine=req.refine,
            exclude_rows=exclude or None,
            **_full_filters(req),
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except RuntimeError as exc:
        raise HTTPException(409, str(exc))


def _retrieve_hits(req: "RetrieveRequest", corpus, vec, exclude) -> dict:
    """A page of the selection, in the envelope the grid already renders."""
    if req.limit <= 0 or req.limit > 1000:
        raise HTTPException(400, "limit must be between 1 and 1000")
    if req.page < 0:
        raise HTTPException(400, "page must not be negative")
    offset = req.page * req.limit
    depth = offset + req.limit
    if depth > _FULL_MAX_DEPTH:
        raise HTTPException(400, f"paging stops at {_FULL_MAX_DEPTH:,} results")
    t0 = time.perf_counter()
    # `k` is the cutoff -- how many results exist -- and `limit` is the page over
    # them, so a k smaller than a page yields fewer hits rather than being
    # widened to fill it. A threshold query has no natural size, so the page
    # depth caps it: materializing millions of rows to show twenty-four would
    # cost the memory this endpoint exists to avoid spending.
    sel = _retrieve_selection(req, corpus, vec, exclude)
    if req.threshold is not None and len(sel) > depth:
        sel = dataclasses.replace(
            sel, rows=sel.rows[:depth], scores=sel.scores[:depth], truncated=True
        )
    rows, scores = sel.rows[offset:depth], sel.scores[offset:depth]
    if req.exact and rows.size:
        try:
            scores = corpus.exact_scores(rows, vec).astype(np.float32)
        except RuntimeError as exc:
            raise HTTPException(409, str(exc))
        order = np.argsort(-scores, kind="stable")
        rows, scores = rows[order], scores[order]
    hits = [
        corpus._hit(int(r), offset + i, float(s), sel.error_bound)
        for i, (r, s) in enumerate(zip(rows, scores), start=1)
    ]
    took_ms = round((time.perf_counter() - t0) * 1000, 1)
    LOGGER.info(
        "retrieve %r -> %d hits over %d rows in %.1fms | band=%d refined=%s%s",
        req.query or "vector", len(hits), corpus.num_rows, took_ms,
        sel.band_rows, sel.refined, _separation_note(hits),
    )
    out = _full_payload(hits, sel.candidates, corpus, req, took_ms,
                        req.query or "vector", page_size=req.limit)
    out["score_kind"] = "exact" if req.exact else sel.score_kind
    out["band_rows"] = sel.band_rows
    out["refined"] = sel.refined
    return out


def _retrieve_export(req: "RetrieveRequest", request: Request, corpus, vec, exclude):
    """The same selection, serialized as a file plus its artifacts.

    Kept behind the same single-flight lock as before: an export holds its rows
    and (with `exact`) a slab of fetched vectors in a process already near its
    memory ceiling, and two at once is what kills the container.
    """
    if not _FULL_EXPORT_LOCK.acquire(blocking=False):
        raise HTTPException(429, "another export is running; retry when it finishes")
    try:
        t0 = time.perf_counter()
        sel = _retrieve_selection(req, corpus, vec, exclude)
        if len(sel) == 0:
            raise HTTPException(404, "nothing matched the query and filters")
        exact_scores, score_kind = None, sel.score_kind
        if req.exact:
            projected = len(sel) * _EXACT_MS_PER_ROW / 1000.0
            if projected > _FULL_EXACT_BUDGET_S:
                raise HTTPException(
                    400,
                    f"exact scoring {len(sel):,} rows is projected at "
                    f"{projected:.0f}s, over the {_FULL_EXACT_BUDGET_S:.0f}s budget",
                )
            try:
                corpus._verify_alignment(sel.rows)
                exact_scores = corpus.exact_scores(
                    sel.rows, vec, deadline=time.monotonic() + _FULL_EXACT_BUDGET_S
                )
            except TimeoutError as exc:
                raise HTTPException(503, str(exc))
            except RuntimeError as exc:
                raise HTTPException(409, str(exc))
            score_kind = "exact"

        # Delegate the artifact half to the existing writer, which already
        # handles intervals, dedupe, parquet, segment sets and publishing.
        shim = FullExportRequest(
            query=req.query or "vector", tag=req.tag, k=req.k,
            threshold=req.threshold, max_rows=req.max_rows or 300_000,
            exact=req.exact, interval=req.interval,
            dedupe_segment=req.dedupe_segment,
            create_segment_set=req.create_segment_set,
            from_date=req.from_date, to_date=req.to_date, vehicle=req.vehicle,
            drive_id=req.drive_id, segment_set_uuid=req.segment_set_uuid,
            filter_lance_uri=req.filter_lance_uri,
        )
        if req.interval:
            return _full_export_intervals(
                shim, request, corpus, sel, vec, t0, exact_scores, score_kind
            )
        return _full_export_write(
            shim, request, corpus, sel, vec, t0, exact_scores, score_kind
        )
    finally:
        _FULL_EXPORT_LOCK.release()


@app.post("/api/full_export")
def full_export(req: FullExportRequest, request: Request) -> Response:
    """Export from the whole 34M-row corpus, in-process -- no Lilypad launch.

    The screening pass that ranks a search already produces every row's score, so
    selection here is the same scan plus a cut: `k` takes the best k, `threshold`
    takes everything at or above a cutoff. What made this an offline job was
    never the scoring (~190ms) but materializing millions of results; the
    columnar path in `full_corpus` is what removes that.

    `exact=true` replaces the screening scores with true 768-d cosine for the
    selected rows. Worth it for a bounded k, and the reason `threshold` mode
    defaults to off: at a permissive cutoff the boundary is not meaningful
    enough to pay a per-100k-row S3 fetch to resolve.
    """
    query = (req.query or "").strip()
    if not query:
        raise HTTPException(400, "query must not be empty")
    if (req.k is None) == (req.threshold is None):
        raise HTTPException(400, "pass exactly one of k or threshold")
    if req.k is not None and req.k <= 0:
        raise HTTPException(400, "k must be a positive integer")
    if not _state.get("model_ready"):
        raise HTTPException(503, "model still loading")
    with _FULL_LOCK:
        corpus = _FULL["corpus"]
    if corpus is None:
        _full_corpus_begin_load()
        raise HTTPException(503, "full corpus is loading; poll /api/full_corpus_status")

    if not _FULL_EXPORT_LOCK.acquire(blocking=False):
        raise HTTPException(429, "another export is running; retry when it finishes")
    try:
        t0 = time.perf_counter()
        vec = search_engine.encode_query(
            query, _state["processor"], _state["model"], _state["cfg"].device
        )
        sel = _full_export_selection(corpus, req, vec)
        if len(sel) == 0:
            raise HTTPException(404, "nothing matched the query and filters")

        exact_scores = None
        score_kind = "bounded_approx"
        if req.exact:
            projected = len(sel) * _EXACT_MS_PER_ROW / 1000.0
            if projected > _FULL_EXACT_BUDGET_S:
                raise HTTPException(
                    400,
                    f"exact scoring {len(sel):,} rows is projected at "
                    f"{projected:.0f}s, over the {_FULL_EXACT_BUDGET_S:.0f}s "
                    "budget. Lower k, raise the threshold, or export with "
                    "exact=false (scores stay bounded-approximate).",
                )
            try:
                corpus._verify_alignment(sel.rows)
                exact_scores = corpus.exact_scores(
                    sel.rows, vec, deadline=time.monotonic() + _FULL_EXACT_BUDGET_S
                )
            except TimeoutError as exc:
                # The estimate was optimistic. 503 rather than 400: the request
                # was reasonable, the object store was slow, and retrying or
                # narrowing are both sensible next moves.
                raise HTTPException(503, str(exc))
            except RuntimeError as exc:
                # Either the snapshot moved under the loaded row positions, or
                # this corpus has no dataset handle to read from. Both mean the
                # exact scores would be wrong or absent -- 409 rather than a
                # confident number attached to the wrong clip.
                raise HTTPException(409, str(exc))
            score_kind = "exact"

        if req.interval:
            return _full_export_intervals(
                req, request, corpus, sel, vec, t0, exact_scores, score_kind
            )

        return _full_export_write(
            req, request, corpus, sel, vec, t0, exact_scores, score_kind
        )
    finally:
        _FULL_EXPORT_LOCK.release()


def _full_export_write(
    req, request: Request, corpus, sel, vec, t0: float,
    exact_scores=None, score_kind: str = "bounded_approx",
) -> Response:
    """Write a clip-granularity export: arrow table -> CSV + parquet, plus
    the segment set, the exp-db row and the Recent-scans entry.

    Extracted from `full_export` so the unified `/api/retrieve` writes the
    same artifacts rather than growing a second copy of this -- duplicated
    selection is what produced most of this module's divergence bugs.
    """
    query = (req.query or "").strip()
    rows, scores = sel.rows, sel.scores
    if exact_scores is not None:
        # Re-rank on the exact numbers: rows that sat within the screening
        # error bound of one another can legitimately swap.
        order = np.argsort(-exact_scores, kind="stable")
        rows, scores, exact_scores = rows[order], scores[order], exact_scores[order]
    table = corpus.to_arrow(
        rows, exact_scores if exact_scores is not None else scores,
        tag=req.tag or query, exact=None,
    )
    if req.dedupe_segment:
        table = _dedupe_arrow_by_segment(table)

    base = _export_base(req.tag or query)
    parquet_uri = _write_arrow_export(table, base)
    segset_label, segset_error = "", ""
    if req.create_segment_set:
        segset_label, segset_error = _create_export_segment_set(
            table.column("segment_id").to_pylist(),
            base,
            provenance={
                "source": "nls_search_app_full_corpus",
                "query": query,
                "embeddings_uri": corpus.corpus_uri,
                "model_uri": _state["cfg"].model_artifact_uri,
                "date_from": req.from_date,
                "date_to": req.to_date,
                "vehicle": req.vehicle,
                "drive_id": req.drive_id,
            },
        )
    took_ms = round((time.perf_counter() - t0) * 1000, 1)
    LOGGER.info(
        "full export %r -> %d rows (%s, %s) over %d rows in %.1fms",
        query, table.num_rows, "k=%d" % req.k if req.k else "tau=%.4f" % req.threshold,
        score_kind, corpus.num_rows, took_ms,
    )
    _record_full_export(req, request, corpus, table, vec, parquet_uri, sel)
    _publish_full_export(
        req, request, base, parquet_uri, segset_label, sel, table.num_rows,
        corpus, score_kind, segset_error=segset_error,
    )
    return Response(
        content=_arrow_csv(table),
        media_type="text/csv",
        headers=_full_export_headers(
            base, parquet_uri, segset_label, segset_error, sel, table.num_rows,
            score_kind, took_ms,
        ),
    )


def _full_export_intervals(
    req: "FullExportRequest", request: Request, corpus, sel, vec, t0: float,
    exact_scores=None, score_kind: str = "bounded_approx",
) -> Response:
    """Interval-granularity full-corpus export: merge the selected clips into
    variable-length spans per drive, then write those instead of clips.

    Merges on the exact scores when they were fetched. This branch used to ignore
    them: `exact=true` paid the S3 read and then projected intervals from the
    screening scores anyway, so the option cost time and changed nothing.

    The cutoff has to move with the scores. `sel.cutoff` is a SCREENING score for
    a top-k selection, and comparing it against exact ones would threshold in the
    wrong space -- so in exact mode a k-selection re-derives it from the exact
    scores, while a threshold selection keeps the caller's own cutoff (which is a
    cosine, and therefore more correctly applied to exact scores than to the
    screen).
    """
    scores = sel.scores if exact_scores is None else np.asarray(exact_scores)
    if exact_scores is None or req.threshold is not None:
        tau = float(sel.cutoff)
    else:
        tau = float(scores.min()) if scores.size else 0.0
    intervals = corpus.intervals(sel.rows, scores, tau=tau)
    if not intervals:
        raise HTTPException(404, "no intervals formed above the cutoff")
    rows = _interval_rows_full(intervals, corpus, req.query.strip())
    base = _export_base(req.tag or "intervals")
    parquet_uri = _write_interval_export_parquet(rows, name=base)
    # Same publish as the clip path: this branch returns early, so leaving it out
    # meant interval exports -- the mode the app defaults to -- never appeared in
    # Recent scans.
    _publish_full_export(
        req, request, base, parquet_uri, "", sel, len(rows), corpus, score_kind,
        segset_error=(
            "intervals: segment-set export not supported yet"
            if req.create_segment_set else ""
        ),
    )
    took_ms = round((time.perf_counter() - t0) * 1000, 1)
    LOGGER.info(
        "full export (intervals) %r -> %d intervals from %d clips (%s) in %.1fms",
        req.query.strip(), len(rows), len(sel), score_kind, took_ms,
    )
    return Response(
        content=_interval_csv(rows),
        media_type="text/csv",
        headers=_full_export_headers(
            base, parquet_uri, "",
            "intervals: segment-set export not supported yet" if req.create_segment_set else "",
            sel, len(rows), score_kind, took_ms,
        ) | {"X-NLS-Interval-Threshold": f"{tau:.6f}"},
    )


def _interval_rows_full(intervals: list, corpus, query: str) -> list[dict]:
    """`_interval_rows` for the full corpus: same output columns, but the peak
    clip's identifiers come from the resident arrays rather than a Corpus."""
    import pyarrow as pa

    peaks = np.asarray([iv.peak_index for iv in intervals], dtype=np.int64)
    valid = peaks >= 0
    lookup = corpus.to_arrow(peaks[valid], np.zeros(int(valid.sum()))).to_pydict()
    it = iter(range(int(valid.sum())))
    rows = []
    for rank, iv in enumerate(intervals, start=1):
        if valid[rank - 1]:
            j = next(it)
            chunk_id, seg, media = (
                lookup["chunk_id"][j], lookup["segment_id"][j],
                lookup["source_media_uri"][j],
            )
        else:
            chunk_id, seg, media = "", None, ""
        rows.append(
            {
                "rank": rank,
                "peak_score": round(float(iv.peak_score), 6),
                "mean_score": round(float(iv.mean_score), 6),
                "run_uuid": iv.run_uuid,
                "start_timestamp_ns": _ns_f(iv.start_unix),
                "end_timestamp_ns": _ns_f(iv.end_unix),
                "duration_s": round(float(iv.end_unix - iv.start_unix), 3),
                "num_chunks": int(iv.num_cells),
                "segment_id": seg or "",
                "chunk_id": chunk_id,
                "source_media_uri": media,
                "tag": query,
            }
        )
    return rows


def _dedupe_arrow_by_segment(table):
    """Keep the best row per segment_id, preserving rank order. Rows with no
    segment_id are all kept -- they cannot be merged into anything."""
    import pyarrow as pa

    seg = table.column("segment_id").to_pylist()
    seen: set = set()
    keep = []
    for i, s in enumerate(seg):
        if not s:
            keep.append(i)
        elif s not in seen:
            seen.add(s)
            keep.append(i)
    if len(keep) == table.num_rows:
        return table
    out = table.take(pa.array(keep))
    return out.set_column(
        out.schema.get_field_index("rank"), "rank",
        pa.array(np.arange(1, out.num_rows + 1, dtype=np.int64)),
    )


def _arrow_csv(table) -> str:
    """Arrow table -> CSV text, streamed through pyarrow rather than the csv
    module: an export can be millions of rows, and building those as Python
    lists costs more than the write."""
    import pyarrow.csv as pacsv

    sink = io.BytesIO()
    # Match what csv.writer produced for the resident export: quote only fields
    # that need it. pyarrow quotes the header row unconditionally whatever the
    # style, so the header is written here and the body appended -- otherwise
    # this emits a valid CSV that is a different file from the one existing
    # consumers parse.
    pacsv.write_csv(
        table,
        sink,
        write_options=pacsv.WriteOptions(
            include_header=False, quoting_style="needed"
        ),
    )
    return ",".join(table.column_names) + "\n" + sink.getvalue().decode()


def _write_arrow_export(table, name: str) -> str:
    """Write an export table as parquet under the OCI export prefix. Returns the
    URI, or "" if export is unconfigured or the upload fails (best-effort, so a
    failed upload never costs the caller their CSV)."""
    prefix = _state["cfg"].export_s3_prefix.strip()
    if not prefix:
        return ""
    import pyarrow.parquet as pq

    sink = io.BytesIO()
    pq.write_table(table, sink, compression="zstd")
    key = f"{prefix.rstrip('/')}/{name}.parquet"
    try:
        oci_s3.put_bytes(key, sink.getvalue(), _state["s3"], "application/octet-stream")
        LOGGER.info("full export parquet written: %s (%d rows)", key, table.num_rows)
        return key
    except (
        ValueError,
        botocore.exceptions.BotoCoreError,
        botocore.exceptions.ClientError,
    ) as exc:
        LOGGER.warning("full export parquet write failed (%s): %s", type(exc).__name__, exc)
        return ""


def _full_export_headers(
    base: str, parquet_uri: str, segset: str, segset_error: str,
    sel, num_rows: int, score_kind: str, took_ms: float,
) -> dict:
    """Response headers for a full-corpus export. `X-NLS-Truncated` and
    `X-NLS-Candidates` are the honest pair: a capped threshold export says how
    much it left behind instead of presenting a partial answer as complete."""
    return {
        "Content-Disposition": f'attachment; filename="{base}.csv"',
        "X-NLS-Export-Name": base,
        "X-NLS-Parquet": parquet_uri,
        "X-NLS-Segset": segset,
        "X-NLS-Segset-Error": segset_error,
        "X-NLS-Rows": str(num_rows),
        "X-NLS-Candidates": str(sel.candidates),
        "X-NLS-Truncated": "1" if sel.truncated else "0",
        "X-NLS-Cutoff": f"{float(sel.cutoff):.6f}",
        "X-NLS-Score-Kind": score_kind,
        "X-NLS-Score-Error-Bound": f"{float(sel.error_bound):.6f}",
        "X-NLS-Elapsed-Ms": str(took_ms),
    }


def _record_full_export(req, request: Request, corpus, table, vec, parquet_uri: str, sel) -> None:
    """Log the export to exp-db. Best-effort, like the resident export path."""
    try:
        db.insert_export(
            {
                "user_email": _current_user(request),
                "query": req.query.strip(),
                "tag": req.tag or req.query.strip(),
                "k": int(req.k or 0),
                "num_results": int(table.num_rows),
                "model_uri": _state["cfg"].model_artifact_uri,
                "embeddings_uri": corpus.corpus_uri,
                "date_from": req.from_date,
                "date_to": req.to_date,
                "vehicle": req.vehicle,
                "drive_id": req.drive_id,
                "thumbs_up": [],
                "thumbs_down": [],
                "search_vector": vec.tolist() if vec is not None else [],
                "parquet_uri": parquet_uri,
            }
        )
    except Exception as exc:  # noqa: BLE001 - never block the download
        LOGGER.warning("full export not recorded: %s", exc)


def _publish_full_export(
    req, request: Request, base: str, parquet_uri: str, segset_label: str,
    sel, num_rows: int, corpus, score_kind: str, segset_error: str = "",
) -> None:
    """Publish a finished export to the Recent scans panel.

    An export is only useful to one person if it lives in their browser's
    downloads. The panel is the shared record of what has been produced and
    where the artifact landed, so an in-app export belongs in it for the same
    reason a Lilypad scan does -- the difference is `source`, not visibility.

    Best-effort: an export that downloaded fine must not fail because the
    listing row could not be written.
    """
    tag = req.tag or req.query.strip()
    try:
        ok = db.insert_scan_job(
            {
                "execution_id": base,
                "user_email": _current_user(request),
                "tags": [tag],
                "thresholds": {tag: float(sel.cutoff)},
                "output_dir": parquet_uri.rsplit("/", 1)[0] if parquet_uri else "",
                "lance_uri": parquet_uri,
                "console_url": "",
                "status": "SUCCEEDED",
                "register_segset": bool(req.create_segment_set),
                "segset_name": segset_label,
                "filters": {
                    "date_from": req.from_date,
                    "date_to": req.to_date,
                    "vehicle": req.vehicle,
                    "drive_id": req.drive_id,
                    "segment_set_uuid": req.segment_set_uuid,
                    "filter_lance_uri": req.filter_lance_uri,
                },
                "source": "app",
            }
        )
        if ok:
            db.set_scan_counts(
                base,
                {
                    "num_segments": int(num_rows),
                    "num_clips_scanned": int(corpus.num_rows),
                    "candidates": int(sel.candidates),
                    "truncated": bool(sel.truncated),
                    "score_kind": score_kind,
                },
            )
        # Always record an OUTCOME, even a negative one. The panel treats
        # "registration requested, nothing recorded" as still-in-progress, so a
        # blank here shows as "pending..." for an export that will never register
        # anything.
        outcome = segset_label or segset_error or (
            "not registered" if req.create_segment_set else ""
        )
        if outcome:
            db.set_scan_segset(base, "", outcome)
    except Exception as exc:  # noqa: BLE001 - never fail a completed download
        LOGGER.warning("full export not published to Recent scans: %s", exc)


@app.post("/api/full_threshold")
def full_threshold(req: FullThresholdRequest, request: Request) -> dict:
    """Fit a cutoff for a full-corpus query from labeled marks, and return the
    next batch of boundary clips to label.

    Same operating-point selection as `/api/threshold_search`, against the full
    corpus's screening scores. Marks are addressed by `row` (a position in the
    34M table), which is what full-corpus hits carry; `index` is a resident-corpus
    address and is ignored here.
    """
    query = (req.query or "").strip()
    if not query:
        raise HTTPException(400, "query must not be empty")
    if not _state.get("model_ready"):
        raise HTTPException(503, "model still loading")
    with _FULL_LOCK:
        corpus = _FULL["corpus"]
    if corpus is None:
        _full_corpus_begin_load()
        raise HTTPException(503, "full corpus is loading; poll /api/full_corpus_status")

    vec = search_engine.encode_query(
        query, _state["processor"], _state["model"], _state["cfg"].device
    )
    scores, err = corpus.score(vec)
    start_unix, end_unix = _date_bounds(req.from_date, req.to_date, corpus)
    mask = corpus.filter_mask(
        **_merge_filters(
            {
                "vehicles": set(_parse_vehicles(req.vehicle)) or None,
                "date_range": (start_unix, end_unix) if (start_unix or end_unix) else None,
                "run_uuids": set(_parse_drive_ids(req.drive_id)) or None,
                "segment_ids": _full_segment_ids(req.segment_set_uuid),
            },
            _full_lance_filter(req.filter_lance_uri),
        )
    )
    allowed = mask if mask is not None else np.ones(corpus.num_rows, dtype=bool)

    def _valid(rows: list) -> list[int]:
        seen: dict[int, None] = {}
        for r in rows:
            if r is not None and 0 <= int(r) < corpus.num_rows:
                seen[int(r)] = None
        return list(seen)

    pos_idx = _valid([m.row for m in req.marks if m.mark == "up"])
    neg_idx = _valid([m.row for m in req.marks if m.mark == "down"])
    dropped = sum(1 for m in req.marks if m.row is None)
    labeled = set(pos_idx) | set(neg_idx)
    pos_scores = scores[np.asarray(pos_idx, dtype=np.int64)] if pos_idx else np.array([])
    neg_scores = scores[np.asarray(neg_idx, dtype=np.int64)] if neg_idx else np.array([])

    fit, note = None, ""
    if pos_idx and neg_idx:
        fit = search_engine.fit_threshold(
            pos_scores, neg_scores, objective=req.objective, beta=req.beta,
            min_precision=req.min_precision, val_fraction=req.val_fraction,
        )
        if fit["objective"] == "precision" and not fit["precision_floor_met"]:
            note = (
                f"No cutoff reaches precision >= {req.min_precision:.2f} on the "
                "current labels; showing the highest-precision cutoff."
            )
    else:
        missing = "positive (\N{THUMBS UP SIGN})" if not pos_idx else "negative (\N{THUMBS DOWN SIGN})"
        note = f"Need at least one {missing} label to fit a threshold."
    if dropped:
        note = (
            f"{dropped} mark(s) carried no full-corpus row and were ignored. "
            + note
        ).strip()

    tau = float(fit["threshold"]) if fit else None
    stats = search_engine.score_stats(scores)
    suggested = search_engine.heuristic_threshold(stats)
    tau_for_sampling = tau if tau is not None else suggested
    sample_idx = search_engine.stratified_boundary_sample(
        scores, allowed, labeled, req.sample_size, tau=tau_for_sampling, band=req.band
    )
    sample_rows = np.asarray(sample_idx, dtype=np.int64)
    sample_hits = [
        {
            "row": int(r), "index": -1,
            "score": round(float(scores[r]), 4),
            "score_error_bound": round(float(err), 4),
            **{
                key: val[j]
                for key, val in corpus.to_arrow(sample_rows, scores[sample_rows])
                .select(["chunk_id", "run_uuid", "segment_id", "source_media_uri", "vehicle"])
                .to_pydict().items()
            },
        }
        for j, r in enumerate(sample_rows)
    ]
    # How many rows the fitted cutoff would actually export -- the number that
    # turns "tau = 0.05" into a decision, since a plausible-looking cutoff can
    # select a tenth of the corpus.
    selected = int((scores[allowed] >= np.float32(tau_for_sampling)).sum())
    return {
        "threshold": tau,
        "suggested_threshold": suggested,
        "fit": fit,
        "note": note,
        "num_up": len(pos_idx),
        "num_down": len(neg_idx),
        "score_kind": "bounded_approx",
        "score_error_bound": round(float(err), 4),
        "up_scores": [round(float(s), 4) for s in pos_scores.tolist()],
        "down_scores": [round(float(s), 4) for s in neg_scores.tolist()],
        "histogram": _score_histogram(scores, tau_for_sampling),
        "sample": sample_hits,
        "num_rows_searched": corpus.num_rows,
        "candidates": int(allowed.sum()),
        "selected_at_threshold": selected,
    }


@app.get("/api/full_corpus_status")
def full_corpus_status() -> dict:
    import full_corpus

    with _FULL_LOCK:
        status, error, started = _FULL["status"], _FULL["error"], _FULL["started"]
        corpus = _FULL["corpus"]
    out = {
        "status": status,
        "error": error,
        "corpus_uri": full_corpus.DEFAULT_CORPUS_TABLE_URI,
        "model": full_corpus.CORPUS_MODEL,
        "num_rows": corpus.num_rows if corpus is not None else 0,
    }
    if status == "loading":
        out["elapsed_s"] = round(time.time() - started, 1)
    return out


@app.post("/api/full_corpus_load")
def full_corpus_load() -> dict:
    """Begin (or report) the on-demand load. Returns immediately."""
    return {"status": _full_corpus_begin_load()}


class FullVectorSearchRequest(BaseModel):
    """Rank the full corpus by a pre-computed vector (resume a saved search, or
    search by an uploaded example). Mirrors FullSearchRequest minus the text."""

    vector: list[float]
    query: str = ""
    limit: int = 50
    page: int = 0
    from_date: str | None = None
    to_date: str | None = None
    vehicle: str | None = None
    drive_id: str | None = None
    segment_set_uuid: str | None = None
    filter_lance_uri: str | None = None


class FullRefineRequest(BaseModel):
    """Re-rank the full corpus from relevance feedback. Marks are addressed by
    `row`; the marked clips' original embeddings are fetched to build the
    prototype, since this corpus holds only the quantized screen."""

    query: str = ""
    marks: list[Mark] = []
    negative_weight: float = 0.5
    text_weight: float = 0.3
    limit: int = 50
    page: int = 0
    from_date: str | None = None
    to_date: str | None = None
    vehicle: str | None = None
    drive_id: str | None = None
    segment_set_uuid: str | None = None
    filter_lance_uri: str | None = None


@app.post("/api/full_search_by_vector")
def full_search_by_vector(req: FullVectorSearchRequest) -> dict:
    """Rank the WHOLE corpus by a supplied vector.

    Resuming a saved search used to fall through to the resident corpus, so a
    reloaded tag was silently ranked over 2M clips while a fresh search of the
    same text covered 34M -- same UI, same wording, different corpus.
    """
    if not req.vector:
        raise HTTPException(400, "empty search vector")
    corpus = _require_full_corpus()
    vec = np.asarray(req.vector, dtype=np.float32)
    if vec.ndim != 1 or vec.shape[0] != corpus.dim:
        raise HTTPException(
            400,
            f"vector dim {vec.shape[0]} != corpus dim {corpus.dim} -- resume "
            "against the corpus the search was originally run on",
        )
    return _full_rank(corpus, vec, req, label=req.query or "saved vector")


@app.post("/api/full_refine")
def full_refine(req: FullRefineRequest) -> dict:
    """Re-rank the whole corpus from 👍/👎 marks (Rocchio prototype).

    The marked clips' 768-d embeddings are read from S3 -- a handful of rows,
    one round trip -- because the resident screen is quantized and PCA-reduced
    and cannot reconstruct a query direction of the quality feedback needs.
    """
    corpus = _require_full_corpus()
    pos = [int(m.row) for m in req.marks if m.mark == "up" and m.row is not None]
    neg = [int(m.row) for m in req.marks if m.mark == "down" and m.row is not None]
    pos = [r for r in dict.fromkeys(pos) if 0 <= r < corpus.num_rows]
    neg = [r for r in dict.fromkeys(neg) if 0 <= r < corpus.num_rows]
    if not pos:
        raise HTTPException(400, "mark at least one positive to re-rank")

    text_vec = None
    if req.text_weight > 0 and (req.query or "").strip():
        if not _state.get("model_ready"):
            raise HTTPException(503, "model still loading")
        text_vec = search_engine.encode_query(
            req.query.strip(), _state["processor"], _state["model"],
            _state["cfg"].device,
        )
    try:
        mat = corpus.vectors_for(pos + neg)
    except RuntimeError as exc:
        raise HTTPException(409, str(exc))
    vec = _rocchio(
        mat[: len(pos)], mat[len(pos):], text_vec,
        req.negative_weight, req.text_weight,
    )
    # Drop the rejected clips from the results. Rocchio moves the query direction
    # away from the negative centroid, which is not the same as removing them: a
    # thumbs-down clip that still sits near the positives came straight back at
    # the top of the re-rank, so the feedback looked ignored. Positives are kept
    # -- seeing them hold their place is how you tell the re-rank worked.
    out = _full_rank(corpus, vec, req, label=req.query or "refined", exclude_rows=neg)
    out["refined_from"] = {
        "num_up": len(pos), "num_down": len(neg), "excluded": len(neg),
    }
    return out


def _rocchio(
    pos: np.ndarray, neg: np.ndarray, text_vec, negative_weight: float,
    text_weight: float,
) -> np.ndarray:
    """Unit-norm prototype: positive centroid, pushed away from the negative
    centroid and optionally blended with the text direction.

    Mirrors `search_engine.refine_query`, which reads vectors off a resident
    Corpus this one does not have -- the arithmetic is the same, the source of
    the vectors is not.
    """
    def _unit(v):
        n = float(np.linalg.norm(v))
        return v / n if n else v

    w = _unit(np.asarray(pos, dtype=np.float32).mean(axis=0))
    if neg.size:
        w = _unit(w - negative_weight * _unit(np.asarray(neg, dtype=np.float32).mean(axis=0)))
    if text_vec is not None and text_weight > 0:
        w = _unit((1.0 - text_weight) * w + text_weight * _unit(np.asarray(text_vec, dtype=np.float32)))
    return w.astype(np.float32)


def _require_full_corpus():
    """The resident full corpus, or a 503 that starts the load."""
    with _FULL_LOCK:
        corpus, status, error = _FULL["corpus"], _FULL["status"], _FULL["error"]
    if corpus is None:
        if status == "error":
            raise HTTPException(503, f"full corpus load failed: {error}")
        _full_corpus_begin_load()
        raise HTTPException(
            503, "full corpus is loading (~2 min); poll /api/full_corpus_status and retry"
        )
    return corpus


def _full_rank(corpus, vec: np.ndarray, req, *, label: str, exclude_rows=None) -> dict:
    """Shared ranking + response envelope for the full-corpus vector endpoints.

    Accepts either field name for the page size: the vector/refine requests use
    `limit`, the older window request uses `page_size`.
    """
    limit = int(getattr(req, "limit", None) or getattr(req, "page_size", None) or 50)
    if limit <= 0 or limit > 1000:
        raise HTTPException(400, "limit must be between 1 and 1000")
    if req.page < 0:
        raise HTTPException(400, "page must not be negative")
    offset = req.page * limit
    if offset + limit > _FULL_MAX_DEPTH:
        raise HTTPException(400, f"full-corpus paging stops at {_FULL_MAX_DEPTH:,} results")
    start_unix, end_unix = _date_bounds(req.from_date, req.to_date, corpus)
    t0 = time.perf_counter()
    hits, candidates = corpus.search(
        vec,
        limit=limit,
        offset=offset,
        **_merge_filters(
            {
                "vehicles": set(_parse_vehicles(req.vehicle)) or None,
                "run_uuids": set(_parse_drive_ids(req.drive_id)) or None,
                "date_range": (start_unix, end_unix) if (start_unix or end_unix) else None,
                "segment_ids": _full_segment_ids(req.segment_set_uuid),
            },
            _full_lance_filter(req.filter_lance_uri),
        ),
        exclude_rows=exclude_rows,
    )
    took_ms = round((time.perf_counter() - t0) * 1000, 1)
    LOGGER.info(
        "full vector search %r -> %d hits over %d rows in %.1fms%s",
        label, len(hits), corpus.num_rows, took_ms, _separation_note(hits),
    )
    return _full_payload(hits, candidates, corpus, req, took_ms, label, page_size=limit)


@app.post("/api/full_search")
def full_search(req: FullSearchRequest) -> dict:
    """Top-`limit` clips over the ENTIRE corpus. No S3 access per query.

    Scores are int8 screening scores: the exact PCA score of each hit lies
    within +/- `score_error_bound`. They are NOT comparable to the resident
    corpus's fp32 scores, so a threshold calibrated there does not transfer --
    hence `score_kind` on every row.
    """
    query = (req.query or "").strip()
    if not query:
        raise HTTPException(400, "query must not be empty")
    if req.limit <= 0 or req.limit > 1000:
        raise HTTPException(400, "limit must be between 1 and 1000")
    if req.page < 0:
        raise HTTPException(400, "page must not be negative")
    offset = req.page * req.limit
    if offset + req.limit > _FULL_MAX_DEPTH:
        raise HTTPException(
            400,
            f"full-corpus paging stops at {_FULL_MAX_DEPTH:,} results; "
            "narrow the filters or export instead",
        )
    if not _state.get("model_ready"):
        raise HTTPException(503, "model still loading")

    with _FULL_LOCK:
        corpus, status, error = _FULL["corpus"], _FULL["status"], _FULL["error"]
    if corpus is None:
        if status == "error":
            raise HTTPException(503, f"full corpus load failed: {error}")
        _full_corpus_begin_load()
        raise HTTPException(
            503,
            "full corpus is loading (~2 min); poll /api/full_corpus_status and retry",
        )

    start_unix, end_unix = _date_bounds(req.from_date, req.to_date, corpus)
    vec = search_engine.encode_query(
        query, _state["processor"], _state["model"], _state["cfg"].device
    )
    t0 = time.perf_counter()
    hits, candidates = corpus.search(
        vec,
        limit=req.limit,
        offset=offset,
        **_merge_filters(
            {
                "vehicles": set(_parse_vehicles(req.vehicle)) or None,
                "run_uuids": set(_parse_drive_ids(req.drive_id)) or None,
                "date_range": (start_unix, end_unix) if (start_unix or end_unix) else None,
                "segment_ids": _full_segment_ids(req.segment_set_uuid),
            },
            _full_lance_filter(req.filter_lance_uri),
        ),
    )
    took_ms = round((time.perf_counter() - t0) * 1000, 1)
    LOGGER.info(
        "full search %r -> %d hits over %d rows in %.1fms%s",
        query, len(hits), corpus.num_rows, took_ms, _separation_note(hits),
    )
    return _full_payload(hits, candidates, corpus, req, took_ms, query)


def _separation_note(hits) -> str:
    """Whether this query actually separated anything, logged with the search.

    A query can be answered fast and correctly over all 34M rows and still be
    useless, because the model puts no distance between the best match and the
    rest. Latency alone never shows that, so a bad tag looked identical to a good
    one in the logs. `spread/eps` is the readable part: below ~1 the top hits sit
    inside the screening error bound of each other and their order is noise.
    """
    if not hits:
        return ""
    sc = [float(h.score) for h in hits]
    eps = float(hits[0].score_error_bound) or 1e-9
    spread = max(sc) - min(sc)
    return (
        f" | top={max(sc):.4f} lo={min(sc):.4f} spread={spread:.4f}"
        f" eps={eps:.4f} spread/eps={spread / eps:.1f}"
    )


def _full_payload(
    hits, candidates: int, corpus, req, took_ms: float, label: str,
    page_size: int | None = None,
) -> dict:
    """The response envelope shared by every full-corpus ranking endpoint.

    Identical in shape to what /api/search returns, so the grid renders these
    with no special-casing. `index` is -1 because there is no resident-corpus
    row to address; `row` is the full-corpus position, which is what
    /api/full_rescore and /api/full_refine key on.
    """
    scores = [float(h.score) for h in hits]
    payload = [
        {
            "rank": h.rank,
            "score": round(float(h.score), 4),
            "score_error_bound": round(float(h.score_error_bound), 4),
            "chunk_id": h.chunk_id,
            "run_uuid": h.run_uuid,
            "segment_id": h.segment_id,
            "start_timestamp_ns": _ns(h.chunk_start_unix),
            "end_timestamp_ns": _ns(h.chunk_end_unix),
            "start_utc": _utc(h.chunk_start_unix),
            "end_utc": _utc(h.chunk_end_unix),
            "source_media_uri": h.source_media_uri,
            "vehicle": h.vehicle,
            "dx_internal_id": h.dx_internal_id,
            # Dataset row position, so /api/full_rescore can address these rows.
            "row": h.row,
            "index": -1,
        }
        for h in hits
    ]
    return {
        "hits": payload,
        # What the pager can reach, not the match count: every filtered row is
        # ranked, but only the first _FULL_MAX_DEPTH are reachable by paging.
        # `candidates` carries the real number.
        "total": min(candidates, _FULL_MAX_DEPTH),
        "candidates": candidates,
        "page": req.page,
        "page_size": page_size if page_size is not None else req.limit,
        "elapsed_ms": took_ms,
        "label": label,
        "score_lo": round(min(scores), 4) if scores else 0.0,
        "score_hi": round(max(scores), 4) if scores else 0.0,
        "funnel": {"corpus_total": corpus.num_rows},
        "full_corpus": True,
        "score_kind": "bounded_approx",
        # Provenance, so the UI can state what was actually searched instead of
        # asking the user to trust that "full corpus" means the whole table.
        "num_rows_searched": corpus.num_rows,
        "candidates_after_filters": candidates,
        "score_error_bound": round(float(hits[0].score_error_bound), 4) if hits else 0.0,
        "corpus_uri": corpus.corpus_uri,
        "corpus_version": corpus.dataset_version,
        "corpus_loaded_utc": _utc(int(corpus.loaded_at)) if corpus.loaded_at else "",
        "filters_applied": {
            "vehicle": sorted(_parse_vehicles(req.vehicle)) or None,
            "drive_id": sorted(_parse_drive_ids(req.drive_id)) or None,
            "from_date": req.from_date or None,
            "to_date": req.to_date or None,
            "segment_set_uuid": req.segment_set_uuid or None,
            "filter_lance_uri": req.filter_lance_uri or None,
        },
    }
