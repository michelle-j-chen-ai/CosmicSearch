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
import contextlib
import dataclasses
import base64
import binascii
import datetime as dt
import gzip
import hashlib
import html
import io
import logging
import os
import random
import re
import shutil
import tempfile
import threading
import time
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
import deployment
import dora_client
import local_cache
import oci_s3
import search_engine
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    RedirectResponse,
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
    # The default project's table, stamped onto persisted rows (exported
    # vectors) as the space those numbers belong to.
    _state["active_uri"] = deployment.get(None).corpus_table_uri
    threading.Thread(target=_warm_engine, name="engine-warmup", daemon=True).start()
    # Pre-warm each project's DORA gRPC channel + the machine token now
    # (independent of the model), so the first segment-set selection doesn't pay
    # channel/auth setup.
    for name in deployment.enabled():
        threading.Thread(
            target=dora_client.prewarm, args=(_dora_host(name),),
            name=f"dora-prewarm-{name}", daemon=True,
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
        import full_corpus

        db.backfill_catalog(project=deployment.default(), model=full_corpus.CORPUS_MODEL)
        # Start every enabled project's corpus load here rather than waiting
        # for someone to search: the read and decode take minutes, and paying
        # that on a user's first query reads as a hang. Each runs on its own
        # thread and never gates readiness.
        try:
            for name in deployment.enabled():
                _full_corpus_begin_load(name)
            LOGGER.info("corpus loads started: %s", ", ".join(deployment.enabled()))
            _start_corpus_refresh_schedule()
        except Exception as exc:  # noqa: BLE001 -- warm-up must not wedge startup
            LOGGER.warning("full corpus pre-warm could not start: %s", exc)
        _state["ready"] = True
    except Exception as exc:  # noqa: BLE001 -- warmup must record failure, not vanish
        _state["load_error"] = str(exc)
        LOGGER.exception("engine warmup failed: %s", exc)


def _project_of(req) -> str:
    """The project a request addresses; the default when the model has no field."""
    return getattr(req, "project", None) or deployment.default()


def _project_or_404(name: str | None) -> "deployment.Project":
    try:
        return deployment.get(name)
    except KeyError:
        raise HTTPException(404, f"unknown project {name!r}; one of {deployment.names()}")


def _dora_host(project: str) -> str | None:
    """The Data Explorer host for a project's segment sets, or None for the env default."""
    return _project_or_404(project).dora_hostname or None


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


def _segment_ids(uuid: str, hostname: str | None = None) -> "frozenset[str] | None":
    """The set's segment_ids if resident, else None while a loader runs.

    Fetched in the BACKGROUND and never inline: a large Data Explorer set is
    millions of ids over thousands of gRPC pages, and a search must not block on
    it. The caller runs unfiltered and the UI re-polls; by the time the user has
    composed a query the ids are usually already here.
    """
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
                ids = dora_client.fetch_segment_ids(uuid, progress=_progress, hostname=hostname)
                _write_seg_cache(uuid, ids)
            rec = {"status": "done", "ids": ids, "count": len(ids), "err": None}
        except Exception as exc:  # noqa: BLE001 -- a background loader must not die silently
            LOGGER.warning("segment-set load failed for %s: %s", uuid, exc)
            rec = {"status": "error", "ids": None, "count": 0, "err": str(exc)}
        with _SEG_LOCK:
            _SEG[uuid] = rec
        LOGGER.info("segment-set %s -> %s (%d ids)", uuid, rec["status"], rec["count"])

    threading.Thread(target=_load, name=f"segload-{uuid[:8]}", daemon=True).start()
    return None


def _seg_state(uuid: str) -> dict:
    with _SEG_LOCK:
        rec = _SEG.get(uuid) or {"status": "idle", "count": 0, "err": None}
    return {
        "status": rec["status"],
        "count": rec["count"],
        "ready": rec["status"] == "done",
        "error": rec.get("err"),
    }


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


def _read_filter_table(uri: str):
    """The whole downsample dataset at ``uri`` as an Arrow table, for exports that
    carry its columns through. Uses the same on-disk copy `_lance_filter_ids`
    keeps, so a dataset is downloaded once however it is used."""
    _lance_filter_ids(uri)  # ensures the local copy exists
    local_dir = _lance_filter_cache_dir(uri)
    if uri.rstrip("/").endswith(".lance"):
        import lance

        return lance.dataset(str(local_dir)).to_table()
    import pyarrow.parquet as pq

    return pq.read_table(str(local_dir))


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


    # Register the exported (deduped) segments as a DORA segment set, named after


# Cap on how many frames a client may send per upload (video sends ~8; the encoder
# resamples to the model's fixed 8 either way). Bounds the request + decode work.
_UPLOAD_MAX_FRAMES = 16
_UPLOAD_MAX_FRAME_BYTES = 8 * 1024 * 1024  # per decoded frame (a 448-ish jpeg is tiny)


@app.get("/ui/segment_sets")
def segment_sets(name_filter: str = "", prefetch: str = "", project: str | None = None):
    """Data Explorer sets whose name matches, for the downsample picker. With
    `prefetch`, start loading that set's ids in the background instead, so the
    pull overlaps the user composing a query."""
    if prefetch.strip():
        _segment_ids(prefetch.strip(), _dora_host(project))
        return _seg_state(prefetch.strip())
    if not name_filter.strip():
        return []
    try:
        sets = dora_client.list_segment_sets(name_filter.strip(), hostname=_dora_host(project))
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


@app.get("/ui/video")
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


app.mount("/static", StaticFiles(directory=os.path.join(HERE, "web")), name="static")


@app.get("/")
def index(request: Request) -> FileResponse:
    # Record a visit for usage analytics, but only for real (IAP-authenticated)
    # users -- skip health-probe / unauthenticated hits so the counts are real.
    user = _current_user(request)
    if user and user != "local":
        analytics.record_visit(user, uuid.uuid4().hex, time.time())
    return FileResponse(os.path.join(HERE, "web", "index.html"))


def _fmt_ts(value) -> str:
    """Render a timestamp/datetime cell compactly (UTC)."""
    if value is None:
        return ""
    try:
        return value.strftime("%Y-%m-%d %H:%M")  # datetime from the driver
    except AttributeError:
        return html.escape(str(value))


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

    # ---- Feedback labels (👍/👎) ----
    # feedback_marks is one row per distinct judgment (query, segment, user), so
    # its counts are the true totals.
    fb = db.feedback_totals()
    fb_total_up = fb["up"]
    fb_total_down = fb["down"]
    fb_queries = fb["queries"]
    unavailable = (
        ""
        if (exports or seg_sets)
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
  <p class=sub>Each 👍/👎 you label during the refine + threshold steps is recorded here
     (from <span class=mono>{_esc(db.SCHEMA_NAME)}.feedback_marks</span>).</p>
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
# The load costs ~2 minutes and ~12GB. It is started by the warm-up thread (and
# by the first request that needs it, if that lands first) rather than in the
# startup event, so uvicorn binds the port immediately and the UI can poll
# `/api/full_corpus_status` while it runs.
_CORPORA_LOCK = threading.Lock()
# One slot per project: {"corpus", "status", "error", "started"}. Created idle on
# first reference so a valid but not-yet-loaded project has a state to report.
_CORPORA: dict[str, dict] = {}


def _slot(project: str) -> dict:
    """This project's slot. Callers hold _CORPORA_LOCK."""
    slot = _CORPORA.get(project)
    if slot is None:
        slot = _CORPORA[project] = {"corpus": None, "status": "idle", "error": "", "started": 0.0}
    return slot


def _full_load_worker(project: str) -> None:
    import full_corpus

    spec = deployment.get(project)
    try:
        corpus = full_corpus.load(table_uri=spec.corpus_table_uri, mp4_prefix=spec.mp4_prefix)
        with _CORPORA_LOCK:
            slot = _slot(project)
            slot["corpus"] = corpus
            slot["status"] = "ready"
            slot["error"] = ""
        LOGGER.info("%s corpus ready: %d rows", project, corpus.num_rows)
    except Exception as exc:  # noqa: BLE001 -- surfaced through the status endpoint
        LOGGER.exception("%s corpus load failed", project)
        with _CORPORA_LOCK:
            slot = _slot(project)
            slot["corpus"] = None
            slot["status"] = "error"
            slot["error"] = f"{type(exc).__name__}: {exc}"


def _full_corpus_begin_load(project: str | None = None) -> str:
    """Start the project's background load if it is not already running/ready."""
    project = _project_or_404(project).name
    with _CORPORA_LOCK:
        slot = _slot(project)
        if slot["status"] in ("loading", "ready"):
            return slot["status"]
        slot["status"] = "loading"
        slot["error"] = ""
        slot["started"] = time.time()
    threading.Thread(
        target=_full_load_worker, args=(project,), name=f"corpus-load-{project}", daemon=True
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


def _full_refresh_worker(project: str | None = None) -> None:
    """Drop the project's resident corpus, then load the current one.

    Dropping FIRST is not a style choice: the process already holds ~24GB (model,
    browse matrix, full corpus) against a 32Gi ceiling, so building a replacement
    beside the old one needs ~37GB and is killed part-way. The cost is that
    full-corpus search is unavailable for the length of the reload; browse is
    unaffected, and searches in that window get the same 503-and-poll they get
    before the first load.
    """
    import gc

    project = project or deployment.default()
    with _CORPORA_LOCK:
        slot = _slot(project)
        previous = slot["corpus"]
        slot["corpus"] = None
        slot["status"] = "loading"
        slot["error"] = ""
        slot["started"] = time.time()
    del previous
    gc.collect()
    _full_load_worker(project)


# ------------------------- daily corpus refresh -----------------------------
# The table this app serves grows by append, and the resident copy is a snapshot
# taken when the instance loaded it. Without a refresh, clips added after a
# deploy stay invisible to every search and export until the next one.
#
# In-process rather than driven by Cloud Scheduler: Cloud Run routes a request
# to ONE instance, and this service runs up to `max_instances`. An HTTP trigger
# would refresh whichever instance the load balancer happened to pick and leave
# the others serving an older version. That is worse than uniform staleness,
# because a row position is only valid against the version it was issued from
# (see Hit.row) -- so a `rows` handle from one instance draws a 409 on another,
# intermittently, depending on routing. Every instance waking on its own timer
# has no addressing problem to get wrong.
#
# NLS_CORPUS_REFRESH_UTC is "HH:MM" in UTC; empty disables the schedule. The
# default is 10:00 UTC (~3am US Pacific), when a few minutes without
# full-corpus search costs least.
# "off" (or empty) disables it. Empty is unusable in practice: the NLS_* vars
# are Secret Manager values so they survive a deploy, and Secret Manager will
# not store an empty payload -- so there has to be a word that means off.
_REFRESH_OFF = {"", "off", "none", "never", "disabled"}
_REFRESH_AT = os.getenv("NLS_CORPUS_REFRESH_UTC", "10:00").strip()
if _REFRESH_AT.lower() in _REFRESH_OFF:
    _REFRESH_AT = ""

# Spread across instances so they do not all drop their corpus in the same
# window: a refresh takes full-corpus search offline on the instance running it,
# and simultaneous refreshes would take the service offline instead of a slice
# of its capacity. Drawn once per process, so each instance keeps a stable time.
_REFRESH_JITTER_S = float(os.getenv("NLS_CORPUS_REFRESH_JITTER_S", "1800"))

_REFRESH: dict = {
    "scheduled_at": "", "next_utc": "", "last_check_utc": "", "last_result": "",
    "refreshes": 0, "skipped": 0, "table_version": None, "error": "",
}


def _seconds_until(hhmm: str, now: float) -> float:
    """Seconds from `now` to the next UTC occurrence of `hhmm`.

    UTC deliberately: it has no DST, so the gap between two runs is always 24h
    and never silently 23 or 25.
    """
    hh, mm = (int(part) for part in hhmm.split(":", 1))
    if not (0 <= hh < 24 and 0 <= mm < 60):
        raise ValueError(f"not a UTC time of day: {hhmm!r}")
    t = time.gmtime(now)
    midnight = now - (t.tm_hour * 3600 + t.tm_min * 60 + t.tm_sec)
    target = midnight + hh * 3600 + mm * 60
    return target - now if target > now else target + 86400 - now


def _corpus_version_check(resident, project: str | None = None) -> dict:
    """Has the project's table moved since `resident` was read? One metadata call."""
    import full_corpus

    project = project or deployment.default()
    held = getattr(resident, "dataset_version", None)
    latest = full_corpus.latest_version(deployment.get(project).corpus_table_uri)
    _REFRESH["table_version"] = latest
    _REFRESH.setdefault("table_versions", {})[project] = latest
    _REFRESH["last_check_utc"] = _utc(int(time.time()))
    # An unknown version is not evidence of sameness. Refresh rather than skip:
    # the cost of a needless reload is minutes of one instance, the cost of a
    # wrongly skipped one is a day of serving clips that no longer match.
    changed = latest is None or held is None or latest != held
    return {"changed": changed, "held_version": held, "table_version": latest}


def _corpus_refresh_tick(project: str | None = None) -> str:
    """One scheduled check of one project. Reloads only if its table has moved."""
    project = project or deployment.default()
    with _CORPORA_LOCK:
        slot = _slot(project)
        status, resident = slot["status"], slot["corpus"]
    if status == "loading":
        return "skipped: a load is already in flight"
    if resident is None:
        # Nothing resident to make stale. Starting a load here would be a
        # surprise on an instance that has not been asked for one.
        return "skipped: no corpus resident"
    verdict = _corpus_version_check(resident, project)
    if not verdict["changed"]:
        _REFRESH["skipped"] += 1
        return f"skipped: version {verdict['table_version']} unchanged"
    _full_refresh_worker(project)  # inline: this thread has nothing else to do
    with _CORPORA_LOCK:
        slot = _slot(project)
        after, error = slot["corpus"], slot["error"]
    if after is None:
        return f"failed: {error}"
    _REFRESH["refreshes"] += 1
    return (
        f"reloaded {verdict['held_version']} -> {after.dataset_version} "
        f"({after.num_rows} rows)"
    )


def _corpus_refresh_all() -> str:
    """One scheduled check across every enabled project."""
    return "; ".join(f"{p}: {_corpus_refresh_tick(p)}" for p in deployment.enabled())


def _corpus_refresh_scheduler() -> None:
    """Wake once a day and refresh this instance's corpus if the table moved."""
    jitter = random.uniform(0, _REFRESH_JITTER_S)
    while True:
        delay = _seconds_until(_REFRESH_AT, time.time()) + jitter
        _REFRESH["next_utc"] = _utc(int(time.time() + delay))
        time.sleep(delay)
        try:
            result = _corpus_refresh_all()
            _REFRESH["last_result"] = result
            _REFRESH["error"] = ""
            LOGGER.info("daily corpus refresh: %s", result)
        except Exception as exc:  # noqa: BLE001 -- a bad night must not end the schedule
            _REFRESH["error"] = f"{type(exc).__name__}: {exc}"
            _REFRESH["last_result"] = "error"
            LOGGER.exception("daily corpus refresh failed")


def _start_corpus_refresh_schedule() -> None:
    """Start the daily schedule, or explain in the log why there isn't one."""
    if not _REFRESH_AT:
        LOGGER.info("daily corpus refresh disabled (NLS_CORPUS_REFRESH_UTC empty)")
        return
    try:
        _seconds_until(_REFRESH_AT, time.time())  # validate before promising it
    except Exception as exc:  # noqa: BLE001
        _REFRESH["error"] = f"invalid NLS_CORPUS_REFRESH_UTC: {exc}"
        LOGGER.error("daily corpus refresh not scheduled: %s", _REFRESH["error"])
        return
    _REFRESH["scheduled_at"] = f"{_REFRESH_AT} UTC"
    threading.Thread(
        target=_corpus_refresh_scheduler, name="corpus-refresh", daemon=True
    ).start()
    LOGGER.info(
        "daily corpus refresh scheduled at %s UTC (+ up to %.0fs jitter)",
        _REFRESH_AT, _REFRESH_JITTER_S,
    )


class CalibrateRequest(BaseModel):
    """Fit a cutoff for a full-corpus query from labeled marks.

    Marks are addressed by `row`, a position in the full corpus. They used to be
    addressed by a resident-corpus index, which full-corpus hits do not carry --
    so every label failed the bounds check and was silently dropped, leaving each
    tag on the label-free mean+3*std heuristic with nothing saying so.
    """

    project: str | None = None
    query: str
    marks: list[Mark] = []
    objective: str = "f1"
    beta: float = 1.0
    min_precision: float = 0.9
    val_fraction: float = 0.0
    sample_size: int = 12
    band: float = 0.02
    # Where to draw the sample from. None centres it on the fitted cutoff, or on
    # the suggested one before there are labels. The sweep sets it to wherever
    # the line has been dragged, so the clips on screen are the clips at the
    # cutoff being considered rather than the one the fit happened to choose.
    at_tau: float | None = None
    from_date: str | None = None
    to_date: str | None = None
    vehicle: str | None = None
    drive_id: str | None = None
    filter_lance_uri: str | None = None
    segment_set_uuid: str | None = None




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
        # An empty intersection is a real answer -- "no row satisfies both" --
        # and stays an empty set. `filter_mask` distinguishes that from None.
        out[key] = val if cur is None else (set(cur) & set(val))
    return out


def _full_filters(req) -> dict:
    """Filter kwargs for `full_corpus` from any request carrying the standard
    filter fields. One definition so no endpoint can quietly support a different
    subset than the others."""
    corpus = _require_full_corpus(_project_of(req))
    start_unix, end_unix = _date_bounds(
        getattr(req, "from_date", None), getattr(req, "to_date", None), corpus
    )
    return _merge_filters(
        _merge_filters(
            {
                "vehicles": set(_parse_vehicles(getattr(req, "vehicle", None))) or None,
                "run_uuids": set(_parse_drive_ids(getattr(req, "drive_id", None))) or None,
                "date_range": (start_unix, end_unix) if (start_unix or end_unix) else None,
            },
            _segment_set_filter(getattr(req, "segment_set_uuid", None), _project_of(req)),
        ),
        _full_lance_filter(getattr(req, "filter_lance_uri", None)),
    )


def _segment_set_filter(uuid: "str | None", project: str | None = None) -> dict:
    """Filter kwargs restricting the corpus to a Data Explorer segment set.

    A set whose ids are still loading raises 503 rather than returning no filter.
    Running unfiltered would answer a narrower question than the one asked, in the
    same wording and with no indication that the set was ignored -- the caller
    should retry once the set reports ready.
    """
    if not uuid or not uuid.strip():
        return {}
    uuid = uuid.strip()
    ids = _segment_ids(uuid, _dora_host(project))
    if ids is None:
        state = _seg_state(uuid)
        if state["status"] == "error":
            raise HTTPException(502, f"segment set could not be read: {state['error']}")
        raise HTTPException(
            503,
            f"segment set still loading ({state['count']:,} ids so far); retry shortly",
        )
    if not ids:
        raise HTTPException(400, "segment set is empty")
    return {"segment_ids": frozenset(ids)}


# An export holds its selected rows, their metadata columns and (when exact) a
# slab of fetched vectors, in a process that already sits near its memory
# ceiling with the model and the 8.8GB screen resident. Two at once is what
# OOM-kills the container, so exports are serialized rather than queued
# per-request.
_FULL_EXPORT_LOCK = threading.Lock()

# A full-corpus scoring pass allocates one float32 per row -- ~216MB at 54M --
# on top of the resident screen and the encoder, against a 32Gi ceiling. Exports
# hold _FULL_EXPORT_LOCK, but tag creation and refinement score too and used to
# be unbounded: enough concurrent callers could OOM the container, and Cloud Run
# answers an OOM by killing the instance, search included. Two at a time leaves
# headroom; a caller that cannot get a slot is told to retry rather than queued
# behind an unbounded wait.
_SCORING_SLOTS = threading.BoundedSemaphore(2)
_SCORING_WAIT_S = 20.0


@contextlib.contextmanager
def scoring_slot():
    """Permission to run one full-corpus scoring pass. 503s when saturated."""
    if not _SCORING_SLOTS.acquire(timeout=_SCORING_WAIT_S):
        raise HTTPException(
            503,
            {"code": "scoring_busy",
             "message": "too many full-corpus scans in flight; retry shortly",
             "status": 503},
            headers={"Retry-After": "10"},
        )
    try:
        yield
    finally:
        _SCORING_SLOTS.release()

# Hard ceiling on a threshold export regardless of what the caller asks for.
# A permissive tau matches a tenth of the corpus (a mean+3*std cutoff selected
# 3.4M rows in practice), and the row count -- not the scan -- is what decides
# how much memory this endpoint uses.
_FULL_EXPORT_MAX_ROWS = 2_000_000


class WindowRef(BaseModel):
    """Footage already in the corpus, used as the query.

    Its clips' original embeddings are mean-pooled into one direction, so this
    is query-by-example without re-encoding anything.
    """

    run_uuid: str = ""
    segment_id: str = ""
    start_ns: int = 0
    end_ns: int = 0


class RetrieveRequest(BaseModel):
    """The one retrieval interface: a query, a cutoff, and how to return it.

    `k` and `threshold` are the same operation -- a top-k HAS a threshold, the
    k-th score -- so they share one cascade rather than two implementations.
    `output` decides serialization only: a page of JSON for browsing, a CSV
    attachment (plus parquet, segment set and a Recent-scans row) for export.
    """

    # A vector source. Text, a saved vector, a window of footage, uploaded
    # frames, or the marks themselves -- these are five ways to arrive at one
    # 768-d direction, not five kinds of search, so they are inputs here rather
    # than endpoints of their own.
    project: str | None = None
    query: str = ""
    # A saved or resent direction. Nested when the ranking it came from used
    # pooling=individual, so a rescore addresses the same set of directions.
    vector: list[float] | list[list[float]] | None = None
    window: "WindowRef | None" = None
    image_b64: str = ""
    frames_b64: list[str] = []

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

    # What to return. All three run on the same resolved vector; only the last
    # two touch the corpus.
    #   vector -> the query direction, encoded and nothing else
    #   scores -> exact 768-d cosine for `rows` you already hold
    #   hits   -> a JSON page of matches
    output: str = "hits"
    rows: list[int] = []          # output="scores" only
    page: int = 0
    limit: int = 50
    # Window search only: `individual` keeps the window's chunks apart and a row
    # scores as its best match against any of them.
    pooling: str = "mean"
    max_rows: int = 0             # threshold-mode ceiling; 0 = the global one

    # A saved tag to resume: its stored vector is the query direction.
    tag: str = ""
    version: int | None = None

    from_date: str | None = None
    to_date: str | None = None
    vehicle: str | None = None
    drive_id: str | None = None
    filter_lance_uri: str | None = None
    segment_set_uuid: str | None = None
    # Marks do two jobs: downvoted rows are excluded from the results, and with
    # `refine_from_marks` they also build the query direction (Rocchio). Both,
    # because moving the vector away from a rejection is not the same as
    # removing it -- a clip near the positives survives the push and returns.
    marks: list[Mark] = []
    refine_from_marks: bool = False
    negative_weight: float = 0.5
    text_weight: float = 0.3


@app.post("/ui/search")
def retrieve(req: RetrieveRequest, request: Request):
    """Search and export, over one selection.

    These were two endpoints over two implementations of the same search, and
    every divergence between them shipped as a bug in one path and not the
    other. Now there is one `full_corpus.select` underneath and this endpoint
    chooses only how to serialize it.
    """
    if req.output not in ("hits", "vector", "scores"):
        raise HTTPException(400, "output must be hits, vector or scores")
    if req.output == "hits" and (req.k is None) == (req.threshold is None):
        raise HTTPException(400, "pass exactly one of k or threshold")
    corpus = _require_full_corpus(_project_of(req))
    vec = _retrieve_vector(req, corpus)

    if req.output == "vector":
        # Encode and stop. Saving a refined or uploaded direction needs the
        # vector itself, not the clips it happens to match today.
        return {"vector": [float(x) for x in vec], "dim": int(vec.shape[0]),
                "label": req.query or "uploaded example",
                "n_frames": len(req.frames_b64) or (1 if req.image_b64 else 0)}

    if req.output == "scores":
        # Exact scores for clips already in hand -- sharpening a page without
        # re-running the search that produced it.
        if not req.rows:
            raise HTTPException(400, "output=scores needs rows")
        if len(req.rows) > 1000:
            raise HTTPException(400, "at most 1000 rows per scores request")
        t0 = time.perf_counter()
        try:
            exact = corpus.exact_scores(np.asarray(req.rows, dtype=np.int64), vec)
        except RuntimeError as exc:
            raise HTTPException(409, str(exc))
        took = round((time.perf_counter() - t0) * 1000, 1)
        LOGGER.info("retrieve scores: %d rows in %.1fms", len(req.rows), took)
        return {"took_ms": took, "score_kind": "exact",
                "scores": [{"row": int(r), "score": round(float(s), 4)}
                           for r, s in zip(req.rows, exact)]}
    exclude = [
        int(m.row) for m in req.marks
        if m.mark == "down" and m.row is not None and 0 <= int(m.row) < corpus.num_rows
    ]

    out = _retrieve_hits(req, corpus, vec, exclude)
    if req.refine_from_marks:
        # Report what the feedback actually did: how many labels shaped the
        # vector, and how many rows were removed outright. Rocchio only rotates
        # the query, so without the exclusion count a caller cannot tell whether
        # a rejection took effect.
        ups = sum(1 for m in req.marks if m.mark == "up" and m.row is not None)
        out["refined_from"] = {
            "num_up": ups, "num_down": len(exclude), "excluded": len(exclude),
        }
    return out


def _window_vector(win: "WindowRef", corpus, pooling: str = "mean") -> np.ndarray:
    """The query direction(s) for the clips inside a window.

    `mean` averages them into one vector; `individual` returns one row per clip
    and leaves the best-of-N to the scorer.

    The window's clips are resolved against the resident metadata, then their
    ORIGINAL 768-d embeddings are read: the screen is quantized and PCA-reduced
    and cannot reconstruct a direction of the quality query-by-example needs.
    """
    if not win.run_uuid.strip() and not win.segment_id.strip():
        raise HTTPException(400, "window needs a run_uuid or a segment_id")
    start_s = int(win.start_ns) // 1_000_000_000 if win.start_ns else 0
    end_s = int(win.end_ns) // 1_000_000_000 if win.end_ns else 0
    try:
        rows = corpus.window_rows(
            run_uuid=win.run_uuid, segment_id=win.segment_id,
            start_unix=start_s, end_unix=end_s,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    if rows.size == 0:
        key = win.run_uuid.strip() or win.segment_id.strip()
        raise HTTPException(404, f"no embedded clips for {key!r} in that window")
    try:
        mat = corpus.vectors_for(rows[:_WINDOW_MAX_POOL])
    except RuntimeError as exc:
        raise HTTPException(409, str(exc))
    if pooling == "individual":
        import full_corpus

        # Every chunk kept as its own direction: a corpus row then scores as its
        # best match against any of them, so a 30s source matches a clip that
        # resembles ANY 8s of it rather than its average.
        try:
            return full_corpus.as_directions(mat)
        except ValueError as exc:
            raise HTTPException(422, str(exc))
    pooled = mat.mean(axis=0)
    norm = float(np.linalg.norm(pooled))
    if norm == 0.0:
        raise HTTPException(422, "the window's embeddings cancelled to a zero vector")
    return (pooled / norm).astype(np.float32)


def _encode_upload(frames_b64: list, image_b64: str) -> np.ndarray:
    """Encode uploaded frames into the joint space.

    A single image is copied into the encoder's 8 slots, so an image query is a
    still clip with no motion in it -- it matches appearance, not movement.
    """
    raw = list(frames_b64) if frames_b64 else ([image_b64] if image_b64 else [])
    if len(raw) > _UPLOAD_MAX_FRAMES:
        raise HTTPException(413, f"too many frames ({len(raw)} > {_UPLOAD_MAX_FRAMES})")
    frames: list[bytes] = []
    for f in raw:
        try:
            data = base64.b64decode(f.split(",", 1)[-1], validate=False)
        except (ValueError, binascii.Error):
            raise HTTPException(400, "invalid base64 frame data")
        if not data:
            raise HTTPException(400, "empty frame")
        if len(data) > _UPLOAD_MAX_FRAME_BYTES:
            raise HTTPException(413, "a frame is too large (max 8MB each)")
        frames.append(data)
    try:
        return search_engine.encode_frames_list(
            frames, _state["processor"], _state["model"], _state["cfg"].device
        )
    except Exception as exc:  # noqa: BLE001 - surfaced as a 400, not a 500
        raise HTTPException(400, f"could not encode upload: {type(exc).__name__}: {exc}")


def _retrieve_vector(req: "RetrieveRequest", corpus) -> np.ndarray:
    """The query direction: a Rocchio prototype, a supplied vector, or the text."""
    if req.refine_from_marks:
        pos = [int(m.row) for m in req.marks if m.mark == "up" and m.row is not None]
        neg = [int(m.row) for m in req.marks if m.mark == "down" and m.row is not None]
        pos = [r for r in dict.fromkeys(pos) if 0 <= r < corpus.num_rows]
        neg = [r for r in dict.fromkeys(neg) if 0 <= r < corpus.num_rows]
        if not pos:
            raise HTTPException(400, "mark at least one positive to re-rank")
        text_vec = None
        if req.text_weight > 0 and req.query.strip():
            if not _state.get("model_ready"):
                raise HTTPException(503, "model still loading")
            text_vec = search_engine.encode_query(
                req.query.strip(), _state["processor"], _state["model"],
                _state["cfg"].device,
            )
        try:
            # The marked clips' ORIGINAL embeddings: the resident screen is
            # quantized and PCA-reduced and cannot reconstruct a direction of
            # the quality feedback needs.
            mat = corpus.vectors_for(pos + neg)
        except RuntimeError as exc:
            raise HTTPException(409, str(exc))
        return _rocchio(
            mat[: len(pos)], mat[len(pos):], text_vec,
            req.negative_weight, req.text_weight,
        )
    if req.tag:
        import catalog

        try:
            stored = catalog.get().version(req.tag, req.version)
        except catalog.UnknownTag:
            raise HTTPException(404, f"unknown tag {req.tag!r}")
        return np.asarray(stored["vector"], dtype=np.float32)
    if req.vector:
        import full_corpus

        try:
            vec = full_corpus.as_directions(np.asarray(req.vector, dtype=np.float32))
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        if vec.shape[1] != corpus.dim:
            raise HTTPException(
                400, f"vector dim {vec.shape[1]} != corpus dim {corpus.dim}"
            )
        return vec[0] if vec.shape[0] == 1 else vec
    if req.window is not None:
        if req.pooling not in ("mean", "individual"):
            raise HTTPException(400, "pooling must be mean or individual")
        return _window_vector(req.window, corpus, req.pooling)
    if req.image_b64 or req.frames_b64:
        if not _state.get("model_ready"):
            raise HTTPException(503, "model still loading")
        return _encode_upload(req.frames_b64, req.image_b64)
    if not req.query.strip():
        raise HTTPException(
            400, "provide one of: query, vector, window, image_b64/frames_b64, "
            "or marks with refine_from_marks"
        )
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
    if req.limit <= 0 or req.limit > _FULL_MAX_DEPTH:
        raise HTTPException(400, f"limit must be between 1 and {_FULL_MAX_DEPTH:,}")
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
    # The direction this ranking used. A later rescore of these rows has to
    # address the SAME vector: for a refine, a window or an upload there is no
    # text that reproduces it, and re-encoding the UI's label would rescore
    # against the cosine of a caption.
    # Shape-preserving: reshape(-1) on a multi-direction query would hand the
    # client one 768*N-long "vector" that rescores against nothing real.
    _v = np.asarray(vec, dtype=np.float32)
    out["vector"] = ([round(float(x), 6) for x in _v] if _v.ndim == 1
                     else [[round(float(x), 6) for x in row] for row in _v])
    return out


def _interval_rows_full(intervals: list, corpus, query: str) -> list[dict]:
    """`_interval_rows` for the full corpus: same output columns, but the peak
    clip's identifiers come from the resident arrays rather than a Corpus."""
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


# A set this large is a symptom (a threshold far too low), and DORA's
# CreateDataSet hangs rather than refusing it -- so report instead of registering.
_SEGSET_MAX_SEGMENTS = int(os.getenv("NLS_SEGSET_MAX_SEGMENTS", "500000"))


def _create_export_segment_set(
    project: str, seg_ids, name: str, *, provenance: dict
) -> tuple[str, str]:
    """Register the exported segments as a DORA dataset. Returns ``(label, error)``.

    Best-effort by construction: a DORA failure comes back in the error string and
    is logged, never raised, so a Data Explorer outage cannot fail the export the
    user actually asked for.
    """
    ids = sorted({s for s in seg_ids if s})
    if not ids:
        return "", "no segments to register"
    if len(ids) > _SEGSET_MAX_SEGMENTS:
        return "", (
            f"not registered: {len(ids):,} segments exceeds "
            f"{_SEGSET_MAX_SEGMENTS:,} cap (raise the threshold or lower k)"
        )
    meta = {k: v for k, v in provenance.items() if v not in (None, "", [])}
    meta["num_segments"] = len(ids)
    try:
        uuid_str, version = dora_client.create_dataset(
            name, ids, custom_metadata=meta, hostname=_dora_host(project)
        )
    except Exception as exc:  # noqa: BLE001 -- see the docstring: never fail the export
        LOGGER.warning(
            "export segment-set create failed (%s): %s", type(exc).__name__, exc,
            exc_info=True,
        )
        return "", f"{type(exc).__name__}: {exc}"
    return f"{uuid_str} v{version} ({len(ids)} segs)", ""


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


@app.post("/ui/calibrate")
def calibrate(req: CalibrateRequest, request: Request) -> dict:
    """What a cutoff would do, and which cutoff to use.

    One endpoint rather than two: a score histogram over the whole corpus is the
    same computation whether or not you have labels. With marks it also fits an
    operating point and returns the next clips worth labelling; without them it
    reports the label-free suggestion over the same distribution. Splitting that
    two separate scoring paths meant two endpoints scoring
    the whole corpus to produce overlapping answers.

    Operating-point selection over the full corpus's screening scores. Marks are
    addressed by `row`, which is what full-corpus hits carry; `index` is a
    resident-corpus address and is ignored here.
    """
    query = (req.query or "").strip()
    if not query:
        raise HTTPException(400, "query must not be empty")
    if not _state.get("model_ready"):
        raise HTTPException(503, "model still loading")
    corpus = _require_full_corpus(req.project)

    vec = search_engine.encode_query(
        query, _state["processor"], _state["model"], _state["cfg"].device
    )
    scores, err = corpus.score(vec)
    # Through the shared builder, so a cutoff is fitted over exactly the rows a
    # search or export would return. Building the filter set by hand here meant
    # segment_set_uuid was declared, sent by the UI, and silently ignored -- the
    # histogram and tau came from the whole corpus while the user believed they
    # were scoped to a segment set.
    mask = corpus.filter_mask(**_full_filters(req))
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
    if req.at_tau is not None:
        if not 0.0 <= float(req.at_tau) <= 1.0:
            raise HTTPException(400, "at_tau must be in [0, 1]")
        tau_for_sampling = float(req.at_tau)
    else:
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


def _require_full_corpus(project: str | None = None):
    """The project's resident corpus, or a 503 that starts its load."""
    project = _project_or_404(project).name
    with _CORPORA_LOCK:
        slot = _slot(project)
        corpus, status, error = slot["corpus"], slot["status"], slot["error"]
    if corpus is None:
        if status == "error":
            raise HTTPException(503, f"{project} corpus load failed: {error}")
        _full_corpus_begin_load(project)
        raise HTTPException(
            503,
            f"{project} corpus is loading (~2 min); poll /api/v1/health and retry",
        )
    return corpus


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

    `index` is -1 because there is no resident-corpus row to address; `row` is
    the full-corpus position, which is how the caller addresses these rows to
    rescore them (`output="scores"`).
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
            # Dataset row position: how `output="scores"` addresses this row.
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
        # Rows whose embedding was null cannot match anything, so a corpus size
        # is not the number of clips a query was actually compared against.
        "embedded_rows": corpus.embedded_rows,
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
            "filter_lance_uri": req.filter_lance_uri or None,
        },
    }


# The versioned API. Its handlers import this module lazily, so the mount is the
# last thing here.
import api_v1  # noqa: E402

app.include_router(api_v1.router)
