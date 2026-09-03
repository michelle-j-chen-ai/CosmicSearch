"""The versioned API: seven endpoints over the tag catalog and the corpus cascade.

Everything here is a view on machinery that already exists -- `full_corpus`
does the search, `catalog` holds the tags, `web_server` owns the resident
corpora and the export writers. This module decides only the request and
response shapes an integration sees.

Every route takes `project`; on reads it defaults to the default project. Every
response that touched a corpus carries `project` and `corpus_version`.
"""

from __future__ import annotations

import base64
import datetime as dt
import logging
import threading
import time
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel

import catalog
import deployment
import oci_s3
import search_engine

LOGGER = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1")

# Limits an integration can plan around; also reported by /health.
LIMITS = {
    "page_size": 500,
    "marks_per_put": 500,
    "image_bytes": 32 * 1024 * 1024,
    "url_bytes": 8192,
    "tag_chars": 64,
}
_TAG_CHARS = set("abcdefghijklmnopqrstuvwxyz0123456789_")
_OUTPUTS = ("json", "csv", "parquet")
# Exports run off the request thread, so the exact-score budget is the job's,
# not a response deadline. A full-corpus export of every above-threshold row is
# a few million rows at ~0.3ms each.
_EXPORT_EXACT_BUDGET_S = 1800.0


def _ws():
    import web_server

    return web_server


# ------------------------------------------------------------------ helpers
def _error(status: int, code: str, message: str) -> HTTPException:
    return HTTPException(status, {"code": code, "message": message, "status": status})


def actor(request: Request) -> str:
    """Who to record as created_by: the IAP identity, else the declared actor."""
    for key in ("x-goog-authenticated-user-email", "x-goog-authenticated-user-id"):
        raw = request.headers.get(key)
        if raw:
            return raw.split(":", 1)[-1]
    return (request.headers.get("x-nls-actor") or "").strip() or "api-key"


def valid_tag(tag: str) -> str:
    t = (tag or "").strip()
    if not t or len(t) > LIMITS["tag_chars"] or any(ch not in _TAG_CHARS for ch in t):
        raise _error(422, "bad_tag", f"tag must be 1-{LIMITS['tag_chars']} chars of [a-z0-9_]")
    return t


def input_prefixes(project: deployment.Project) -> list[str]:
    """Where image inputs may be read from: the project's own corpus bucket."""
    bucket, _ = oci_s3.parse_s3_uri(project.corpus_table_uri)
    return [f"s3://{bucket}/"]


def project_for_clip(uri: str) -> deployment.Project | None:
    """The project whose clip prefix `uri` sits under, or None."""
    for name in deployment.names():
        p = deployment.get(name)
        if uri.startswith(p.mp4_prefix):
            return p
    return None


def _utc(ts: float) -> str:
    return dt.datetime.fromtimestamp(ts, tz=dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _resolve_project(name: str | None) -> deployment.Project:
    try:
        return deployment.get(name)
    except KeyError:
        raise _error(404, "unknown_project", f"unknown project {name!r}; one of {deployment.names()}")


def _corpus(project: str):
    ws = _ws()
    try:
        return ws._require_full_corpus(project)
    except HTTPException as exc:
        if exc.status_code == 503:
            raise HTTPException(503, exc.detail, headers={"Retry-After": "60"})
        raise


def _storage_error(exc: BaseException) -> bool:
    """Whether `exc` is the object store being unreachable rather than bad input.

    Lance surfaces IO failures as OSError or ValueError carrying a LanceError
    string, and ValueError is also how `select` reports bad arguments -- so the
    two are told apart by content. Getting this wrong in the lenient direction
    turns a 400 into a 503, which a caller retries; the strict direction hides
    an outage behind "bad request".
    """
    if isinstance(exc, (OSError, oci_s3.CredentialsMissing)):
        return True
    return "LanceError" in str(exc) or "SignatureDoesNotMatch" in str(exc)


def _unreachable(exc: BaseException) -> HTTPException:
    LOGGER.warning("corpus unreachable: %s: %s", type(exc).__name__, str(exc)[:300])
    return HTTPException(
        503,
        {"code": "corpus_unreachable",
         "message": "the corpus could not be read from object storage",
         "status": 503},
        headers={"Retry-After": "60"},
    )


def _model_ready() -> None:
    if not _ws()._state.get("model_ready"):
        raise HTTPException(503, {"code": "loading", "message": "model is still loading",
                                  "status": 503}, headers={"Retry-After": "60"})


# ------------------------------------------------------------------- models
class InputType(BaseModel):
    type: str
    text: str = ""
    uri: str = ""
    run_uuid: str = ""
    segment_id: str = ""
    pooling: str = "mean"
    start_ns: int = 0
    end_ns: int = 0


class CreateTag(BaseModel):
    tag: str
    project: str
    input: InputType
    threshold_mode: str
    threshold: float | None = None
    description: str = ""
    save: bool = True
    distribution: bool = False


class MarkIn(BaseModel):
    chunk_id: str
    mark: str


class UpdateTag(BaseModel):
    project: str | None = None
    marks: list[MarkIn] = []
    objective: str = "f1"
    negative_weight: float = 0.5
    anchor_weight: float = 0.3
    threshold_mode: str | None = None
    threshold: float | None = None
    description: str | None = None
    pinned_version: int | None = None


# ------------------------------------------------------------ vector + tau
def vector_for_input(inp: InputType, project: deployment.Project, corpus) -> np.ndarray:
    """The 768-d query direction for an InputType, in the corpus's model space."""
    ws = _ws()
    if inp.type == "text":
        text = inp.text.strip()
        if not text or len(text) > 512:
            raise _error(422, "bad_input", "text must be 1-512 chars")
        _model_ready()
        return search_engine.encode_query(text, ws._state["processor"], ws._state["model"],
                                          ws._state["cfg"].device)
    if inp.type == "image":
        uri = inp.uri.strip()
        if not uri.startswith("s3://"):
            raise _error(422, "bad_input", "image input needs an s3:// uri")
        if not any(uri.startswith(p) for p in input_prefixes(project)):
            raise _error(422, "bad_input",
                         f"uri must sit under one of {input_prefixes(project)}")
        _model_ready()
        data = _read_object(uri, LIMITS["image_bytes"])
        # A still image is a clip with no motion in it: the encoder's 8 frames
        # are all this one, so it matches appearance rather than movement.
        return ws._encode_upload([], base64.b64encode(data).decode())
    if inp.type == "video":
        if inp.uri.strip():
            raise _error(422, "bad_input", "video by uri is not supported; give run_uuid or segment_id")
        if inp.pooling != "mean":
            raise _error(422, "bad_input", "a tag holds one vector, so pooling must be mean")
        if not (inp.run_uuid.strip() or inp.segment_id.strip()):
            raise _error(422, "bad_input", "video input needs run_uuid or segment_id")
        win = ws.WindowRef(run_uuid=inp.run_uuid, segment_id=inp.segment_id,
                           start_ns=inp.start_ns, end_ns=inp.end_ns)
        return ws._window_vector(win, corpus)
    raise _error(422, "bad_input", "input.type must be text, image or video")


def _read_object(uri: str, max_bytes: int) -> bytes:
    import botocore.exceptions

    bucket, key = oci_s3.parse_s3_uri(uri)
    try:
        obj = _ws()._state["s3"].get_object(Bucket=bucket, Key=key)
        size = int(obj.get("ContentLength") or 0)
        if size > max_bytes:
            raise _error(413, "too_large", f"image is {size} bytes; limit {max_bytes}")
        data = obj["Body"].read(max_bytes + 1)
    except botocore.exceptions.ClientError as exc:
        raise _error(404, "not_found", f"could not read {uri}: {exc}")
    if len(data) > max_bytes:
        raise _error(413, "too_large", f"image exceeds {max_bytes} bytes")
    return data


def threshold_for(mode: str, value: float | None, vec: np.ndarray, corpus,
                  want_distribution: bool = False) -> tuple[dict, dict | None]:
    """The threshold record for one project, and the histogram if asked for.

    `suggested` is a property of this corpus's score distribution for this
    vector; `explicit` is the caller's number. Both record how many rows they
    select at this corpus version, so the number can be read back later.
    """
    if mode not in ("suggested", "explicit"):
        raise _error(422, "bad_threshold_mode", "threshold_mode must be suggested or explicit")
    if mode == "explicit" and value is None:
        raise _error(422, "threshold_required", "threshold is required when threshold_mode is explicit")
    with _ws().scoring_slot():
        scores, _err = corpus.score(vec)
    if mode == "suggested":
        tau = float(search_engine.heuristic_threshold(search_engine.score_stats(scores)))
    else:
        tau = float(value)
        if not 0.0 <= tau <= 1.0:
            raise _error(422, "bad_threshold", "threshold must be in [0, 1]")
    selected = int((scores >= np.float32(tau)).sum())
    rec = {"value": round(tau, 4), "mode": mode, "selected": selected,
           "corpus_version": corpus.dataset_version}
    hist = _ws()._score_histogram(scores, tau) if want_distribution else None
    return rec, hist


# -------------------------------------------------------------------- tags
@router.post("/tags", status_code=201)
def create_tag(req: CreateTag, request: Request, response: Response) -> dict:
    tag = valid_tag(req.tag)
    project = _resolve_project(req.project)
    corpus = _corpus(project.name)
    vec = vector_for_input(req.input, project, corpus)
    th, hist = threshold_for(req.threshold_mode, req.threshold, vec, corpus, req.distribution)
    source = req.input.model_dump(exclude_defaults=True)
    source["type"] = req.input.type
    if not req.save:
        response.status_code = 200
        out = {
            "tag": tag, "version": None, "pinned_version": None, "description": req.description,
            "created_at": None, "created_by": actor(request), "model": corpus.model_id or "black_dwarf",
            "source": source, "vector": [round(float(x), 6) for x in vec],
            "thresholds": {project.name: {**th, "set_at": None}},
        }
    else:
        try:
            out = catalog.get().create(
                tag=tag, project=project.name, source=source, vector=[float(x) for x in vec],
                model=corpus.model_id or "black_dwarf", threshold=th,
                description=req.description, created_by=actor(request),
            )
        except catalog.TagExists as exc:
            raise _error(409, "tag_exists", str(exc))
    if hist is not None:
        out["distribution"] = hist
    out["project"] = project.name
    out["corpus_version"] = corpus.dataset_version
    return out


@router.get("/tags")
def list_tags(project: str | None = None, q: str | None = None, created_by: str | None = None,
              page: int = 1, page_size: int = 50) -> dict:
    if project:
        project = _resolve_project(project).name
    if not 1 <= page_size <= LIMITS["page_size"]:
        raise _error(422, "bad_page_size", f"page_size must be 1-{LIMITS['page_size']}")
    return catalog.get().list(project=project, q=q, created_by=created_by, page=max(page, 1),
                              page_size=page_size)


@router.get("/tags/{tag}")
def read_tag(
    tag: str, request: Request, response: Response,
    project: str | None = None, version: int | None = None, output: str | None = None,
    k: int | None = None, page: int = 1, page_size: int = 50,
    interval: bool = False, segment_mode: bool = True, confidence: bool = False,
    include_below_threshold: bool = False, passthrough_columns: bool = False,
    create_segment_set: bool = False,
    from_date: str | None = None, to_date: str | None = None, vehicle: str | None = None,
    segment: str | None = None, segment_set_uuid: str | None = None,
    filter_lance_uri: str | None = None, since_corpus_version: int | None = None,
) -> dict:
    cat = catalog.get()
    try:
        if output is None:
            return _with_download_links(cat.record(tag))
        if output not in _OUTPUTS:
            raise _error(422, "bad_output", "output must be json, csv or parquet")
        v = cat.resolve_version(tag, version)
        ver = cat.version(tag, v)
    except catalog.UnknownTag as exc:
        raise _error(404, "unknown_tag", f"unknown tag or version: {exc}")
    proj = _resolve_project(project)
    th = ver["thresholds"].get(proj.name)
    if th is None and k is None:
        raise _error(409, "no_threshold",
                     f"{tag} v{v} has no {proj.name} threshold; POST it for this project or pass k")
    filters = {"from_date": from_date, "to_date": to_date, "vehicle": vehicle, "segment": segment,
               "segment_set_uuid": segment_set_uuid, "filter_lance_uri": filter_lance_uri,
               "since_corpus_version": since_corpus_version}
    if output == "json":
        return _live_page(tag, v, ver, proj, th, k, page, page_size, filters, segment_mode, confidence)
    params = {**filters, "k": k, "interval": interval, "segment_mode": segment_mode,
              "confidence": confidence, "include_below_threshold": include_below_threshold,
              "passthrough_columns": passthrough_columns, "create_segment_set": create_segment_set}
    return _materialize(tag, v, ver, proj, th, output, params, request, response)


def _filter_kwargs(proj: deployment.Project, corpus, f: dict) -> dict:
    """Corpus filter kwargs from the doc's Filters, through the same builder the
    old routes use so no filter is silently ignored here and honoured there."""
    ws = _ws()

    class _Req:  # what _full_filters reads
        project = proj.name
        from_date = f.get("from_date"); to_date = f.get("to_date")
        vehicle = f.get("vehicle"); drive_id = None
        segment_set_uuid = f.get("segment_set_uuid"); filter_lance_uri = f.get("filter_lance_uri")

    try:
        kw = ws._full_filters(_Req)
    except HTTPException as exc:
        if exc.status_code == 503:
            raise HTTPException(503, exc.detail, headers={"Retry-After": "30"})
        raise
    if f.get("segment"):
        ids = frozenset(s.strip() for s in str(f["segment"]).split(",") if s.strip())
        kw = ws._merge_filters(kw, {"segment_ids": ids})
    return kw


def _since_mask(corpus, since_version: int | None) -> np.ndarray | None:
    """Rows added after `since_version`. The corpus records each row's insert
    time, and a table version maps to the time it was committed; rows created at
    or after that instant are the ones a watermark reader has not seen."""
    if since_version is None:
        return None
    ds = corpus.dataset
    if ds is None or not hasattr(ds, "versions"):
        raise _error(422, "unsupported", "since_corpus_version needs a versioned dataset")
    committed = None
    for ver in ds.versions():
        if int(ver.get("version", -1)) == int(since_version):
            committed = ver.get("timestamp")
    if committed is None:
        raise _error(404, "unknown_version", f"corpus version {since_version} not found")
    created = corpus.created_at_unix()
    return created >= np.float64(committed.timestamp())


def _live_page(tag, v, ver, proj, th, k, page, page_size, filters, segment_mode, confidence) -> dict:
    ws = _ws()
    if not 1 <= page_size <= LIMITS["page_size"]:
        raise _error(422, "bad_page_size", f"page_size must be 1-{LIMITS['page_size']}")
    depth = page * page_size
    # Each page re-runs the cascade, and its cost grows with depth, so paging is
    # for the head of a ranked list. Reading deeper than this is an export.
    if depth > ws._FULL_MAX_DEPTH:
        raise _error(422, "page_too_deep",
                     f"paging stops at {ws._FULL_MAX_DEPTH:,} results; use output=csv "
                     "or output=parquet for bulk")
    corpus = _corpus(proj.name)
    vec = np.asarray(ver["vector"], dtype=np.float32)
    kw = _filter_kwargs(proj, corpus, filters)
    try:
        with ws.scoring_slot():
            sel = corpus.select(vec, k=k, tau=None if k is not None else float(th["value"]),
                                max_rows=depth if k is None else 0, refine="auto", **kw)
    except ValueError as exc:
        if _storage_error(exc):
            raise _unreachable(exc)
        raise _error(400, "bad_request", str(exc))
    except OSError as exc:
        raise _unreachable(exc)
    rows, scores = sel.rows, sel.scores
    if segment_mode:
        rows, scores = _first_per_segment(corpus, rows, scores)
    total = int(rows.size)
    lo, hi = (page - 1) * page_size, page * page_size
    hits = [corpus._hit(int(r), lo + i, float(s), sel.error_bound)
            for i, (r, s) in enumerate(zip(rows[lo:hi], scores[lo:hi]), start=1)]
    conf = _confidence(np.asarray([h.score for h in hits]), float(th["value"]) if th else None) if confidence else None
    return {
        "tag": tag, "version": v, "project": proj.name, "corpus_version": corpus.dataset_version,
        "threshold": th["value"] if th else None, "k": k,
        "hits": [
            {"chunk_id": h.chunk_id, "run_uuid": h.run_uuid, "segment_id": h.segment_id,
             "dx_internal_id": h.dx_internal_id, "vehicle": h.vehicle,
             "start_unix": h.chunk_start_unix, "end_unix": h.chunk_end_unix,
             "score": round(float(h.score), 4),
             **({"confidence": round(float(conf[i]), 3)} if conf is not None else {}),
             "source_media_uri": h.source_media_uri}
            for i, h in enumerate(hits)
        ],
        "page": page, "page_size": page_size, "total": total,
    }


def _first_per_segment(corpus, rows: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Keep the best-scoring clip per segment. `rows` arrive score-descending, so
    the first occurrence of each segment is its best; clips without a segment
    are all kept."""
    import pyarrow as pa

    if rows.size == 0:
        return rows, scores
    segs = corpus._meta["segment_id"].take(pa.array(rows)).to_pylist()
    seen: set[str] = set()
    keep = []
    for i, seg in enumerate(segs):
        if seg is None:
            keep.append(i)
        elif seg not in seen:
            seen.add(seg)
            keep.append(i)
    idx = np.asarray(keep, dtype=np.int64)
    return rows[idx], scores[idx]


def _confidence(scores: np.ndarray, tau: float | None) -> np.ndarray:
    """0..1 from the score's position between the cutoff and the best score seen.
    A margin, not a probability: without labels there is no calibration to
    offer, and this is stated in the export as confidence_basis=score_margin."""
    if scores.size == 0:
        return scores
    lo = float(tau) if tau is not None else float(scores.min())
    hi = max(float(scores.max()), lo + 1e-6)
    return np.clip((scores - lo) / (hi - lo), 0.0, 1.0)


# ----------------------------------------------------------------- exports
def _materialize(tag, v, ver, proj, th, output, params, request, response) -> dict:
    cat = catalog.get()
    export, claimed = cat.export_claim(tag=tag, version=v, project=proj.name, output=output,
                                       params=params, created_by=actor(request))
    if claimed:
        threading.Thread(target=_run_export, args=(export["export_id"], tag, v, ver, proj, th, output, params),
                         name=f"export-{export['export_id'][:8]}", daemon=True).start()
        export = cat.export_get(export["export_id"])
    if export["status"] == "error":
        # Report it once, then forget it so the next identical call runs again.
        cat.export_forget(export["export_id"])
    return _export_response(tag, v, proj, export, response)


def _with_download_links(record: dict) -> dict:
    """Presign each ready export's artifact so the record is directly usable."""
    ws = _ws()
    ttl = ws._state["cfg"].presign_ttl_s
    for e in record.get("exports", []):
        if e.get("status") == "ready" and e.get("uri"):
            try:
                e["download_url"] = oci_s3.presign_get(e["uri"], ws._state["s3"], ttl)
            except Exception as exc:  # noqa: BLE001 -- the uri is still there
                LOGGER.debug("presign skipped for %s: %s", e["uri"], exc)
    return record


def _export_response(tag, v, proj, export: dict, response: Response) -> dict:
    ws = _ws()
    block = {k: val for k, val in export.items() if k not in ("tag", "version", "project", "params")}
    if export["status"] == "ready" and export.get("uri"):
        ttl = ws._state["cfg"].presign_ttl_s
        try:
            block["download_url"] = oci_s3.presign_get(export["uri"], ws._state["s3"], ttl)
            block["expires_at"] = _utc(time.time() + ttl)
        except Exception as exc:  # noqa: BLE001 -- the artifact exists; the link is a convenience
            LOGGER.warning("presign failed for %s: %s", export["uri"], exc)
    if export["status"] in ("running", "pending"):
        response.status_code = 202
        response.headers["Retry-After"] = "15"
    return {"tag": tag, "version": v, "project": proj.name,
            "corpus_version": export.get("corpus_version"), "export": block}


def _run_export(eid: str, tag: str, v: int, ver: dict, proj: deployment.Project, th: dict | None,
                output: str, params: dict) -> None:
    """The export itself, off the request thread. Holds the same single-flight
    lock as the old export path: two exports at once is what OOM-kills the
    container."""
    ws = _ws()
    cat = catalog.get()
    try:
        corpus = ws._require_full_corpus(proj.name)
        vec = np.asarray(ver["vector"], dtype=np.float32)
        with ws._FULL_EXPORT_LOCK, ws.scoring_slot():
            result = _build_export(eid, tag, v, proj, corpus, vec, th, output, params)
        cat.export_finish(eid, **result)
        LOGGER.info("export %s ready: %s (%s rows)", eid, result.get("uri"), result.get("num_rows"))
    except Exception as exc:  # noqa: BLE001 -- recorded on the row, read back by the poll
        LOGGER.exception("export %s failed", eid)
        detail = getattr(exc, "detail", None)
        msg = detail.get("message") if isinstance(detail, dict) else str(detail or exc)
        cat.export_fail(eid, f"{type(exc).__name__}: {msg}")


def _build_export(eid, tag, v, proj, corpus, vec, th, output, params) -> dict:
    import pyarrow as pa

    ws = _ws()
    tau = float(th["value"]) if th else None
    k = params.get("k")
    kw = _filter_kwargs(proj, corpus, params)
    since = _since_mask(corpus, params.get("since_corpus_version"))
    filters_applied = {key: val for key, val in params.items()
                       if key in ("from_date", "to_date", "vehicle", "segment", "segment_set_uuid",
                                  "filter_lance_uri", "since_corpus_version") and val not in (None, "")}
    if filters_applied.get("vehicle"):
        filters_applied["vehicle"] = [s.strip() for s in str(filters_applied["vehicle"]).split(",") if s.strip()]

    if params.get("include_below_threshold"):
        scores_all, _err = corpus.score(vec)
        mask = corpus.filter_mask(**kw)
        allowed = mask if mask is not None else np.ones(corpus.num_rows, dtype=bool)
        if since is not None:
            allowed &= since
        rows = np.flatnonzero(allowed)
        scores = scores_all[rows]
        order = np.argsort(-scores, kind="stable")[: ws._FULL_EXPORT_MAX_ROWS]
        rows, scores = rows[order], scores[order]
        cutoff = tau if tau is not None else (float(scores[k - 1]) if k and scores.size >= k else float("nan"))
    else:
        sel = corpus.select(vec, k=k, tau=None if k is not None else tau,
                            max_rows=ws._FULL_EXPORT_MAX_ROWS, refine="auto", **kw)
        rows, scores = sel.rows, sel.scores
        if since is not None and rows.size:
            keep = since[rows]
            rows, scores = rows[keep], scores[keep]
        cutoff = float(sel.cutoff)
    if rows.size == 0:
        raise RuntimeError("nothing matched the tag and filters")

    # Exact 768-d scores, always: the artifact is the thing downstream trusts.
    corpus._verify_alignment(rows)
    exact = corpus.exact_scores(rows, vec, deadline=time.monotonic() + _EXPORT_EXACT_BUDGET_S)
    order = np.argsort(-exact, kind="stable")
    rows, exact = rows[order], exact[order]
    unscorable = int(np.isnan(exact).sum())

    base = ws._export_base(f"{tag}_v{v}_{proj.name}")
    result: dict[str, Any] = {"corpus_version": corpus.dataset_version, "cutoff": cutoff,
                              "filters_applied": filters_applied, "rows_unscorable": unscorable}
    if params.get("interval"):
        intervals = corpus.intervals(rows, exact, tau=cutoff if cutoff == cutoff else float(exact.min()))
        if not intervals:
            raise RuntimeError("no intervals formed above the cutoff")
        irows = ws._interval_rows_full(intervals, corpus, tag)
        for r in irows:
            r["project"] = proj.name; r["tag_version"] = v; r["corpus_version"] = corpus.dataset_version
        uri = ws._write_interval_export_parquet(irows, name=base) if output == "parquet" else \
            _write_csv(ws._interval_csv(irows), base)
        seg_ids = [r["segment_id"] for r in irows]
        result.update(uri=uri, num_rows=len(irows))
    else:
        table = corpus.to_arrow(rows, exact, tag=tag)
        if params.get("segment_mode"):
            table = ws._dedupe_arrow_by_segment(table)
        n = table.num_rows
        sc = np.asarray(table.column("score").to_pylist(), dtype=np.float64)
        table = table.append_column("project", pa.array([proj.name] * n))
        table = table.append_column("tag_version", pa.array([v] * n, pa.int32()))
        table = table.append_column("corpus_version", pa.array([corpus.dataset_version] * n))
        if params.get("include_below_threshold"):
            table = table.append_column("above_threshold", pa.array((sc >= cutoff).tolist()))
            result["rows_above_threshold"] = int((sc >= cutoff).sum())
        if params.get("confidence"):
            table = table.append_column("confidence", pa.array(_confidence(sc, cutoff).tolist()))
            result["confidence_basis"] = "score_margin"
        if params.get("passthrough_columns") and params.get("filter_lance_uri"):
            table = _passthrough(table, params["filter_lance_uri"])
        uri = ws._write_arrow_export(table, base) if output == "parquet" else _write_csv(ws._arrow_csv(table), base)
        seg_ids = table.column("segment_id").to_pylist()
        result.update(uri=uri, num_rows=n)
    if not result["uri"]:
        raise RuntimeError("artifact upload failed; see server log")
    if params.get("create_segment_set"):
        label, err = ws._create_export_segment_set(
            proj.name, seg_ids, base,
            provenance={"source": "nls_api_v1", "tag": tag, "version": v, "project": proj.name,
                        "embeddings_uri": corpus.corpus_uri, **filters_applied})
        if label:
            result["segment_set_uuid"] = label.split()[0]
        elif err:
            LOGGER.warning("export %s: segment set not created: %s", eid, err)
    return result


def _write_csv(text: str, base: str) -> str:
    ws = _ws()
    prefix = ws._state["cfg"].export_s3_prefix.strip().rstrip("/")
    if not prefix:
        return ""
    key = f"{prefix}/{base}.csv"
    oci_s3.put_bytes(key, text.encode(), ws._state["s3"], "text/csv")
    return key


def _passthrough(table, source_uri: str):
    """Join the source dataset's columns onto the export by segment_id."""
    src = _ws()._read_filter_table(source_uri)
    if "segment_id" not in src.column_names:
        return table
    dup = [c for c in src.column_names if c in table.column_names and c != "segment_id"]
    src = src.drop_columns(dup) if dup else src
    return table.join(src, keys="segment_id", join_type="left outer")


# -------------------------------------------------------------- PUT / DELETE
@router.put("/tags/{tag}")
def update_tag(tag: str, req: UpdateTag, request: Request) -> dict:
    cat = catalog.get()
    try:
        ver = cat.version(tag)
    except catalog.UnknownTag:
        raise _error(404, "unknown_tag", f"unknown tag {tag!r}")
    who = actor(request)
    touched = False
    if req.marks:
        if len(req.marks) > LIMITS["marks_per_put"]:
            raise _error(413, "too_many_marks", f"at most {LIMITS['marks_per_put']} marks per call")
        if not req.project:
            raise _error(422, "project_required", "project is required with marks")
        out = _refine(tag, ver, _resolve_project(req.project), req, who)
        touched = True
    elif req.threshold_mode:
        if not req.project:
            raise _error(422, "project_required", "project is required with threshold_mode")
        proj = _resolve_project(req.project)
        corpus = _corpus(proj.name)
        th, _ = threshold_for(req.threshold_mode, req.threshold, np.asarray(ver["vector"], dtype=np.float32), corpus)
        out = cat.set_threshold(tag, proj.name, th)
        out["project"], out["corpus_version"] = proj.name, corpus.dataset_version
        touched = True
    else:
        out = ver
    fields: dict[str, Any] = {}
    if req.description is not None:
        fields["description"] = req.description
    if "pinned_version" in req.model_fields_set:
        fields["pinned_version"] = req.pinned_version
    if fields:
        try:
            out = cat.update(tag, **fields) | {k: out[k] for k in ("project", "corpus_version") if k in out}
        except catalog.UnknownTag as exc:
            raise _error(404, "unknown_version", str(exc))
        touched = True
    if not touched:
        raise _error(422, "nothing_to_do", "send marks, threshold_mode, description or pinned_version")
    return out


def _refine(tag: str, ver: dict, proj: deployment.Project, req: UpdateTag, who: str) -> dict:
    ws = _ws()
    corpus = _corpus(proj.name)
    ids = [m.chunk_id for m in req.marks]
    found = corpus.rows_for_chunk_ids(ids)
    unresolved = [c for c in ids if c not in found]
    pos = [found[m.chunk_id] for m in req.marks if m.mark == "up" and m.chunk_id in found]
    neg = [found[m.chunk_id] for m in req.marks if m.mark == "down" and m.chunk_id in found]
    if any(m.mark not in ("up", "down") for m in req.marks):
        raise _error(422, "bad_mark", "mark must be up or down")
    if not pos:
        raise _error(422, "no_positive", "mark at least one clip up to refine")
    try:
        mat = corpus.vectors_for(pos + neg)
    except (OSError, ValueError) as exc:
        if _storage_error(exc):
            raise _unreachable(exc)
        raise
    prev = np.asarray(ver["vector"], dtype=np.float32)
    vec = ws._rocchio(mat[: len(pos)], mat[len(pos):], prev, req.negative_weight, req.anchor_weight)
    with ws.scoring_slot():
        scores, _err = corpus.score(vec)
    pos_s, neg_s = scores[np.asarray(pos)], scores[np.asarray(neg)] if neg else np.array([])
    if neg:
        fit = search_engine.fit_threshold(pos_s, neg_s, objective=req.objective)
        th = {"value": round(float(fit["threshold"]), 4), "mode": "fitted",
              "precision": fit.get("precision"), "recall": fit.get("recall"), "f1": fit.get("f1")}
    else:
        th = {"value": round(float(search_engine.heuristic_threshold(search_engine.score_stats(scores))), 4),
              "mode": "suggested"}
    th["selected"] = int((scores >= np.float32(th["value"])).sum())
    th["corpus_version"] = corpus.dataset_version
    positive = [m.chunk_id for m in req.marks if m.mark == "up" and m.chunk_id in found]
    negative = [m.chunk_id for m in req.marks if m.mark == "down" and m.chunk_id in found]
    out = catalog.get().new_version(
        tag=tag, project=proj.name,
        source={"type": "marks", "parent_version": ver["version"], "project": proj.name,
                "positive": positive, "negative": negative},
        vector=[float(x) for x in vec], threshold=th,
        refine={"positive_count": len(positive), "negative_count": len(negative), "prototype_count": 1,
                "excluded": negative, "unresolved": unresolved},
        created_by=who,
    )
    catalog.get().add_marks(tag, out["version"], proj.name,
                            [{"chunk_id": m.chunk_id, "mark": m.mark} for m in req.marks if m.chunk_id in found], who)
    top = np.argsort(-scores)[:3]
    out["sample"] = [{"chunk_id": corpus._hit(int(r), i + 1, float(scores[r]), 0.0).chunk_id,
                      "score": round(float(scores[r]), 4)} for i, r in enumerate(top)]
    out["project"], out["corpus_version"] = proj.name, corpus.dataset_version
    return out


@router.delete("/tags/{tag}", status_code=204)
def delete_tag(tag: str) -> Response:
    try:
        catalog.get().delete(tag)
    except catalog.UnknownTag:
        raise _error(404, "unknown_tag", f"unknown tag {tag!r}")
    except catalog.TagPinned:
        raise _error(409, "pinned", f"{tag} is pinned; unpin it first")
    return Response(status_code=204)


# ------------------------------------------------------------ video/health
@router.get("/video")
def video(uri: str) -> dict:
    import botocore.exceptions

    proj = project_for_clip(uri)
    if proj is None:
        raise _error(404, "not_a_clip", "uri is not under any project's clip prefix")
    ws = _ws()
    ttl = ws._state["cfg"].presign_ttl_s
    try:
        url = oci_s3.presign_get(uri, ws._state["s3"], ttl)
    except (ValueError, botocore.exceptions.ClientError) as exc:
        raise _error(404, "not_found", f"video unavailable: {exc}")
    return {"uri": uri, "project": proj.name, "url": url, "expires_at": _utc(time.time() + ttl)}


@router.get("/health")
def health(response: Response) -> dict:
    ws = _ws()
    projects: dict[str, dict] = {}
    any_ready = False
    for name in deployment.enabled():
        spec = deployment.get(name)
        with ws._CORPORA_LOCK:
            slot = ws._slot(name)
            corpus, status, error, started = slot["corpus"], slot["status"], slot["error"], slot["started"]
        info: dict[str, Any] = {"ready": corpus is not None, "status": status,
                                "corpus_table_uri": spec.corpus_table_uri,
                                "clip_prefix": spec.mp4_prefix, "input_prefixes": input_prefixes(spec)}
        if corpus is not None:
            any_ready = True
            lo, hi = corpus.time_span()
            info.update({
                "corpus_version": corpus.dataset_version, "rows": corpus.num_rows,
                "embedded_rows": corpus.embedded_rows,
                "date_span": [_utc(lo)[:10], _utc(hi)[:10]],
                "vehicles": corpus.vehicles(), "last_refresh": _utc(corpus.loaded_at),
            })
        elif status == "loading":
            info["elapsed_s"] = round(time.time() - started, 1)
        elif status == "error":
            info["error"] = error
        projects[name] = info
    body = {"status": "ok" if any_ready else "loading",
            "model": {"id": "black_dwarf", "ready": bool(ws._state.get("model_ready"))},
            "projects": projects, "limits": LIMITS}
    if not any_ready:
        response.status_code = 503
        response.headers["Retry-After"] = "60"
    return body
