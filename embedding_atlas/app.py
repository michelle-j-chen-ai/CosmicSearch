"""FastAPI server for the VLM embedding atlas.

Serves the precomputed 2D projection of a Cosmos-Embed video embedding table
(see precompute_atlas.py) as an interactive scatterplot, with click-through to
the source MP4 chunk and exact nearest-neighbour lookup in PCA space.

Config comes from the environment so the same code runs locally and on Apps
Platform:
    ATLAS_URI          s3:// path to atlas.parquet (required)
    ATLAS_CACHE_ROOT   local dir for the downloaded artifact
    ATLAS_PRESIGN_TTL_S presigned MP4 URL lifetime
    AWS_*              OCI S3-compat credentials and endpoint
"""

from __future__ import annotations

import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import oci_s3
from atlas_store import AtlasStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger("embedding_atlas")

WEB_DIR = Path(__file__).parent / "web"

_STATE: dict[str, object] = {}


def _video_url(uri: "str | None") -> str:
    """Presigned clip URL, or "" when the row has no media.

    A null source_media_uri is ordinary in this corpus. Presigning it raised
    inside parse_s3_uri, so one null row failed an entire neighbours or lasso
    response rather than yielding one card without a clip.
    """
    if not uri:
        return ""
    try:
        return oci_s3.presign_get(uri, _STATE["s3"], _presign_ttl_s())
    except Exception as exc:  # noqa: BLE001 -- one bad row must not fail the rest
        LOGGER.warning("could not presign %s: %s", uri, exc)
        return ""


def _presign_ttl_s() -> int:
    return int(os.environ.get("ATLAS_PRESIGN_TTL_S", "3600"))


def _load_atlas() -> AtlasStore:
    atlas_uri = os.environ["ATLAS_URI"].strip()
    cache_root = Path(os.environ.get("ATLAS_CACHE_ROOT", "/tmp/atlas_cache"))
    local_path = cache_root / atlas_uri.removeprefix("s3://").replace("/", "_")

    client = oci_s3.s3_client()
    if not local_path.exists():
        LOGGER.info("downloading %s -> %s", atlas_uri, local_path)
        started = time.perf_counter()
        oci_s3.download_object(atlas_uri, local_path, client)
        LOGGER.info("downloaded in %.1fs", time.perf_counter() - started)
    return AtlasStore(local_path)


def _load_atlas_background() -> None:
    try:
        _STATE["atlas"] = _load_atlas()
    except Exception as exc:  # surfaced via /healthz rather than killing serving
        LOGGER.exception("atlas load failed")
        _STATE["error"] = f"{type(exc).__name__}: {exc}"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Loaded on a background thread so the port opens immediately. Blocking here
    # would (a) risk the Cloud Run startup probe expiring during the 63MB
    # download and parse, and (b) crash-loop a service whose config secrets are
    # not attached yet -- secrets cannot be created until the service exists, so
    # the very first deploy of a new service necessarily boots unconfigured.
    _STATE["s3"] = oci_s3.s3_client()
    threading.Thread(target=_load_atlas_background, daemon=True).start()
    yield
    _STATE.clear()


app = FastAPI(title="VLM Embedding Atlas", lifespan=lifespan)


def _atlas() -> AtlasStore:
    atlas = _STATE.get("atlas")
    if atlas is None:
        # 503 + Retry-After, not 500: the load is either still running or failed
        # on config the operator can fix without a code change.
        raise HTTPException(
            status_code=503,
            detail=_STATE.get("error", "atlas still loading"),
            headers={"Retry-After": "10"},
        )
    return atlas  # type: ignore[return-value]


@app.get("/healthz")
def healthz() -> JSONResponse:
    """Readiness, not liveness: reports whether the atlas is actually resident."""
    atlas = _STATE.get("atlas")
    return JSONResponse(
        {
            "ok": atlas is not None,
            "points": getattr(atlas, "count", 0),
            "error": _STATE.get("error"),
        },
        status_code=200 if atlas is not None else 503,
    )


@app.get("/api/meta")
def meta() -> dict:
    atlas = _atlas()
    return {
        "count": atlas.count,
        "pca_dim": int(atlas.pca.shape[1]),
        "color_fields": atlas.color_fields,
        "atlas_uri": os.environ.get("ATLAS_URI", ""),
        "revision": os.environ.get("K_REVISION", "local"),
    }


@app.get("/api/points")
def points() -> Response:
    """Interleaved xy Float32Array. ~2MB at 250k points."""
    return Response(
        content=_atlas().positions,
        media_type="application/octet-stream",
        # Immutable: the artifact is fixed for the lifetime of a revision, so the
        # browser re-fetches only on deploy.
        headers={"Cache-Control": "public, max-age=86400, immutable"},
    )


@app.get("/api/coloring/{field}")
def coloring(field: str) -> Response:
    """Per-point category index as Uint8Array. Labels come from /api/legend.

    The legend is a separate request rather than a response header: category
    labels are arbitrary text, and any separator that survives arbitrary labels
    is a control character, which is not a legal HTTP header value.
    """
    atlas = _atlas()
    if field not in atlas.color_fields:
        raise HTTPException(status_code=404, detail=f"unknown color field {field}")
    indices, _ = atlas.coloring(field)
    return Response(
        content=indices,
        media_type="application/octet-stream",
        headers={"Cache-Control": "public, max-age=86400, immutable"},
    )


@app.get("/api/legend/{field}")
def legend(field: str) -> dict:
    atlas = _atlas()
    if field not in atlas.color_fields:
        raise HTTPException(status_code=404, detail=f"unknown color field {field}")
    _, labels = atlas.coloring(field)
    return {"field": field, "legend": labels}


@app.get("/api/point/{index}")
def point(index: int) -> dict:
    atlas = _atlas()
    if not 0 <= index < atlas.count:
        raise HTTPException(status_code=404, detail="index out of range")
    detail = atlas.detail(index)
    detail["video_url"] = _video_url(detail.get("source_media_uri"))
    return detail


@app.get("/api/neighbors/{index}")
def neighbors(index: int, k: int = 24) -> dict:
    atlas = _atlas()
    if not 0 <= index < atlas.count:
        raise HTTPException(status_code=404, detail="index out of range")
    results = atlas.neighbors(index, min(k, 96))
    for item in results:
        item["video_url"] = _video_url(item.get("source_media_uri"))
    return {"query": atlas.detail(index), "neighbors": results}


class LassoRequest(BaseModel):
    polygon: list[list[float]]
    limit: int = 48


@app.post("/api/lasso")
def lasso(request: LassoRequest) -> dict:
    atlas = _atlas()
    if len(request.polygon) < 3:
        raise HTTPException(status_code=400, detail="polygon needs >= 3 vertices")
    indices = atlas.lasso(request.polygon, min(request.limit, 96))
    items = []
    for index in indices:
        detail = atlas.detail(index)
        detail["video_url"] = _video_url(detail.get("source_media_uri"))
        items.append(detail)
    return {"count": len(items), "items": items}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
