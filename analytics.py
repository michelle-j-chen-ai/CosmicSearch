"""Lightweight usage analytics for the NLS search app: who visits, and when.

A "visit" is one Streamlit session. We record it two ways:

- A structured ``NLS_VISIT`` log line -> immediately queryable in Cloud Logging.
- One small JSON file per visit under a durable, GCS-fuse-backed directory
  (``NLS_ANALYTICS_ROOT`` or ``/mnt/data/nls_analytics``). One file per visit
  means concurrent instances never contend on a shared file, and the records
  survive instance restarts and log-retention windows. ``load_visits`` globs
  them back for the in-app usage view.

Records are intentionally minimal -- user email (from the IAP-injected header),
unix timestamp, and the session id -- so this never stores query content.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path

LOGGER = logging.getLogger(__name__)

_VISITS_SUBDIR = "visits"
_SEARCHES_SUBDIR = "searches"


def analytics_dir() -> Path:
    """Durable directory for visit records. Prefers the GCS-fuse mount so the
    history persists across instances and revisions; falls back to a temp dir
    locally."""
    override = os.environ.get("NLS_ANALYTICS_ROOT", "").strip()
    if override:
        base = Path(override)
    else:
        base = (
            next(
                (m for m in (Path("/mnt/data"), Path("/gcs")) if m.is_dir()),
                Path(tempfile.gettempdir()),
            )
            / "nls_analytics"
        )
    return base


def record_visit(user: str, session_id: str, ts_unix: float) -> None:
    """Record one visit: a structured log line + a durable per-visit JSON file.

    Best-effort -- analytics must never break the app, so persistence failures
    are logged and swallowed (the log line still captures the visit).
    """
    rec = {"user": user, "session_id": session_id, "ts_unix": float(ts_unix)}
    LOGGER.info("NLS_VISIT %s", json.dumps(rec))
    try:
        d = analytics_dir() / _VISITS_SUBDIR
        d.mkdir(parents=True, exist_ok=True)
        # One file per visit -> no cross-instance write contention.
        (d / f"{int(ts_unix)}_{session_id[:8]}.json").write_text(json.dumps(rec))
    except OSError as exc:
        LOGGER.warning("could not persist visit record: %s", exc)


def load_visits() -> list[dict]:
    """Load all persisted visit records (newest first). Empty if none/unreadable."""
    return _load(_VISITS_SUBDIR, "ts_unix")


def load_searches() -> list[dict]:
    """Load all persisted search records (newest first). Empty if none/unreadable."""
    return _load(_SEARCHES_SUBDIR, "ts_unix")


def _load(subdir: str, sort_key: str) -> list[dict]:
    d = analytics_dir() / subdir
    if not d.is_dir():
        return []
    out: list[dict] = []
    for f in d.glob("*.json"):
        try:
            out.append(json.loads(f.read_text()))
        except (OSError, ValueError):
            continue
    out.sort(key=lambda r: r.get(sort_key, 0), reverse=True)
    return out
