"""Apps Platform Cloud SQL (Postgres) access for the export log.

Mirrors the repo template (onroad/tools/entity_upramp_monitor/db.py):
- Cloud Run: APV2 injects INSTANCE_CONNECTION_NAME / DB_USER / DB_NAME; connect via
  the Cloud SQL Python connector with IAM auth + private IP.
- Local: no INSTANCE_CONNECTION_NAME -> connect to localhost:5432 over the
  `apps-platform app connect-db` proxy.

Per-app Postgres SCHEMA = K_SERVICE with hyphens -> underscores (vlm_nls_search).

Every write here is BEST-EFFORT: ``insert_export`` and ``init_schema`` log and
swallow connection/SQL errors so a download (or app startup) never breaks when the
DB is unreachable. ``insert_export`` returns True only when the row was committed.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any

import sqlalchemy
from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

logger = logging.getLogger(__name__)

SCHEMA_NAME = os.getenv("K_SERVICE", "vlm-nls-search").replace("-", "_")
_TABLE = f"{SCHEMA_NAME}.export_log"
# One row per DISTINCT human relevance judgment: (query, segment, user) -> 👍/👎.
# This is the honest feedback ledger. It is written the moment a mark is USED
# (refine / threshold-sweep / export), not only on Download, and re-marking the
# same (query, segment) upserts in place -- so a COUNT here is the true number of
# thumbs.
_MARKS_TABLE = f"{SCHEMA_NAME}.feedback_marks"
_engine: Engine | None = None
_engine_lock = threading.Lock()
_schema_ready = False

_DDL = [
    # NB: do NOT CREATE SCHEMA here. Apps Platform pre-provisions the per-app
    # schema ({SCHEMA_NAME}) and grants the service account on it; the SA has no
    # CREATE privilege on the `postgres` database itself, so a CREATE SCHEMA
    # raises "permission denied for database postgres". We only create our table
    # inside the already-provisioned schema (mirrors entity_upramp_monitor/db.py).
    f"""CREATE TABLE IF NOT EXISTS {_TABLE} (
        id                BIGSERIAL PRIMARY KEY,
        created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
        user_email        TEXT,
        query             TEXT,
        tag               TEXT,
        k                 INTEGER,
        num_results       INTEGER,
        model_uri         TEXT,
        embeddings_uri    TEXT,
        segment_set_uuid  TEXT,
        segment_set_name  TEXT,
        date_from         TEXT,
        date_to           TEXT,
        filter_lance_uri  TEXT,
        vehicle           TEXT,
        drive_id          TEXT,
        threshold         DOUBLE PRECISION,
        thumbs_up         JSONB,
        thumbs_down       JSONB,
        search_vector     JSONB,
        parquet_uri       TEXT
    )""",
    # Migrate already-provisioned tables that predate the parquet_uri column.
    f"ALTER TABLE {_TABLE} ADD COLUMN IF NOT EXISTS parquet_uri TEXT",
    # ... and the lance/vehicle filters, so Resume can restore the full filter set.
    f"ALTER TABLE {_TABLE} ADD COLUMN IF NOT EXISTS filter_lance_uri TEXT",
    f"ALTER TABLE {_TABLE} ADD COLUMN IF NOT EXISTS vehicle TEXT",
    f"ALTER TABLE {_TABLE} ADD COLUMN IF NOT EXISTS drive_id TEXT",
    # Per-tag scan cosine threshold, so the Export table remembers each tag's cutoff (like k).
    f"ALTER TABLE {_TABLE} ADD COLUMN IF NOT EXISTS threshold DOUBLE PRECISION",
    # Make `tag` a TRUE key: one row per non-empty tag, so iterating on a tag (re-search,
    # re-save, re-export) updates the SAME history row instead of appending. First collapse
    # any existing duplicate tags -- keep the best row per tag (one with a stored vector
    # wins, then newest) -- then add a partial UNIQUE index on non-empty tags (untagged rows
    # stay un-constrained). insert_export upserts via ON CONFLICT against this index.
    f"""DELETE FROM {_TABLE} e USING (
            SELECT id, row_number() OVER (
                PARTITION BY tag
                ORDER BY (CASE WHEN search_vector IS NOT NULL
                                AND jsonb_array_length(search_vector) > 0 THEN 1 ELSE 0 END) DESC,
                         created_at DESC, id DESC
            ) AS rn
            FROM {_TABLE} WHERE tag IS NOT NULL AND tag <> ''
        ) d
        WHERE e.id = d.id AND d.rn > 1""",
    f"""CREATE UNIQUE INDEX IF NOT EXISTS export_log_tag_uidx
        ON {_TABLE} (tag) WHERE tag IS NOT NULL AND tag <> ''""",
    # Feedback ledger: one row per distinct (query, segment, user) judgment.
    f"""CREATE TABLE IF NOT EXISTS {_MARKS_TABLE} (
        id                BIGSERIAL PRIMARY KEY,
        created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
        updated_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
        user_email        TEXT NOT NULL DEFAULT '',
        query             TEXT NOT NULL,
        query_key         TEXT NOT NULL,
        segment_id        TEXT NOT NULL,
        chunk_id          TEXT,
        label             BOOLEAN NOT NULL,
        source            TEXT,
        embeddings_uri    TEXT,
        model_uri         TEXT
    )""",
    # A judgment is unique per (query, segment, user); re-marking upserts in place
    # so the ledger never double-counts the same thumb.
    f"""CREATE UNIQUE INDEX IF NOT EXISTS feedback_marks_uidx
        ON {_MARKS_TABLE} (query_key, segment_id, user_email)""",
    # One-time (idempotent) backfill from the historical export_log thumbs, so the
    # ledger reflects prior downloaded feedback immediately. ON CONFLICT DO NOTHING
    # keeps any live mark authoritative over the backfilled copy.
    f"""INSERT INTO {_MARKS_TABLE}
            (user_email, query, query_key, segment_id, chunk_id, label, source,
             embeddings_uri, model_uri, created_at)
        SELECT COALESCE(el.user_email, ''),
               el.query,
               lower(btrim(regexp_replace(el.query, '\\s+', ' ', 'g'))),
               COALESCE(NULLIF(m->>'segment_id', ''), m->>'chunk_id'),
               m->>'chunk_id',
               (t.lab = 'up'),
               'export_backfill',
               el.embeddings_uri, el.model_uri, el.created_at
        FROM {_TABLE} el
        CROSS JOIN LATERAL (VALUES ('up', el.thumbs_up), ('down', el.thumbs_down)) AS t(lab, arr)
        CROSS JOIN LATERAL jsonb_array_elements(COALESCE(t.arr, '[]'::jsonb)) AS m
        WHERE el.query IS NOT NULL AND btrim(el.query) <> ''
          AND COALESCE(NULLIF(m->>'segment_id', ''), m->>'chunk_id') IS NOT NULL
        ON CONFLICT (query_key, segment_id, user_email) DO NOTHING""",
]

def _make_engine() -> Engine:
    db_user = os.environ.get("DB_USER", "")
    db_name = os.environ.get("DB_NAME", "postgres")
    instance_conn = os.environ.get("INSTANCE_CONNECTION_NAME", "")

    if not instance_conn:
        # Local: connect to localhost via the apps-platform connect-db proxy.
        password = os.environ.get("DB_PASSWORD", "")
        auth = f"{db_user}:{password}" if password else db_user
        url = f"postgresql+psycopg2://{auth}@localhost:5432/{db_name}"
        logger.info("DB: local Postgres at localhost:5432 (schema %s)", SCHEMA_NAME)
        return sqlalchemy.create_engine(url, pool_pre_ping=True)

    from google.cloud.sql.connector import Connector  # type: ignore[import-untyped]

    connector = Connector()

    def getconn() -> Any:
        return connector.connect(
            instance_conn,
            "pg8000",
            user=db_user,
            db=db_name,
            enable_iam_auth=True,
            ip_type="private",
        )

    logger.info(
        "DB: Cloud SQL %s as %s (schema %s)", instance_conn, db_user, SCHEMA_NAME
    )
    return sqlalchemy.create_engine(
        "postgresql+pg8000://", creator=getconn, pool_pre_ping=True
    )


def _get_engine() -> Engine:
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                _engine = _make_engine()
    return _engine


def init_schema() -> bool:
    """Create the schema + table if needed. Best-effort; returns success."""
    global _schema_ready
    try:
        with _get_engine().begin() as conn:
            for stmt in _DDL:
                conn.execute(text(stmt))
        _schema_ready = True
        logger.info("DB: schema ready (%s)", _TABLE)
    except (SQLAlchemyError, OSError) as exc:
        logger.warning("DB: init_schema failed (%s): %s", type(exc).__name__, exc)
        return False
    try:
        import catalog

        catalog.get().init()
        logger.info("DB: tag catalog ready (%s)", SCHEMA_NAME)
    except (SQLAlchemyError, OSError) as exc:
        logger.warning("DB: catalog init failed (%s): %s", type(exc).__name__, exc)
        return False
    return True


_BACKFILL_ROWS = text(
    f"""SELECT tag, query, threshold, user_email, created_at, parquet_uri, num_results,
               search_vector::text AS search_vector
        FROM {_TABLE}
        WHERE tag IS NOT NULL AND tag <> ''
        ORDER BY created_at"""
)


def backfill_catalog(*, project: str, model: str) -> int:
    """Seed the tag catalog from export_log once per tag. Idempotent and
    best-effort; returns the number of tags added (0 when nothing to do or the
    database is unreachable)."""
    try:
        import catalog

        if not _schema_ready and not init_schema():
            return 0
        with _get_engine().begin() as conn:
            conn.execute(text(f"SET search_path TO {SCHEMA_NAME}"))
            rows = [dict(r) for r in conn.execute(_BACKFILL_ROWS).mappings()]
        n = catalog.get().backfill_export_log(rows, project=project, model=model)
        if n:
            logger.info("DB: catalog backfilled %d tags from export_log", n)
        return n
    except (SQLAlchemyError, OSError) as exc:
        logger.warning("DB: catalog backfill failed (%s): %s", type(exc).__name__, exc)
        return 0


def feedback_totals() -> dict:
    """Distinct human feedback counts from the ledger -- the HONEST totals.

    Counts one row per (query, segment, user) judgment. Best-effort; returns
    zeros when exp-db is unreachable."""
    try:
        if not _schema_ready:
            init_schema()
        with _get_engine().begin() as conn:
            conn.execute(text(f"SET search_path TO {SCHEMA_NAME}"))
            row = conn.execute(text(
                f"""SELECT
                    count(*) FILTER (WHERE label)            AS up,
                    count(*) FILTER (WHERE NOT label)        AS down,
                    count(DISTINCT segment_id)               AS segments,
                    count(DISTINCT query_key)                AS queries
                FROM {_MARKS_TABLE}"""
            )).mappings().one()
        return {"up": int(row["up"]), "down": int(row["down"]),
                "segments": int(row["segments"]), "queries": int(row["queries"])}
    except (SQLAlchemyError, OSError) as exc:
        logger.warning("DB: feedback_totals failed (%s): %s", type(exc).__name__, exc)
        return {"up": 0, "down": 0, "segments": 0, "queries": 0}


# Read side -- backs the /analytics view. The search_vector column is large
# (a 768-float JSON array per row) so it is deliberately excluded here; only its
# length is surfaced. Both helpers are best-effort and return [] when exp-db is
# unreachable so the page renders an "unavailable" state instead of erroring.
_RECENT = text(
    f"""SELECT id, created_at, user_email, query, tag, k, num_results,
               model_uri, embeddings_uri, segment_set_uuid, segment_set_name,
               date_from, date_to,
               COALESCE(jsonb_array_length(thumbs_up), 0)        AS num_up,
               COALESCE(jsonb_array_length(thumbs_down), 0)      AS num_down,
               COALESCE(jsonb_array_length(search_vector), 0)    AS vec_dim,
               thumbs_up::text                                   AS thumbs_up_json,
               thumbs_down::text                                 AS thumbs_down_json,
               search_vector::text                               AS search_vector_json
        FROM {_TABLE}
        ORDER BY created_at DESC
        LIMIT :limit"""
)

# Distinct segment sets ever used in an export, with usage stats -- "the db
# stored previously saved segment sets".
_SEG_SETS = text(
    f"""SELECT segment_set_uuid,
               max(segment_set_name)                  AS segment_set_name,
               count(*)                               AS times_used,
               max(created_at)                        AS last_used,
               sum(num_results)                       AS total_results
        FROM {_TABLE}
        WHERE segment_set_uuid IS NOT NULL AND segment_set_uuid <> ''
        GROUP BY segment_set_uuid
        ORDER BY last_used DESC"""
)


def recent_exports(limit: int = 200) -> list[dict]:
    """Most recent export-log rows (newest first); [] if exp-db is unreachable."""
    try:
        if not _schema_ready:
            init_schema()
        with _get_engine().begin() as conn:
            conn.execute(text(f"SET search_path TO {SCHEMA_NAME}"))
            rows = conn.execute(_RECENT, {"limit": int(limit)}).mappings().all()
        return [dict(r) for r in rows]
    except (SQLAlchemyError, OSError) as exc:
        logger.warning("DB: recent_exports failed (%s): %s", type(exc).__name__, exc)
        return []


def segment_sets_used() -> list[dict]:
    """Distinct segment sets referenced by past exports, with usage stats."""
    try:
        if not _schema_ready:
            init_schema()
        with _get_engine().begin() as conn:
            conn.execute(text(f"SET search_path TO {SCHEMA_NAME}"))
            rows = conn.execute(_SEG_SETS).mappings().all()
        return [dict(r) for r in rows]
    except (SQLAlchemyError, OSError) as exc:
        logger.warning("DB: segment_sets_used failed (%s): %s", type(exc).__name__, exc)
        return []


# ----- launched per-segment scans (Lilypad workloads) -----------------------------

