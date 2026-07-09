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

import json
import logging
import os
import threading
from typing import Any

import sqlalchemy
from sqlalchemy import bindparam, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

logger = logging.getLogger(__name__)

SCHEMA_NAME = os.getenv("K_SERVICE", "vlm-nls-search").replace("-", "_")
_TABLE = f"{SCHEMA_NAME}.export_log"
# Bare relation name (no schema). Used to reference the target row inside an
# INSERT ... ON CONFLICT DO UPDATE, where Postgres expects the unqualified table name.
_BARE = _TABLE.split(".")[-1]
# Launched per-segment scans (Lilypad workloads). A durable record so the Export
# tab can show what is running / completed and its workload id across reloads.
_SCAN_TABLE = f"{SCHEMA_NAME}.scan_jobs"
# Threshold-tuning episodes: one row per labeled fit, holding the query's score
# features + the fitted tau + metrics. Training data for a future learned
# threshold policy (see scripts/fit_threshold_policy.py); append-only.
_EPISODE_TABLE = f"{SCHEMA_NAME}.threshold_episodes"
_BARE_EPISODE = _EPISODE_TABLE.split(".")[-1]
# Learned threshold policy: one row per embedding space (corpus). Holds the ridge
# weights fitted from that corpus's episodes, so serving predicts the suggested tau
# with a cheap dot product (no fit on the request path). See search_engine.fit_threshold_policy.
_POLICY_TABLE = f"{SCHEMA_NAME}.threshold_policy"
_BARE_POLICY = _POLICY_TABLE.split(".")[-1]

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
    # Launched per-segment scans, keyed by Lilypad workload id. thresholds is the
    # per-tag cosine cutoff map ({tag: float}); status is the last-polled phase.
    f"""CREATE TABLE IF NOT EXISTS {_SCAN_TABLE} (
        id            BIGSERIAL PRIMARY KEY,
        created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
        updated_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
        execution_id  TEXT NOT NULL,
        user_email    TEXT,
        tags          JSONB,
        thresholds    JSONB,
        output_dir    TEXT,
        lance_uri     TEXT,
        console_url   TEXT,
        status        TEXT,
        error         TEXT,
        register_segset BOOLEAN DEFAULT false,
        segset_name   TEXT,
        segset_uuid   TEXT,
        segset_label  TEXT,
        filters       JSONB
    )""",
    f"ALTER TABLE {_SCAN_TABLE} ADD COLUMN IF NOT EXISTS lance_uri TEXT",
    f"ALTER TABLE {_SCAN_TABLE} ADD COLUMN IF NOT EXISTS filters JSONB",
    f"ALTER TABLE {_SCAN_TABLE} ADD COLUMN IF NOT EXISTS register_segset BOOLEAN DEFAULT false",
    f"ALTER TABLE {_SCAN_TABLE} ADD COLUMN IF NOT EXISTS segset_name TEXT",
    f"ALTER TABLE {_SCAN_TABLE} ADD COLUMN IF NOT EXISTS segset_uuid TEXT",
    f"ALTER TABLE {_SCAN_TABLE} ADD COLUMN IF NOT EXISTS segset_label TEXT",
    f"CREATE UNIQUE INDEX IF NOT EXISTS scan_jobs_exec_idx ON {_SCAN_TABLE} (execution_id)",
    # Idempotency key for server-side launch dedup: identical concurrent requests
    # (e.g. a Spark stage firing the same scan from every executor) coalesce to one
    # Lilypad workload. The partial UNIQUE index is the cross-instance authority.
    f"ALTER TABLE {_SCAN_TABLE} ADD COLUMN IF NOT EXISTS idem_key TEXT",
    f"CREATE UNIQUE INDEX IF NOT EXISTS scan_jobs_idem_idx ON {_SCAN_TABLE} (idem_key) "
    f"WHERE idem_key IS NOT NULL",
    # Threshold-tuning episodes: (score-distribution features, suggested tau, fitted
    # tau, metrics) captured each time a labeled fit is produced. Append-only training
    # data for a future learned threshold policy; no unique key (many tunes per tag).
    f"""CREATE TABLE IF NOT EXISTS {_EPISODE_TABLE} (
        id                 BIGSERIAL PRIMARY KEY,
        created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
        user_email         TEXT,
        query              TEXT,
        tag                TEXT,
        model_uri          TEXT,
        embeddings_uri     TEXT,
        features           JSONB,
        suggested_tau      DOUBLE PRECISION,
        fit_tau            DOUBLE PRECISION,
        f1                 DOUBLE PRECISION,
        precision          DOUBLE PRECISION,
        recall             DOUBLE PRECISION,
        average_precision  DOUBLE PRECISION,
        objective          TEXT,
        n_pos              INTEGER,
        n_neg              INTEGER
    )""",
    f"""CREATE TABLE IF NOT EXISTS {_POLICY_TABLE} (
        embeddings_uri    TEXT PRIMARY KEY,
        feature_names     JSONB NOT NULL,
        weights           JSONB NOT NULL,
        n_episodes        INTEGER NOT NULL,
        mae_policy        DOUBLE PRECISION,
        mae_heuristic     DOUBLE PRECISION,
        fitted_at         TIMESTAMPTZ NOT NULL DEFAULT now()
    )""",
]

# Upsert keyed on `tag` (the true key): a row with a non-empty tag conflicts with the
# partial UNIQUE index (export_log_tag_uidx) and UPDATEs the existing same-tag row --
# so iterating on a tag (re-search/refine, re-save, re-export) refreshes one history row
# instead of appending. Untagged rows (tag '' / NULL) are outside the index, so they just
# insert. The merge is conservative: thumbs/search_vector are only overwritten when the
# incoming value is non-empty, so a write that doesn't carry them (e.g. a config export)
# never wipes a previously-saved refined vector or its marks. created_at is bumped so the
# freshly-touched tag sorts to the top of Search history.
_INSERT = text(
    f"""INSERT INTO {_TABLE} (
        user_email, query, tag, k, num_results, model_uri, embeddings_uri,
        segment_set_uuid, segment_set_name, date_from, date_to,
        filter_lance_uri, vehicle, drive_id, threshold,
        thumbs_up, thumbs_down, search_vector, parquet_uri
    ) VALUES (
        :user_email, :query, :tag, :k, :num_results, :model_uri, :embeddings_uri,
        :segment_set_uuid, :segment_set_name, :date_from, :date_to,
        :filter_lance_uri, :vehicle, :drive_id, :threshold,
        CAST(:thumbs_up AS JSONB), CAST(:thumbs_down AS JSONB),
        CAST(:search_vector AS JSONB), :parquet_uri
    )
    ON CONFLICT (tag) WHERE tag IS NOT NULL AND tag <> ''
    DO UPDATE SET
        created_at = now(), user_email = EXCLUDED.user_email, query = EXCLUDED.query,
        k = EXCLUDED.k, num_results = EXCLUDED.num_results, model_uri = EXCLUDED.model_uri,
        embeddings_uri = EXCLUDED.embeddings_uri, segment_set_uuid = EXCLUDED.segment_set_uuid,
        segment_set_name = EXCLUDED.segment_set_name, date_from = EXCLUDED.date_from,
        date_to = EXCLUDED.date_to, filter_lance_uri = EXCLUDED.filter_lance_uri,
        vehicle = EXCLUDED.vehicle, drive_id = EXCLUDED.drive_id, parquet_uri = EXCLUDED.parquet_uri,
        threshold = CASE WHEN EXCLUDED.threshold IS NOT NULL AND EXCLUDED.threshold > 0
                         THEN EXCLUDED.threshold ELSE {_BARE}.threshold END,
        thumbs_up = CASE WHEN COALESCE(jsonb_array_length(EXCLUDED.thumbs_up), 0) > 0
                         THEN EXCLUDED.thumbs_up ELSE {_BARE}.thumbs_up END,
        thumbs_down = CASE WHEN COALESCE(jsonb_array_length(EXCLUDED.thumbs_down), 0) > 0
                           THEN EXCLUDED.thumbs_down ELSE {_BARE}.thumbs_down END,
        search_vector = CASE WHEN COALESCE(jsonb_array_length(EXCLUDED.search_vector), 0) > 0
                             THEN EXCLUDED.search_vector ELSE {_BARE}.search_vector END
    RETURNING id"""
)


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
        return True
    except (SQLAlchemyError, OSError) as exc:
        logger.warning("DB: init_schema failed (%s): %s", type(exc).__name__, exc)
        return False


def insert_export(record: dict, upsert_by_tag: bool = True) -> bool:
    """Write one export-log row. Best-effort; returns True iff committed.

    ``tag`` is the true key: a non-empty tag UPSERTs (refreshes the existing same-tag row
    via the partial UNIQUE index) so iterating on a tag updates one history row instead of
    appending; an empty tag always inserts. The write is conservative -- thumbs/vector are
    only overwritten when the incoming record carries them (see the _INSERT comment). The
    ``upsert_by_tag`` arg is retained for call-site compatibility but no longer changes
    behavior: tag uniqueness is now enforced by the schema, not the caller."""
    del upsert_by_tag  # tag is unique at the schema level now; always upsert-on-conflict.
    params = {
        "user_email": record.get("user_email"),
        "query": record.get("query"),
        "tag": record.get("tag"),
        "k": record.get("k"),
        "num_results": record.get("num_results"),
        "model_uri": record.get("model_uri"),
        "embeddings_uri": record.get("embeddings_uri"),
        "segment_set_uuid": record.get("segment_set_uuid"),
        "segment_set_name": record.get("segment_set_name"),
        "date_from": record.get("date_from"),
        "date_to": record.get("date_to"),
        "filter_lance_uri": record.get("filter_lance_uri"),
        "vehicle": record.get("vehicle"),
        "drive_id": record.get("drive_id"),
        "threshold": record.get("threshold"),
        "thumbs_up": json.dumps(record.get("thumbs_up", [])),
        "thumbs_down": json.dumps(record.get("thumbs_down", [])),
        "search_vector": json.dumps(record.get("search_vector", [])),
        "parquet_uri": record.get("parquet_uri") or "",
    }
    tag = (params["tag"] or "").strip()
    try:
        if not _schema_ready:
            init_schema()
        with _get_engine().begin() as conn:
            conn.execute(text(f"SET search_path TO {SCHEMA_NAME}"))
            # Non-empty tag -> upsert on the unique index; empty tag -> plain insert (the
            # ON CONFLICT predicate doesn't cover it, so no conflict arises).
            row_id = conn.execute(_INSERT, params).scalar_one()
        logger.info("DB: export_log row %s written (tag=%r)", row_id, tag)
        return True
    except (SQLAlchemyError, OSError) as exc:
        logger.warning("DB: insert_export failed (%s): %s", type(exc).__name__, exc)
        return False


_INSERT_EPISODE = text(
    f"""INSERT INTO {_BARE_EPISODE}
        (user_email, query, tag, model_uri, embeddings_uri, features, suggested_tau,
         fit_tau, f1, precision, recall, average_precision, objective, n_pos, n_neg)
        VALUES (:user_email, :query, :tag, :model_uri, :embeddings_uri, CAST(:features AS JSONB),
                :suggested_tau, :fit_tau, :f1, :precision, :recall, :average_precision,
                :objective, :n_pos, :n_neg)
        RETURNING id"""
)


def insert_threshold_episode(record: dict) -> bool:
    """Append one threshold-tuning episode (features + fitted tau + metrics).

    Best-effort training-data capture for a future learned threshold policy;
    swallows connection/SQL errors so tuning never breaks when the DB is
    unreachable. Returns True iff the row was committed."""
    params = {
        "user_email": record.get("user_email"),
        "query": record.get("query"),
        "tag": record.get("tag"),
        "model_uri": record.get("model_uri"),
        "embeddings_uri": record.get("embeddings_uri"),
        "features": json.dumps(record.get("features", {})),
        "suggested_tau": record.get("suggested_tau"),
        "fit_tau": record.get("fit_tau"),
        "f1": record.get("f1"),
        "precision": record.get("precision"),
        "recall": record.get("recall"),
        "average_precision": record.get("average_precision"),
        "objective": record.get("objective"),
        "n_pos": record.get("n_pos"),
        "n_neg": record.get("n_neg"),
    }
    try:
        if not _schema_ready:
            init_schema()
        with _get_engine().begin() as conn:
            conn.execute(text(f"SET search_path TO {SCHEMA_NAME}"))
            row_id = conn.execute(_INSERT_EPISODE, params).scalar_one()
        logger.info("DB: threshold_episode row %s written (tag=%r)", row_id, params["tag"])
        return True
    except (SQLAlchemyError, OSError) as exc:
        logger.warning("DB: insert_threshold_episode failed (%s): %s", type(exc).__name__, exc)
        return False


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


# Search history: every export (one row per Download), newest first -- the
# user-given tag (may be blank), the natural-language query (interpretability) +
# its search vector, the top-k cutoff, the model/embedding version, and the
# parquet path written to S3. Backs the /tags ("Search history") page. Untagged
# exports are included (no tag filter) so every Download shows up.
_TAGS = text(
    f"""SELECT id, tag, query, k, num_results, model_uri, embeddings_uri,
               segment_set_uuid, segment_set_name, date_from, date_to,
               filter_lance_uri, vehicle, drive_id, threshold, parquet_uri,
               COALESCE(jsonb_array_length(search_vector), 0) AS vec_dim,
               search_vector::text                            AS search_vector_json
        FROM {_TABLE}
        ORDER BY created_at DESC
        LIMIT :limit"""
)

# Persist a tag's last-used scan threshold so the Export table pre-fills it next time. Only
# touches existing tagged rows (no insert); other columns are untouched.
_SET_TAG_THRESHOLD = text(
    f"""UPDATE {_TABLE} SET threshold = :threshold
        WHERE tag = :tag AND tag IS NOT NULL AND tag <> ''"""
)

# A single saved session by row id -- the query, vector, and filters needed to
# RESUME it. The full search_vector is returned here (unlike the list views that
# only surface its length) because resuming re-ranks with the exact vector.
_SESSION = text(
    f"""SELECT id, tag, query, embeddings_uri, segment_set_uuid, segment_set_name,
               date_from, date_to, filter_lance_uri, vehicle, drive_id,
               search_vector::text AS search_vector_json,
               thumbs_up::text   AS thumbs_up_json,
               thumbs_down::text AS thumbs_down_json
        FROM {_TABLE}
        WHERE id = :id"""
)


def tags_catalog(limit: int = 500) -> list[dict]:
    """Every export (one row per Download), newest first -- backs Search history."""
    try:
        if not _schema_ready:
            init_schema()
        with _get_engine().begin() as conn:
            conn.execute(text(f"SET search_path TO {SCHEMA_NAME}"))
            rows = conn.execute(_TAGS, {"limit": int(limit)}).mappings().all()
        return [dict(r) for r in rows]
    except (SQLAlchemyError, OSError) as exc:
        logger.warning("DB: tags_catalog failed (%s): %s", type(exc).__name__, exc)
        return []


def set_tag_thresholds(thresholds: dict[str, float]) -> bool:
    """Remember each tag's last-used scan threshold (Export table pre-fills it). Best-effort;
    updates only existing tagged rows, leaving other columns intact."""
    rows = [
        {"tag": t.strip(), "threshold": float(v)}
        for t, v in (thresholds or {}).items()
        if t and t.strip() and v is not None and float(v) > 0
    ]
    if not rows:
        return False
    try:
        if not _schema_ready:
            init_schema()
        with _get_engine().begin() as conn:
            conn.execute(text(f"SET search_path TO {SCHEMA_NAME}"))
            conn.execute(_SET_TAG_THRESHOLD, rows)
        return True
    except (SQLAlchemyError, OSError) as exc:
        logger.warning("DB: set_tag_thresholds failed (%s): %s", type(exc).__name__, exc)
        return False


def threshold_episodes(limit: int = 10000) -> list[dict]:
    """All logged threshold-tuning episodes, newest-first. Training data for the
    offline threshold-policy fit (scripts/fit_threshold_policy.py). Best-effort:
    returns [] when exp-db is unreachable."""
    try:
        if not _schema_ready:
            init_schema()
        with _get_engine().begin() as conn:
            conn.execute(text(f"SET search_path TO {SCHEMA_NAME}"))
            rows = conn.execute(
                text(
                    f"SELECT features, suggested_tau, fit_tau, f1, tag, objective, "
                    f"n_pos, n_neg, embeddings_uri, model_uri FROM {_BARE_EPISODE} "
                    f"ORDER BY created_at DESC LIMIT :lim"
                ),
                {"lim": int(limit)},
            ).mappings().all()
        return [dict(r) for r in rows]
    except (SQLAlchemyError, OSError) as exc:
        logger.warning("DB: threshold_episodes failed (%s): %s", type(exc).__name__, exc)
        return []


_INSERT_POLICY = text(
    f"""INSERT INTO {_BARE_POLICY}
        (embeddings_uri, feature_names, weights, n_episodes, mae_policy, mae_heuristic, fitted_at)
        VALUES (:embeddings_uri, CAST(:feature_names AS JSONB), CAST(:weights AS JSONB),
                :n_episodes, :mae_policy, :mae_heuristic, now())
        ON CONFLICT (embeddings_uri) DO UPDATE SET
            feature_names = EXCLUDED.feature_names, weights = EXCLUDED.weights,
            n_episodes = EXCLUDED.n_episodes, mae_policy = EXCLUDED.mae_policy,
            mae_heuristic = EXCLUDED.mae_heuristic, fitted_at = now()"""
)


def upsert_threshold_policy(record: dict) -> bool:
    """Persist the fitted ridge policy for one embedding space. Best-effort."""
    try:
        if not _schema_ready:
            init_schema()
        with _get_engine().begin() as conn:
            conn.execute(text(f"SET search_path TO {SCHEMA_NAME}"))
            conn.execute(_INSERT_POLICY, {
                "embeddings_uri": record.get("embeddings_uri") or "",
                "feature_names": json.dumps(record.get("feature_names") or []),
                "weights": json.dumps(record.get("weights") or []),
                "n_episodes": int(record.get("n_episodes") or 0),
                "mae_policy": record.get("mae_policy"),
                "mae_heuristic": record.get("mae_heuristic"),
            })
        return True
    except (SQLAlchemyError, OSError) as exc:
        logger.warning("DB: upsert_threshold_policy failed (%s): %s", type(exc).__name__, exc)
        return False


def get_threshold_policy(embeddings_uri: str) -> dict | None:
    """The fitted policy for one embedding space, or None. Best-effort."""
    try:
        if not _schema_ready:
            init_schema()
        with _get_engine().begin() as conn:
            conn.execute(text(f"SET search_path TO {SCHEMA_NAME}"))
            row = conn.execute(
                text(f"SELECT feature_names, weights, n_episodes, mae_policy, mae_heuristic "
                     f"FROM {_BARE_POLICY} WHERE embeddings_uri = :uri"),
                {"uri": embeddings_uri or ""},
            ).mappings().first()
        return dict(row) if row else None
    except (SQLAlchemyError, OSError) as exc:
        logger.warning("DB: get_threshold_policy failed (%s): %s", type(exc).__name__, exc)
        return None


def get_session(session_id: int) -> dict | None:
    """One saved session (query + vector + filters) by row id; None if missing or
    exp-db is unreachable. Backs Resume on the Search-history page."""
    try:
        if not _schema_ready:
            init_schema()
        with _get_engine().begin() as conn:
            conn.execute(text(f"SET search_path TO {SCHEMA_NAME}"))
            row = conn.execute(_SESSION, {"id": int(session_id)}).mappings().first()
        return dict(row) if row else None
    except (SQLAlchemyError, OSError) as exc:
        logger.warning("DB: get_session failed (%s): %s", type(exc).__name__, exc)
        return None


# Newest stored search vector for an EXACT (query text, corpus, model). Backs the
# "Export from config" reuse: a query whose vector was persisted by a prior export
# is re-ranked without re-encoding. The (embeddings_uri, model_uri) match keeps a
# vector from being reused against a corpus/model it was not encoded for.
_FIND_BY_QUERY = text(
    f"""SELECT search_vector::text AS search_vector_json
        FROM {_TABLE}
        WHERE query = :q AND embeddings_uri = :uri AND model_uri = :model
              AND search_vector IS NOT NULL
              AND jsonb_array_length(search_vector) > 0
        ORDER BY created_at DESC
        LIMIT 1"""
)


def find_vector_by_query(
    query: str, embeddings_uri: str, model_uri: str
) -> list[float] | None:
    """The most recent stored search_vector for an exact (query, corpus, model),
    or None if there is none / exp-db is unreachable (best-effort: never raises
    into the request -- callers re-encode on None)."""
    try:
        if not _schema_ready:
            init_schema()
        with _get_engine().begin() as conn:
            conn.execute(text(f"SET search_path TO {SCHEMA_NAME}"))
            row = (
                conn.execute(
                    _FIND_BY_QUERY,
                    {"q": query, "uri": embeddings_uri, "model": model_uri},
                )
                .mappings()
                .first()
            )
        if not row or not row.get("search_vector_json"):
            return None
        val = json.loads(row["search_vector_json"])
        return val if isinstance(val, list) and val else None
    except (SQLAlchemyError, OSError, ValueError) as exc:
        logger.warning(
            "DB: find_vector_by_query failed (%s): %s", type(exc).__name__, exc
        )
        return None


# Newest stored search vector for an exact (TAG, corpus, model). Backs config-export
# reuse: the per-query tag IS the query text, so a prior config run's persisted row is
# reused without re-encoding (and not re-appended). corpus+model match keeps a vector
# from being reused against a space it was not encoded for.
_FIND_BY_TAG = text(
    f"""SELECT search_vector::text AS search_vector_json
        FROM {_TABLE}
        WHERE tag = :tag AND embeddings_uri = :uri AND model_uri = :model
              AND search_vector IS NOT NULL
              AND jsonb_array_length(search_vector) > 0
        ORDER BY created_at DESC
        LIMIT 1"""
)


def find_vector_by_tag(
    tag: str, embeddings_uri: str, model_uri: str
) -> list[float] | None:
    """The most recent stored search_vector for an exact (tag, corpus, model), or
    None if none / exp-db unreachable (best-effort; callers re-encode on None)."""
    try:
        if not _schema_ready:
            init_schema()
        with _get_engine().begin() as conn:
            conn.execute(text(f"SET search_path TO {SCHEMA_NAME}"))
            row = (
                conn.execute(
                    _FIND_BY_TAG,
                    {"tag": tag, "uri": embeddings_uri, "model": model_uri},
                )
                .mappings()
                .first()
            )
        if not row or not row.get("search_vector_json"):
            return None
        val = json.loads(row["search_vector_json"])
        return val if isinstance(val, list) and val else None
    except (SQLAlchemyError, OSError, ValueError) as exc:
        logger.warning(
            "DB: find_vector_by_tag failed (%s): %s", type(exc).__name__, exc
        )
        return None


# Most-recent stored search vector per tag, scoped by MODEL only (not corpus). Backs the
# per-segment scan launch over an arbitrary set of saved tags: the text/refined vector lives
# in the model's joint space, so it is reused across corpora (the interactive search saves
# under the resident corpus uri while the offline scan runs on the full corpus).
_VECTORS_FOR_TAGS = text(
    f"""SELECT DISTINCT ON (tag) tag, search_vector::text AS search_vector_json
        FROM {_TABLE}
        WHERE tag IN :tags AND model_uri = :model
              AND search_vector IS NOT NULL
              AND jsonb_array_length(search_vector) > 0
        ORDER BY tag, created_at DESC"""
).bindparams(bindparam("tags", expanding=True))


def vectors_for_tags(tags: list[str], model_uri: str) -> dict[str, list[float]]:
    """``{tag: vector}`` for the given tags in the given model's space -- the newest stored
    vector per tag. Best-effort: returns only the tags it found (missing/unreachable -> absent),
    so callers re-encode the gaps."""
    wanted = sorted({t.strip() for t in tags if t and t.strip()})
    if not wanted:
        return {}
    out: dict[str, list[float]] = {}
    try:
        if not _schema_ready:
            init_schema()
        with _get_engine().begin() as conn:
            conn.execute(text(f"SET search_path TO {SCHEMA_NAME}"))
            rows = (
                conn.execute(_VECTORS_FOR_TAGS, {"tags": wanted, "model": model_uri})
                .mappings()
                .all()
            )
        for row in rows:
            sv = row.get("search_vector_json")
            if not sv:
                continue
            val = json.loads(sv)
            if isinstance(val, list) and val:
                out[row["tag"]] = val
    except (SQLAlchemyError, OSError, ValueError) as exc:
        logger.warning("DB: vectors_for_tags failed (%s): %s", type(exc).__name__, exc)
    return out


# ----- launched per-segment scans (Lilypad workloads) -----------------------------

_INSERT_SCAN = text(
    f"""INSERT INTO {_SCAN_TABLE} (
        execution_id, user_email, tags, thresholds, output_dir, lance_uri, console_url,
        status, register_segset, segset_name, filters, idem_key
    ) VALUES (
        :execution_id, :user_email, CAST(:tags AS JSONB), CAST(:thresholds AS JSONB),
        :output_dir, :lance_uri, :console_url, :status, :register_segset, :segset_name,
        CAST(:filters AS JSONB), :idem_key
    )
    ON CONFLICT (execution_id) DO UPDATE SET
        user_email = EXCLUDED.user_email, tags = EXCLUDED.tags,
        thresholds = EXCLUDED.thresholds, output_dir = EXCLUDED.output_dir,
        lance_uri = EXCLUDED.lance_uri, console_url = EXCLUDED.console_url,
        status = EXCLUDED.status, register_segset = EXCLUDED.register_segset,
        segset_name = EXCLUDED.segset_name, filters = EXCLUDED.filters,
        idem_key = COALESCE({_SCAN_TABLE}.idem_key, EXCLUDED.idem_key), updated_at = now()
    RETURNING id"""
)

_UPDATE_SCAN = text(
    f"""UPDATE {_SCAN_TABLE} SET status = :status, error = :error, updated_at = now()
        WHERE execution_id = :execution_id"""
)

# Record the DORA segment set created from a completed scan's output (once).
_UPDATE_SCAN_SEGSET = text(
    f"""UPDATE {_SCAN_TABLE}
        SET segset_uuid = :segset_uuid, segset_label = :segset_label, updated_at = now()
        WHERE execution_id = :execution_id"""
)

_LIST_SCANS = text(
    f"""SELECT execution_id, created_at, updated_at, user_email, output_dir, lance_uri,
               console_url, status, error,
               register_segset, segset_name, segset_uuid, segset_label,
               tags::text       AS tags_json,
               thresholds::text AS thresholds_json,
               filters::text    AS filters_json
        FROM {_SCAN_TABLE}
        ORDER BY created_at DESC
        LIMIT :limit"""
)


def _scan_params(record: dict) -> dict:
    """Build the _INSERT_SCAN bind params from a launch record dict."""
    return {
        "execution_id": record.get("execution_id"),
        "user_email": record.get("user_email"),
        "tags": json.dumps(record.get("tags", [])),
        "thresholds": json.dumps(record.get("thresholds", {})),
        "output_dir": record.get("output_dir") or "",
        "lance_uri": record.get("lance_uri") or "",
        "console_url": record.get("console_url") or "",
        "status": record.get("status") or "LAUNCHED",
        "register_segset": bool(record.get("register_segset")),
        "segset_name": record.get("segset_name") or "",
        "filters": json.dumps(record.get("filters") or {}),
        "idem_key": record.get("idem_key") or None,
    }


def insert_scan_job(record: dict) -> bool:
    """Record a launched per-segment scan (upsert by workload id). Best-effort."""
    params = _scan_params(record)
    if not params["execution_id"]:
        return False
    try:
        if not _schema_ready:
            init_schema()
        with _get_engine().begin() as conn:
            conn.execute(text(f"SET search_path TO {SCHEMA_NAME}"))
            conn.execute(_INSERT_SCAN, params)
        return True
    except (SQLAlchemyError, OSError) as exc:
        logger.warning("DB: insert_scan_job failed (%s): %s", type(exc).__name__, exc)
        return False


# Cross-instance single-flight for scan launches. A transaction-scoped advisory
# lock on the idempotency key serializes ONLY same-key callers and auto-releases
# on commit / rollback / crash / disconnect (no stale-lease cleanup needed); the
# launch + its record commit atomically in that same transaction.
_SCAN_ADVISORY_LOCK = text("SELECT pg_advisory_xact_lock(hashtextextended(:k, 0))")
_SELECT_SCAN_BY_IDEM = text(
    f"SELECT execution_id, status FROM {_SCAN_TABLE} WHERE idem_key = :idem"
)

# Release a failed execution's idempotency key (keep the row for history) so an
# identical relaunch can own the key instead of deduplicating to the dead workload.
_RELEASE_SCAN_IDEM = text(
    f"UPDATE {_SCAN_TABLE} SET idem_key = NULL WHERE idem_key = :idem"
)


def launch_or_get(idem_key: str, launch_and_record) -> tuple[dict, bool]:
    """Single-flight a scan launch across all instances on ``idem_key``.

    Holds an advisory lock on the key, then: if a row already carries this key,
    return its existing workload id (dedup hit); otherwise call
    ``launch_and_record() -> (result, record)`` -- the ONE real launch -- and
    persist the record (with ``idem_key``) in the SAME transaction. Returns
    ``(result, deduplicated)``. On a dedup hit ``result`` is
    ``{"execution_id": <existing>}``.

    Raises if the DB is unreachable (so the caller fails loudly rather than
    double-launching) or if ``launch_and_record`` raises (the transaction rolls
    back, the lock releases, and the next caller retries cleanly -- nothing is
    recorded).
    """
    if not _schema_ready:
        init_schema()
    with _get_engine().begin() as conn:
        conn.execute(text(f"SET search_path TO {SCHEMA_NAME}"))
        conn.execute(_SCAN_ADVISORY_LOCK, {"k": idem_key})
        row = conn.execute(_SELECT_SCAN_BY_IDEM, {"idem": idem_key}).first()
        if row and row.execution_id:
            # A FAILED/ABORTED execution must not satisfy dedup -- returning it hands
            # the caller a dead workload (and a lance_uri that will never exist).
            # Release its key (the row stays for history) and fall through to a
            # fresh launch that takes over the key.
            up = (row.status or "").upper()
            if any(t in up for t in ("FAILED", "ABORTED")):
                conn.execute(_RELEASE_SCAN_IDEM, {"idem": idem_key})
            else:
                return {"execution_id": row.execution_id}, True
        result, record = launch_and_record()
        params = _scan_params({**record, "idem_key": idem_key})
        if params["execution_id"]:
            conn.execute(_INSERT_SCAN, params)
        return result, False


def release_scan_idem(idem_key: str) -> bool:
    """Release a (dead) execution's idempotency key so an identical relaunch can own
    it. The row keeps its history; only the dedup key is cleared. Best-effort."""
    if not idem_key:
        return False
    try:
        if not _schema_ready:
            init_schema()
        with _get_engine().begin() as conn:
            conn.execute(text(f"SET search_path TO {SCHEMA_NAME}"))
            conn.execute(_RELEASE_SCAN_IDEM, {"idem": idem_key})
        return True
    except (SQLAlchemyError, OSError) as exc:
        logger.warning("DB: release_scan_idem failed (%s): %s", type(exc).__name__, exc)
        return False


def get_scan_job(execution_id: str) -> dict | None:
    """One scan job by workload id (parsed tags/thresholds), or None. Best-effort."""
    if not execution_id:
        return None
    try:
        if not _schema_ready:
            init_schema()
        with _get_engine().begin() as conn:
            conn.execute(text(f"SET search_path TO {SCHEMA_NAME}"))
            row = (
                conn.execute(_LIST_SCANS, {"limit": 500})
                .mappings()
                .all()
            )
        for r in row:
            if r.get("execution_id") == execution_id:
                d = dict(r)
                try:
                    d["tags"] = json.loads(d.pop("tags_json") or "[]")
                except ValueError:
                    d["tags"] = []
                d.pop("thresholds_json", None)
                return d
        return None
    except (SQLAlchemyError, OSError) as exc:
        logger.warning("DB: get_scan_job failed (%s): %s", type(exc).__name__, exc)
        return None


def set_scan_segset(execution_id: str, segset_uuid: str, segset_label: str) -> bool:
    """Record the DORA segment set registered from a completed scan. Best-effort."""
    if not execution_id:
        return False
    try:
        if not _schema_ready:
            init_schema()
        with _get_engine().begin() as conn:
            conn.execute(text(f"SET search_path TO {SCHEMA_NAME}"))
            conn.execute(
                _UPDATE_SCAN_SEGSET,
                {
                    "execution_id": execution_id,
                    "segset_uuid": segset_uuid or "",
                    "segset_label": segset_label or "",
                },
            )
        return True
    except (SQLAlchemyError, OSError) as exc:
        logger.warning("DB: set_scan_segset failed (%s): %s", type(exc).__name__, exc)
        return False


def update_scan_job(execution_id: str, status: str, error: str = "") -> bool:
    """Update a scan job's last-polled status/error. Best-effort."""
    if not execution_id:
        return False
    try:
        if not _schema_ready:
            init_schema()
        with _get_engine().begin() as conn:
            conn.execute(text(f"SET search_path TO {SCHEMA_NAME}"))
            conn.execute(
                _UPDATE_SCAN,
                {"execution_id": execution_id, "status": status, "error": error or ""},
            )
        return True
    except (SQLAlchemyError, OSError) as exc:
        logger.warning("DB: update_scan_job failed (%s): %s", type(exc).__name__, exc)
        return False


def list_scan_jobs(limit: int = 50) -> list[dict]:
    """Recent launched scans (newest first); [] if exp-db is unreachable. Each row
    parses tags (list) and thresholds ({tag: float}) from their JSON columns."""
    out: list[dict] = []
    try:
        if not _schema_ready:
            init_schema()
        with _get_engine().begin() as conn:
            conn.execute(text(f"SET search_path TO {SCHEMA_NAME}"))
            rows = conn.execute(_LIST_SCANS, {"limit": int(limit)}).mappings().all()
        for row in rows:
            d = dict(row)
            try:
                d["tags"] = json.loads(d.pop("tags_json") or "[]")
            except ValueError:
                d["tags"] = []
            try:
                d["thresholds"] = json.loads(d.pop("thresholds_json") or "{}")
            except ValueError:
                d["thresholds"] = {}
            try:
                d["filters"] = json.loads(d.pop("filters_json") or "{}")
            except ValueError:
                d["filters"] = {}
            out.append(d)
    except (SQLAlchemyError, OSError) as exc:
        logger.warning("DB: list_scan_jobs failed (%s): %s", type(exc).__name__, exc)
    return out
