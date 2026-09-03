"""The tag catalog: what a tag is, per version and per project, and what has been
exported from it.

A tag is one search vector with a history of versions and one threshold per
project. The vector is fleet-independent (every project is encoded by the same
model); the threshold is not, because each project's corpus has its own PCA
basis and score distribution. Exports are recorded here too, keyed by the
parameters that produced them, so an identical request returns the artifact it
already made.

Three tables. `tags` is one row per (tag, version): the version's vector and its
per-project thresholds, plus the tag-level fields (description, model, pin,
delete) repeated on every version of that tag. The repetition is deliberate --
those fields change rarely and always through `update`/`delete`, which write
every row of the tag in one transaction, and folding them in keeps a tag's whole
state one row wide instead of a three-way join on every read.

Tables are declared with SQLAlchemy Core so the same code runs against Cloud SQL
in production and SQLite in tests. Every public method opens its own short
transaction; none is on a search's hot path.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import threading
from typing import Any

import sqlalchemy as sa
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)

THRESHOLD_MODES = ("suggested", "explicit", "fitted")
EXPORT_STATUSES = ("pending", "running", "ready", "error")
# Fields of a threshold record that are stored; anything else a caller passes is
# dropped rather than persisted under a name nothing reads back.
_THRESHOLD_FIELDS = ("value", "mode", "selected", "precision", "recall", "f1",
                     "corpus_version", "stale", "set_at")


class TagExists(Exception):
    """The tag already has a threshold on this project; refine it instead."""


class UnknownTag(KeyError):
    pass


class TagPinned(Exception):
    """A pinned tag cannot be deleted."""


def _now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _iso(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, dt.datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=dt.timezone.utc)
        return value.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return str(value)


def export_id(tag: str, version: int, project: str, output: str, params: dict) -> str:
    """Stable id for one materialized export: the same request maps to the same
    artifact, so a retry finds what it already made."""
    canon = json.dumps(
        {"tag": tag, "version": int(version), "project": project, "output": output,
         "params": {k: v for k, v in sorted(params.items()) if v not in (None, "", [], False)}},
        sort_keys=True, separators=(",", ":"),
    )
    return hashlib.sha1(canon.encode()).hexdigest()[:20]


def _projects_key(thresholds: dict) -> str:
    """The `projects` column for a thresholds map: comma-delimited with sentinel
    commas, so `LIKE '%,neuron,%'` is an exact membership test on every backend.
    JSON key containment is not portable between Postgres and SQLite; this is."""
    return "," + ",".join(sorted(thresholds)) + "," if thresholds else ","


def _tables(schema: str | None) -> tuple[sa.MetaData, dict[str, sa.Table]]:
    md = sa.MetaData(schema=schema)
    t: dict[str, sa.Table] = {}
    # One row per (tag, version). `thresholds` is {project: {value, mode, ...}};
    # `projects` mirrors its keys so the list query can filter by project in SQL.
    # description/model/pinned_version/deleted_at are tag-level, held on every
    # version row and written together.
    t["tags"] = sa.Table(
        "tags", md,
        sa.Column("tag", sa.Text, primary_key=True),
        sa.Column("version", sa.Integer, primary_key=True),
        sa.Column("description", sa.Text, nullable=False, default=""),
        sa.Column("model", sa.Text, nullable=False),
        sa.Column("pinned_version", sa.Integer, nullable=True),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, default=_now),
        sa.Column("created_by", sa.Text, nullable=False, default=""),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, default=_now, onupdate=_now),
        sa.Column("source", sa.JSON, nullable=False),
        sa.Column("vector", sa.JSON, nullable=False),
        sa.Column("refine", sa.JSON, nullable=True),
        sa.Column("thresholds", sa.JSON, nullable=False, default=dict),
        sa.Column("projects", sa.Text, nullable=False, default=","),
    )
    t["marks"] = sa.Table(
        "marks", md,
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("tag", sa.Text, nullable=False),
        sa.Column("version", sa.Integer, nullable=False),
        sa.Column("project", sa.Text, nullable=False),
        sa.Column("chunk_id", sa.Text, nullable=False),
        sa.Column("mark", sa.Text, nullable=False),
        sa.Column("user_email", sa.Text, nullable=False, default=""),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, default=_now),
        sa.Index("marks_tag_version_idx", "tag", "version"),
    )
    t["exports"] = sa.Table(
        "exports", md,
        sa.Column("export_id", sa.Text, primary_key=True),
        sa.Column("tag", sa.Text, nullable=False),
        sa.Column("version", sa.Integer, nullable=False),
        sa.Column("project", sa.Text, nullable=False),
        sa.Column("output", sa.Text, nullable=False),
        sa.Column("params", sa.JSON, nullable=False),
        sa.Column("status", sa.Text, nullable=False),
        sa.Column("error", sa.Text, nullable=False, default=""),
        sa.Column("uri", sa.Text, nullable=False, default=""),
        sa.Column("num_rows", sa.Integer, nullable=True),
        sa.Column("rows_above_threshold", sa.Integer, nullable=True),
        sa.Column("rows_unscorable", sa.Integer, nullable=True),
        sa.Column("cutoff", sa.Float, nullable=True),
        sa.Column("confidence_basis", sa.Text, nullable=True),
        sa.Column("segment_set_uuid", sa.Text, nullable=True),
        sa.Column("filters_applied", sa.JSON, nullable=True),
        sa.Column("corpus_version", sa.Integer, nullable=True),
        sa.Column("created_by", sa.Text, nullable=False, default=""),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, default=_now),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Index("exports_tag_idx", "tag"),
    )
    return md, t


class Catalog:
    def __init__(self, engine: Engine, schema: str | None = None) -> None:
        self.engine = engine
        self.schema = schema
        self.md, self.t = _tables(schema)

    # ------------------------------------------------------------- lifecycle
    def init(self) -> None:
        self.md.create_all(self.engine)

    def _begin(self):
        return self.engine.begin()

    # ------------------------------------------------------------------ reads
    def _row(self, conn, tag: str, version: int | None = None, include_deleted: bool = False):
        """One version row, or the latest when `version` is None."""
        T = self.t["tags"]
        where = [T.c.tag == tag]
        if version is not None:
            where.append(T.c.version == int(version))
        row = conn.execute(
            sa.select(T).where(*where).order_by(T.c.version.desc()).limit(1)
        ).mappings().first()
        if row is None or (row["deleted_at"] is not None and not include_deleted):
            raise UnknownTag(tag if version is None else f"{tag} v{version}")
        return row

    def _latest_version(self, conn, tag: str) -> int:
        T = self.t["tags"]
        return int(conn.execute(sa.select(sa.func.max(T.c.version)).where(T.c.tag == tag)).scalar() or 0)

    def resolve_version(self, tag: str, version: int | None) -> int:
        """`version` if given, else the pinned version, else the latest."""
        with self._begin() as conn:
            row = self._row(conn, tag)
            if version is not None:
                self._row(conn, tag, int(version))
                return int(version)
            if row["pinned_version"] is not None:
                return int(row["pinned_version"])
            return int(row["version"])

    def _thresholds(self, row) -> dict:
        """The public thresholds map for a row: stored fields only, and the
        optional ones omitted when unset so a caller sees what was measured."""
        out = {}
        for project, th in (row["thresholds"] or {}).items():
            d = {"value": th["value"], "mode": th["mode"],
                 "corpus_version": th.get("corpus_version"), "set_at": _iso(th.get("set_at"))}
            for k in ("selected", "precision", "recall", "f1"):
                if th.get(k) is not None:
                    d[k] = th[k]
            if th.get("stale"):
                d["stale"] = True
            out[project] = d
        return out

    def version(self, tag: str, version: int | None = None, with_vector: bool = True) -> dict:
        """One version as the API returns it, with its per-project thresholds."""
        v = self.resolve_version(tag, version)
        with self._begin() as conn:
            row = self._row(conn, tag, v)
            out = {
                "tag": tag, "version": v, "pinned_version": row["pinned_version"],
                "description": row["description"], "created_at": _iso(row["created_at"]),
                "created_by": row["created_by"], "model": row["model"],
                "source": row["source"], "thresholds": self._thresholds(row),
            }
            if with_vector:
                out["vector"] = row["vector"]
            if row["refine"]:
                out["refine"] = row["refine"]
            return out

    def record(self, tag: str) -> dict:
        """The metadata-mode read: every version, every threshold, past exports."""
        T, E = self.t["tags"], self.t["exports"]
        with self._begin() as conn:
            rows = conn.execute(
                sa.select(T).where(T.c.tag == tag).order_by(T.c.version)
            ).mappings().all()
            if not rows or rows[-1]["deleted_at"] is not None:
                raise UnknownTag(tag)
            versions = [{
                "version": r["version"], "created_at": _iso(r["created_at"]),
                "created_by": r["created_by"], "source": r["source"],
                "thresholds": self._thresholds(r),
            } for r in rows]
            exports = [self._export_public(r) for r in conn.execute(
                sa.select(E).where(E.c.tag == tag).order_by(E.c.created_at.desc())).mappings()]
            first, latest = rows[0], rows[-1]
            return {
                "tag": tag, "description": latest["description"], "model": latest["model"],
                "created_by": first["created_by"], "created_at": _iso(first["created_at"]),
                "pinned_version": latest["pinned_version"], "versions": versions, "exports": exports,
            }

    def list(self, project: str | None = None, q: str | None = None, created_by: str | None = None,
             page: int = 1, page_size: int = 50) -> dict:
        """A page of tags at their latest version, most recently touched first."""
        T = self.t["tags"]
        with self._begin() as conn:
            newest = (
                sa.select(T.c.tag, sa.func.max(T.c.version).label("version"))
                .group_by(T.c.tag).subquery()
            )
            where = [T.c.deleted_at.is_(None)]
            if q:
                like = f"%{q.lower()}%"
                where.append(sa.or_(sa.func.lower(T.c.tag).like(like),
                                    sa.func.lower(T.c.description).like(like)))
            if created_by:
                where.append(T.c.created_by == created_by)
            if project:
                where.append(T.c.projects.like(f"%,{project},%"))
            base = sa.select(T).join(
                newest, sa.and_(T.c.tag == newest.c.tag, T.c.version == newest.c.version)
            ).where(*where)
            total = conn.execute(
                sa.select(sa.func.count()).select_from(base.subquery())).scalar() or 0
            rows = conn.execute(
                base.order_by(T.c.updated_at.desc())
                .offset((max(page, 1) - 1) * page_size).limit(page_size)
            ).mappings().all()
            tags = [{
                "tag": r["tag"], "description": r["description"], "version": r["version"],
                "pinned_version": r["pinned_version"], "model": r["model"],
                "created_by": r["created_by"], "updated_at": _iso(r["updated_at"]),
                "thresholds": {
                    p: {k: v for k, v in d.items() if k in ("value", "mode", "precision", "recall", "stale")}
                    for p, d in self._thresholds(r).items()
                },
            } for r in rows]
            return {"tags": tags, "page": page, "page_size": page_size, "total": int(total)}

    # ----------------------------------------------------------------- writes
    def _threshold_values(self, th: dict, *, stale: bool = False) -> dict:
        if th.get("mode") not in THRESHOLD_MODES:
            raise ValueError(f"threshold mode must be one of {THRESHOLD_MODES}")
        out = {k: th.get(k) for k in _THRESHOLD_FIELDS if k in th}
        out.update({"value": float(th["value"]), "mode": th["mode"], "stale": stale,
                    "set_at": _iso(_now())})
        return out

    def _write_thresholds(self, conn, tag: str, version: int, thresholds: dict) -> None:
        T = self.t["tags"]
        conn.execute(sa.update(T).where(T.c.tag == tag, T.c.version == version).values(
            thresholds=thresholds, projects=_projects_key(thresholds), updated_at=_now()))

    def _put_threshold(self, conn, tag: str, version: int, project: str, th: dict) -> None:
        row = self._row(conn, tag, version)
        thresholds = dict(row["thresholds"] or {})
        thresholds[project] = self._threshold_values(th)
        self._write_thresholds(conn, tag, version, thresholds)

    def create(self, *, tag: str, project: str, source: dict, vector: list[float], model: str,
               threshold: dict, description: str = "", created_by: str = "") -> dict:
        """Create a tag at version 1, or add `project`'s threshold to an existing
        tag's latest version. Raises TagExists if that project already has one."""
        T = self.t["tags"]
        with self._begin() as conn:
            existing = conn.execute(
                sa.select(T).where(T.c.tag == tag).order_by(T.c.version.desc()).limit(1)
            ).mappings().first()
            if existing is not None and existing["deleted_at"] is None:
                version = int(existing["version"])
                if project in (existing["thresholds"] or {}):
                    raise TagExists(f"{tag} already has a {project} threshold; use PUT to refine")
                self._put_threshold(conn, tag, version, project, threshold)
                if description:
                    conn.execute(sa.update(T).where(T.c.tag == tag).values(description=description))
            else:
                if existing is not None:  # revive a deleted tag as a fresh v1
                    conn.execute(sa.delete(T).where(T.c.tag == tag))
                    conn.execute(sa.delete(self.t["marks"]).where(self.t["marks"].c.tag == tag))
                th = {project: self._threshold_values(threshold)}
                conn.execute(sa.insert(T).values(
                    tag=tag, version=1, description=description, model=model,
                    created_by=created_by, source=source, vector=list(map(float, vector)),
                    thresholds=th, projects=_projects_key(th)))
                version = 1
        return self.version(tag, version)

    def set_threshold(self, tag: str, project: str, threshold: dict, version: int | None = None) -> dict:
        """Re-threshold one project at a version (latest by default). No new version."""
        with self._begin() as conn:
            v = int(self._row(conn, tag, version)["version"])
            self._put_threshold(conn, tag, v, project, threshold)
        return self.version(tag, v)

    def new_version(self, *, tag: str, project: str, source: dict, vector: list[float],
                    threshold: dict, refine: dict | None = None, created_by: str = "") -> dict:
        """Append a version with a new vector. The refining project gets its fitted
        threshold; every other project's threshold is carried over marked stale,
        since the vector it was calibrated against has changed."""
        T = self.t["tags"]
        with self._begin() as conn:
            prev = self._row(conn, tag)
            v = int(prev["version"]) + 1
            thresholds = {p: {**th, "stale": True}
                          for p, th in (prev["thresholds"] or {}).items() if p != project}
            thresholds[project] = self._threshold_values(threshold)
            conn.execute(sa.insert(T).values(
                tag=tag, version=v, description=prev["description"], model=prev["model"],
                pinned_version=prev["pinned_version"], created_by=created_by, source=source,
                vector=list(map(float, vector)), refine=refine,
                thresholds=thresholds, projects=_projects_key(thresholds)))
        return self.version(tag, v)

    def update(self, tag: str, *, description: str | None = None,
               pinned_version: int | None | object = ...) -> dict:
        """Set tag-level fields. These live on every version row, so one statement
        writes them all -- no version may disagree about the pin or the name."""
        T = self.t["tags"]
        with self._begin() as conn:
            self._row(conn, tag)
            values: dict[str, Any] = {"updated_at": _now()}
            if description is not None:
                values["description"] = description
            if pinned_version is not ...:
                if pinned_version is not None and pinned_version > self._latest_version(conn, tag):
                    raise UnknownTag(f"{tag} v{pinned_version}")
                values["pinned_version"] = pinned_version
            conn.execute(sa.update(T).where(T.c.tag == tag).values(**values))
        return self.version(tag)

    def delete(self, tag: str) -> None:
        """Soft delete. Exports stay readable by URI; a pinned tag must be unpinned first."""
        T = self.t["tags"]
        with self._begin() as conn:
            row = self._row(conn, tag)
            if row["pinned_version"] is not None:
                raise TagPinned(tag)
            conn.execute(sa.update(T).where(T.c.tag == tag).values(deleted_at=_now(), updated_at=_now()))

    def add_marks(self, tag: str, version: int, project: str, marks: list[dict], user_email: str = "") -> int:
        M = self.t["marks"]
        with self._begin() as conn:
            self._row(conn, tag)
            n = 0
            for m in marks:
                cid, mark = str(m["chunk_id"]), str(m["mark"])
                if mark not in ("up", "down"):
                    raise ValueError("mark must be up or down")
                conn.execute(sa.delete(M).where(M.c.tag == tag, M.c.version == version, M.c.chunk_id == cid))
                conn.execute(sa.insert(M).values(tag=tag, version=version, project=project, chunk_id=cid,
                                                 mark=mark, user_email=user_email))
                n += 1
            return n

    def marks(self, tag: str, version: int) -> list[dict]:
        M = self.t["marks"]
        with self._begin() as conn:
            return [dict(r) for r in conn.execute(
                sa.select(M.c.chunk_id, M.c.mark, M.c.project, M.c.user_email)
                .where(M.c.tag == tag, M.c.version == version).order_by(M.c.id)).mappings()]

    # ---------------------------------------------------------------- exports
    def _export_public(self, r) -> dict:
        out = {"export_id": r["export_id"], "tag": r["tag"], "version": r["version"], "project": r["project"],
               "output": r["output"], "status": r["status"], "params": r["params"],
               "created_at": _iso(r["created_at"]), "started_at": _iso(r["started_at"]),
               "finished_at": _iso(r["finished_at"]), "corpus_version": r["corpus_version"]}
        if r["status"] == "ready":
            out.update({"uri": r["uri"], "num_rows": r["num_rows"]})
            for k in ("rows_above_threshold", "rows_unscorable", "cutoff", "confidence_basis",
                      "segment_set_uuid", "filters_applied"):
                if r[k] not in (None, ""):
                    out[k] = r[k]
        if r["status"] == "error":
            out["error"] = r["error"]
        return out

    def export_get(self, eid: str) -> dict | None:
        E = self.t["exports"]
        with self._begin() as conn:
            r = conn.execute(sa.select(E).where(E.c.export_id == eid)).mappings().first()
            return self._export_public(r) if r else None

    def export_claim(self, *, tag: str, version: int, project: str, output: str, params: dict,
                     created_by: str = "") -> tuple[dict, bool]:
        """Return (export, claimed). `claimed` is True only for the caller that
        created the row, so exactly one worker runs a given export."""
        eid = export_id(tag, version, project, output, params)
        E = self.t["exports"]
        with self._begin() as conn:
            r = conn.execute(sa.select(E).where(E.c.export_id == eid)).mappings().first()
            if r is not None:
                return self._export_public(r), False
            conn.execute(sa.insert(E).values(export_id=eid, tag=tag, version=version, project=project,
                                             output=output, params=params, status="running",
                                             created_by=created_by, started_at=_now()))
        return self.export_get(eid), True

    def export_finish(self, eid: str, **fields) -> None:
        E = self.t["exports"]
        allowed = {"uri", "num_rows", "rows_above_threshold", "rows_unscorable", "cutoff",
                   "confidence_basis", "segment_set_uuid", "filters_applied", "corpus_version"}
        values = {k: v for k, v in fields.items() if k in allowed}
        with self._begin() as conn:
            conn.execute(sa.update(E).where(E.c.export_id == eid).values(status="ready", finished_at=_now(), **values))

    def export_fail(self, eid: str, error: str) -> None:
        E = self.t["exports"]
        with self._begin() as conn:
            conn.execute(sa.update(E).where(E.c.export_id == eid).values(status="error", error=error[:2000],
                                                                          finished_at=_now()))

    def export_forget(self, eid: str) -> None:
        """Drop a failed row so the next identical request runs again."""
        E = self.t["exports"]
        with self._begin() as conn:
            conn.execute(sa.delete(E).where(E.c.export_id == eid, E.c.status == "error"))

    # --------------------------------------------------------------- backfill
    def backfill_export_log(self, rows: list[dict], *, project: str, model: str) -> int:
        """Seed the catalog from the old one-row-per-tag store. Each tagged row with
        a vector becomes version 1 on `project`; a stored cosine threshold becomes
        an explicit threshold; a parquet URI becomes a ready export. Tags already
        in the catalog are left alone, so this is safe to run at every start."""
        T = self.t["tags"]
        n = 0
        with self._begin() as conn:
            have = {r[0] for r in conn.execute(sa.select(T.c.tag))}
        for r in rows:
            tag = (r.get("tag") or "").strip()
            vec = r.get("search_vector")
            if isinstance(vec, str):
                try:
                    vec = json.loads(vec)
                except ValueError:
                    vec = None
            if not tag or tag in have or not isinstance(vec, list) or not vec:
                continue
            th = r.get("threshold")
            source = {"type": "text", "text": r.get("query") or tag}
            created = r.get("created_at") or _now()
            thresholds = {}
            if th is not None and float(th) > 0:
                thresholds[project] = self._threshold_values({"value": float(th), "mode": "explicit"})
            with self._begin() as conn:
                conn.execute(sa.insert(T).values(
                    tag=tag, version=1, description="", model=model,
                    created_by=r.get("user_email") or "", created_at=created, updated_at=created,
                    source=source, vector=list(map(float, vec)),
                    thresholds=thresholds, projects=_projects_key(thresholds)))
                if r.get("parquet_uri"):
                    params = {"legacy": True}
                    eid = export_id(tag, 1, project, "parquet", params)
                    conn.execute(sa.insert(self.t["exports"]).values(
                        export_id=eid, tag=tag, version=1, project=project, output="parquet",
                        params=params, status="ready", uri=r["parquet_uri"], num_rows=r.get("num_results"),
                        created_by=r.get("user_email") or "", created_at=created, finished_at=created))
            have.add(tag)
            n += 1
        return n


# ------------------------------------------------------------------ singleton
_CATALOG: Catalog | None = None
_LOCK = threading.Lock()


def get() -> Catalog:
    """The process-wide catalog on the app's Cloud SQL engine and schema."""
    global _CATALOG
    if _CATALOG is None:
        with _LOCK:
            if _CATALOG is None:
                import db

                _CATALOG = Catalog(db._get_engine(), schema=db.SCHEMA_NAME)
    return _CATALOG
