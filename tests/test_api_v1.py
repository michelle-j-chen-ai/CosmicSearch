"""The versioned API: request shapes, validation, and wiring through the app."""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest
import sqlalchemy as sa
from fastapi.testclient import TestClient

import api_v1
import catalog
import deployment
import full_corpus
import web_server as ws


# ------------------------------------------------------------------ helpers
def test_tag_names_are_constrained():
    assert api_v1.valid_tag(" motorcycle_filtering ") == "motorcycle_filtering"
    for bad in ("", "Has-Dash", "x" * 65, "sp ace"):
        with pytest.raises(Exception) as e:
            api_v1.valid_tag(bad)
        assert e.value.status_code == 422


def test_clip_uri_resolves_to_its_project():
    neuron = deployment.get("neuron").mp4_prefix
    assert api_v1.project_for_clip(neuron + "dt=2026-08-20/r_t1.mp4").name == "neuron"
    assert api_v1.project_for_clip("s3://frontier-perception-datasets/sibogeng/vlm/chunks_mp4_full/x.mp4").name == "frontier"
    assert api_v1.project_for_clip("s3://elsewhere/x.mp4") is None


def test_image_inputs_must_come_from_the_projects_bucket():
    assert api_v1.input_prefixes(deployment.get("frontier")) == ["s3://frontier-perception-datasets/"]


def test_confidence_is_a_margin_between_cutoff_and_best():
    c = api_v1._confidence(np.array([0.9, 0.5, 0.1]), tau=0.5)
    assert c.tolist() == [1.0, 0.0, 0.0]
    assert api_v1._confidence(np.array([]), 0.5).size == 0


class _Corpus:
    """Just enough corpus for threshold_for and the live page."""
    dataset_version = 661
    model_id = "black_dwarf"
    num_rows = 5
    dim = 4

    def score(self, vec):
        return np.array([0.9, 0.6, 0.3, 0.2, 0.1], dtype=np.float32), 0.01


def test_threshold_for_suggested_and_explicit(monkeypatch):
    monkeypatch.setattr(ws, "_score_histogram", lambda scores, tau: {"bins": [0, 1], "counts": [5]})
    th, hist = api_v1.threshold_for("explicit", 0.5, np.zeros(4), _Corpus(), want_distribution=True)
    assert th == {"value": 0.5, "mode": "explicit", "selected": 2, "corpus_version": 661}
    assert hist == {"bins": [0, 1], "counts": [5]}
    th, hist = api_v1.threshold_for("suggested", None, np.zeros(4), _Corpus())
    assert th["mode"] == "suggested" and 0 <= th["value"] <= 1 and hist is None
    for mode, val in (("explicit", None), ("weird", 0.1), ("explicit", 1.5)):
        with pytest.raises(Exception) as e:
            api_v1.threshold_for(mode, val, np.zeros(4), _Corpus())
        assert e.value.status_code == 422


def _tiny_full_corpus():
    n = 4
    meta = {
        "vehicle": np.array([0, 0, 1, 1], dtype=np.int32), "vehicle_uniques": ["v1", "v2"],
        "run_uuid": np.array([0, 1, 0, 1], dtype=np.int32), "run_uuid_uniques": ["run-a", "run-b"],
        "dt": np.zeros(n, dtype=np.int32), "dt_uniques": ["2026-01-01"],
        "chunk_start_unix": np.array([10, 20, 30, 40], dtype=np.int64),
        "chunk_end_unix": np.array([18, 28, 38, 48], dtype=np.int64),
        "dx_internal_id": np.full(n, -1, dtype=np.int64),
        "segment_id": pa.array(["s1", "s2", "s3", "s4"], pa.large_string()),
    }
    pca = np.eye(4, dtype=np.float32)
    return full_corpus.FullCorpus(np.zeros((n, 4), dtype=np.int8), pca, np.ones(4, dtype=np.float32), meta)


def test_rows_for_chunk_ids_resolves_only_real_clips():
    c = _tiny_full_corpus()
    found = c.rows_for_chunk_ids(["run-a#t10", "run-b#t40", "run-a#t99", "run-z#t10", "garbage"])
    assert found == {"run-a#t10": 0, "run-b#t40": 3}
    assert c.rows_for_chunk_ids([]) == {}


def test_media_uri_uses_the_corpus_prefix():
    c = _tiny_full_corpus()
    c.mp4_prefix = "s3://b/clips/"
    assert c._hit(0, 1, 0.5, 0.0).source_media_uri == "s3://b/clips/dt=2026-01-01/run-a_t10.mp4"


# -------------------------------------------------------------- app wiring
@pytest.fixture
def client(monkeypatch):
    # One shared in-memory database: the test client runs handlers on a worker
    # thread, and a plain sqlite:// engine would hand each thread its own empty db.
    from sqlalchemy.pool import StaticPool
    engine = sa.create_engine("sqlite://", poolclass=StaticPool,
                              connect_args={"check_same_thread": False})
    cat = catalog.Catalog(engine, schema=None)
    cat.init()
    monkeypatch.setattr(catalog, "_CATALOG", cat)
    monkeypatch.setenv("NLS_PROJECTS", "neuron,frontier")
    monkeypatch.setattr(ws, "_CORPORA", {})
    return TestClient(ws.app), cat


def test_health_is_503_until_a_corpus_is_resident(client):
    c, _ = client
    r = c.get("/api/v1/health")
    assert r.status_code == 503 and r.headers["retry-after"]
    body = r.json()
    assert set(body["projects"]) == {"neuron", "frontier"}
    assert body["projects"]["frontier"]["clip_prefix"].startswith("s3://frontier-perception-datasets/")
    assert body["limits"]["marks_per_put"] == 500


def test_video_rejects_a_uri_outside_every_clip_prefix(client):
    c, _ = client
    r = c.get("/api/v1/video", params={"uri": "s3://elsewhere/x.mp4"})
    assert r.status_code == 404 and r.json()["detail"]["code"] == "not_a_clip"


def test_unknown_tag_and_project_are_404(client):
    c, _ = client
    assert c.get("/api/v1/tags/nope").status_code == 404
    assert c.get("/api/v1/tags", params={"project": "cars"}).status_code == 404
    assert c.delete("/api/v1/tags/nope").status_code == 404


def test_tag_lifecycle_through_the_api(client, monkeypatch):
    c, cat = client
    corpus = _Corpus()
    monkeypatch.setitem(ws._CORPORA, "neuron", {"corpus": corpus, "status": "ready", "error": "", "started": 0.0})
    monkeypatch.setitem(ws._state, "model_ready", True)
    monkeypatch.setitem(ws._state, "processor", None)
    monkeypatch.setitem(ws._state, "model", None)
    monkeypatch.setitem(ws._state, "cfg", type("C", (), {"device": "cpu", "presign_ttl_s": 60, "export_s3_prefix": ""})())
    monkeypatch.setattr(api_v1.search_engine, "encode_query", lambda *a, **k: np.array([1, 0, 0, 0], dtype=np.float32))

    body = {"tag": "motorcycle_filtering", "project": "neuron", "description": "lanes",
            "input": {"type": "text", "text": "motorcycle filtering"}, "threshold_mode": "explicit",
            "threshold": 0.5}
    r = c.post("/api/v1/tags", json=body, headers={"X-NLS-Actor": "pipe@x"})
    assert r.status_code == 201, r.text
    rec = r.json()
    assert rec["version"] == 1 and rec["thresholds"]["neuron"]["value"] == 0.5
    assert rec["created_by"] == "pipe@x" and rec["project"] == "neuron" and rec["corpus_version"] == 661

    # same project again is a conflict; try-mode never writes
    assert c.post("/api/v1/tags", json=body).status_code == 409
    r = c.post("/api/v1/tags", json={**body, "tag": "scratch", "save": False, "distribution": True})
    assert r.status_code == 200 and r.json()["version"] is None and "distribution" in r.json()
    assert c.get("/api/v1/tags/scratch").status_code == 404

    # list, read, pin, re-threshold, delete
    assert c.get("/api/v1/tags", params={"project": "neuron"}).json()["total"] == 1
    assert c.get("/api/v1/tags/motorcycle_filtering").json()["versions"][0]["version"] == 1
    r = c.put("/api/v1/tags/motorcycle_filtering", json={"project": "neuron", "threshold_mode": "explicit",
                                                       "threshold": 0.3, "pinned_version": 1})
    assert r.status_code == 200 and r.json()["thresholds"]["neuron"]["value"] == 0.3
    assert r.json()["pinned_version"] == 1
    assert c.delete("/api/v1/tags/motorcycle_filtering").status_code == 409  # pinned
    assert c.put("/api/v1/tags/motorcycle_filtering", json={"pinned_version": None}).status_code == 200
    assert c.delete("/api/v1/tags/motorcycle_filtering").status_code == 204
    assert c.get("/api/v1/tags/motorcycle_filtering").status_code == 404


def test_export_on_a_project_without_a_threshold_is_a_409(client, monkeypatch):
    c, cat = client
    cat.create(tag="t", project="neuron", source={"type": "text", "text": "x"}, vector=[1, 0, 0, 0],
               model="black_dwarf", threshold={"value": 0.5, "mode": "explicit"})
    r = c.get("/api/v1/tags/t", params={"project": "frontier", "output": "parquet"})
    assert r.status_code == 409 and r.json()["detail"]["code"] == "no_threshold"
    r = c.get("/api/v1/tags/t", params={"output": "xlsx"})
    assert r.status_code == 422


def test_first_per_segment_keeps_the_best_clip_of_each_segment():
    c = _tiny_full_corpus()
    c._meta["segment_id"] = pa.array(["s1", "s1", None, "s2"], pa.large_string())
    rows, scores = api_v1._first_per_segment(c, np.array([1, 0, 2, 3]), np.array([0.9, 0.8, 0.7, 0.6]))
    assert rows.tolist() == [1, 2, 3] and scores.tolist() == [0.9, 0.7, 0.6]
