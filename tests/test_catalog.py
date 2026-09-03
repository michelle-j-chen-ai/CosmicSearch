"""The tag catalog: versions, per-project thresholds, pins, marks, exports, backfill."""

from __future__ import annotations

import pytest
import sqlalchemy as sa

import catalog


@pytest.fixture
def cat():
    c = catalog.Catalog(sa.create_engine("sqlite://"), schema=None)
    c.init()
    return c


VEC = [0.1] * 8
TH = {"value": 0.3184, "mode": "suggested", "selected": 41837, "corpus_version": 661}


def _make(cat, tag="motorcycle_filtering", project="neuron", **kw):
    return cat.create(tag=tag, project=project, source={"type": "text", "text": "motorcycle"},
                      vector=VEC, model="black_dwarf", threshold=TH, created_by="m@x", **kw)


def test_create_is_version_one_with_one_project_threshold(cat):
    rec = _make(cat, description="lanes")
    assert rec["version"] == 1 and rec["pinned_version"] is None
    assert rec["thresholds"] == {"neuron": {"value": 0.3184, "mode": "suggested", "selected": 41837,
                                             "corpus_version": 661, "set_at": rec["thresholds"]["neuron"]["set_at"]}}
    assert rec["vector"] == VEC and rec["description"] == "lanes"


def test_posting_another_project_adds_its_threshold_without_a_new_version(cat):
    _make(cat)
    rec = _make(cat, project="frontier")
    assert rec["version"] == 1
    assert set(rec["thresholds"]) == {"neuron", "frontier"}


def test_posting_the_same_project_again_is_a_conflict(cat):
    _make(cat)
    with pytest.raises(catalog.TagExists):
        _make(cat)


def test_refine_makes_a_version_and_marks_other_projects_stale(cat):
    _make(cat)
    _make(cat, project="frontier")
    rec = cat.new_version(tag="motorcycle_filtering", project="neuron",
                          source={"type": "marks", "parent_version": 1, "project": "neuron",
                                  "positive": ["a#t1"], "negative": []},
                          vector=[0.2] * 8, threshold={"value": 0.4467, "mode": "fitted", "f1": 0.81},
                          refine={"positive_count": 1, "negative_count": 0}, created_by="m@x")
    assert rec["version"] == 2
    assert rec["thresholds"]["neuron"]["mode"] == "fitted" and "stale" not in rec["thresholds"]["neuron"]
    assert rec["thresholds"]["frontier"]["stale"] is True
    assert rec["refine"]["positive_count"] == 1


def test_set_threshold_changes_one_project_without_a_new_version(cat):
    _make(cat)
    rec = cat.set_threshold("motorcycle_filtering", "frontier", {"value": 0.41, "mode": "explicit"})
    assert rec["version"] == 1 and rec["thresholds"]["frontier"]["value"] == 0.41


def test_version_resolution_prefers_explicit_then_pinned_then_latest(cat):
    _make(cat)
    cat.new_version(tag="motorcycle_filtering", project="neuron", source={"type": "text", "text": "x"},
                    vector=VEC, threshold=TH)
    assert cat.resolve_version("motorcycle_filtering", None) == 2
    cat.update("motorcycle_filtering", pinned_version=1)
    assert cat.resolve_version("motorcycle_filtering", None) == 1
    assert cat.resolve_version("motorcycle_filtering", 2) == 2
    with pytest.raises(catalog.UnknownTag):
        cat.resolve_version("motorcycle_filtering", 9)


def test_pin_past_latest_is_rejected(cat):
    _make(cat)
    with pytest.raises(catalog.UnknownTag):
        cat.update("motorcycle_filtering", pinned_version=3)


def test_delete_is_soft_and_refuses_a_pinned_tag(cat):
    _make(cat)
    cat.update("motorcycle_filtering", pinned_version=1)
    with pytest.raises(catalog.TagPinned):
        cat.delete("motorcycle_filtering")
    cat.update("motorcycle_filtering", pinned_version=None)
    cat.delete("motorcycle_filtering")
    with pytest.raises(catalog.UnknownTag):
        cat.version("motorcycle_filtering")
    assert cat.list()["total"] == 0
    # re-creating a deleted tag starts fresh at version 1
    rec = _make(cat)
    assert rec["version"] == 1


def test_list_filters_by_project_and_text(cat):
    _make(cat, description="Motorcycle filtering between lanes")
    _make(cat, tag="unprotected_left", project="frontier")
    assert [t["tag"] for t in cat.list(project="frontier")["tags"]] == ["unprotected_left"]
    assert [t["tag"] for t in cat.list(q="LANES")["tags"]] == ["motorcycle_filtering"]
    page = cat.list(page=1, page_size=1)
    assert page["total"] == 2 and len(page["tags"]) == 1
    assert "vector" not in page["tags"][0]


def test_marks_are_per_version_and_upsert_by_chunk(cat):
    _make(cat)
    n = cat.add_marks("motorcycle_filtering", 1, "neuron",
                      [{"chunk_id": "a#t1", "mark": "up"}, {"chunk_id": "a#t1", "mark": "down"}], "m@x")
    assert n == 2
    assert cat.marks("motorcycle_filtering", 1) == [{"chunk_id": "a#t1", "mark": "down", "project": "neuron",
                                                     "user_email": "m@x"}]
    with pytest.raises(ValueError):
        cat.add_marks("motorcycle_filtering", 1, "neuron", [{"chunk_id": "b", "mark": "meh"}])


def test_export_claim_is_idempotent_on_its_parameters(cat):
    _make(cat)
    params = {"interval": True, "from_date": "2026-01-01", "vehicle": None, "segment_mode": True}
    first, claimed = cat.export_claim(tag="motorcycle_filtering", version=1, project="neuron",
                                      output="parquet", params=params, created_by="m@x")
    assert claimed and first["status"] == "running"
    again, claimed2 = cat.export_claim(tag="motorcycle_filtering", version=1, project="neuron",
                                       output="parquet", params={**params, "vehicle": ""})
    assert not claimed2 and again["export_id"] == first["export_id"]
    cat.export_finish(first["export_id"], uri="s3://b/x.parquet", num_rows=41837, corpus_version=661,
                      filters_applied={"from_date": "2026-01-01"})
    done = cat.export_get(first["export_id"])
    assert done["status"] == "ready" and done["uri"] == "s3://b/x.parquet" and done["num_rows"] == 41837
    assert done["filters_applied"] == {"from_date": "2026-01-01"}
    assert cat.record("motorcycle_filtering")["exports"][0]["export_id"] == first["export_id"]


def test_failed_export_can_be_forgotten_and_rerun(cat):
    _make(cat)
    e, _ = cat.export_claim(tag="motorcycle_filtering", version=1, project="neuron", output="csv", params={})
    cat.export_fail(e["export_id"], "boom")
    assert cat.export_get(e["export_id"])["status"] == "error"
    cat.export_forget(e["export_id"])
    _, claimed = cat.export_claim(tag="motorcycle_filtering", version=1, project="neuron", output="csv", params={})
    assert claimed


def test_record_has_every_version_and_export(cat):
    _make(cat)
    cat.new_version(tag="motorcycle_filtering", project="neuron", source={"type": "text", "text": "x"},
                    vector=VEC, threshold=TH)
    rec = cat.record("motorcycle_filtering")
    assert [v["version"] for v in rec["versions"]] == [1, 2]
    assert rec["exports"] == [] and "vector" not in rec


def test_backfill_from_export_log_rows(cat):
    rows = [
        {"tag": "a", "query": "q a", "threshold": 0.42, "user_email": "u", "search_vector": "[0.1, 0.2]",
         "parquet_uri": "s3://b/a.parquet", "num_results": 10},
        {"tag": "b", "query": "q b", "threshold": 0.0, "user_email": "u", "search_vector": [0.3]},
        {"tag": "", "query": "untagged", "search_vector": [0.1]},
        {"tag": "c", "query": "no vector", "search_vector": None},
    ]
    assert cat.backfill_export_log(rows, project="neuron", model="black_dwarf") == 2
    a = cat.version("a")
    assert a["thresholds"]["neuron"] == {**a["thresholds"]["neuron"], "value": 0.42, "mode": "explicit"}
    assert a["source"] == {"type": "text", "text": "q a"} and a["vector"] == [0.1, 0.2]
    assert cat.record("a")["exports"][0]["uri"] == "s3://b/a.parquet"
    assert cat.version("b")["thresholds"] == {}
    assert cat.backfill_export_log(rows, project="neuron", model="black_dwarf") == 0  # idempotent


def _rows(cat, tag):
    T = cat.t["tags"]
    with cat.engine.begin() as conn:
        return conn.execute(sa.select(T).where(T.c.tag == tag).order_by(T.c.version)).mappings().all()


def test_tag_level_fields_agree_across_every_version(cat):
    """description/pin/delete live on every version row; one write sets them all,
    so no version can disagree with another about what the tag is."""
    _make(cat, description="lanes")
    cat.new_version(tag="motorcycle_filtering", project="neuron", source={"type": "text", "text": "x"},
                    vector=VEC, threshold=TH)
    cat.update("motorcycle_filtering", description="filtering between lanes", pinned_version=1)
    rows = _rows(cat, "motorcycle_filtering")
    assert len(rows) == 2
    assert {r["description"] for r in rows} == {"filtering between lanes"}
    assert {r["pinned_version"] for r in rows} == {1}
    assert {r["model"] for r in rows} == {"black_dwarf"}
    cat.update("motorcycle_filtering", pinned_version=None)
    cat.delete("motorcycle_filtering")
    assert all(r["deleted_at"] is not None for r in _rows(cat, "motorcycle_filtering"))
    with pytest.raises(catalog.UnknownTag):
        cat.version("motorcycle_filtering", 1)


def test_projects_column_tracks_the_thresholds_map(cat):
    """`projects` mirrors the thresholds keys, because that is what the list
    query filters on -- a drift between the two would hide a tag from its fleet."""
    _make(cat)
    assert _rows(cat, "motorcycle_filtering")[0]["projects"] == ",neuron,"
    cat.set_threshold("motorcycle_filtering", "frontier", {"value": 0.4, "mode": "explicit"})
    assert _rows(cat, "motorcycle_filtering")[0]["projects"] == ",frontier,neuron,"
    assert [t["tag"] for t in cat.list(project="frontier")["tags"]] == ["motorcycle_filtering"]
    cat.new_version(tag="motorcycle_filtering", project="neuron", source={"type": "text", "text": "x"},
                    vector=VEC, threshold=TH)
    latest = _rows(cat, "motorcycle_filtering")[1]
    assert latest["projects"] == ",frontier,neuron,"
    assert latest["thresholds"]["frontier"]["stale"] is True
    assert latest["thresholds"]["neuron"]["stale"] is False
