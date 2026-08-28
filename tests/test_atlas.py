"""The atlas, as served by the search app.

Two things matter here. Nulls are ordinary in this corpus — 19.7M of 54.2M rows
carry no embedding, and metadata columns are nullable — so a colouring built by
sorting an object array, or a presign handed a null URI, takes down a response
that should have degraded to one blank card. And the atlas must be optional: an
app with no atlas configured is not a broken app.
"""

from __future__ import annotations

import numpy as np
import pytest

import atlas_store
import web_server as ws


class _Store:
    """Stands in for AtlasStore: only `_build_coloring` is under test."""

    _build_coloring = atlas_store.AtlasStore._build_coloring

    def __init__(self, values):
        self.dt = values


def test_label_names_the_missing_category():
    assert atlas_store._label(None) == "(none)"
    assert atlas_store._label("2026-08-27") == "2026-08-27"


def test_coloring_survives_null_values():
    """np.unique sorts, and sorting None against str raises TypeError."""
    indices, legend = _Store(["a", None, "b", "a", None])._build_coloring("dt")
    assert len(indices) == 5
    assert "(none)" in legend


def test_coloring_counts_by_frequency_not_order():
    values = ["rare"] + ["common"] * 10 + ["mid"] * 5
    _indices, legend = _Store(values)._build_coloring("dt")
    assert {"common", "mid"} <= set(legend)


def test_coloring_indices_address_the_legend():
    values = ["a", "b", None, "a"]
    indices, legend = _Store(values)._build_coloring("dt")
    assert indices.max() < len(legend)


# ------------------------------ presigning ---------------------------------

def test_video_url_is_empty_for_a_null_uri(monkeypatch):
    """One null row must not fail a whole neighbours or lasso response."""
    def _boom(*a, **k):
        raise AssertionError("presigned a null uri")
    monkeypatch.setattr(ws.oci_s3, "presign_get", _boom)
    assert ws._atlas_video_url(None) == ""
    assert ws._atlas_video_url("") == ""


def test_video_url_is_empty_when_presigning_fails(monkeypatch):
    monkeypatch.setattr(ws.oci_s3, "presign_get",
                        lambda *a, **k: (_ for _ in ()).throw(ValueError("bad uri")))
    monkeypatch.setitem(ws._state, "s3", object())
    monkeypatch.setitem(ws._state, "cfg", type("C", (), {"presign_ttl_s": 60})())
    assert ws._atlas_video_url("s3://bucket/clip.mp4") == ""


# ------------------------------ optionality --------------------------------

def test_unconfigured_atlas_is_absent_not_broken(monkeypatch):
    monkeypatch.setattr(ws, "_ATLAS_URI", "")
    with pytest.raises(ws.HTTPException) as exc:
        ws._atlas()
    assert exc.value.status_code == 404


def test_a_loading_atlas_asks_the_caller_to_retry(monkeypatch):
    monkeypatch.setattr(ws, "_ATLAS_URI", "s3://bucket/atlas.parquet")
    monkeypatch.setattr(ws, "_ATLAS", {"store": None, "status": "loading", "error": ""})
    with pytest.raises(ws.HTTPException) as exc:
        ws._atlas()
    assert exc.value.status_code == 503


def test_a_failed_atlas_reports_why(monkeypatch):
    monkeypatch.setattr(ws, "_ATLAS_URI", "s3://bucket/atlas.parquet")
    monkeypatch.setattr(ws, "_ATLAS",
                        {"store": None, "status": "error", "error": "OSError: nope"})
    with pytest.raises(ws.HTTPException) as exc:
        ws._atlas()
    assert exc.value.status_code == 503
    assert "OSError: nope" in exc.value.detail


def test_atlas_failure_does_not_touch_retrieval():
    """The atlas is lazily loaded and holds its own state; a corpus search must
    not consult it at all."""
    import inspect
    for fn in (ws._retrieve_selection, ws._retrieve_hits):
        assert "_atlas" not in inspect.getsource(fn)
