"""The project registry: which table a request is served from is decided here."""

from __future__ import annotations

import pytest

import deployment
import full_corpus


def test_default_project_is_the_neuron_table():
    p = deployment.get(deployment.FALLBACK)
    assert p.name == "neuron"
    assert p.corpus_table_uri == full_corpus.DEFAULT_CORPUS_TABLE_URI


def test_frontier_has_its_own_table_prefix_and_cluster():
    p = deployment.get("frontier")
    assert p.corpus_table_uri.startswith("s3://frontier-perception-datasets/")
    assert p.mp4_prefix.startswith("s3://frontier-perception-datasets/")
    assert p.dora_hostname == "grpc.frontier.prod.applied.dev"


def test_names_are_case_insensitive_and_none_means_default():
    assert deployment.get("FRONTIER").name == "frontier"
    assert deployment.get(None).name == deployment.default()


def test_unknown_project_is_an_error_not_the_default():
    with pytest.raises(KeyError):
        deployment.get("cars")


def test_per_project_overrides_come_from_env(monkeypatch):
    monkeypatch.setenv("NLS_FRONTIER_CORPUS_TABLE_URI", "s3://bucket/other.lance")
    monkeypatch.setenv("NLS_FRONTIER_MP4_PREFIX", "s3://bucket/clips/")
    monkeypatch.setenv("NLS_FRONTIER_DORA_HOSTNAME", "grpc.example.test")
    p = deployment.get("frontier")
    assert (p.corpus_table_uri, p.mp4_prefix, p.dora_hostname) == (
        "s3://bucket/other.lance", "s3://bucket/clips/", "grpc.example.test",
    )
    assert deployment.get("neuron").mp4_prefix != "s3://bucket/clips/"


def test_neuron_inherits_the_shared_defaults(monkeypatch):
    monkeypatch.setenv("NLS_MP4_PREFIX", "s3://shared/clips/")
    monkeypatch.setenv("URSA_SDK_GRPC_HOSTNAME", "grpc.shared.test")
    p = deployment.get("neuron")
    assert p.mp4_prefix == "s3://shared/clips/"
    assert p.dora_hostname == "grpc.shared.test"


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("neuron", ["neuron"]),
        ("neuron,frontier", ["neuron", "frontier"]),
        (" Frontier , neuron ", ["frontier", "neuron"]),
        ("frontier,frontier", ["frontier"]),
        ("cars,trucks", ["neuron"]),
        ("", ["neuron"]),
    ],
)
def test_enabled_parses_nls_projects(monkeypatch, raw, expected):
    monkeypatch.setenv("NLS_PROJECTS", raw)
    assert deployment.enabled() == expected


def test_enabled_defaults_to_the_fallback_project(monkeypatch):
    monkeypatch.delenv("NLS_PROJECTS", raising=False)
    assert deployment.enabled() == [deployment.FALLBACK]


def test_a_single_project_service_defaults_to_the_corpus_it_holds(monkeypatch):
    """neuron and frontier deploy as two services off one image. A frontier-only
    service must answer an unqualified request from frontier -- defaulting to a
    project it never loaded would 503 every call that omitted one."""
    monkeypatch.setenv("NLS_PROJECTS", "frontier")
    assert deployment.default() == "frontier"
    assert deployment.get(None).name == "frontier"
    assert deployment.get(None).corpus_table_uri.startswith("s3://frontier-perception-datasets/")


def test_the_default_is_the_first_listed_project(monkeypatch):
    monkeypatch.setenv("NLS_PROJECTS", "frontier,neuron")
    assert deployment.default() == "frontier"


def test_media_uri_is_the_project_prefix_plus_the_shared_layout():
    uri = full_corpus.media_uri("s3://b/clips/", "2026-08-20", "run-1", 1766502218)
    assert uri == "s3://b/clips/dt=2026-08-20/run-1_t1766502218.mp4"


def test_refresh_all_reports_every_enabled_project(monkeypatch):
    import web_server as ws
    monkeypatch.setenv("NLS_PROJECTS", "neuron,frontier")
    monkeypatch.setattr(ws, "_corpus_refresh_tick", lambda p: f"ok-{p}")
    assert ws._corpus_refresh_all() == "neuron: ok-neuron; frontier: ok-frontier"


def test_unknown_project_is_a_404_on_the_api():
    import web_server as ws
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as e:
        ws._require_full_corpus("cars")
    assert e.value.status_code == 404
