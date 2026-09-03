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


def test_unconfigured_dora_host_names_the_config_not_a_missing_module(monkeypatch):
    """The slim image omits trino/simian on purpose. With no hostname set,
    _get_stub used to fall through to the SDK's own get_stub and fail with
    ModuleNotFoundError, pointing at a dependency instead of the empty config."""
    import dora_client

    monkeypatch.delenv("DATA_EXPLORER_SDK_GRPC_HOSTNAME", raising=False)
    monkeypatch.delenv("URSA_SDK_GRPC_HOSTNAME", raising=False)
    # Enter at _build_secure_stub, where the guard is: _get_stub imports the
    # vendored proto stubs first, and adp/ is a gitignored artifact that a clean
    # checkout does not have -- so going through it would test the environment
    # rather than the guard.
    with pytest.raises(dora_client.DoraUnavailable) as exc:
        dora_client._build_secure_stub(object, "")
    assert "URSA_SDK_GRPC_HOSTNAME" in str(exc.value)


def test_absent_credentials_raise_instead_of_reaching_the_metadata_server(monkeypatch):
    """object_store treats an incomplete option set as "find credentials
    yourself" and walks to the GCE metadata server, which answers 403 Missing
    required header: Metadata-Flavor -- naming neither this app's config nor the
    bucket. Fail on the config instead."""
    import oci_s3

    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
    with pytest.raises(oci_s3.CredentialsMissing) as exc:
        oci_s3.lance_storage_options()
    assert "AWS_ACCESS_KEY_ID" in str(exc.value)

    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "id")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")
    opts = oci_s3.lance_storage_options()
    assert opts["aws_access_key_id"] == "id" and opts["aws_secret_access_key"] == "secret"


def _fake_dataset(*, model="black_dwarf", model_id="black-dwarf", storage="2.0",
                  drop=(), pca_dim=4, artifact=""):
    """A stand-in with just the surface `full_corpus.validate` reads."""
    import base64
    import io

    import numpy as np
    import pyarrow as pa

    import full_corpus

    def enc(a):
        buf = io.BytesIO()
        np.save(buf, a)
        return base64.b64encode(buf.getvalue())

    meta = {
        full_corpus.META_KEY_PCA_COMPONENTS: enc(np.zeros((pca_dim, 8), dtype=np.float32)),
        full_corpus.META_KEY_QUANT_SCALES: enc(np.ones(pca_dim, dtype=np.float32)),
    }
    if model_id is not None:
        meta[full_corpus.META_KEY_MODEL_ID] = model_id.encode()
    if artifact:
        meta[full_corpus.META_KEY_MODEL_ARTIFACT_URI] = artifact.encode()

    fields = [
        pa.field(full_corpus.embedding_column(model), pa.list_(pa.int8(), pca_dim)),
        pa.field(full_corpus.vector_fp_column(model), pa.list_(pa.float32(), pca_dim)),
        pa.field(full_corpus.vector_full_column(model), pa.list_(pa.float32(), 8), metadata=meta),
        *(pa.field(c, pa.string()) for c in full_corpus.REQUIRED_COLUMNS),
    ]
    fields = [f for f in fields if f.name not in drop]

    class _DS:
        schema = pa.schema(fields)
        data_storage_version = storage

    return _DS()


def test_validate_accepts_the_production_tables_as_they_are_written():
    """Both fleets' tables report model_id "black-dwarf" with a hyphen, against a
    "black_dwarf" column suffix, at data_storage_version 2.0. An earlier version
    of this check required an exact model_id match and a 2.1 floor, so it
    rejected both of the tables it was meant to protect."""
    import full_corpus

    assert full_corpus.validate(_fake_dataset()) == "black-dwarf"
    assert full_corpus.validate(_fake_dataset(model_id="black_dwarf")) == "black_dwarf"


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({"model_id": None}, "nls.model_id"),
        ({"model_id": "some_other_encoder"}, "some_other_encoder"),
        ({"drop": ("segment_id",)}, "segment_id"),
        ({"drop": ("vector_fp_black_dwarf",)}, "vector_fp_black_dwarf"),
    ],
)
def test_validate_rejects_a_table_the_app_cannot_serve(kwargs, expected):
    """A wrong-model corpus returns a confident ranked list with nothing in it to
    signal every score is meaningless, so the identity check has to be fatal."""
    import full_corpus

    with pytest.raises(full_corpus.CorpusContractError) as exc:
        full_corpus.validate(_fake_dataset(**kwargs))
    assert expected in str(exc.value)


def test_a_checkpoint_mismatch_warns_rather_than_rejecting(monkeypatch, caplog):
    """Every checkpoint of this family reports the same model_id, so only the
    artifact URI separates them -- but each fleet keeps its own copy under its
    own bucket, so paths differ where the checkpoint does not. Compare the
    checkpoint name, and only log: a false rejection here takes a fleet down."""
    import logging

    import full_corpus

    monkeypatch.setenv("NLS_MODEL_ARTIFACT_URI",
                       "s3://neuron-prod-data-intelligence-exploratory/x/models/maxsim-mainfull-ckpt14500/")
    same = _fake_dataset(artifact="s3://frontier-perception-datasets/model_assets/maxsim-mainfull-ckpt14500/")
    with caplog.at_level(logging.WARNING):
        assert full_corpus.validate(same) == "black-dwarf"
    assert "checkpoint" not in caplog.text

    other = _fake_dataset(artifact="s3://frontier-perception-datasets/model_assets/maxsim-mainfull-ckpt9000/")
    with caplog.at_level(logging.WARNING):
        assert full_corpus.validate(other) == "black-dwarf"
    assert "ckpt9000" in caplog.text
