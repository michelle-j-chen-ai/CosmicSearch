"""The merged snapshot carries its own architecture; load it from there.

A snapshot written by the eval pipeline ships modeling_embed1.py and the rest
next to model.safetensors, but its config.json inherits the upstream `auto_map`,
whose entries read "nvidia/Cosmos-Embed1-448p--modeling_embed1.CosmosEmbed1".
That form tells transformers to fetch the class from that repo, so every cold
start downloaded and exec'd unpinned Python from huggingface.co while the
identical files sat unread in the snapshot.
"""

from __future__ import annotations

import json
import pathlib

import search_engine


def _snapshot(tmp_path: pathlib.Path, auto_map: dict, modules=("modeling_embed1",)) -> pathlib.Path:
    for name in modules:
        (tmp_path / f"{name}.py").write_text("")
    (tmp_path / "config.json").write_text(
        json.dumps({"auto_map": auto_map, "embed_dim": 768})
    )
    return tmp_path


def _auto_map(path: pathlib.Path) -> dict:
    return json.loads((path / "config.json").read_text())["auto_map"]


def test_a_hub_qualified_reference_is_pointed_at_the_local_file():
    import tempfile

    d = _snapshot(
        pathlib.Path(tempfile.mkdtemp()),
        {"AutoModel": "nvidia/Cosmos-Embed1-448p--modeling_embed1.CosmosEmbed1"},
    )
    search_engine._localize_auto_map(d)
    assert _auto_map(d) == {"AutoModel": "modeling_embed1.CosmosEmbed1"}


def test_a_module_the_snapshot_does_not_have_keeps_its_remote_reference():
    """Localizing a module that is not present would stop the model loading at
    all -- worse than the fetch this avoids."""
    import tempfile

    d = _snapshot(
        pathlib.Path(tempfile.mkdtemp()),
        {"AutoModel": "nvidia/Cosmos-Embed1-448p--absent_module.Cls"},
        modules=(),
    )
    search_engine._localize_auto_map(d)
    assert _auto_map(d) == {"AutoModel": "nvidia/Cosmos-Embed1-448p--absent_module.Cls"}


def test_the_rest_of_the_config_survives():
    """The config carries the architecture's own hyperparameters; rewriting one
    key must not drop the others."""
    import tempfile

    d = _snapshot(
        pathlib.Path(tempfile.mkdtemp()),
        {"AutoModel": "nvidia/Cosmos-Embed1-448p--modeling_embed1.CosmosEmbed1"},
    )
    search_engine._localize_auto_map(d)
    assert json.loads((d / "config.json").read_text())["embed_dim"] == 768


def test_rewriting_twice_changes_nothing():
    """The snapshot cache is on a shared volume and outlives the process, so
    this runs again on every start against an already-rewritten config."""
    import tempfile

    d = _snapshot(
        pathlib.Path(tempfile.mkdtemp()),
        {"AutoModel": "nvidia/Cosmos-Embed1-448p--modeling_embed1.CosmosEmbed1"},
    )
    search_engine._localize_auto_map(d)
    once = _auto_map(d)
    search_engine._localize_auto_map(d)
    assert _auto_map(d) == once


def test_an_already_local_auto_map_is_left_alone():
    import tempfile

    d = _snapshot(
        pathlib.Path(tempfile.mkdtemp()), {"AutoModel": "modeling_embed1.CosmosEmbed1"}
    )
    search_engine._localize_auto_map(d)
    assert _auto_map(d) == {"AutoModel": "modeling_embed1.CosmosEmbed1"}


def test_a_snapshot_without_an_auto_map_is_not_touched():
    import tempfile

    d = pathlib.Path(tempfile.mkdtemp())
    (d / "config.json").write_text(json.dumps({"embed_dim": 768}))
    search_engine._localize_auto_map(d)
    assert json.loads((d / "config.json").read_text()) == {"embed_dim": 768}
