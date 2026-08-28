"""The /api/retrieve -> export shim must not drop request options.

`_retrieve_export` rebuilds the caller's RetrieveRequest as a FullExportRequest
before handing it to the writer. When that translation was written out field by
field, `create_segment_set` was left off it -- so exports asked to register a
Data Explorer segment set silently did not, with nothing in the logs to say so.
"""

from __future__ import annotations

import web_server as ws


def test_every_export_field_has_a_retrieve_source():
    """A field on the writer's model that /api/retrieve cannot set is a request
    option that is always at its default, no matter what the caller sends."""
    orphans = set(ws.FullExportRequest.model_fields) - set(ws.RetrieveRequest.model_fields)
    assert orphans == set(), f"unreachable export options: {sorted(orphans)}"


def test_segment_set_options_survive_the_shim():
    req = ws.RetrieveRequest(
        query="tunnel entry",
        output="csv",
        threshold=0.4,
        create_segment_set=True,
        segment_set_uuid="abc-123",
    )
    shim = ws.FullExportRequest(
        **{
            name: getattr(req, name)
            for name in ws.FullExportRequest.model_fields
            if hasattr(req, name)
        }
    )
    assert shim.create_segment_set is True
    assert shim.segment_set_uuid == "abc-123"
