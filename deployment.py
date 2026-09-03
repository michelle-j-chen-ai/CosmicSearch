"""Which corpora this service serves, and what differs between them.

One image serves every project. A project is a fleet's corpus and the few values
that travel with it: the Lance table, the prefix its clips are stored under, and
the Data Explorer cluster its segment sets live on. Everything else -- the
encoder, the S3 endpoint and credentials, the export prefix -- is shared, so a
request differs between projects only in which table it is scored against.

`NLS_PROJECTS` is the comma-separated list of projects this instance loads. A
service that lists one project serves only that project, and it is what a
request that names none gets: neuron and frontier are deployed as two services
off this image, each with its own corpus, its own Data Explorer cluster and its
own Postgres schema. A project's values can be overridden per deployment with
NLS_<PROJECT>_CORPUS_TABLE_URI, NLS_<PROJECT>_MP4_PREFIX and
NLS_<PROJECT>_DORA_HOSTNAME.
"""

from __future__ import annotations

import dataclasses
import os

import config

# The project a build falls back to when NLS_PROJECTS is unset. Not the default
# for a running instance -- that is `default()`, the first project it loads.
FALLBACK = "neuron"


@dataclasses.dataclass(frozen=True)
class Project:
    name: str
    label: str
    corpus_table_uri: str
    mp4_prefix: str
    dora_hostname: str


# Built-in values. None means "the shared default": the neuron clip prefix is
# NLS_MP4_PREFIX and its Data Explorer host is URSA_SDK_GRPC_HOSTNAME, as they
# were before there was a second project.
_BUILTIN: dict[str, dict] = {
    "neuron": {
        "label": "NEURON",
        "corpus_table_uri": (
            "s3://neuron-prod-data-intelligence-exploratory/vlm/corpus/video_embeddings.lance"
        ),
        "mp4_prefix": None,
        "dora_hostname": None,
    },
    "frontier": {
        "label": "FRONTIER",
        "corpus_table_uri": (
            "s3://frontier-perception-datasets/vlm/corpus/video_embeddings.lance"
        ),
        "mp4_prefix": "s3://frontier-perception-datasets/sibogeng/vlm/chunks_mp4_full/",
        "dora_hostname": "grpc.frontier.prod.applied.dev",
    },
}


def _override(project: str, key: str) -> str | None:
    return os.environ.get(f"NLS_{project.upper()}_{key}", "").strip() or None


def names() -> list[str]:
    """Every project this build knows, enabled or not."""
    return list(_BUILTIN)


def is_known(name: str | None) -> bool:
    return (name or "").strip().lower() in _BUILTIN


def get(name: str | None) -> Project:
    """The project's resolved values. Raises KeyError for a name this build does
    not know, so a typo cannot fall through to the default corpus."""
    key = (name or default()).strip().lower()
    if key not in _BUILTIN:
        raise KeyError(key)
    built = _BUILTIN[key]
    return Project(
        name=key,
        label=built["label"],
        corpus_table_uri=_override(key, "CORPUS_TABLE_URI") or built["corpus_table_uri"],
        mp4_prefix=_override(key, "MP4_PREFIX") or built["mp4_prefix"] or config.mp4_prefix(),
        dora_hostname=(
            _override(key, "DORA_HOSTNAME")
            or built["dora_hostname"]
            or os.environ.get("URSA_SDK_GRPC_HOSTNAME", "").strip()
        ),
    )


def enabled() -> list[str]:
    """Projects this instance loads, in NLS_PROJECTS order. Unknown names are
    dropped; an empty result falls back to the default so the service always
    serves something."""
    raw = os.environ.get("NLS_PROJECTS", FALLBACK)
    out: list[str] = []
    for part in raw.split(","):
        key = part.strip().lower()
        if key in _BUILTIN and key not in out:
            out.append(key)
    return out or [FALLBACK]


def default() -> str:
    """The project a request that names none is served from: this instance's
    first enabled project. A single-project service therefore answers unqualified
    requests from the corpus it actually holds, rather than 503-ing on one it
    was never configured to load."""
    return enabled()[0]
