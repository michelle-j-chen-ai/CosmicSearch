"""Thin client for listing Data Explorer (DORA) segment sets and their members.

Wraps the same gRPC pattern used by
``onroad/tools/offboard/common/auto_labeling/nightly_curation_regression/curation_step.py``:
``adp.public.strada.dora.get_stub`` + ``ListDataSets`` / ``DescribeDataSet``.

The ``adp.*`` SDK (``data-explorer-py`` + ``ursa-py``, internal index) is imported
lazily so the app still starts when it is absent; callers get a clear error only
if they actually use the downsample feature. Auth + endpoint come from the
``URSA_SDK_GRPC_*`` env vars (synced to ``DATA_EXPLORER_SDK_GRPC_*``).
"""

from __future__ import annotations

import dataclasses
import logging
import os
import threading
import time

import machine_auth

LOGGER = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class SegmentSet:
    name: str
    version: int
    dataset_uuid: str
    num_segments: int

    def label(self) -> str:
        return f"{self.name}  (v{self.version}, {self.num_segments} segments)"


class DoraUnavailable(RuntimeError):
    """Raised when the DORA SDK is not installed or the service is unreachable."""


def _sync_dora_env() -> None:
    """Mirror URSA_SDK_GRPC_* into DATA_EXPLORER_SDK_GRPC_* (curation_step.py)."""
    for ursa_var, dora_var in [
        ("URSA_SDK_GRPC_HOSTNAME", "DATA_EXPLORER_SDK_GRPC_HOSTNAME"),
        ("URSA_SDK_GRPC_AUTH_TOKEN", "DATA_EXPLORER_SDK_GRPC_AUTH_TOKEN"),
    ]:
        if ursa_var in os.environ and dora_var not in os.environ:
            os.environ[dora_var] = os.environ[ursa_var]
    os.environ.setdefault("CLOUD_ENVIRONMENT", "OCI")


_STUBS: dict[str, object] = {}  # one per hostname: get_stub() opens a Connector
# + gRPC channel + threads, so each is created ONCE. Creating one per call leaks
# threads/channels and starves the rest of the process.
_STUBS_LOCK = threading.Lock()

# gRPC's default 4 MiB receive cap throttles segment-id fetches: at 4 MiB a page
# tops out near 80k external_ids, so a multi-million-segment set needs hundreds of
# sequential round trips (~5k ids/s end to end). Raising the cap lets one page
# carry 100k+ ids -- measured ~80k ids/s, a ~15x speedup. 256 MiB is a generous
# ceiling (a 100k-id page is ~5 MiB).
_MAX_RECV_BYTES = 256 * 1024 * 1024

# gRPC keepalive options (mirrors adp.public.strada.dora.GRPC_CHANNEL_OPTIONS).
# Defined locally so the deployed cloud path never imports the heavy
# ``adp.public.strada.dora`` helper -- that module pulls a deep dependency chain
# (job_metadata -> simian -> ...) that is not vendored in the slim image. We only
# need the proto stubs + raw grpc here; see _build_secure_stub.
_GRPC_CHANNEL_OPTIONS = [
    ("grpc.keepalive_time_ms", 60000),
    ("grpc.keepalive_timeout_ms", 10000),
    ("grpc.keepalive_permit_without_calls", True),
    ("grpc.http2.max_pings_without_data", 1),
]


def _default_hostname() -> str:
    _sync_dora_env()
    return os.getenv("DATA_EXPLORER_SDK_GRPC_HOSTNAME", "")


def _get_stub(hostname: str | None = None):
    """The stub for `hostname` (the env default when None), built once per host."""
    host = hostname or _default_hostname()
    with _STUBS_LOCK:
        stub = _STUBS.get(host)
    if stub is not None:
        return stub
    try:
        from adp.services.dora_event_management.proto import (
            dora_event_management_service_pb2_grpc,
        )
    except ImportError as exc:  # SDK not bundled in this environment
        raise DoraUnavailable(
            "Data Explorer SDK not installed (need data-explorer-py + ursa-py). "
            "Segment-set downsampling is unavailable in this deployment."
        ) from exc
    stub = _build_secure_stub(
        dora_event_management_service_pb2_grpc.DoraEventManagementStub, host
    )
    with _STUBS_LOCK:
        _STUBS[host] = stub
    return stub


def _wait_ready(channel, hostname) -> None:
    import grpc

    try:
        grpc.channel_ready_future(channel).result(timeout=30)
    except grpc.FutureTimeoutError as exc:
        channel.close()
        raise DoraUnavailable(
            f"DORA gRPC channel to {hostname} not ready: {exc}"
        ) from exc


def _build_secure_stub(stub_type, hostname: str = ""):
    """A DORA stub whose channel allows large receive messages and is authed.

    Adds ``grpc.max_receive_message_length`` (so segment-id pages can be ~100k
    ids each) and attaches a bearer token. Auth precedence:

    - **Machine credentials** (``MACHINE_CLIENT_ID`` / ``MACHINE_CLIENT_SECRET``):
      a fresh, auto-refreshing token is injected **per RPC** via gRPC call
      credentials (``composite_channel_credentials`` -- the pattern required for
      grpc.neuron.oci.applied.dev), so a long-lived channel never serves an
      expired token. This is what deployed services use.
    - **Static token** (``DATA_EXPLORER_SDK_GRPC_AUTH_TOKEN``, local dev): a
      header interceptor with the personal token.

    For non-cloud hosts (``localhost`` dev / in-cluster discovery) the SDK's own
    ``get_stub`` is used (lazy import -- only reachable in local dev, where the
    full SDK is installed; the slim image only vendors the proto stubs).
    """
    import grpc

    hostname = hostname or os.getenv("DATA_EXPLORER_SDK_GRPC_HOSTNAME", "")
    if not hostname:
        # An unconfigured deployment must say so. Falling through to the SDK's
        # own get_stub below would import adp.public.strada.dora, which pulls
        # trino + simian -- deliberately absent from the slim image -- so the
        # error would name a missing module instead of the missing hostname.
        raise DoraUnavailable(
            "no Data Explorer hostname configured; set URSA_SDK_GRPC_HOSTNAME "
            "or NLS_<PROJECT>_DORA_HOSTNAME for this deployment"
        )
    if not hostname.startswith("grpc."):
        from adp.public.strada import dora

        return dora.get_stub(stub_type)

    options = list(_GRPC_CHANNEL_OPTIONS) + [
        ("grpc.max_receive_message_length", _MAX_RECV_BYTES),
    ]
    ssl_creds = grpc.ssl_channel_credentials()

    if machine_auth.available():

        def _bearer_plugin(context, callback):
            try:
                token = machine_auth.get_machine_token()
                callback((("authorization", f"Bearer {token}"),), None)
            except Exception as exc:  # noqa: BLE001 -- report to grpc, never raise
                callback((), exc)

        creds = grpc.composite_channel_credentials(
            ssl_creds, grpc.metadata_call_credentials(_bearer_plugin)
        )
        channel = grpc.secure_channel(hostname, creds, options=options)
        _wait_ready(channel, hostname)
        LOGGER.info("DORA channel to %s using machine auth", hostname)
        return stub_type(channel)

    channel = grpc.secure_channel(hostname, ssl_creds, options=options)
    _wait_ready(channel, hostname)
    auth_token = os.getenv("DATA_EXPLORER_SDK_GRPC_AUTH_TOKEN")
    if auth_token:
        from adp.public.strada.dora import grpc_interceptor

        interceptor = grpc_interceptor.create(
            grpc_interceptor.add_header_intercept_call(
                [("authorization", f"Bearer {auth_token}")]
            )
        )
        channel = grpc.intercept_channel(channel, interceptor)
    return stub_type(channel)


def prewarm(hostname: str | None = None) -> bool:
    """Open the gRPC channel + fetch the machine token now (best-effort).

    Called at app startup so the FIRST user segment-set selection doesn't pay the
    ~1s channel + auth setup. Returns True if the stub is ready. Safe to call when
    the SDK is absent -- it just logs and returns False, and the lazy path retries
    on real use."""
    try:
        _get_stub(hostname)
        LOGGER.info("DORA stub pre-warmed (%s)", hostname or "default host")
        return True
    except Exception as exc:  # noqa: BLE001 -- prewarm is best-effort
        LOGGER.warning("DORA prewarm failed (will retry lazily): %s", exc)
        return False


# Cache the (server-filtered) ListDataSets result per filter token. Incremental
# typing keeps the same most-selective token, so each extra character reuses one
# fetch instead of re-paginating DORA from scratch. Dataset versions are
# effectively immutable, so a short TTL is safe.
_LIST_CACHE: dict[tuple[str, str], tuple[float, list]] = {}
_LIST_CACHE_LOCK = threading.Lock()
_LIST_TTL_S = 300


def _list_datasets(dora_filter: str, hostname: str | None = None) -> list[SegmentSet]:
    """Latest version of each dataset matching DORA's fuzzy server filter (TTL-cached)."""
    now = time.time()
    with _LIST_CACHE_LOCK:
        ent = _LIST_CACHE.get((hostname or "", dora_filter))
        if ent and ent[0] > now:
            return ent[1]

    stub = _get_stub(hostname)
    import grpc
    from adp.public.proto import pagination_pb2
    from adp.services.dora_event_management.proto import (
        dora_event_management_service_pb2 as pb,
    )

    out: list[SegmentSet] = []
    cursor = ""
    try:
        while True:
            resp = stub.ListDataSets(
                pb.ListDataSetsRequest(
                    page_request=pagination_pb2.PageRequest(
                        # 1000/page (vs 200) cuts the round trips ~5x; each row is
                        # just a name + counts, so the page stays small.
                        page_size=1000,
                        after_cursor=cursor,
                    ),
                    name_filter=dora_filter,
                    # Only the most recent version of each set (not every version,
                    # not just PROD-labelled ones -- prod-only hid too many sets).
                    request_type=pb.LIST_DATA_SETS_REQUEST_TYPE_ONLY_LATEST,
                )
            )
            for ds in resp.datasets:
                if ds.version == 0 or not ds.dataset_uuid or not ds.name:
                    continue
                out.append(
                    SegmentSet(
                        name=ds.name,
                        version=ds.version,
                        dataset_uuid=ds.dataset_uuid,
                        num_segments=ds.num_segments,
                    )
                )
            if not resp.page_info or not resp.page_info.end_cursor:
                break
            cursor = resp.page_info.end_cursor
    except grpc.RpcError as exc:
        raise DoraUnavailable(f"ListDataSets failed: {exc}") from exc

    with _LIST_CACHE_LOCK:
        _LIST_CACHE[(hostname or "", dora_filter)] = (now + _LIST_TTL_S, out)
    return out


def list_segment_sets(name_filter: str = "", hostname: str | None = None) -> list[SegmentSet]:
    """Latest version of each DORA dataset matching every whitespace token.

    DORA's name_filter is fuzzy and can't handle multi-word queries, so the
    server side is narrowed to the single most-selective (longest) token (and
    that fetch is cached); the full multi-token match + sort is applied
    client-side, so typing more words reuses the cached fetch.
    """
    # Resolve the stub first so a missing SDK is a clean DoraUnavailable (503),
    # even on a cache hit, rather than a raw ImportError.
    _get_stub(hostname)
    tokens = [t for t in name_filter.lower().split() if t]
    dora_filter = max(tokens, key=len) if tokens else name_filter

    out = list(_list_datasets(dora_filter, hostname=hostname))
    if tokens:
        out = [s for s in out if all(t in s.name.lower() for t in tokens)]
    out.sort(key=lambda s: (s.name, -s.version))
    LOGGER.info("Listed %d DORA dataset versions matching %r", len(out), name_filter)
    return out


def fetch_segment_ids(
    dataset_uuid: str, progress=None, hostname: str | None = None
) -> frozenset[str]:
    """The set of segment external_ids belonging to a dataset (paginated).

    ``progress`` (optional) is called with the running id count after each page,
    so a background loader can surface live progress for very large sets.
    """
    stub = _get_stub(hostname)  # clean DoraUnavailable if the SDK isn't installed

    import grpc
    from adp.public.proto import pagination_pb2
    from adp.services.dora_event_management.proto import (
        dora_event_management_service_pb2 as pb,
    )

    ids: set[str] = set()
    cursor = ""
    try:
        while True:
            resp = stub.DescribeDataSet(
                pb.DescribeDataSetRequest(
                    uuid=dataset_uuid,
                    segments_page_request=pagination_pb2.PageRequest(
                        # 100k ids/page (~5 MiB) gave the best measured
                        # throughput (~80k ids/s); relies on the raised channel
                        # receive limit set in _build_secure_stub.
                        page_size=100000,
                        after_cursor=cursor,
                    ),
                    mode=pb.DESCRIBE_DATA_SET_MODE_EXTERNAL_IDS_ONLY,
                )
            )
            for seg in resp.dataset.segments:
                if seg.external_id:
                    ids.add(seg.external_id)
            if progress is not None:
                progress(len(ids))
            if not resp.segments_page_info or not resp.segments_page_info.end_cursor:
                break
            cursor = resp.segments_page_info.end_cursor
    except grpc.RpcError as exc:
        raise DoraUnavailable(f"DescribeDataSet failed: {exc}") from exc
    LOGGER.info("Fetched %d segment ids from dataset %s", len(ids), dataset_uuid)
    return frozenset(ids)


def fetch_segment_bitmap(dataset_uuid: str, hostname: str | None = None):
    """The dataset's membership as a ``pyroaring.BitMap`` of global internal
    segment counters, fetched in a SINGLE ``DescribeDataSet(include_bitmap=True)``
    call -- no pagination, regardless of set size.

    DORA builds each dataset's roaring bitmap over a stable, global per-segment
    counter (the same id space across all datasets), so this bitmap can be
    intersected directly against a corpus that carries ``dx_internal_id``. Used
    by the fast segment-set downsample; returns a ``pyroaring.BitMap``.
    """
    stub = _get_stub(hostname)  # clean DoraUnavailable if the SDK isn't installed

    import grpc
    from adp.public.proto import pagination_pb2
    from adp.services.dora_event_management.proto import (
        dora_event_management_service_pb2 as pb,
    )
    from pyroaring import BitMap

    try:
        resp = stub.DescribeDataSet(
            pb.DescribeDataSetRequest(
                uuid=dataset_uuid,
                # We only need the bitmap, not the segment page; ask for 1 row.
                segments_page_request=pagination_pb2.PageRequest(page_size=1),
                include_bitmap=True,
            )
        )
    except grpc.RpcError as exc:
        raise DoraUnavailable(f"DescribeDataSet(include_bitmap) failed: {exc}") from exc

    bitmap = BitMap.deserialize(bytes(resp.dataset.roaring_bitmap))
    LOGGER.info(
        "Fetched roaring bitmap for dataset %s: %d segments (1 call)",
        dataset_uuid,
        len(bitmap),
    )
    return bitmap


# DORA accepts segments inline on CreateDataSet, but a single request body is
# bounded; the proven enrich path (scripts/enrich_corpus_with_dx_internal_id.py)
# batches AddSegmentsToDataSet at 1000 ids. We do the inline single call for small
# sets and fall back to create + batched-add above this threshold.
_CREATE_INLINE_MAX = 1000


def create_dataset(
    name: str,
    external_ids: list[str],
    version: int | None = None,
    custom_metadata: dict | None = None,
    hostname: str | None = None,
) -> tuple[str, int]:
    """Create a DORA curation dataset (segment set) from segment ``external_id``s.

    Mirrors the proven ``CreateDataSet`` / ``AddSegmentsToDataSet`` pattern in
    ``scripts/enrich_corpus_with_dx_internal_id.py``. When ``version`` is given,
    DORA upserts on ``(name, version)``. ``custom_metadata`` (a plain dict) is
    attached as a ``google.protobuf.Struct`` for provenance.

    Returns ``(dataset_uuid, version)``. Raises ``DoraUnavailable`` if the SDK is
    absent or the RPC fails -- the caller surfaces that to the UI.
    """
    stub = _get_stub(hostname)  # clean DoraUnavailable if the SDK isn't installed

    import grpc
    from adp.services.dora_event_management.proto import (
        dora_event_management_service_pb2 as pb,
    )
    from google.protobuf import struct_pb2

    meta_struct = None
    if custom_metadata:
        meta_struct = struct_pb2.Struct()
        meta_struct.update(custom_metadata)

    # dataset_type is omitted on purpose: the server defaults it to
    # DATA_SET_TYPE_CURATION_DATASET, and the enum value is not a top-level
    # attribute of this generated module (setting it raises AttributeError). This
    # matches scripts/enrich_corpus_with_dx_internal_id.py, which also omits it.
    req = pb.CreateDataSetRequest(name=name)
    if meta_struct is not None:
        req.custom_metadata.CopyFrom(meta_struct)
    if version is not None:
        req.version = int(version)

    inline = len(external_ids) <= _CREATE_INLINE_MAX
    if inline:
        req.external_ids.extend(external_ids)

    try:
        resp = stub.CreateDataSet(req)
        uuid = resp.dataset_uuid
        if not inline:
            for i in range(0, len(external_ids), _CREATE_INLINE_MAX):
                stub.AddSegmentsToDataSet(
                    pb.AddSegmentsToDataSetRequest(
                        dataset_uuid=uuid,
                        external_ids=external_ids[i : i + _CREATE_INLINE_MAX],
                    )
                )
    except grpc.RpcError as exc:
        raise DoraUnavailable(f"CreateDataSet failed: {exc}") from exc

    LOGGER.info(
        "Created DORA dataset %s (v%d) %r with %d segments",
        uuid,
        resp.version,
        name,
        len(external_ids),
    )
    return uuid, resp.version
