"""Submit the NLS scans as Lilypad CPU (Ray) workloads, directly from the app.

Replaces flyte_client: the app builds a Lilypad WorkloadConfig and submits it to the
Lilypad gRPC service, referencing the prebuilt, pushed worker image by digest
(NLS_SCAN_IMAGE -- keep in sync with nls_scan/purpose/prod.yaml). The bazel launcher
(nls_scan/launch_scan.py) is the dev-container equivalent of this module.

The full lilypad SDK (`lilypad_sdk`) is unusable inside this lean Cloud Run app: its
module imports ray at load time, and its LaunchWorkload validates the image with a
`docker pull` (no Docker daemon here). So we use only the SUBMIT PRIMITIVES, which import
without ray -- build the request with `workload_utils.make_launch_workload_request` and
call `LilypadStub.LaunchWorkload` directly -- and set SIMIAN_CLOUD_SERVICE=1, which the SDK
itself uses to skip the docker-pull validation (we pin a known-pushed digest instead).
Only the single `lilypad_py` wheel is vendored (vendor/); its import-closure deps are
light and come from public PyPI (requirements.txt). The bundled `adp.services.lilypad`
protos merge with the app's vendored DORA `adp` (both PEP420 namespace packages).

Auth reuses the app's EXISTING ADP machine credentials -- the same machine_auth flow
dora_client uses for the data-explorer / DORA APIs (MACHINE_CLIENT_ID / MACHINE_CLIENT_SECRET,
Apps Platform secrets). machine_auth exchanges them at accounts.applied.co for a token; we
put it in AUTH_TOKEN / URSA_SDK_GRPC_AUTH_TOKEN, which is what get_auth_token_from_env reads.
The gRPC hostname defaults to prod (ml-infra.applied.dev), so no endpoint config is needed.

Deploy requirements (apps-platform): the vendored lilypad_py wheel + light deps installed,
and egress to ml-infra.applied.dev. `available()` reports whether a launch can be made.
"""

from __future__ import annotations

import logging
import os

import machine_auth

LOGGER = logging.getLogger("nls_search_app.nls_launcher")

# Pinned worker image -- the pushed form of //...:nls_scan_lilypad_image. Keep in sync
# with nls_scan/purpose/prod.yaml; overridable per-deploy via NLS_SCAN_IMAGE.
_DEFAULT_IMAGE = (
    "us-phoenix-1.ocir.io/idskhu5vqvtl/lilypad/sds"
    "@sha256:c47d518453a039ab86bf6d358ba3727d966de2cbed28ac2ece3eb4e1386f3b5f"
)
_OCI_ENDPOINT = "https://idskhu5vqvtl.compat.objectstorage.us-phoenix-1.oraclecloud.com"
_CUSTOMER_RESOURCE = "autolabeling"
_ALLOWED_REGIONS = ["us-chicago-1"]
_NUM_CPU_NODES = 16

_SEGMENT_FN = "tools.offboard.common.auto_labeling.vlm.nls_scan.nls_segment_scan.run"

# Lilypad model identifier (mandatory: the backend rejects launches without one).
# This CPU workload scores query embeddings against the precomputed VLM (Cosmos-Embed)
# corpus, so it uses the `vlm_embedding` category.
#
# Must be the full <category>.<mode>.<model_id>.<variant>.<version>. The bare
# category passed the SDK's client-side check (alphanumeric + `-_.`, <=100 chars)
# and was accepted by the backend until it began enforcing the five-part shape,
# at which point every launch failed with INVALID_ARGUMENT and the app returned
# 502 with no server-side log of why.
_MODEL_IDENTIFIER = "vlm_embedding.search.black-dwarf.default.v1"


class LauncherUnavailable(RuntimeError):
    """Raised when a Lilypad launch cannot be made (SDK absent / no creds / no egress)."""


def _image() -> str:
    return os.environ.get("NLS_SCAN_IMAGE", "").strip() or _DEFAULT_IMAGE


def available() -> bool:
    """True iff we can import the lilypad submit primitives and obtain a machine token.

    NB: we deliberately do NOT import lilypad.public.sdk_py.lilypad_sdk -- that module
    imports ray (via workspace_utils) at load time. We only need the request builder +
    gRPC stub, which import without ray."""
    try:
        import lilypad.public.schemas.workload_config  # noqa: F401
        import lilypad.public.sdk_py.workload_utils  # noqa: F401
    except ImportError:
        return False
    return machine_auth.available() or bool(os.environ.get("URSA_SDK_GRPC_AUTH_TOKEN"))


def _auth() -> None:
    """Prepare the env the lilypad submit path reads: the auth token (from the app's ADP
    machine creds) + SIMIAN_CLOUD_SERVICE so make_launch_workload_request skips the
    docker-pull image validation (there is no Docker daemon in Cloud Run)."""
    # Skip the SDK's docker-pull validation of the image (build_and_upload_runtime_environment
    # gates it on `not SIMIAN_CLOUD_SERVICE`); we pin a known-pushed digest instead.
    os.environ.setdefault("SIMIAN_CLOUD_SERVICE", "1")
    if machine_auth.available():
        token = machine_auth.get_machine_token()
        os.environ["URSA_SDK_GRPC_AUTH_TOKEN"] = token
        os.environ["AUTH_TOKEN"] = token
    elif not os.environ.get("URSA_SDK_GRPC_AUTH_TOKEN"):
        raise LauncherUnavailable("no machine creds and no URSA_SDK_GRPC_AUTH_TOKEN")


def _lilypad_stub():
    """A LilypadStub bound to the prod gRPC endpoint + the app's auth token. Imports the
    submit primitives only (no lilypad_sdk -> no ray)."""
    from adp.services.lilypad.proto import lilypad_service_pb2_grpc
    from lilypad.public.utils import env as lp_env, grpc as lp_grpc

    return lp_grpc.get_stub(
        lilypad_service_pb2_grpc.LilypadStub,
        lp_env.get_hostname_from_env(),
        lp_env.get_auth_token_from_env(),
    )


def _workload_cfg(
    *, name: str, entrypoint_fn: str, entrypoint_fn_config: dict, enable_boto3_cache: bool
) -> dict:
    return {
        "name": name,
        "model_identifier": _MODEL_IDENTIFIER,
        "workload_variant_config": {
            "workload_type": "generic",
            "entrypoint_fn": entrypoint_fn,
            "entrypoint_fn_config": entrypoint_fn_config,
        },
        "cluster_resources": {
            "num_cpu_nodes": _NUM_CPU_NODES,
            "num_gpus": 0,
            "customer_resource": _CUSTOMER_RESOURCE,
            "preemptible": "when_quota_full",
            "allowed_regions": list(_ALLOWED_REGIONS),
        },
        "experimental_flags": {
            "file_cache_configuration": {
                "enable_lance_cache": True,
                "enable_boto3_cache": enable_boto3_cache,
            }
        },
        "runtime_environment": {
            "code_assets": {"docker_image": _image()},
            "constant_environment_variables": {
                "AWS_ENDPOINT_URL_S3": _OCI_ENDPOINT,
                "AWS_DEFAULT_REGION": "us-phoenix-1",
                "URSA_SDK_GRPC_HOSTNAME": "grpc.neuron.oci.applied.dev",
                "LANCE_IO_THREADS": "16",
            },
        },
    }


def _workload_url(workload_id: str) -> str:
    """Best-effort Lilypad workloads page (the run dashboard needs the internal uuid,
    not the user-facing id, so link to the filterable list)."""
    return f"https://ml-infra.applied.dev/lilypad?user_facing_id={workload_id}"


def _submit(cfg: dict) -> dict:
    if not available():
        raise LauncherUnavailable("lilypad submit primitives or machine creds unavailable")
    _auth()
    from lilypad.public.schemas.workload_config import WorkloadConfig
    from lilypad.public.sdk_py import workload_utils

    try:
        req = workload_utils.make_launch_workload_request(WorkloadConfig(**cfg))
        resp = _lilypad_stub().LaunchWorkload(req)
        workload_id = str(resp.user_facing_id)
    except Exception as exc:  # noqa: BLE001 -- surface a clean 502 to the UI
        raise LauncherUnavailable(f"lilypad launch failed: {exc}") from exc
    # UI-compatible shape (the front end was written against the Flyte client): the
    # workload id stands in for execution_id, plus a console url.
    return {"execution_id": workload_id, "url": _workload_url(workload_id)}


def launch_segment_scan(
    *,
    search_vectors: dict[str, list[float]],
    thresholds: dict[str, float],
    embeddings_uri: str,
    output_dir: str,
    scan_id: str = "",
    chunks_metadata_uri: str = "",
    start_date: str = "",
    end_date: str = "",
    segment_set_uuid: str = "",
    segment_set_name: str = "",
    filter_lance_uri: str = "",
    filter_key: str = "",
    filter_segment_ids: list[str] | None = None,
    filter_ids_uri: str = "",
    segment_set_ids: list[str] | None = None,
    segment_set_ids_uri: str = "",
    vehicle: str = "",
    drive_id: str = "",
    merge_intervals: bool = True,
    top_k: int | None = None,
    name: str = "nls-segment-scan",
) -> dict:
    """Per-segment multi-tag scan: {tag: [768 floats]} -> per-segment Lance table.

    ``thresholds`` is the per-tag cosine cutoff ({tag: float}); the worker applies each
    tag's own cutoff. A scalar ``threshold`` (the min over tags) is also sent as a
    back-compat fallback for older worker images that predate per-tag thresholds.

    ``scan_id`` pins the output subdir so the caller knows the exact Lance path up front
    (the worker writes ``output_dir/<scan_id>/segments.lance``); the returned dict carries
    ``lance_uri`` / ``output_root`` for that path."""
    per_tag = {t: float(thresholds.get(t, 0.3)) for t in search_vectors}
    scalar = min(per_tag.values()) if per_tag else 0.3
    sid = (scan_id or "").strip()
    ec = {
        "embeddings_uri": embeddings_uri,
        "output_dir": output_dir,
        "thresholds": per_tag,
        "threshold": float(scalar),
        "start_date": start_date or "",
        "end_date": end_date or "",
        "num_blocks": 64,
        # Output mode (always segment-keyed): True merges contiguous clips into intervals per
        # segment; False emits one best clip per segment. Worker defaults True if omitted.
        "merge_intervals": bool(merge_intervals),
        "search_vectors": {t: [float(x) for x in v] for t, v in search_vectors.items()},
    }
    # Top-K retrieval (only when set): the worker returns the K highest-scoring segments
    # per tag, scoped to the downsample/segment set. Omitted otherwise so a threshold scan
    # config is unchanged (and an old worker image that predates the feature is unaffected).
    if top_k:
        ec["top_k"] = int(top_k)
    # Carry the full active filter set into the workflow inputs so the launched scan
    # includes the same filters the search used. The worker enforces date today and reads
    # the rest by name; passing them as runtime config needs no image rebuild and makes the
    # workflow self-documenting (and future-enforceable) for segment-set/lance/vehicle/drive.
    filters = {
        "from_date": start_date or "",
        "to_date": end_date or "",
        "segment_set_uuid": segment_set_uuid or "",
        "segment_set_name": segment_set_name or "",
        "filter_lance_uri": filter_lance_uri or "",
        "vehicle": vehicle or "",
        "drive_id": drive_id or "",
        # Pre-resolved downsample membership: inline for small sets, else by parquet
        # reference (filter_ids_uri / segment_set_ids_uri -- any cardinality; the
        # worker reads them in one GET). Output is restricted to these segments.
        "filter_key": filter_key or "",
        "filter_segment_ids": list(filter_segment_ids or []),
        "filter_ids_uri": filter_ids_uri or "",
        # DX segment-set membership (external segment_ids resolved app-side); the worker
        # keeps segments in this set. Separate from the lance downsample so both can
        # apply together.
        "segment_set_ids": list(segment_set_ids or []),
        "segment_set_ids_uri": segment_set_ids_uri or "",
    }
    ec["filters"] = filters
    if sid:
        ec["scan_id"] = sid
    if chunks_metadata_uri:
        ec["chunks_metadata_uri"] = chunks_metadata_uri
    res = _submit(_workload_cfg(
        name=name, entrypoint_fn=_SEGMENT_FN, entrypoint_fn_config=ec, enable_boto3_cache=True
    ))
    if sid:
        root = f"{output_dir.rstrip('/')}/{sid}"
        res["output_root"] = root
        res["lance_uri"] = f"{root}/segments.lance"
    return res


def _status_name(info) -> str:
    """Human-readable workload status NAME. The proto ``status`` field is an enum, so a
    plain ``str()`` yields its integer value (e.g. "4"); resolve the symbolic name via the
    message descriptor so the UI shows RUNNING/COMPLETED/... instead of a number."""
    raw = getattr(info, "status", None)
    if raw is None:
        return ""
    try:
        fld = info.DESCRIPTOR.fields_by_name.get("status")
        if fld is not None and fld.enum_type is not None:
            ev = fld.enum_type.values_by_number.get(int(raw))
            if ev is not None:
                return ev.name
    except Exception:  # noqa: BLE001 -- fall back to the raw value
        pass
    return str(raw)


def scan_status(workload_id: str, timeout: float = 8.0) -> dict:
    """Phase of a launched scan workload, in the UI's {done, phase, error} shape.

    ``timeout`` bounds the Lilypad gRPC call (seconds) so an unresponsive backend or a
    stuck workload fails fast instead of blocking the caller -- essential when this is
    called in a loop over many jobs on a request path.

    Lilypad status (e.g. EXPERIMENT_RUNNING / EXPERIMENT_COMPLETED / EXPERIMENT_FAILED)
    maps to: done iff terminal; phase=SUCCEEDED on success (the UI checks that), else
    the human-readable status."""
    if not available():
        raise LauncherUnavailable("lilypad submit primitives or machine creds unavailable")
    _auth()
    from adp.services.lilypad.proto import lilypad_service_pb2

    try:
        req = lilypad_service_pb2.GetWorkloadRequest(user_facing_id=workload_id)
        info = _lilypad_stub().GetWorkload(req, timeout=timeout).workload
    except Exception as exc:  # noqa: BLE001
        raise LauncherUnavailable(f"lilypad status failed: {exc}") from exc
    status = _status_name(info).upper()
    succeeded = any(s in status for s in ("COMPLETED", "SUCCEEDED", "SUCCESS"))
    terminal = succeeded or any(s in status for s in ("FAILED", "STOPPED", "CANCEL", "ERROR"))
    # Strip common enum prefixes so the UI shows a short phase (RUNNING, PENDING, ...).
    short = status.replace("EXPERIMENT_STATUS_", "").replace("EXPERIMENT_", "").replace("WORKLOAD_", "")
    phase = "SUCCEEDED" if succeeded else (short or "RUNNING")
    return {"done": terminal, "phase": phase, "error": ""}
