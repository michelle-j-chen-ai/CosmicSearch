"""Environment-driven configuration for the NLS video search app.

All values are sourced from environment variables so the same code runs
locally (export the vars) and on Apps Platform (set via `apps-platform app
secret set`). Nothing here is secret on its own; the AWS creds come from the
standard AWS_* env the container already receives.
"""

from __future__ import annotations

import dataclasses
import os


@dataclasses.dataclass(frozen=True)
class AppConfig:
    # Optional default Lance corpus URI to prefill the search box. The user can
    # type any rank=NNNNN/-sharded embeddings URI at query time; empty is fine.
    default_embeddings_uri: str
    # S3 URI of the merged fine-tuned text-encoder snapshot. Empty -> base HF
    # model. Downloaded to the disk cache on first load.
    model_artifact_uri: str
    # torch device for query encoding. "cpu" is the validated default
    # (~33ms/query); the model needs no GPU for text-only encoding.
    device: str
    # In-memory dtype for the embedding matrix. fp32 is the default because
    # numpy has no BLAS kernel for fp16: at 1M x 768 an fp32 gemv is ~58ms vs
    # ~1.2s for fp16. Use fp16 only to halve RAM on a memory-constrained
    # deploy, accepting the ~20x slower matmul.
    matrix_dtype: str
    # Presigned URL lifetime for the OCI MP4 objects, in seconds.
    presign_ttl_s: int
    # S3 prefix (OCI) under which each Download also writes a parquet copy of the
    # exported top-k rows; the written path is recorded with the export so Search
    # history can link to it. Empty -> parquet export disabled.
    export_s3_prefix: str
    # Emails allowed to see the usage-analytics view ("maintainers"). Visits are
    # recorded for everyone, but only these users can view them. Sourced from
    # NLS_OWNER_EMAIL as a comma-separated list. Empty -> nobody sees the view
    # (fail closed), so a misconfigured deploy never leaks usage to all users.
    maintainer_emails: frozenset[str]
    # Offline full-corpus interval scan: the FULL embedding dataset the scan worker
    # reads (the resident in-app corpus is a small subset). MUST be encoded by the
    # same model as the app's query encoder + resident corpus (black dwarf) -- a
    # query vector scored against a different model's video embeddings yields
    # systematically depressed cosines, so a threshold calibrated in the app UI
    # rejects everything. The scan writes intervals.{parquet,csv} under
    # scan_output_bucket/scan_output_prefix/<id>/. Trucking overrides this with its
    # own corpus via the NLS_SCAN_EMBEDDINGS_URI env var.
    scan_embeddings_uri: str
    scan_output_bucket: str
    scan_output_prefix: str
    # Default start date for the offline full-corpus scan -> latest. The scan
    # covers the full dataset history from here, independent of the interactive
    # date filter (which only scopes the in-app resident search).
    scan_default_from_date: str

    @staticmethod
    def from_env() -> "AppConfig":
        return AppConfig(
            default_embeddings_uri=os.environ.get("NLS_EMBEDDINGS_URI", "").strip(),
            model_artifact_uri=os.environ.get("NLS_MODEL_ARTIFACT_URI", "").strip(),
            device=os.environ.get("NLS_DEVICE", "cpu").strip(),
            matrix_dtype=os.environ.get("NLS_MATRIX_DTYPE", "float32").strip(),
            presign_ttl_s=int(os.environ.get("NLS_PRESIGN_TTL_S", "3600")),
            export_s3_prefix=os.environ.get(
                "NLS_EXPORT_S3_PREFIX",
                "s3://neuron-prod-data-intelligence-exploratory/michelle/nls_search/exports",
            ).strip(),
            maintainer_emails=frozenset(
                e.strip().lower()
                for e in os.environ.get("NLS_OWNER_EMAIL", "").split(",")
                if e.strip()
            ),
            scan_embeddings_uri=os.environ.get(
                "NLS_SCAN_EMBEDDINGS_URI",
                "s3://neuron-prod-data-intelligence-exploratory/michelle/nls_search/"
                "black-dwarf/embeddings/cars/",
            ).strip(),
            scan_output_bucket=os.environ.get(
                "NLS_SCAN_OUTPUT_BUCKET", "neuron-prod-data-intelligence-exploratory"
            ).strip(),
            scan_output_prefix=os.environ.get(
                "NLS_SCAN_OUTPUT_PREFIX", "michelle/nls_search/nls_scans"
            ).strip(),
            scan_default_from_date=os.environ.get(
                "NLS_SCAN_DEFAULT_FROM_DATE", "2025-07-01"
            ).strip(),
        )


# Base model identity, must match fine_tuned_embed_inference.py so query
# embeddings land in the same joint space as the indexed video embeddings.
BASE_MODEL_URI = "nvidia/Cosmos-Embed1-448p"
BASE_MODEL_REVISION = "f60ec73636eb7c9cc25267367713b7b1b0cffaf3"
# Lance table name written by the inference workload.
OUTPUT_TABLE_NAME = "video_embeddings"
