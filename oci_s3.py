"""OCI S3-compat helpers: client construction, Lance storage options, and
presigned URLs for streaming MP4 chunks directly to the browser.

The video objects are tiny (8 frames at 1Hz), so we hand the browser a
presigned GET URL and let it stream from OCI with range requests rather than
proxying bytes through the app.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from urllib.parse import urlparse

import boto3
import botocore.config
import botocore.exceptions

LOGGER = logging.getLogger(__name__)


def _endpoint_url() -> str | None:
    return (
        os.environ.get("AWS_ENDPOINT_URL_S3")
        or os.environ.get("AWS_ENDPOINT_URL")
        or os.environ.get("S3_ENDPOINT_URL")
    )


def _region() -> str:
    return (
        os.environ.get("AWS_REGION")
        or os.environ.get("AWS_DEFAULT_REGION")
        or "us-phoenix-1"
    )


def s3_client(fast_fail: bool = False) -> object:
    """boto3 S3 client pointed at the OCI S3-compat endpoint.

    Mirrors fine_tuned_embed_inference._s3_client: OCI rejects AWS CLI v2's
    default streaming checksums, so request_checksum_calculation is forced to
    "when_required".

    ``fast_fail`` trims retries + timeouts so a request against an unreachable
    bucket (e.g. a downsample-dataset path on a bucket the app can't read)
    surfaces an error in a few seconds instead of stalling on retry backoff.
    """
    cfg_kwargs: dict[str, object] = {
        "retries": {
            "total_max_attempts": 2 if fast_fail else 8,
            "mode": "standard",
        },
        "connect_timeout": 5 if fast_fail else 30,
        "read_timeout": 15 if fast_fail else 120,
        # OCI S3-compat requires SigV4 and path-style addressing. Without
        # signature_version, boto3 presigns legacy SigV2 URLs (AWSAccessKeyId/
        # Signature/Expires) that OCI rejects with 404.
        "signature_version": "s3v4",
        "s3": {"addressing_style": "path"},
    }
    try:
        config = botocore.config.Config(
            **cfg_kwargs,
            request_checksum_calculation="when_required",
            response_checksum_validation="when_required",
        )
    except TypeError:
        config = botocore.config.Config(**cfg_kwargs)
    # region_name must be passed explicitly: the SigV4 signature embeds the
    # region, and OCI rejects a mismatch (e.g. a default us-west-2) with 400.
    return boto3.client(
        "s3", endpoint_url=_endpoint_url(), region_name=_region(), config=config
    )


class CredentialsMissing(RuntimeError):
    """AWS_* credentials for the OCI endpoint are not configured."""


def lance_storage_options() -> dict[str, str]:
    """Storage options for lancedb.connect() against OCI S3-compat.

    lance uses the Rust object_store crate, whose keys differ from boto3's.

    Raises if the credentials are absent rather than omitting them. object_store
    treats an incomplete option set as "discover credentials yourself" and walks
    its own provider chain to the GCE metadata server, which answers
    ``403 Missing required header: Metadata-Flavor`` -- an error that names
    neither this app's configuration nor the bucket it failed to read.
    """
    access_key = os.environ.get("AWS_ACCESS_KEY_ID")
    secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
    if not access_key or not secret_key:
        absent = [
            name
            for name, val in (
                ("AWS_ACCESS_KEY_ID", access_key),
                ("AWS_SECRET_ACCESS_KEY", secret_key),
            )
            if not val
        ]
        raise CredentialsMissing(
            f"{' and '.join(absent)} not set; the OCI corpus and model cannot be read"
        )
    opts: dict[str, str] = {
        "aws_access_key_id": access_key,
        "aws_secret_access_key": secret_key,
    }
    endpoint = _endpoint_url()
    region = _region()
    if endpoint:
        opts["aws_endpoint"] = endpoint
        opts["aws_virtual_hosted_style_request"] = "false"
        opts["aws_allow_http"] = "true" if endpoint.startswith("http://") else "false"
    opts["aws_region"] = region
    return opts


def parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri.replace("s3a://", "s3://", 1))
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path:
        raise ValueError(f"Expected S3 URI, got {uri}")
    return parsed.netloc, parsed.path.lstrip("/")


def discover_rank_tables(embeddings_uri: str, client: object) -> list[str]:
    """List rank=NNNNN/ shard URIs under the embeddings parent prefix."""
    bucket, key_prefix = parse_s3_uri(embeddings_uri.rstrip("/") + "/")
    paginator = client.get_paginator("list_objects_v2")
    rank_prefixes: set[str] = set()
    for page in paginator.paginate(Bucket=bucket, Prefix=key_prefix, Delimiter="/"):
        for entry in page.get("CommonPrefixes", []) or []:
            prefix = entry.get("Prefix", "")
            tail = prefix[len(key_prefix) :].rstrip("/")
            if tail.startswith("rank="):
                rank_prefixes.add(prefix)
    return sorted(f"s3://{bucket}/{p.rstrip('/')}" for p in rank_prefixes)


def download_s3_prefix(prefix_uri: str, destination: Path, client: object) -> None:
    """Download every object under an S3 prefix into a local directory.

    Used to materialize a fine-tuned merged-model snapshot locally so
    transformers can load it (transformers cannot read s3:// paths directly).
    """
    bucket, key_prefix = parse_s3_uri(prefix_uri.rstrip("/") + "/")
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=key_prefix):
        for item in page.get("Contents", []) or []:
            key = item["Key"]
            if key.endswith("/"):
                continue
            relative = key[len(key_prefix) :].lstrip("/")
            local_path = destination / relative
            local_path.parent.mkdir(parents=True, exist_ok=True)
            LOGGER.info("downloading %s -> %s", key, local_path)
            client.download_file(bucket, key, str(local_path))


def object_exists(uri: str, client: object) -> bool:
    """True if a single S3 object exists (used to detect the fast npy format)."""
    bucket, key = parse_s3_uri(uri)
    try:
        client.head_object(Bucket=bucket, Key=key)
        return True
    except botocore.exceptions.ClientError:
        return False


def download_object(uri: str, destination: Path, client: object) -> None:
    """Download a single S3 object to a local path."""
    bucket, key = parse_s3_uri(uri)
    destination.parent.mkdir(parents=True, exist_ok=True)
    client.download_file(bucket, key, str(destination))


def put_bytes(uri: str, data: bytes, client: object, content_type: str) -> None:
    """Upload an in-memory blob to an S3 object (used to write export parquet)."""
    bucket, key = parse_s3_uri(uri)
    client.put_object(Bucket=bucket, Key=key, Body=data, ContentType=content_type)


def presign_get(source_media_uri: str, client: object, ttl_s: int) -> str:
    """Presigned GET URL for an MP4 chunk, streamable by a browser <video>."""
    bucket, key = parse_s3_uri(source_media_uri)
    return client.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=ttl_s,
    )
