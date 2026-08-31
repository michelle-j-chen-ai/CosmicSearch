"""OCI S3-compat helpers: client construction and presigned MP4 URLs.

The chunk MP4s are small (8 frames), so the browser gets a presigned GET URL and
streams from OCI directly rather than proxying bytes through Cloud Run.
"""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlparse

import boto3
import botocore.config


def _endpoint_url() -> str | None:
    return os.environ.get("AWS_ENDPOINT_URL_S3") or os.environ.get("AWS_ENDPOINT_URL")


def _region() -> str:
    return os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-phoenix-1"


def s3_client() -> object:
    # OCI rejects AWS CLI v2's default streaming checksums. SigV4 with path-style
    # addressing is mandatory: without signature_version, boto3 presigns legacy
    # SigV2 URLs that OCI answers with 404, and the signature embeds the region,
    # so a default us-west-2 is rejected with 400.
    kwargs: dict[str, object] = {
        "retries": {"total_max_attempts": 8, "mode": "standard"},
        "connect_timeout": 30,
        "read_timeout": 300,
        "signature_version": "s3v4",
        "s3": {"addressing_style": "path"},
    }
    try:
        config = botocore.config.Config(
            **kwargs,
            request_checksum_calculation="when_required",
            response_checksum_validation="when_required",
        )
    except TypeError:
        config = botocore.config.Config(**kwargs)
    return boto3.client(
        "s3", endpoint_url=_endpoint_url(), region_name=_region(), config=config
    )


def parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri.replace("s3a://", "s3://", 1))
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path:
        raise ValueError(f"Expected S3 URI, got {uri}")
    return parsed.netloc, parsed.path.lstrip("/")


def download_object(uri: str, destination: Path, client: object) -> None:
    bucket, key = parse_s3_uri(uri)
    destination.parent.mkdir(parents=True, exist_ok=True)
    client.download_file(bucket, key, str(destination))


def presign_get(uri: str, client: object, ttl_s: int) -> str:
    bucket, key = parse_s3_uri(uri)
    return client.generate_presigned_url(
        "get_object", Params={"Bucket": bucket, "Key": key}, ExpiresIn=ttl_s
    )
