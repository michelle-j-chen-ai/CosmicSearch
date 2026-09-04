"""Offline precompute: Lance embedding table -> 2D atlas artifact for the app.

The source table is ~34M x 768 fp32 (~104GB). Neither UMAP nor a browser can
consume that, and a 1000x1000 canvas only resolves ~1e6 points anyway, so the
artifact is a stratified sample of ~250k rows projected to 2D.

Pipeline: uniform row sample -> per-run cap -> L2 normalize -> PCA 768->50
-> UMAP 50->2.

PCA before UMAP is not just a speedup. High-dimensional distances concentrate
(the nearest/farthest neighbour ratio drifts toward 1), so a kNN graph built on
raw 768-d encodes more noise than one built on the top-50 principal directions.

The PCA basis and the fitted UMAP model are written alongside the coordinates.
Without them, later points (a new checkpoint's embeddings, a text query) cannot
be placed on THIS map -- coordinates from two independent UMAP fits are not
comparable, so re-fitting produces an unrelated picture.

Usage:
    python3 precompute_atlas.py \
        --embeddings-uri s3://.../video_embeddings.lance \
        --output-uri s3://.../nls_search/black-dwarf/atlas
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import time
from pathlib import Path

import boto3
import botocore.config
import lance
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch
import umap

LOGGER = logging.getLogger("precompute_atlas")

# Columns carried into the artifact. `vector` is fetched for the projection but
# never written -- the app only needs coordinates and click-through metadata.
METADATA_COLUMNS = [
    "chunk_id",
    "run_uuid",
    "chunk_start_unix",
    "dt",
    "source_media_uri",
]

OCI_ENDPOINT = (
    "https://idskhu5vqvtl.compat.objectstorage.us-phoenix-1.oraclecloud.com"
)
OCI_REGION = "us-phoenix-1"


def _frozen_credentials(profile: str | None) -> object:
    session = boto3.Session(profile_name=profile) if profile else boto3.Session()
    credentials = session.get_credentials()
    assert credentials is not None, f"no AWS credentials for profile {profile!r}"
    return credentials.get_frozen_credentials()


def lance_storage_options(profile: str | None) -> dict[str, str]:
    """object_store-style options for OCI S3-compat.

    lance uses the Rust object_store crate, whose keys differ from boto3's;
    path-style addressing is required because OCI has no virtual-hosted form.
    """
    creds = _frozen_credentials(profile)
    return {
        "aws_access_key_id": creds.access_key,
        "aws_secret_access_key": creds.secret_key,
        "aws_endpoint": OCI_ENDPOINT,
        "aws_region": OCI_REGION,
        "aws_virtual_hosted_style_request": "false",
    }


def s3_client(profile: str | None) -> object:
    session = boto3.Session(profile_name=profile) if profile else boto3.Session()
    # OCI rejects AWS CLI v2's default streaming checksums, and SigV4 with
    # path-style addressing is mandatory.
    config = botocore.config.Config(
        retries={"total_max_attempts": 8, "mode": "standard"},
        connect_timeout=30,
        read_timeout=300,
        signature_version="s3v4",
        s3={"addressing_style": "path"},
        request_checksum_calculation="when_required",
        response_checksum_validation="when_required",
    )
    return session.client(
        "s3", endpoint_url=OCI_ENDPOINT, region_name=OCI_REGION, config=config
    )


def sample_rows(
    dataset: lance.LanceDataset,
    sample_size: int,
    per_run_cap: int,
    oversample: float,
    batch_size: int,
    seed: int,
    vector_column: str = "vector",
) -> pa.Table:
    """Uniform row sample, then capped per run_uuid.

    Uniform-then-cap rather than a true stratified draw: reading `run_uuid` for
    all 34M rows to build exact strata costs a full column scan, while the cap
    achieves the thing that actually matters -- stopping one long drive from
    dominating the map.
    """
    total_rows = dataset.count_rows()
    draw = min(total_rows, int(sample_size * oversample))
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(total_rows, size=draw, replace=False))
    LOGGER.info("drawing %d of %d rows (target %d)", draw, total_rows, sample_size)

    # take() in batches: one request for 500k random offsets across 94 fragments
    # spends a long time buffering before anything is observable.
    batches = []
    started = time.perf_counter()
    for start in range(0, len(indices), batch_size):
        chunk = indices[start : start + batch_size]
        batches.append(
            dataset.take(
                chunk.tolist(),
                # Aliased so the rest of this script, and the artifact schema,
                # do not depend on what the source table calls its embedding.
                columns={**{c: c for c in METADATA_COLUMNS}, "vector": vector_column},
            )
        )
        fetched = min(start + batch_size, len(indices))
        LOGGER.info(
            "fetched %d/%d rows (%.1fs elapsed)",
            fetched,
            len(indices),
            time.perf_counter() - started,
        )
    table = pa.concat_tables(batches)

    # A corpus row exists before it is embedded, so `vector` is nullable and a
    # null cannot be projected. Dropped before the cap so the trim still lands
    # on sample_size rather than slightly under it.
    valid = pc.is_valid(table["vector"])
    dropped = table.num_rows - pc.sum(valid).as_py()
    if dropped:
        LOGGER.info("dropped %d rows with no embedding", dropped)
        table = table.filter(valid)

    keep = _cap_per_run(table["run_uuid"].to_numpy(zero_copy_only=False), per_run_cap)
    # Logged before the trim: afterwards the count is sample_size regardless, so
    # it would say nothing about whether the cap actually bound.
    LOGGER.info(
        "cap at %d per run_uuid dropped %d of %d rows",
        per_run_cap,
        table.num_rows - len(keep),
        table.num_rows,
    )
    rng.shuffle(keep)
    keep = np.sort(keep[:sample_size])
    LOGGER.info("trimmed to %d rows", len(keep))
    return table.take(keep)


def _cap_per_run(run_uuids: np.ndarray, per_run_cap: int) -> np.ndarray:
    """Row positions keeping at most `per_run_cap` rows per run_uuid.

    `run_uuid` is nullable, and sorting an object array containing None raises
    TypeError comparing NoneType to str -- after the full multi-hour read that
    produced this array. Nulls become one group of their own instead.
    """
    run_uuids = np.asarray(
        ["" if v is None else str(v) for v in run_uuids], dtype=object
    )
    order = np.argsort(run_uuids, kind="stable")
    sorted_ids = run_uuids[order]
    # Position of each row within its own run_uuid group.
    group_starts = np.flatnonzero(
        np.concatenate(([True], sorted_ids[1:] != sorted_ids[:-1]))
    )
    rank_in_group = np.arange(len(sorted_ids)) - np.repeat(
        group_starts, np.diff(np.append(group_starts, len(sorted_ids)))
    )
    return order[rank_in_group < per_run_cap]


def fit_pca(vectors: np.ndarray, pca_dim: int, device: str) -> dict[str, np.ndarray]:
    """L2-normalize, then PCA to `pca_dim` via randomized SVD on the GPU.

    Returns the basis (mean + components) so the same projection can be applied
    to new vectors later, plus the computed singular values. Their share of the
    TOTAL variance is the cheapest check for embedding collapse; the spectrum is
    truncated at q, so it is a lower bound on how much structure exists.
    """
    matrix = torch.from_numpy(vectors).to(device)
    matrix = torch.nn.functional.normalize(matrix, dim=1)
    mean = matrix.mean(dim=0, keepdim=True)
    centered = matrix - mean

    # q > pca_dim oversamples the randomized range finder; the extra directions
    # also give a longer spectrum to inspect for collapse.
    q = min(centered.shape[1], max(pca_dim * 2, pca_dim + 32))
    _, singular_values, components = torch.pca_lowrank(centered, q=q, niter=4)

    projected = centered @ components[:, :pca_dim]
    # Normalize by the TOTAL variance, not by the truncated spectrum.
    # pca_lowrank returns only q singular values, so dividing by their sum makes
    # the ratios sum to 1 over whatever was computed -- reporting "retains 100%"
    # for any q and hiding exactly the collapse this number exists to detect.
    total_var = float((centered**2).sum())
    sv2 = (singular_values**2).cpu().numpy()
    explained = (sv2 / total_var).astype(np.float32) if total_var > 0 else sv2 * 0.0
    LOGGER.info(
        "PCA %d->%d retains %.1f%% of sampled variance (top-10: %.1f%%, "
        "%d of %d directions computed)",
        vectors.shape[1],
        pca_dim,
        100.0 * float(explained[:pca_dim].sum()),
        100.0 * float(explained[:10].sum()),
        q,
        vectors.shape[1],
    )
    return {
        "projected": projected.cpu().numpy().astype(np.float32),
        "mean": mean.cpu().numpy().astype(np.float32),
        "components": components[:, :pca_dim].cpu().numpy().astype(np.float32),
        "singular_values": singular_values.cpu().numpy().astype(np.float32),
        "explained_variance_ratio": explained.astype(np.float32),
    }


def fit_umap(
    projected: np.ndarray, n_neighbors: int, min_dist: float, seed: int
) -> tuple[np.ndarray, umap.UMAP]:
    """UMAP to 2D. Cosine metric: these are normalized embeddings, and the
    retrieval path they are evaluated by scores with cosine similarity."""
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="cosine",
        random_state=seed,
        verbose=True,
    )
    started = time.perf_counter()
    coords = reducer.fit_transform(projected)
    LOGGER.info("UMAP fit in %.1fs", time.perf_counter() - started)
    return coords.astype(np.float32), reducer


def build_artifact(table: pa.Table, coords: np.ndarray, pca: dict) -> pa.Table:
    """Coordinates + click-through metadata, normalized to a [-1, 1] box.

    Rescaling keeps the frontend from having to discover the extent of an
    arbitrary UMAP output before it can set up a viewport.
    """
    centered = coords - coords.mean(axis=0)
    scaled = centered / np.abs(centered).max()
    return pa.table(
        {
            "x": pa.array(scaled[:, 0], type=pa.float32()),
            "y": pa.array(scaled[:, 1], type=pa.float32()),
            **{name: table[name] for name in METADATA_COLUMNS},
            # Retained so the map can be re-laid-out with different UMAP
            # hyperparameters without re-reading the 104GB source table.
            "pca": pa.array(list(pca["projected"]), type=pa.list_(pa.float32())),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings-uri", required=True)
    parser.add_argument(
        "--output-uri",
        required=True,
        help="S3 prefix receiving atlas.parquet, projection.pkl, spectrum.json",
    )
    parser.add_argument("--local-dir", default="/tmp/embedding_atlas")
    parser.add_argument("--sample-size", type=int, default=250_000)
    parser.add_argument(
        "--per-run-cap",
        type=int,
        default=400,
        help="Max rows per run_uuid, so one long drive cannot dominate the map",
    )
    parser.add_argument("--oversample", type=float, default=2.0)
    parser.add_argument("--take-batch-size", type=int, default=50_000)
    # Corpus tables name the embedding per model (`vector_black_dwarf`); the
    # older standalone atlas table calls it `vector`.
    parser.add_argument("--vector-column", default="vector")
    parser.add_argument("--pca-dim", type=int, default=50)
    parser.add_argument("--umap-neighbors", type=int, default=25)
    parser.add_argument("--umap-min-dist", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--aws-profile", default="oci")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    dataset = lance.dataset(
        args.embeddings_uri,
        storage_options=lance_storage_options(args.aws_profile),
    )
    table = sample_rows(
        dataset,
        sample_size=args.sample_size,
        per_run_cap=args.per_run_cap,
        oversample=args.oversample,
        batch_size=args.take_batch_size,
        seed=args.seed,
        vector_column=args.vector_column,
    )

    vectors = np.stack(table["vector"].to_numpy(zero_copy_only=False)).astype(np.float32)
    LOGGER.info("sampled vectors: %s", vectors.shape)

    pca = fit_pca(vectors, pca_dim=args.pca_dim, device=args.device)
    coords, reducer = fit_umap(
        pca["projected"],
        n_neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist,
        seed=args.seed,
    )

    local_dir = Path(args.local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    artifact = build_artifact(table, coords, pca)
    atlas_path = local_dir / "atlas.parquet"
    pq.write_table(artifact, atlas_path, compression="zstd")

    projection_path = local_dir / "projection.pkl"
    with open(projection_path, "wb") as handle:
        pickle.dump(
            {
                "pca_mean": pca["mean"],
                "pca_components": pca["components"],
                "umap": reducer,
                "embeddings_uri": args.embeddings_uri,
                "seed": args.seed,
            },
            handle,
        )

    spectrum = {
        "rows_sampled": artifact.num_rows,
        "source_rows": dataset.count_rows(),
        "embedding_dim": int(vectors.shape[1]),
        "pca_dim": args.pca_dim,
        "explained_variance_ratio": pca["explained_variance_ratio"].tolist(),
        "singular_values": pca["singular_values"].tolist(),
        "effective_rank_95pct": int(
            np.searchsorted(np.cumsum(pca["explained_variance_ratio"]), 0.95) + 1
        ),
        "umap_neighbors": args.umap_neighbors,
        "umap_min_dist": args.umap_min_dist,
    }
    spectrum_path = local_dir / "spectrum.json"
    spectrum_path.write_text(json.dumps(spectrum, indent=2))
    LOGGER.info(
        "effective rank (95%% variance): %d of %d dims",
        spectrum["effective_rank_95pct"],
        vectors.shape[1],
    )

    client = s3_client(args.aws_profile)
    prefix = args.output_uri.rstrip("/")
    for path, content_type in (
        (atlas_path, "application/octet-stream"),
        (projection_path, "application/octet-stream"),
        (spectrum_path, "application/json"),
    ):
        bucket, _, key_prefix = prefix.removeprefix("s3://").partition("/")
        key = f"{key_prefix}/{path.name}"
        client.put_object(
            Bucket=bucket,
            Key=key,
            Body=path.read_bytes(),
            ContentType=content_type,
        )
        LOGGER.info("uploaded s3://%s/%s (%.1f MB)", bucket, key, path.stat().st_size / 1e6)


if __name__ == "__main__":
    main()
