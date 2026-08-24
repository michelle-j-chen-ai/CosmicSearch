"""Build an exact-threshold Lance corpus from a fast-format npy corpus.

The fast corpus format (`embeddings.npy` + `metadata.parquet`, see
`local_cache.NPY_MATRIX_FILE`) stores raw fp32 768-d vectors and is what the
trucking service currently loads. The exact-threshold format is the int8 PCA
Lance dataset `threshold_search` scores against. This converts the former into
the latter: fit PCA-256 over the corpus, project, quantize, and hand the
artifact set to `lance_writer.build_dataset`.

Unlike the production artifact pipeline, the raw fp32 vectors are available
here, so the true pre-quantization projection is emitted as
`pca_projection_fp32.npy`. `vector_fp` then holds a real re-rank column rather
than dequantized int8, which is the condition `eps_bound.py`'s zero-false-
negative guarantee needs to bound the true score (see `lance_writer`'s module
docstring).

USAGE::

    PYTHONPATH=. python tools/build_threshold_corpus_from_npy.py \\
        --source-dir /path/to/corpus \\
        --out-dir    /path/to/build

`--source-dir` holds `embeddings.npy` + `metadata.parquet`; the Lance dataset
lands at `<out-dir>/corpus.lance` with the artifact set beside it. Writing to
S3 is deliberately not supported -- build locally, inspect, then upload.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import lance_writer
from bench_common import MODEL_DIM, int8_quantize, pca_basis
from local_cache import NPY_MATRIX_FILE

# Rows projected/quantized per block. The projection is (rows, 256) fp32 and the
# source block (rows, 768) fp32, so this bounds transient memory independently
# of corpus size.
_PROJECT_BLOCK_ROWS = 250_000

# A corpus row whose L2 norm is further than this from 1.0 was not normalized by
# the embedding pipeline. Cosine scoring downstream assumes unit rows, and the
# PCA scales are fit in whatever units the input carries, so a non-unit corpus
# silently changes the score scale the app's thresholds were calibrated in.
_NORM_TOLERANCE = 1e-3
_NORM_SAMPLE_ROWS = 20_000

_DASHED_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_TRAILING_NS = re.compile(r"(-\d{15,})+$")


def vehicle_from_segment_id(segment_id: str | None) -> str | None:
    """Derive the vehicle from neuron and frontier segment-id conventions.

    Frontier ids lead with a dashed date (`2026-02-27_10-05-15_truck-809-<ns>-<ns>`)
    and carry the vehicle last; neuron ids lead with the vehicle. Mirrors the
    derivation the embedding pipeline uses so a corpus built here filters the
    same way as one built there.
    """
    if not segment_id:
        return None
    core = _TRAILING_NS.sub("", segment_id)
    parts = core.split("_")
    if _DASHED_DATE.match(parts[0]):
        return parts[-1]
    return parts[0]


def assert_rows_normalized(embeddings: np.ndarray) -> float:
    """Check a sample of rows is unit-norm; return the worst deviation seen."""
    rng = np.random.default_rng(0)
    n = embeddings.shape[0]
    idx = rng.choice(n, size=min(_NORM_SAMPLE_ROWS, n), replace=False)
    norms = np.linalg.norm(np.asarray(embeddings[np.sort(idx)], dtype=np.float32), axis=1)
    worst = float(np.abs(norms - 1.0).max())
    if worst > _NORM_TOLERANCE:
        raise ValueError(
            f"corpus rows are not L2-normalized (worst |‖v‖-1| = {worst:.6f} over "
            f"{len(idx)} sampled rows). Scoring and the fitted quantization scales "
            f"both assume unit rows; normalize the source corpus first."
        )
    return worst


def project_and_quantize(
    embeddings: np.ndarray, pca: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project all rows through `pca`, then int8-quantize with per-dim scales.

    Returns `(projected fp32 (N, D), corpus_int8 (N, D), scale (D,))`. The
    projection is materialized in blocks and the scales are taken over the whole
    projection, matching `bench_common.int8_quantize` on a single array.
    """
    n = embeddings.shape[0]
    d = pca.shape[0]
    projected = np.empty((n, d), dtype=np.float32)
    for start in range(0, n, _PROJECT_BLOCK_ROWS):
        stop = min(start + _PROJECT_BLOCK_ROWS, n)
        block = np.asarray(embeddings[start:stop], dtype=np.float32)
        projected[start:stop] = block @ pca.T
    corpus_i8, scale = int8_quantize(projected)
    return projected, corpus_i8, scale


def build_metadata(metadata_path: Path, n_rows: int) -> pa.Table:
    """Read the source metadata and add the `vehicle` column the writer sorts on.

    The fast corpus carries no vehicle column, so filtering by vehicle is
    unavailable to a service reading it. Deriving one from `segment_id` here
    makes the built dataset match the columns `search_engine._vehicle_from_arrow`
    looks for, and gives `lance_writer` a real key for its BITMAP index.
    """
    meta = pq.read_table(metadata_path)
    if meta.num_rows != n_rows:
        raise ValueError(
            f"metadata row count ({meta.num_rows}) does not match embeddings row "
            f"count ({n_rows})"
        )
    if any(c in meta.column_names for c in lance_writer._VEHICLE_COLUMNS):
        return meta
    segment_ids = meta.column("segment_id").to_pylist()
    vehicles = [vehicle_from_segment_id(s) for s in segment_ids]
    return meta.append_column("vehicle", pa.array(vehicles, type=pa.string()))


def build(source_dir: Path, out_dir: Path) -> str:
    """Convert the fast corpus in `source_dir` into a Lance dataset in `out_dir`."""
    source_dir = Path(source_dir)
    out_dir = Path(out_dir)
    artifact_dir = out_dir / "artifact"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    embeddings = np.load(source_dir / NPY_MATRIX_FILE, mmap_mode="r")
    if embeddings.ndim != 2 or embeddings.shape[1] != MODEL_DIM:
        raise ValueError(
            f"expected an (N, {MODEL_DIM}) embedding matrix, got {embeddings.shape}"
        )
    n_rows = embeddings.shape[0]
    print(f"corpus: {n_rows:,} rows x {embeddings.shape[1]} dims ({embeddings.dtype})")

    # Resolve the scalar metadata before the fit: it is the cheap half, and a
    # missing file or a row-count mismatch should not cost a full PCA pass first.
    metadata = build_metadata(source_dir / lance_writer.METADATA_FILE, n_rows)
    print(f"metadata: {metadata.num_rows:,} rows, columns {metadata.column_names}")

    worst = assert_rows_normalized(embeddings)
    print(f"rows are unit-norm (worst deviation {worst:.2e})")

    print("fitting PCA-256 over the full corpus ...")
    pca = pca_basis(embeddings)
    print(f"pca_components: {pca.shape} {pca.dtype}")

    print("projecting + quantizing ...")
    projected, corpus_i8, scale = project_and_quantize(embeddings, pca)
    retained = float((projected.astype(np.float64) ** 2).sum() / n_rows)
    print(f"mean retained energy per row: {retained:.6f} of 1.0")

    np.save(artifact_dir / lance_writer.PCA_FILE, pca)
    np.save(artifact_dir / lance_writer.SCALE_FILE, scale)
    np.save(artifact_dir / lance_writer.CORPUS_INT8_FILE, corpus_i8)
    np.save(artifact_dir / lance_writer.PRE_QUANT_FP32_FILE, projected)
    pq.write_table(metadata, artifact_dir / lance_writer.METADATA_FILE)
    print(f"artifacts written to {artifact_dir}")

    corpus_uri = str(out_dir / "corpus.lance")
    print(f"building Lance dataset at {corpus_uri} ...")
    lance_writer.build_dataset(artifact_dir, corpus_uri)
    return corpus_uri


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        required=True,
        type=Path,
        help="directory holding embeddings.npy + metadata.parquet",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        type=Path,
        help="directory to write artifact/ and corpus.lance into",
    )
    args = parser.parse_args(argv)
    corpus_uri = build(args.source_dir, args.out_dir)
    print(f"done: {corpus_uri}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
