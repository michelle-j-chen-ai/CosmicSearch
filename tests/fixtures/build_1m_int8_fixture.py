"""One-time, offline fixture-generation tool: real ~1M-row embeddings -> a
pre-built Lance 2.1 dataset for `tests/test_real_corpus_integration.py`.

This is NOT invoked by the test suite at runtime and is NOT part of the
shipped production writer. `lance_writer.py` only ever converts an *existing*
`gpu_corpus`-style int8 PCA artifact (pca_components.npy + quant_scales.npy +
corpus_int8.npy + metadata.parquet) -- it deliberately never re-fits SVD.
`tests/test_real_corpus_integration.py` only ever opens an already-built
Lance 2.1 dataset (`lance.dataset(...)`) via the `NLS_REAL_CORPUS_LANCE_DIR`
env var; it never calls into this script. Whoever needs to (re)generate that
fixture -- e.g. because the source embeddings changed, or the fixture needs
to be copied to a new environment -- runs this script by hand, once, followed
by `lance_writer.build_dataset(...)` on its output, and stores the resulting
`.lance` directory wherever `NLS_REAL_CORPUS_LANCE_DIR` will point.

Regeneration procedure (run once, whenever the fixture needs rebuilding):
    1. Download the raw embeddings (see below).
    2. Run this script to produce the 3 int8-artifact files (uncentered SVD +
       per-dim int8 quant) -- this is the one live SVD fit, done here, not in
       the test.
    3. Build the Lance 2.1 dataset from those artifacts:
           python3 -c "import lance_writer; lance_writer.build_dataset(
               '<artifact-dir>', '/path/to/nls_real_corpus.lance')"
    4. Point `NLS_REAL_CORPUS_LANCE_DIR` at `/path/to/nls_real_corpus.lance`
       when running `tests/test_real_corpus_integration.py`.

Input data (download once, not committed):
    aws --profile oci.phx --region us-phoenix-1 \\
        --endpoint-url https://idskhu5vqvtl.compat.objectstorage.us-phoenix-1.oraclecloud.com \\
        s3 cp s3://neuron-prod-data-intelligence-exploratory/sibogeng/nls_search/embeddings/v3_lr_5e5-ckpt-6549_npy/embeddings.npy <dest>
    aws --profile oci.phx --region us-phoenix-1 \\
        --endpoint-url https://idskhu5vqvtl.compat.objectstorage.us-phoenix-1.oraclecloud.com \\
        s3 cp s3://neuron-prod-data-intelligence-exploratory/sibogeng/nls_search/embeddings/v3_lr_5e5-ckpt-6549_npy/metadata.parquet <dest>

`embeddings.npy`: (902827, 768) fp32, C order, L2-unit-norm rows.
`metadata.parquet`: columns chunk_id, run_uuid, chunk_start_unix,
source_media_uri (902827 rows). No segment_id/vehicle/chunk_end_unix -- see
the metadata-mapping notes below.

Usage (run once; ~1-2 minutes on a modern multi-core CPU, needs scikit-learn):
    python3 tests/fixtures/build_1m_int8_fixture.py \\
        --embeddings-npy <dest>/embeddings.npy \\
        --metadata-parquet <dest>/metadata.parquet \\
        --out-dir <artifact-dir>

Conversion, matching gpu_corpus.py's documented convention exactly:
  * uncentered truncated SVD, 768 -> 256 (`sklearn.decomposition.TruncatedSVD`,
    which -- unlike PCA -- does not mean-center before the decomposition, so
    the projected dot product still equals the original cosine: score-lossless
    per gpu_corpus.py's module docstring).
  * per-dim symmetric int8 quantization: `scale_d = max(|projected[:, d]|)`,
    `int8 = round(projected * 127 / scale)`, dequant = `int8 * scale / 127`
    (byte-identical formula to gpu_corpus.py / lance_writer.py / eps_bound.py).

Metadata mapping (real data lacks segment_id/vehicle/chunk_end_unix):
  * run_uuid, chunk_start_unix: copied directly from metadata.parquet.
  * segment_id: `lance_writer.build_table` requires this column and uses it
    only as an opaque per-row identifier (not parsed). `chunk_id` in the real
    metadata is already a unique, row-aligned string
    (f"{run_uuid}#t{chunk_start_unix}"), so it is reused as segment_id
    directly -- no synthetic ID needed.
  * vehicle: absent; `lance_writer._read_metadata_table` already treats
    `vehicle` as optional (falls back to an all-null string column when none
    of `vehicle`/`vehicle_name`/`vehicle_id` are present), so it is simply
    omitted here rather than filled with a placeholder value.
  * chunk_end_unix: absent; also optional (`lance_writer` fills it with nulls
    when missing), omitted here for the same reason.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

PCA_FILE = "pca_components.npy"
SCALE_FILE = "quant_scales.npy"
CORPUS_INT8_FILE = "corpus_int8.npy"
METADATA_FILE = "metadata.parquet"

_N_COMPONENTS = 256


def _quantize_per_dim(projected: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-dim symmetric int8 quant: scale_d = max|v[:, d]|, matching gpu_corpus."""
    scale = np.abs(projected).max(axis=0).astype("float32")
    scale = np.where(scale == 0, 1.0, scale)  # guard degenerate all-zero dims
    corpus_i8 = np.round(projected * (127.0 / scale)).clip(-127, 127).astype("int8")
    return corpus_i8, scale


def build_fixture(
    embeddings_npy: Path, metadata_parquet: Path, out_dir: Path, seed: int = 0
) -> None:
    from sklearn.decomposition import TruncatedSVD

    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    embeddings = np.load(embeddings_npy)  # (N, 768) fp32, full load (~2.8GB)
    n, d_model = embeddings.shape
    print(f"loaded embeddings: shape={embeddings.shape} dtype={embeddings.dtype} "
          f"({time.time() - t0:.1f}s)")

    t0 = time.time()
    svd = TruncatedSVD(n_components=_N_COMPONENTS, algorithm="randomized", random_state=seed)
    projected = svd.fit_transform(embeddings).astype("float32")  # (N, 256)
    pca = svd.components_.astype("float32")  # (256, 768), rows orthonormal
    print(f"fit truncated SVD 768->{_N_COMPONENTS}: explained_variance_ratio sum="
          f"{svd.explained_variance_ratio_.sum():.6f} ({time.time() - t0:.1f}s)")

    corpus_i8, scale = _quantize_per_dim(projected)

    np.save(out_dir / PCA_FILE, pca)
    np.save(out_dir / SCALE_FILE, scale)
    np.save(out_dir / CORPUS_INT8_FILE, corpus_i8)

    meta = pq.read_table(metadata_parquet)
    if meta.num_rows != n:
        raise ValueError(
            f"metadata row count ({meta.num_rows}) != embeddings row count ({n})"
        )
    out_meta = pa.table(
        {
            "run_uuid": meta.column("run_uuid"),
            "chunk_start_unix": meta.column("chunk_start_unix").cast(pa.int64()),
            "segment_id": meta.column("chunk_id"),  # unique, row-aligned; see docstring
        }
    )
    pq.write_table(out_meta, out_dir / METADATA_FILE)

    print(f"wrote fixture artifacts to {out_dir}: "
          f"corpus_int8={corpus_i8.shape} pca={pca.shape} scale={scale.shape}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings-npy", type=Path, required=True)
    parser.add_argument("--metadata-parquet", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    build_fixture(args.embeddings_npy, args.metadata_parquet, args.out_dir, seed=args.seed)


if __name__ == "__main__":
    main()
