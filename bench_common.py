"""Shared helpers for benchmark + corpus-builder tools.

`pca_basis` is the single canonical home for the uncentered-Gram PCA fit;
`int8_quantize` is the single canonical per-dim symmetric quantize. The
synthetic/fixture/real embeddings generators and `build_corpus` also live here
so every bench and test imports them from one place.
"""

from __future__ import annotations

import ast
import io
import struct
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import lance_writer

D = 256
MODEL_DIM = 768
_GRAM_BLOCK_ROWS = 100_000

# Per-dim energy decay of the synthetic latent. Real Cosmos-Embed corpora are
# strongly concentrated in their leading dimensions; measured against
# v3_lr_5e5-ckpt-6549 (902,827 rows), the top 64 of 256 components hold 0.9798
# of the energy and the top 128 hold 0.9953. d**-0.95 reproduces that closely
# (0.9822 / 0.9938) and lands eps within ~4% of the real corpus's. Flat
# (uniform-energy) latents do not: they leave the top 64 at ~0.29 and shrink
# the per-dim quantization-scale spread from ~146x to ~1.4x, which is what the
# eps bound is sized against.
_SPECTRUM_DECAY = 0.95


def pca_basis(embeddings: np.ndarray) -> np.ndarray:
    """Energy-ordered orthonormal (D, MODEL_DIM) basis: uncentered SVD via Gram.

    The 768x768 Gram matrix makes this exact and dependency-free -- its top-D
    eigenvectors are the right singular vectors a truncated SVD would return.
    Accumulated in row blocks so the float64 working copy stays bounded rather
    than scaling with the corpus (a whole-matrix upcast is ~8 bytes/value,
    which at tens of millions of rows exceeds RAM).
    """
    gram = np.zeros((embeddings.shape[1], embeddings.shape[1]), dtype=np.float64)
    for start in range(0, embeddings.shape[0], _GRAM_BLOCK_ROWS):
        block = embeddings[start : start + _GRAM_BLOCK_ROWS].astype(np.float64)
        gram += block.T @ block
    _eigenvalues, eigenvectors = np.linalg.eigh(gram)
    return np.ascontiguousarray(eigenvectors[:, ::-1][:, :D].T.astype("float32"))


def int8_quantize(projected: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-dim symmetric int8 quantization.

    Returns ``(corpus_int8, scale)`` where ``corpus_int8`` is ``int8`` and
    ``scale`` is a ``(D,)`` fp32 array of per-dim max-abs. Dequantization is
    ``int8 * scale / 127``.
    """
    scale = np.abs(projected).max(axis=0).astype("float32")
    scale = np.where(scale == 0, np.float32(1.0), scale)  # avoid div-by-zero
    corpus_i8 = np.clip(np.round(projected * 127.0 / scale), -127, 127).astype(np.int8)
    return corpus_i8, scale


# A 10,000-row seeded random sample of the production corpus, stored as its
# PCA-256 projection plus the basis (derived from all 902,827 rows) rather
# than raw 768-d vectors: the corpus is rank-256, so reconstruction is exact
# to ~2e-4 in score -- 60x below the eps bound -- at a third of the bytes.
# Regenerate with tools/make_real_fixture.py.
FIXTURE_PATH = Path(__file__).parent / "tests" / "fixtures" / "real_corpus_sample.npz"


def _metadata_for(n: int) -> pa.Table:
    """Synthetic scalar metadata: it takes no part in scoring, only in the
    writer's sort/index and the fields a hit carries back."""
    return pa.table(
        {
            "run_uuid": pa.array([f"run-{i % 100}" for i in range(n)]),
            "chunk_start_unix": pa.array(np.arange(n, dtype="int64")),
            "segment_id": pa.array([f"seg-{i}" for i in range(n)]),
        }
    )


def synthetic_embeddings(n: int, seed: int = 0) -> tuple[np.ndarray, pa.Table]:
    """`n` unit-norm model-space rows with a realistic energy spectrum.

    Rows live IN a 256-d subspace, so the PCA-256 projection is exactly
    score-lossless -- a property the real corpus has too (its top 256 of 768
    components hold 1.00000 of the energy), and the precondition that makes
    the model-space and PCA-space scores comparable at all.
    """
    rng = np.random.default_rng(seed)
    basis, _ = np.linalg.qr(rng.standard_normal((MODEL_DIM, D)))
    pca = np.ascontiguousarray(basis.T.astype("float32"))  # (256, 768), energy-ordered

    decay = (np.arange(1, D + 1) ** -_SPECTRUM_DECAY).astype("float32")
    latent = (rng.standard_normal((n, D)) * decay).astype("float32")
    latent /= np.linalg.norm(latent, axis=1, keepdims=True)
    embeddings = (latent @ pca).astype("float32")

    return embeddings, _metadata_for(n)


def fixture_embeddings(rows: int | None = None) -> tuple[np.ndarray, pa.Table, np.ndarray]:
    """The committed real-corpus sample, reconstructed to model space.

    Returns `(embeddings, metadata, pca)` -- the basis comes with the fixture
    because it was derived from the full corpus, which a 10k sample cannot
    reproduce on its own.
    """
    with np.load(FIXTURE_PATH) as data:
        latent, pca = data["latent"], data["pca"]
    if rows:
        latent = latent[:rows]
    embeddings = np.ascontiguousarray(latent @ pca, dtype="float32")
    return embeddings, _metadata_for(embeddings.shape[0]), pca


def real_embeddings(source: str, rows: int) -> tuple[np.ndarray, pa.Table]:
    """Leading `rows` of a real `embeddings.npy` + its sibling metadata.

    `source` is a local path or an `s3://` URI; only the bytes for those rows
    are fetched, so pointing at a multi-GB corpus does not download it.
    """
    meta_source = source.rsplit("/", 1)[0] + "/metadata.parquet"
    if source.startswith("s3://"):
        import oci_s3

        client = oci_s3.s3_client()

        def _get(uri: str, byte_range: str | None = None) -> bytes:
            bucket, key = uri[len("s3://"):].split("/", 1)
            kwargs = {"Range": byte_range} if byte_range else {}
            return client.get_object(Bucket=bucket, Key=key, **kwargs)["Body"].read()

        head = _get(source, "bytes=0-255")
        preamble = 10 + struct.unpack("<H", head[8:10])[0]
        header = ast.literal_eval(head[10:preamble].decode().strip())
        n_total, dim = header["shape"]
        rows = min(rows, n_total)
        raw = _get(source, f"bytes={preamble}-{preamble + rows * dim * 4 - 1}")
        embeddings = np.frombuffer(raw, dtype=header["descr"]).reshape(rows, dim)
        metadata = pq.read_table(io.BytesIO(_get(meta_source)))[:rows]
    else:
        embeddings = np.array(np.load(source, mmap_mode="r")[:rows])
        metadata = pq.read_table(meta_source)[:rows]
        rows = embeddings.shape[0]

    columns = set(metadata.column_names)
    if "segment_id" not in columns:
        # The production artifact carries chunk_id, not segment_id; the writer
        # requires one per-row id it can index.
        metadata = metadata.append_column("segment_id", metadata.column("chunk_id"))
    return np.ascontiguousarray(embeddings, dtype="float32"), metadata.select(
        ["run_uuid", "chunk_start_unix", "segment_id"]
    )


def build_corpus(
    embeddings: np.ndarray, metadata: pa.Table, out_dir: Path, pca: np.ndarray
) -> str:
    """Quantize + write the exact-threshold dataset; returns its URI.

    Quantization goes through `int8_quantize` so there is one quantize path.
    """
    import lance

    artifact_dir = out_dir / "artifact"
    artifact_dir.mkdir(parents=True)

    projected = (embeddings @ pca.T).astype("float32")
    corpus_i8, scale = int8_quantize(projected)

    np.save(artifact_dir / lance_writer.PCA_FILE, pca)
    np.save(artifact_dir / lance_writer.SCALE_FILE, scale)
    np.save(artifact_dir / lance_writer.CORPUS_INT8_FILE, corpus_i8)
    # A genuine pre-quantization projection, so vector_fp is a real fp32
    # reference rather than dequant(int8) -- otherwise the exactness check
    # would only re-derive the screen in higher precision.
    np.save(artifact_dir / lance_writer.PRE_QUANT_FP32_FILE, projected)
    pq.write_table(metadata, artifact_dir / lance_writer.METADATA_FILE)

    uri = str(out_dir / "corpus.lance")
    lance_writer.build_dataset(artifact_dir, uri)
    return uri
