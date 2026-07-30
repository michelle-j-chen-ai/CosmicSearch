"""Shared helpers for benchmark + corpus-builder tools.

`pca_basis` is the single canonical home for the uncentered-Gram PCA fit;
bench_threshold_search and tools/build_test_corpora both import it from here.
"""

from __future__ import annotations

import numpy as np

D = 256
MODEL_DIM = 768
_GRAM_BLOCK_ROWS = 100_000


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
