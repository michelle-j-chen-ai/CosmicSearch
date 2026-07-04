"""Single-node, provably-complete cosine-threshold scan over a Lance 2.1 dataset.

The threshold-retrieval workload needs ALL rows with score >= tau, not an
approximate top-k (see `eps_bound.py`). This module wires the two accepted
pieces together against a real `lance_writer.build_dataset` output:

  1. Screen the resident `embedding_i8` column with `gpu_corpus`'s numba int8
     kernel (the same ~49GB/s kernel the GPU-corpus CPU backend uses) at
     `tau - eps`, where `eps` is `eps_bound.eps_cauchy_schwarz`'s hard error
     bound. This can only produce false positives, never false negatives.
  2. `eps_bound.classify` splits the screened scores into ABOVE (guaranteed
     member) / BAND (ambiguous, needs an exact score) / BELOW (provably
     excluded, never touched again).
  3. The dataset's `vector_fp` column is this system's exact, full-precision
     score reference (see `lance_writer.py`'s module docstring: it is derived
     from the same int8 artifact, not an independently-stored pre-quantization
     corpus, since no such corpus is available to this builder). ABOVE and
     BAND rows both get their exact score recomputed from `vector_fp` via
     `Dataset.take()` -- one IOP per row, never a full-column read -- so every
     returned score is this exact reference value rather than the int8
     screening approximation. BAND rows are then filtered to `score >= tau`;
     ABOVE rows need no filtering (the eps bound already guarantees it) but
     are re-scored for consistency.
  4. Return ABOVE union filtered-BAND, sorted by score descending.

`rank_top_k` / `score_corpus` in `search_engine.py` are a different code path
(approximate top-k over a resident matrix) and are untouched by this module.
"""

from __future__ import annotations

import dataclasses

import lance
import numpy as np

import eps_bound
import gpu_corpus
import lance_writer


@dataclasses.dataclass(frozen=True)
class ThresholdHit:
    """One row of a `threshold_search` result."""

    # 0-based row index into the dataset (stable for a given dataset version;
    # matches the row order `Dataset.take()` addresses).
    row_id: int
    # Exact fp32 cosine score, i.e. `vector_fp[row_id] . query_pca`.
    score: float


def _fixed_size_list_matrix(table: object, column: str, dtype: np.dtype) -> np.ndarray:
    """(N, D) ndarray from an Arrow table's FixedSizeList column."""
    col = table.column(column).combine_chunks()
    n = len(col)
    d = col.type.list_size
    flat = np.asarray(col.values.to_numpy(zero_copy_only=False), dtype=dtype)
    return flat.reshape(n, d)


def _project_query(query: np.ndarray, pca: np.ndarray) -> np.ndarray:
    """Project a raw model-space `query` into PCA-256 space (fp32).

    Shared by `threshold_search` and its test's brute-force oracle so both
    score every row from the identical fp32 query projection -- otherwise an
    independently-rounded oracle projection would make an exact score
    comparison meaningless.
    """
    query = np.ascontiguousarray(query, dtype=np.float32)
    return (pca.astype(np.float32) @ query).astype(np.float32)


def threshold_search(
    query: np.ndarray, tau: float, dataset: lance.LanceDataset
) -> list[ThresholdHit]:
    """Every row of `dataset` with exact fp32 cosine score >= `tau`.

    `query` is a unit-norm vector in the original (pre-PCA) model space, same
    convention as `gpu_corpus.GpuCorpus.gpu_score`. `dataset` must be a Lance
    2.1 exact-threshold dataset (see `lance_writer.build_dataset`).

    Zero false negatives: `eps_bound.eps_cauchy_schwarz` is a hard upper bound
    on the gap between the int8 screening score and the `vector_fp` exact
    score for a unit-norm query, so screening at `tau - eps` can only ever
    produce extra rows to re-rank (BAND), never drop a row whose exact score
    is >= tau. Sorted by score descending.
    """
    if not lance_writer.is_v21_dataset(dataset):
        raise ValueError(
            "threshold_search requires a Lance 2.1 exact-threshold dataset "
            "(embedding_i8 + vector_fp columns, data_storage_version == '2.1')"
        )

    n = dataset.count_rows()
    if n == 0:
        return []

    pca, scale = lance_writer.read_pca_metadata(dataset)
    query_pca = _project_query(query, pca)  # (D,)

    # 1) Screen the resident int8 column with gpu_corpus's numba kernel -- the
    # same kernel the CPU GpuCorpus backend uses, reused rather than
    # reimplemented.
    i8_table = dataset.to_table(columns=[lance_writer.EMBEDDING_I8_COLUMN])
    corpus_i8 = _fixed_size_list_matrix(i8_table, lance_writer.EMBEDDING_I8_COLUMN, np.int8)
    w = (query_pca * (scale.astype(np.float32) / 127.0)).astype(np.float32)
    screening_scores = np.empty(n, dtype=np.float32)
    gpu_corpus._cpu_score_kernel()(np.ascontiguousarray(corpus_i8), w, screening_scores)

    eps = eps_bound.eps_cauchy_schwarz(scale)
    above, band, _below = eps_bound.classify(screening_scores, tau, eps)
    above_idx = np.nonzero(above)[0]
    band_idx = np.nonzero(band)[0]

    # 2) Exact re-rank via take() on vector_fp -- 1 IOP/row, never a full
    # column read. ABOVE and BAND are each typically a tiny fraction of the
    # corpus, so this is bounded by the match count, not corpus size.
    query_pca64 = query_pca.astype(np.float64)

    def _exact_scores(idx: np.ndarray) -> np.ndarray:
        if idx.size == 0:
            return np.empty(0, dtype=np.float64)
        rows = dataset.take(idx.tolist(), columns=[lance_writer.VECTOR_FP_COLUMN])
        fp = _fixed_size_list_matrix(rows, lance_writer.VECTOR_FP_COLUMN, np.float64)
        return fp @ query_pca64

    above_scores = _exact_scores(above_idx)
    band_scores = _exact_scores(band_idx)
    band_keep = band_scores >= tau
    band_idx = band_idx[band_keep]
    band_scores = band_scores[band_keep]

    all_idx = np.concatenate([above_idx, band_idx])
    all_scores = np.concatenate([above_scores, band_scores])
    order = np.argsort(-all_scores, kind="stable")
    return [
        ThresholdHit(row_id=int(all_idx[i]), score=float(all_scores[i])) for i in order
    ]
