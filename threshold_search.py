"""Single-node, provably-complete cosine-threshold scan over a Lance dataset.

The threshold-retrieval workload needs ALL rows with score >= tau, not an
approximate top-k (see `eps_bound.py`). This module wires the two accepted
pieces together against a real `lance_writer.build_dataset` output:

  1. Screen the resident `embedding_i8` column with `gpu_corpus`'s numba int8
     kernel (the same ~49GB/s kernel the GPU-corpus CPU backend uses) at
     `tau - eps`, where `eps` is `eps_bound.eps_cauchy_schwarz`'s hard error
     bound. This can only produce false positives, never false negatives
     (against `vector_fp` -- see `eps_bound.py`'s module docstring for the
     precise scope of that guarantee).
  2. `eps_bound.classify` splits the screened scores into ABOVE (guaranteed
     member) / BAND (ambiguous, needs an exact score) / BELOW (provably
     excluded, never touched again).
  3. The dataset's `vector_fp` column is this system's re-rank reference (see
     `lance_writer.py`'s module docstring for what it holds and when it is a
     true pre-quantization signal vs a fallback dequantized-int8 value).
     ABOVE and BAND rows both get their exact score recomputed from
     `vector_fp` via `Dataset.take()` -- one IOP per row, never a full-column
     read. Both ABOVE and BAND are then filtered to `score >= tau`: BAND
     because the eps window does not by itself decide membership, and ABOVE
     as a cheap unconditional safety net so "every returned score >= tau"
     never rests solely on the (empirically large but formally unbounded)
     margin between the fastmath-fp32 screening kernel and the fp64 re-rank.
  4. Return ABOVE union filtered-BAND, sorted by score descending, with each
     hit's scalar metadata (fetched via the same `take()` pass).

`ThresholdCorpus` decodes and holds `embedding_i8` resident in memory once, at
construction, so repeated `.threshold_search()` calls scan an in-memory array
rather than re-reading the column from the dataset on every query (the
`design.md` "resident int8 shard" architecture, at single-node scope). The
module-level `threshold_search()` function is the standalone entry point used
by tests and one-off callers; it decodes fresh on every call and does not
cache anything across calls.

`rank_top_k` / `score_corpus` in `search_engine.py` are a different code path
(approximate top-k over a resident matrix) and are untouched by this module.

No date/segment/vehicle prefilter yet: `threshold_search` always screens the
entire `embedding_i8` column, so the writer's physical sort and its
BTREE/BITMAP scalar indices (`lance_writer.build_dataset`) are not consulted
here. Deliberately deferred rather than added unsafely: a filtered scan's
row positions no longer address the dataset the way `Dataset.take()` expects
(see `_load_resident_corpus`'s docstring on why `scan_in_order=True` on an
UNfiltered scan is what makes positional indexing safe); resolving a
filtered scan's rows back to `take()`-addressable positions needs either
Lance's stable row ids (design.md explicitly avoids them -- "experimental")
or its internal, unstable `_take_rows` row-address API. Implement this only
by adopting one of those two mechanisms deliberately, not by passing
`filter=` to the `embedding_i8` scan below.
"""

from __future__ import annotations

import dataclasses

import lance
import numpy as np

import eps_bound
import gpu_corpus
import lance_writer

# Scalar columns fetched alongside the exact re-rank score so a ThresholdHit
# can be resolved to a clip without a second pass over the dataset.
_METADATA_COLUMNS = ("run_uuid", "segment_id", "chunk_start_unix", "chunk_end_unix", "vehicle")

# eps_cauchy_schwarz's bound assumes ||query||_2 == 1 (see eps_bound.py); this
# is the tolerance for treating a query as unit-norm.
_UNIT_NORM_TOLERANCE = 1e-3


@dataclasses.dataclass(frozen=True)
class ThresholdHit:
    """One row of a `threshold_search` result."""

    # Position of this row in the dataset's canonical (fragment, in-fragment)
    # scan order at the time of the query. Valid only against the same open
    # dataset version -- compaction or a new append changes fragment layout
    # and invalidates it as an address into a later version.
    row_id: int
    # Exact re-rank score, i.e. `vector_fp[row_id] . query_pca` (see
    # `lance_writer.py`'s module docstring for what `vector_fp` holds).
    score: float
    run_uuid: str
    segment_id: str
    chunk_start_unix: int
    chunk_end_unix: int | None
    vehicle: str | None


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


def _assert_unit_norm(query: np.ndarray) -> None:
    norm = float(np.linalg.norm(query))
    if abs(norm - 1.0) > _UNIT_NORM_TOLERANCE:
        raise ValueError(
            f"threshold_search requires a unit-norm query -- "
            f"eps_bound.eps_cauchy_schwarz's bound assumes ||query||_2 == 1 "
            f"(see its docstring); got ||query||_2 = {norm:.6f}"
        )


@dataclasses.dataclass(frozen=True)
class _ResidentCorpus:
    """Decoded-once resident int8 screen matrix + PCA metadata for one dataset.

    Building this is the only full read of `embedding_i8`; `ThresholdCorpus`
    builds it exactly once in `__init__` and every subsequent
    `.threshold_search()` call reuses it.
    """

    corpus_i8: np.ndarray  # (N, D) int8, resident
    pca: np.ndarray  # (D, model_dim) fp32
    scale: np.ndarray  # (D,) fp32
    eps: float


def _load_resident_corpus(dataset: lance.LanceDataset) -> _ResidentCorpus:
    """Decode `dataset`'s `embedding_i8` column + PCA metadata into memory.

    `scan_in_order=True` is required, not incidental: `Dataset.take()`
    addresses rows by position in the dataset's canonical (fragment,
    in-fragment) order, and callers resolve `take()` indices from *positions
    in this scan's output*. Without pinning in-order scanning, a
    multi-fragment out-of-order scan (`scan_in_order=False`, which
    `design.md`'s Scan tuning section recommends for throughput) can return
    rows in a different order than `take()` addresses them, silently
    mismatching every screened index against the wrong row's `vector_fp` --
    see `tests/test_threshold_search.py`'s multi-fragment regression test.
    """
    pca, scale = lance_writer.read_pca_metadata(dataset)
    i8_table = dataset.to_table(columns=[lance_writer.EMBEDDING_I8_COLUMN], scan_in_order=True)
    corpus_i8 = _fixed_size_list_matrix(i8_table, lance_writer.EMBEDDING_I8_COLUMN, np.int8)
    eps = eps_bound.eps_cauchy_schwarz(scale)
    return _ResidentCorpus(corpus_i8=corpus_i8, pca=pca, scale=scale, eps=eps)


def _search_resident(
    query: np.ndarray, tau: float, dataset: lance.LanceDataset, resident: _ResidentCorpus
) -> list[ThresholdHit]:
    """Screen+re-rank `query` against an already-decoded `_ResidentCorpus`."""
    _assert_unit_norm(query)
    n = resident.corpus_i8.shape[0]
    if n == 0:
        return []

    query_pca = _project_query(query, resident.pca)
    w = (query_pca * (resident.scale.astype(np.float32) / np.float32(127.0))).astype(np.float32)
    screening_scores = np.empty(n, dtype=np.float32)
    gpu_corpus._cpu_score_kernel()(np.ascontiguousarray(resident.corpus_i8), w, screening_scores)

    above, band, _below = eps_bound.classify(screening_scores, tau, resident.eps)
    above_idx = np.nonzero(above)[0]
    band_idx = np.nonzero(band)[0]

    # Exact re-rank via take() on vector_fp -- 1 IOP/row, never a full column
    # read. ABOVE and BAND are each typically a tiny fraction of the corpus
    # for a selective query, so this is bounded by the match count, not
    # corpus size (a broad/low-tau query can still make this large -- see
    # design.md's latency budget, which this module does not attempt to cap).
    def _exact_scores(idx: np.ndarray) -> np.ndarray:
        if idx.size == 0:
            return np.empty(0, dtype=np.float64)
        rows = dataset.take(idx.tolist(), columns=[lance_writer.VECTOR_FP_COLUMN])
        fp = _fixed_size_list_matrix(rows, lance_writer.VECTOR_FP_COLUMN, np.float32)
        # Accumulate in float64 without materializing a float64 copy of the
        # take() result. Reading the column as float64 instead costs more in
        # the upcast than the dot product itself, and dominates this stage
        # once a query matches thousands of rows; `dtype` keeps the
        # accumulator at full precision, so scores are unchanged.
        return np.einsum("ij,j->i", fp, query_pca, dtype=np.float64)

    above_scores = _exact_scores(above_idx)
    band_scores = _exact_scores(band_idx)

    # Filter both ABOVE and BAND to score >= tau. BAND needs it because the
    # eps window alone does not decide membership; ABOVE is filtered too as a
    # cheap, unconditional safety net (see module docstring point 3).
    above_keep = above_scores >= tau
    above_idx = above_idx[above_keep]
    above_scores = above_scores[above_keep]
    band_keep = band_scores >= tau
    band_idx = band_idx[band_keep]
    band_scores = band_scores[band_keep]

    all_idx = np.concatenate([above_idx, band_idx])
    all_scores = np.concatenate([above_scores, band_scores])
    order = np.argsort(-all_scores, kind="stable")
    sorted_idx = all_idx[order]
    sorted_scores = all_scores[order]

    if sorted_idx.size == 0:
        return []

    meta_rows = dataset.take(sorted_idx.tolist(), columns=list(_METADATA_COLUMNS))
    meta = {name: meta_rows.column(name).to_pylist() for name in _METADATA_COLUMNS}
    return [
        ThresholdHit(
            row_id=int(sorted_idx[i]),
            score=float(sorted_scores[i]),
            run_uuid=meta["run_uuid"][i],
            segment_id=meta["segment_id"][i],
            chunk_start_unix=meta["chunk_start_unix"][i],
            chunk_end_unix=meta["chunk_end_unix"][i],
            vehicle=meta["vehicle"][i],
        )
        for i in range(sorted_idx.size)
    ]


def threshold_search(
    query: np.ndarray, tau: float, dataset: lance.LanceDataset
) -> list[ThresholdHit]:
    """Every row of `dataset` with re-rank score >= `tau`.

    `query` is a unit-norm vector in the original (pre-PCA) model space, same
    convention as `gpu_corpus.GpuCorpus.gpu_score`. `dataset` must be an
    exact-threshold dataset (see `lance_writer.build_dataset`).

    Zero false negatives (against `vector_fp`; see `eps_bound.py`'s module
    docstring for what that guarantees against the true pre-quantization
    score): `eps_bound.eps_cauchy_schwarz` is a hard upper bound on the gap
    between the int8 screening score and `vector_fp`'s score for a unit-norm
    query, so screening at `tau - eps` can only ever produce extra rows to
    re-rank (BAND), never drop a row whose exact score is >= tau. Sorted by
    score descending.

    This is the standalone entry point: it decodes `embedding_i8` fresh on
    every call. A caller making repeated queries against the same dataset
    should use `ThresholdCorpus` instead, which decodes once and reuses the
    resident matrix across queries.
    """
    if not lance_writer.is_exact_threshold_dataset(dataset):
        raise ValueError(
            "threshold_search requires an exact-threshold Lance dataset "
            "(embedding_i8 + vector_fp columns, data_storage_version >= "
            f"'{lance_writer.MIN_DATA_STORAGE_VERSION}', PCA schema metadata present)"
        )
    resident = _load_resident_corpus(dataset)
    return _search_resident(query, tau, dataset, resident)


class ThresholdCorpus:
    """Duck-type corpus wrapper around an exact-threshold Lance dataset.

    Decodes and holds `embedding_i8` resident in memory once, at
    construction -- repeated `.threshold_search()` calls scan that in-memory
    array rather than re-reading the column from the dataset every query.

    This is NOT a drop-in replacement for `search_engine.Corpus`: it has no
    resident 768-d matrix, so it cannot support `rank_top_k` / `score_corpus`
    (top-k ranking is a different workload with a different residency shape).
    Callers doing threshold retrieval get one from
    `search_engine.load_threshold_corpus`, a separate entry point from the
    shared `search_engine.load_corpus` (which never returns a
    `ThresholdCorpus`, so every existing `Corpus`-typed consumer keeps working
    unchanged).
    """

    def __init__(self, dataset: lance.LanceDataset) -> None:
        if not lance_writer.is_exact_threshold_dataset(dataset):
            raise ValueError(
                "ThresholdCorpus requires an exact-threshold Lance dataset "
                "(embedding_i8 + vector_fp columns, data_storage_version >= "
                f"'{lance_writer.MIN_DATA_STORAGE_VERSION}', PCA schema metadata present)"
            )
        self._dataset = dataset
        self._resident = _load_resident_corpus(dataset)

    @property
    def num_rows(self) -> int:
        return self._resident.corpus_i8.shape[0]

    def threshold_search(self, query: np.ndarray, tau: float) -> list[ThresholdHit]:
        """Every row with re-rank score >= `tau`; see `threshold_search`."""
        return _search_resident(query, tau, self._dataset, self._resident)
