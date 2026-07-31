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
     true pre-quantization signal vs a fallback dequantized-int8 value). The
     shipped default (`fast_curation=True`) accepts ABOVE rows with their
     bounded int8 screening score and no `take()` re-rank -- membership is
     proven by the eps bound (`screen - eps >= tau`), so the screen score is a
     bounded approximation, not an exact value. BAND rows are always re-ranked
     from `vector_fp` via `Dataset.take()` (one IOP per row, never a full-column
     read) and filtered to `score >= tau`, because the eps window alone does
     not decide their membership. The exact path (`fast_curation=False`)
     re-ranks ABOVE too and filters it to `score >= tau`; it is the internal
     reference used for benchmark comparison. The eps bound is proven against
     `vector_fp`, but the fastmath-fp32 screening kernel's own numeric error
     relative to that bound is empirically large yet formally unbounded -- the
     exact path exists precisely to measure that gap.
  4. Return ABOVE union filtered-BAND (the match set, in canonical row order),
     hit's scalar metadata (fetched via the same `take()` pass).

`ThresholdCorpus` decodes and holds `embedding_i8` resident in memory once, at
construction, so repeated `.threshold_search()` calls scan an in-memory array
rather than re-reading the column from the dataset on every query (resident
int8 screen, out-of-core `vector_fp` refine — single-node scope). The
module-level `threshold_search()` function is the standalone entry point used
by tests and one-off callers; it decodes fresh on every call and does not
cache anything across calls.

`rank_top_k` / `score_corpus` in `search_engine.py` are a different code path
(approximate top-k over a resident matrix) and are untouched by this module.

Pre-retrieval filters (`vehicle`, `date_range`, `run_uuids`) are applied
before screening via a canonical-order resident mask: the int8 screen runs
only over surviving rows, but every index handed to `Dataset.take()` or
emitted as a `ThresholdHit.row_id` is a canonical position (see
`_filter_mask` and the subset-to-canonical remap in `_search_resident`).
This keeps positional `take()` addressing safe -- the invariant
`_load_resident_corpus`'s `scan_in_order=True` establishes for the
unfiltered path is preserved under a filter by remapping subset positions
back to canonical ids before any `take()`/metadata/`row_id` use. Lance
`filter=` scans remain unused; pushing the predicate into a Lance scalar
index scan is a separate concern for out-of-core corpora and is not needed
here because the int8 screen is already resident.
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

# Tolerance for treating a model-space query as unit-norm (see _assert_unit_norm).
_UNIT_NORM_TOLERANCE = 1e-3


@dataclasses.dataclass(frozen=True)
class ThresholdHit:
    """One row of a `threshold_search` result."""

    # Position of this row in the dataset's canonical (fragment, in-fragment)
    # scan order at the time of the query. Valid only against the same open
    # dataset version -- compaction or a new append changes fragment layout
    # and invalidates it as an address into a later version.
    row_id: int
    # Re-rank score. For score_kind="exact" this is `vector_fp[row_id] .
    # query_pca`; for score_kind="bounded_approx" it is the int8 screening
    # score (see `score_kind` / `score_error_bound` below).
    score: float
    run_uuid: str
    segment_id: str
    chunk_start_unix: int
    chunk_end_unix: int | None
    vehicle: str | None
    # "exact" = recomputed from `vector_fp` via take(); "bounded_approx" = the
    # int8 screening score, kept without a take() in fast_curation mode (ABOVE
    # rows only). For bounded rows, the true score lies within
    # [score - score_error_bound, score + score_error_bound].
    score_kind: str = "exact"
    score_error_bound: float = 0.0


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
    """Reject a query that is not unit-norm in model space.

    Scores are cosine similarities only for a unit-norm query, so `tau` is
    not comparable across queries otherwise. (The eps bound itself no longer
    depends on this -- `_search_resident` scales it by the projected query's
    norm -- but a caller passing an unnormalized query is choosing a
    threshold against a different scale than they think.)
    """
    norm = float(np.linalg.norm(query))
    if abs(norm - 1.0) > _UNIT_NORM_TOLERANCE:
        raise ValueError(
            f"threshold_search requires a unit-norm query so that scores are "
            f"cosine similarities and tau means the same thing across "
            f"queries; got ||query||_2 = {norm:.6f}"
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
    vehicle: np.ndarray | None  # (N,) object (str or None); None in hybrid mode
    chunk_start_unix: np.ndarray | None  # (N,) int64; None in hybrid mode
    run_uuid: np.ndarray | None  # (N,) object; None in hybrid mode
    row_ids: np.ndarray | None  # (N,) int64 ascending; None in resident mode


def _assert_ascending_row_ids(row_ids: np.ndarray) -> None:
    """Hard-guard: the row-id map must be strictly ascending.

    `scan_in_order=True` yields fragments in fragment-list order, and a
    compaction can interleave ids so scan-order row ids descend. `searchsorted`
    over a non-ascending array returns silently wrong positions -- a
    zero-false-dismissal violation. This guard turns that into a loud error.
    """
    if row_ids.size > 1 and not np.all(np.diff(row_ids) > 0):
        raise ValueError(
            "resident row_ids are not strictly ascending -- a compaction likely "
            "interleaved fragments; the hybrid prefilter cannot safely map "
            "filtered row ids to resident-screen positions. Rebuild the dataset "
            "or use prefilter_mode='resident'."
        )


def _load_resident_corpus(
    dataset: lance.LanceDataset, *, capture_row_ids: bool = False
) -> _ResidentCorpus:
    """Decode `dataset`'s `embedding_i8` column + PCA metadata into memory.

    `scan_in_order=True` is required, not incidental: `Dataset.take()`
    addresses rows by position in the dataset's canonical (fragment,
    in-fragment) order, and callers resolve `take()` indices from *positions
    in this scan's output*. Without pinning in-order scanning, a
    multi-fragment out-of-order scan (`scan_in_order=False`, often tempting
    for throughput) can return rows in a different order than `take()`
    addresses them, silently mismatching every screened index against the
    wrong row's `vector_fp` -- see `tests/test_threshold_search.py`'s
    multi-fragment regression test.

    `capture_row_ids=True` (hybrid mode) reads the screen column with
    `with_row_id=True` so the row-id -> position map is captured for filter
    pushdown, and skips the metadata `to_table` entirely (the metadata filter
    arrays are not held in hybrid mode -- the predicate is pushed to Lance).
    """
    pca, scale = lance_writer.read_pca_metadata(dataset)
    if capture_row_ids:
        i8_table = dataset.to_table(
            columns=[lance_writer.EMBEDDING_I8_COLUMN],
            with_row_id=True, scan_in_order=True,
        )
        corpus_i8 = _fixed_size_list_matrix(i8_table, lance_writer.EMBEDDING_I8_COLUMN, np.int8)
        # pylance 0.34 exposes the physical row id as `_rowid` (uint64).
        row_ids = i8_table.column("_rowid").to_numpy().astype(np.int64)
        _assert_ascending_row_ids(row_ids)
        return _ResidentCorpus(
            corpus_i8=corpus_i8, pca=pca, scale=scale,
            eps=eps_bound.eps_cauchy_schwarz(scale),
            vehicle=None, chunk_start_unix=None, run_uuid=None, row_ids=row_ids,
        )
    i8_table = dataset.to_table(columns=[lance_writer.EMBEDDING_I8_COLUMN], scan_in_order=True)
    corpus_i8 = _fixed_size_list_matrix(i8_table, lance_writer.EMBEDDING_I8_COLUMN, np.int8)
    meta_table = dataset.to_table(
        columns=["vehicle", "chunk_start_unix", "run_uuid"], scan_in_order=True
    )
    return _ResidentCorpus(
        corpus_i8=corpus_i8, pca=pca, scale=scale,
        eps=eps_bound.eps_cauchy_schwarz(scale),
        vehicle=np.array(meta_table.column("vehicle").to_pylist()),
        chunk_start_unix=np.array(meta_table.column("chunk_start_unix").to_pylist()),
        run_uuid=np.array(meta_table.column("run_uuid").to_pylist()),
        row_ids=None,
    )


def _filter_mask(
    resident: _ResidentCorpus,
    vehicle: str | None,
    date_range: tuple[int | None, int | None] | None,
    run_uuids: "set[str] | None",
) -> np.ndarray | None:
    """AND of the given filters over canonical order; None when no filter given."""
    if vehicle is None and date_range is None and run_uuids is None:
        return None
    n = resident.corpus_i8.shape[0]
    mask = np.ones(n, dtype=bool)
    if vehicle is not None:
        mask &= resident.vehicle == vehicle
    if date_range is not None:
        lo, hi = date_range
        if lo is not None:
            mask &= resident.chunk_start_unix >= lo
        if hi is not None:
            mask &= resident.chunk_start_unix < hi
    if run_uuids is not None:
        mask &= np.isin(resident.run_uuid, list(run_uuids))
    return mask


def _lance_filter_sql(
    vehicle: str | None,
    date_range: tuple[int | None, int | None] | None,
    run_uuids: "set[str] | None",
) -> str | None:
    """Build a Lance (DataFusion) SQL predicate mirroring `_filter_mask`.

    Returns `None` (no filter / empty match) when no clause is given or a set
    filter is empty. Single-quotes in string literals are doubled to escape
    them. NULL vehicles are excluded by SQL `=`, matching the numpy
    object-array `==` mask (NULL != 'veh_a' on both sides). Boundary
    inclusivity matches `_filter_mask`: `>= lo`, `< hi`.
    """
    if vehicle is None and date_range is None and run_uuids is None:
        return None
    clauses: list[str] = []
    if vehicle is not None:
        clauses.append(f"vehicle = '{vehicle.replace(chr(39), chr(39)*2)}'")
    if date_range is not None:
        lo, hi = date_range
        if lo is not None:
            clauses.append(f"chunk_start_unix >= {int(lo)}")
        if hi is not None:
            clauses.append(f"chunk_start_unix < {int(hi)}")
    if run_uuids is not None:
        # An empty set matches nothing (parity with np.isin(x, [])); emit a
        # literal false rather than invalid `IN ()` SQL. `_hybrid_sub_idx`
        # short-circuits empty sets before calling here, so this is a
        # defense-in-depth for any other caller.
        if not run_uuids:
            return "false"
        escaped = ", ".join(f"'{r.replace(chr(39), chr(39)*2)}'" for r in sorted(run_uuids))
        clauses.append(f"run_uuid IN ({escaped})")
    if not clauses:
        return None  # e.g. date_range=(None, None) -> no actual predicate
    return " AND ".join(clauses)


def _hybrid_sub_idx(
    dataset: lance.LanceDataset,
    row_ids: np.ndarray,
    vehicle: str | None,
    date_range: tuple[int | None, int | None] | None,
    run_uuids: "set[str] | None",
) -> np.ndarray | None:
    """Push the filter predicate to Lance's scalar indexes; return the matching
    positions into the resident int8 screen (canonical scan order).

    Returns `None` when there is no filter (screen the whole corpus). The
    predicate is evaluated by Lance's BTREE/BITMAP indexes -- no data columns
    are read for the filter. Matched row ids are sorted ascending (filtered-scan
    output order is not guaranteed), mapped to positions via `searchsorted` over
    the ascending `row_ids` captured at hydrate, and verified: the row ids at
    those positions must equal the matched set, else the layout/version
    assumption is violated and we raise rather than return wrong positions.
    """
    # An empty run_uuids set matches nothing (parity with the resident path's
    # np.isin(x, [])). Check before _lance_filter_sql, which collapses an empty
    # set to None ("no filter") and would otherwise screen the whole corpus.
    if run_uuids is not None and not run_uuids:
        return np.empty(0, dtype=np.intp)
    sql = _lance_filter_sql(vehicle, date_range, run_uuids)
    if sql is None:
        return None
    matched_table = dataset.to_table(
        filter=sql, columns=[], with_row_id=True, scan_in_order=True
    )
    matched = np.sort(matched_table.column("_rowid").to_numpy().astype(np.int64))
    if matched.size == 0:
        return np.empty(0, dtype=np.intp)
    sub_idx = np.searchsorted(row_ids, matched)
    # Per-query guard: the positions must point at exactly the matched row ids.
    # Catches any fragment-layout skew or a matched id absent from `row_ids`
    # (searchsorted would otherwise map it to a wrong neighbor, or return
    # row_ids.size for an id past the end). Bounds-check before indexing so the
    # diagnostic fires rather than an opaque IndexError.
    if sub_idx.size != matched.size or np.any(sub_idx >= row_ids.size):
        raise ValueError(
            "hybrid prefilter position mismatch -- a filtered row id is absent "
            "from the resident map; the dataset likely changed between hydrate "
            "and query. Rebuild the ThresholdCorpus."
        )
    if not np.array_equal(row_ids[sub_idx], matched):
        raise ValueError(
            "hybrid prefilter position mismatch -- filtered row ids do not map "
            "back to resident positions; the dataset likely changed between "
            "hydrate and query. Rebuild the ThresholdCorpus."
        )
    return sub_idx


def _search_resident(
    query: np.ndarray, tau: float, dataset: lance.LanceDataset, resident: _ResidentCorpus,
    *, fast_curation: bool = True,
    sub_idx: np.ndarray | None = None,
) -> list[ThresholdHit]:
    """Screen+re-rank `query` against an already-decoded `_ResidentCorpus`.

    `sub_idx` is the only filter input: the canonical positions to screen.
    `None` screens the whole corpus. Callers resolve filters to `sub_idx`
    (e.g. via `_filter_mask` + `np.nonzero`) before calling.

    When `fast_curation` is true, ABOVE rows keep their bounded screening score
    and skip `take()`; BAND rows are always re-ranked. Membership is identical
    in both modes."""
    _assert_unit_norm(query)
    n = resident.corpus_i8.shape[0]
    if n == 0:
        return []

    query_pca = _project_query(query, resident.pca)
    w = (query_pca * (resident.scale.astype(np.float32) / np.float32(127.0))).astype(np.float32)

    if sub_idx is None:
        corpus_i8 = np.ascontiguousarray(resident.corpus_i8)
    else:
        if sub_idx.size == 0:
            return []
        corpus_i8 = np.ascontiguousarray(resident.corpus_i8[sub_idx])

    screening_scores = np.empty(corpus_i8.shape[0], dtype=np.float32)
    gpu_corpus._cpu_score_kernel()(corpus_i8, w, screening_scores)

    # Cauchy-Schwarz bounds the screening error by ||query_pca|| * ||e||, and
    # `resident.eps` is the ||e|| half. The scan happens in PCA space, so the
    # norm that matters is the PROJECTED query's, not the model-space query's:
    # a basis whose rows are not orthonormal can amplify the projection past
    # 1 and would leave an unscaled bound too small to be a bound at all. For
    # the orthonormal basis a real SVD produces this factor is <= 1, so the
    # window is also tighter than the corpus-wide constant.
    eps = resident.eps * float(np.linalg.norm(query_pca))
    above, band, _below = eps_bound.classify(screening_scores, tau, eps)
    # SUBSET positions (index into screening_scores / corpus_i8), NOT canonical
    # ids. Score lookups below use these; take()/row_id/metadata use canonical.
    above_pos = np.nonzero(above)[0]
    band_pos = np.nonzero(band)[0]

    # fast_curation ABOVE acceptance reads the bounded score at SUBSET position.
    # Remap to canonical ids AFTER all subset-position score lookups are done,
    # so take()/row_id/metadata address the right rows.
    if sub_idx is None:
        above_idx = above_pos  # unfiltered: subset position == canonical id
        band_idx = band_pos
    else:
        above_idx = sub_idx[above_pos]  # canonical ids for take()/metadata/row_id
        band_idx = sub_idx[band_pos]

    # Exact re-rank via take() on vector_fp -- 1 IOP/row, never a full column
    # read. ABOVE and BAND are each typically a tiny fraction of the corpus
    # for a selective query, so this is bounded by the match count, not
    # corpus size (a broad/low-tau query can still make this large; this
    # module does not attempt to cap refine cost).
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

    if fast_curation:
        # ABOVE rows are provable members (screen - eps >= tau): accept the
        # int8 screening score as a bounded approximation, no take() re-rank.
        above_scores = screening_scores[above_pos].astype(np.float64)
        above_kind = "bounded_approx"
        above_bound = eps
    else:
        above_scores = _exact_scores(above_idx)
        above_keep = above_scores >= tau
        above_idx = above_idx[above_keep]
        above_scores = above_scores[above_keep]
        above_kind = "exact"
        above_bound = 0.0

    # BAND rows are always re-ranked: the eps window alone does not decide
    # membership, so the exact score is required to filter to >= tau.
    band_scores = _exact_scores(band_idx)
    band_keep = band_scores >= tau
    band_idx = band_idx[band_keep]
    band_scores = band_scores[band_keep]

    # Results are returned in canonical row order (ABOVE then BAND), NOT
    # sorted by score: the curation workload consumes the match SET, and a
    # consumer that needs score order sorts the returned hits itself.
    all_idx = np.concatenate([above_idx, band_idx])
    all_scores = np.concatenate([above_scores, band_scores])
    all_kind = [above_kind] * above_idx.size + ["exact"] * band_idx.size
    all_bound = [above_bound] * above_idx.size + [0.0] * band_idx.size

    if all_idx.size == 0:
        return []

    meta_rows = dataset.take(all_idx.tolist(), columns=list(_METADATA_COLUMNS))
    meta = {name: meta_rows.column(name).to_pylist() for name in _METADATA_COLUMNS}
    return [
        ThresholdHit(
            row_id=int(all_idx[i]),
            score=float(all_scores[i]),
            run_uuid=meta["run_uuid"][i],
            segment_id=meta["segment_id"][i],
            chunk_start_unix=meta["chunk_start_unix"][i],
            chunk_end_unix=meta["chunk_end_unix"][i],
            vehicle=meta["vehicle"][i],
            score_kind=all_kind[i],
            score_error_bound=all_bound[i],
        )
        for i in range(all_idx.size)
    ]


def threshold_search(
    query: np.ndarray, tau: float, dataset: lance.LanceDataset,
    *, fast_curation: bool = True,
    vehicle: str | None = None,
    date_range: tuple[int | None, int | None] | None = None,
    run_uuids: "set[str] | None" = None,
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
    re-rank (BAND), never drop a row whose exact score is >= tau. Results are
    returned in canonical row order (the match set), not sorted by score.

    `fast_curation` is the shipped mode: ABOVE rows (provably above tau by the
    eps bound) keep their bounded int8 screening score and skip `take()`;
    BAND rows are always re-ranked exactly. `fast_curation=False` refines
    ABOVE too, used internally for benchmark comparison. Membership is
    identical in both modes.

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
    allowed = _filter_mask(resident, vehicle, date_range, run_uuids)
    sub_idx = None if allowed is None else np.nonzero(allowed)[0]
    return _search_resident(
        query, tau, dataset, resident, fast_curation=fast_curation, sub_idx=sub_idx,
    )


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

    `prefilter_mode` selects how filters narrow the resident screen:

    - ``"resident"`` (default): decode the metadata columns (vehicle,
      chunk_start_unix, run_uuid) resident and build a dense boolean mask per
      query. The simplest path; holds the metadata arrays in memory.
    - ``"hybrid"``: push the filter predicate to Lance's scalar indexes
      (BTREE/BITMAP), receive the matching row ids, and map them to positions
      into the already-resident int8 screen. Drops the metadata arrays (saves
      RAM) and uses the writer's scalar indexes at query time.
      Requires a stable dataset version between construction and query (a
      read-only serving corpus satisfies this); the hydrate and per-query
      guards raise if the row-id map is unsafe.
    """

    def __init__(
        self, dataset: lance.LanceDataset, *, prefilter_mode: str = "resident"
    ) -> None:
        if prefilter_mode not in ("resident", "hybrid"):
            raise ValueError(
                f"prefilter_mode must be 'resident' or 'hybrid', got {prefilter_mode!r}"
            )
        if not lance_writer.is_exact_threshold_dataset(dataset):
            raise ValueError(
                "ThresholdCorpus requires an exact-threshold Lance dataset "
                "(embedding_i8 + vector_fp columns, data_storage_version >= "
                f"'{lance_writer.MIN_DATA_STORAGE_VERSION}', PCA schema metadata present)"
            )
        self._dataset = dataset
        self._prefilter_mode = prefilter_mode
        self._resident = _load_resident_corpus(
            dataset, capture_row_ids=(prefilter_mode == "hybrid")
        )

    @property
    def num_rows(self) -> int:
        return self._resident.corpus_i8.shape[0]

    def threshold_search(
        self, query: np.ndarray, tau: float, *, fast_curation: bool = True,
        vehicle: str | None = None,
        date_range: tuple[int | None, int | None] | None = None,
        run_uuids: "set[str] | None" = None,
    ) -> list[ThresholdHit]:
        """Every row with re-rank score >= `tau`; see `threshold_search`."""
        if self._prefilter_mode == "hybrid":
            row_ids = self._resident.row_ids
            assert row_ids is not None  # hybrid hydrate sets this or raises
            sub_idx = _hybrid_sub_idx(
                self._dataset, row_ids, vehicle, date_range, run_uuids
            )
        else:
            allowed = _filter_mask(self._resident, vehicle, date_range, run_uuids)
            sub_idx = None if allowed is None else np.nonzero(allowed)[0]
        return _search_resident(
            query, tau, self._dataset, self._resident,
            fast_curation=fast_curation, sub_idx=sub_idx,
        )
