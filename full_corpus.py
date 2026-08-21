"""Live top-k search over the whole consolidated VLM corpus.

The app's resident `search_engine.Corpus` holds fp32 768-d vectors, which caps
it at a few million clips. This module serves the FULL corpus by holding the
int8/PCA-256 screen resident instead: 34.4M x 256 = 8.8GB, swept by
`gpu_corpus`'s numba kernel in ~180ms on 8 cores, versus ~105GB if the same
rows were held as fp32.

Scores are the int8 screening scores, whose error against the exact PCA score
is bounded by `eps_bound.eps_cauchy_schwarz` scaled by the projected query's
norm (`Hit.score_error_bound`). Ranking off the screen needs no S3 at all.
Exact scores are a separate, slower path (`threshold_search`), which fetches
`vector_fp` per row and is worth it for curation, not for browsing.

Why the corpus URI is a module constant and not a caller argument: pointing a
run at the wrong table silently produces plausible-looking results, because a
query vector scored against another model's embeddings still returns a ranked
list -- just a meaningless one. Naming it once here means a caller cannot get
it wrong.

The consolidated table stores one column family per encoder
(`embedding_i8_<model>`), so it can carry several models side by side; the
active one is `CORPUS_MODEL`. Its PCA basis and quantization scales live on the
FIELD metadata of that family's float column, alongside the encoder identity, so
they are read from the corpus itself. A sibling table carries a copy, but two
copies can drift: projecting a query through a basis that no longer matches the
vectors it is scored against still returns a confident ranked list, just a
meaningless one, and nothing errors.
"""

from __future__ import annotations

import dataclasses
import logging
import time

import lance
import numpy as np

import config
import eps_bound
import lance_writer
import oci_s3

LOGGER = logging.getLogger(__name__)


# The one corpus this app reads. Pinned, NOT a caller/workflow input -- see the
# module docstring. Every other embedding source has been retired; anything that
# needs "the corpus" resolves to this.
DEFAULT_CORPUS_TABLE_URI = (
    "s3://neuron-prod-data-intelligence-exploratory/vlm/corpus/video_embeddings.lance"
)
# Encoder whose column family this module scores against. Must match the model
# the app encodes queries with, or every score is meaningless.
CORPUS_MODEL = "black_dwarf"

# Rebuilt per hit rather than held resident: 34.4M source_media_uri strings are
# ~4.5GB, and the value is a pure function of (dt, run_uuid, chunk_start_unix).
# Verified against the corpus: reconstruction matched on every sampled row.
# Shares NLS_MP4_PREFIX with gpu_corpus so the two cannot disagree about where
# the clips live.
_MEDIA_URI_TEMPLATE = (
    config.mp4_prefix() + "dt={dt}/{run_uuid}_t{chunk_start_unix}.mp4"
)

# Rows per take() when fetching exact vectors for an export. take() has a ~1.65s
# floor per call regardless of size, so bigger is cheaper -- but each chunk holds
# rows x 768 x 4 bytes (~300MB at 100k) on top of a process already near its
# ceiling, so this trades a few extra round trips for a bounded peak.
_EXACT_CHUNK_ROWS = 100_000

# Clip length used when chunk_end_unix is NULL (-1 in the resident array), so
# interval projection still has a span to work with. Matches interval_core.
_DEFAULT_WINDOW_S = 8

# Filterable/emittable metadata held resident. source_media_uri and chunk_id are
# excluded on purpose (both reconstructed above).
_METADATA_COLUMNS = (
    "run_uuid",
    "segment_id",
    "chunk_start_unix",
    "chunk_end_unix",
    "vehicle",
    "dt",
    "dx_internal_id",
)


def embedding_column(model: str = CORPUS_MODEL) -> str:
    return f"embedding_i8_{model}"


def vector_fp_column(model: str = CORPUS_MODEL) -> str:
    return f"vector_fp_{model}"


def _read_field_pca(
    dataset: "lance.LanceDataset", model: str
) -> "tuple[np.ndarray, np.ndarray, str]":
    """(pca, scales, model_id) from the float column's FIELD metadata.

    `lance_writer.read_pca_metadata` looks at SCHEMA-level metadata, where this
    table carries only `embedding_dim`/`pca_dim`. The basis itself sits on the
    field, which is the copy that travels with the vectors it describes, and it
    brings the encoder identity with it -- so a corpus built by a different model
    can be rejected rather than silently scored against.
    """
    column = vector_full_column(model)
    field = dataset.schema.field(column)
    meta = field.metadata or {}
    missing = [
        k.decode()
        for k in (lance_writer.META_KEY_PCA_COMPONENTS, lance_writer.META_KEY_QUANT_SCALES)
        if k not in meta
    ]
    if missing:
        raise ValueError(
            f"{column} field metadata is missing {missing}; the basis must travel "
            "with the vectors it describes"
        )
    pca = lance_writer.decode_array(meta[lance_writer.META_KEY_PCA_COMPONENTS])
    scale = lance_writer.decode_array(meta[lance_writer.META_KEY_QUANT_SCALES])
    model_id = (meta.get(b"nls.model_id") or b"").decode()
    return pca, scale, model_id


def vector_full_column(model: str = CORPUS_MODEL) -> str:
    """The original pre-PCA embedding. Exact scores come from here rather than
    `vector_fp`, which is full precision but still PCA-reduced: only this column
    yields a cosine comparable to the app's float corpus."""
    return f"vector_{model}"


@dataclasses.dataclass(frozen=True)
class Hit:
    """One ranked result. `score` is the int8 screening score: the exact PCA
    score lies within +/- `score_error_bound` of it, until `rescore` replaces it
    with the true 768-d cosine."""

    # Position in the dataset's canonical scan order. This is how the next tier
    # is addressed -- `take()` works by position, not by id -- and it is only
    # valid against the dataset version this corpus was loaded from.
    row: int
    rank: int
    score: float
    score_error_bound: float
    segment_id: str | None
    run_uuid: str
    chunk_id: str
    source_media_uri: str
    chunk_start_unix: int
    chunk_end_unix: int | None
    vehicle: str | None
    dx_internal_id: int | None


@dataclasses.dataclass(frozen=True)
class Selection:
    """An export-sized result set, held as arrays rather than objects.

    `rows` are corpus row positions in score-descending order and `scores` are
    the int8 screening scores for exactly those rows. `cutoff` is the score of
    the worst included row (top-k) or the requested threshold (tau), so a caller
    can hand either mode's result to the same interval projection.

    `candidates` counts everything that passed the filters and the cutoff BEFORE
    `max_rows` was applied, so a truncated export still reports how much it left
    behind instead of quietly presenting a partial answer as complete.
    """

    rows: np.ndarray
    scores: np.ndarray
    error_bound: float
    candidates: int
    cutoff: float
    truncated: bool

    def __len__(self) -> int:
        return int(self.rows.size)


def _dictionary_codes(column: object) -> tuple[np.ndarray, list]:
    """Arrow string column -> (int32 codes, uniques). Code -1 is NULL.

    Mirrors `threshold_search._dictionary_codes`: filters then compare int32
    codes (SIMD) instead of Python strings, and the uniques list is tiny next
    to the per-row column it replaces.
    """
    import pyarrow as pa
    import pyarrow.compute as pc

    encoded = pc.dictionary_encode(column.combine_chunks())
    if isinstance(encoded, pa.ChunkedArray):
        encoded = encoded.combine_chunks()
    codes = encoded.indices.to_numpy(zero_copy_only=False)
    codes = np.where(np.isnan(codes.astype(np.float64)), -1, codes).astype(np.int32) \
        if codes.dtype.kind == "f" else codes.astype(np.int32)
    if encoded.indices.null_count:
        codes = codes.copy()
        codes[np.asarray(encoded.indices.is_null())] = -1
    return codes, encoded.dictionary.to_pylist()


class FullCorpus:
    """The whole corpus, resident as an int8 screen, searchable top-k."""

    def __init__(
        self,
        corpus_i8: np.ndarray,
        pca: np.ndarray,
        scale: np.ndarray,
        meta: dict,
    ) -> None:
        self.corpus_i8 = corpus_i8
        self.pca = pca
        self.scale = scale
        self.eps = eps_bound.eps_cauchy_schwarz(scale)
        self._meta = meta
        # Provenance the UI surfaces, so a result can state which snapshot
        # answered it rather than leaving "is this really the whole corpus?"
        # unanswerable. Set on the instance at load; defaulted here so a
        # directly-constructed corpus still has them.
        self.corpus_uri = DEFAULT_CORPUS_TABLE_URI
        self.dataset_version = None
        self.loaded_at = 0.0
        # Held open from load time. Row positions address THIS version; the table
        # grows daily, so reopening it per request would let positions drift onto
        # different clips with nothing to signal it.
        self.dataset = None
        # Encoder identity recorded next to the basis, so a mismatched corpus is
        # detectable rather than assumed from a constant.
        self.model_id = ""
        # Set at load by `probe_vector_fp`. False means the middle cascade tier
        # is the quantization it would be there to resolve.
        self.vector_fp_usable = False

    @property
    def num_rows(self) -> int:
        return int(self.corpus_i8.shape[0])

    @property
    def dim(self) -> int:
        return int(self.pca.shape[1])

    def time_span(self) -> tuple[int, int]:
        """(min, max) chunk_start_unix. Same contract as `search_engine.Corpus`
        so the web server's date-bounds helper works against either."""
        starts = self._meta["chunk_start_unix"]
        if starts.size == 0:
            return (0, 0)
        return (int(starts.min()), int(starts.max()))

    def has_segment_id(self) -> bool:
        """Whether any row carries a segment_id (drives the UI's capability note)."""
        import pyarrow.compute as pc

        seg = self._meta["segment_id"]
        if seg is None:
            return False
        return bool(pc.any(pc.not_equal(seg, "")).as_py())

    def vehicles(self) -> list[str]:
        """Distinct vehicle ids present, for populating a filter UI."""
        return sorted(v for v in self._meta["vehicle_uniques"] if v)

    # ---- scoring -------------------------------------------------------

    def _weights(self, query: np.ndarray) -> tuple[np.ndarray, float]:
        """(int8-space weight vector, error bound) for a unit-norm 768-d query.

        Dequantization folds into the query: scoring int8 rows against
        `pca @ q * scale / 127` equals scoring dequantized rows against
        `pca @ q`, without materializing a float copy of the corpus.
        """
        q = np.asarray(query, dtype=np.float32).reshape(-1)
        if q.shape[0] != self.dim:
            raise ValueError(f"query must be {self.dim}-d, got {q.shape[0]}")
        norm = float(np.linalg.norm(q))
        if norm == 0.0:
            raise ValueError("query vector is all zeros")
        q = q / norm
        q_pca = (self.pca @ q).astype(np.float32)
        w = (q_pca * (self.scale.astype(np.float32) / np.float32(127.0))).astype(
            np.float32
        )
        return w, float(self.eps * np.linalg.norm(q_pca))

    def score(self, query: np.ndarray) -> tuple[np.ndarray, float]:
        """Screening score for every row. One sweep of the resident matrix."""
        import gpu_corpus

        w, err = self._weights(query)
        out = np.empty(self.num_rows, dtype=np.float32)
        gpu_corpus._cpu_score_kernel()(self.corpus_i8, w, out)
        return out, err

    def warm(self) -> None:
        """Compile the scoring kernel before any request can hit it.

        The kernel is numba-jitted on first call, and compiling it took ~85s on
        the deployed service -- paid by whoever ran the first search after a
        load, who sees an apparently hung request. Compiling here against a
        small slice moves that cost into the load, which is already asynchronous
        and reports progress. Cheap to get wrong silently, so failure is logged
        rather than raised: a warm-up that fails only restores the old behaviour.
        """
        try:
            import gpu_corpus

            t0 = time.perf_counter()
            probe = np.ascontiguousarray(self.corpus_i8[: min(1024, self.num_rows)])
            out = np.empty(probe.shape[0], dtype=np.float32)
            gpu_corpus._cpu_score_kernel()(
                probe, np.zeros(probe.shape[1], dtype=np.float32), out
            )
            LOGGER.info("scoring kernel warm in %.1fs", time.perf_counter() - t0)
        except Exception as exc:  # noqa: BLE001 -- warm-up is an optimization
            LOGGER.warning("scoring kernel warm-up failed (%s); the first search "
                           "will pay the compile", exc)

    # ---- filtering -----------------------------------------------------

    def segment_mask(self, segment_ids: "set[str] | frozenset[str]") -> np.ndarray:
        """Rows whose segment_id is in `segment_ids`.

        `search_engine.segment_mask` tests membership row by row in Python, which
        is fine over a few million rows and takes tens of seconds over 34M. The
        resident segment_id column is Arrow, so `is_in` does the same test in one
        vectorized pass.
        """
        import pyarrow as pa
        import pyarrow.compute as pc

        seg = self._meta["segment_id"]
        if seg is None:
            return np.zeros(self.num_rows, dtype=bool)
        hit = pc.is_in(seg, value_set=pa.array(sorted(segment_ids), type=pa.string()))
        return pc.fill_null(hit, False).to_numpy(zero_copy_only=False).astype(bool)

    def filter_mask(
        self,
        vehicles: "set[str] | None" = None,
        date_range: "tuple[int | None, int | None] | None" = None,
        run_uuids: "set[str] | None" = None,
        segment_ids: "set[str] | frozenset[str] | None" = None,
        exclude_rows: "list[int] | np.ndarray | None" = None,
    ) -> np.ndarray | None:
        """AND of the given filters; None when nothing was asked for.

        Unlike `threshold_search._filter_mask`, `vehicles` is a SET: the app's
        vehicle box accepts a list, and a single-value filter silently dropped
        every vehicle but one.

        `exclude_rows` drops specific rows outright. Relevance feedback needs it:
        Rocchio moves the query DIRECTION away from the negatives, which is not
        the same as removing them, so a rejected clip that still sits near the
        positive centroid comes back near the top of the very next re-rank.
        """
        has_exclude = exclude_rows is not None and len(exclude_rows) > 0
        if (
            not vehicles and date_range is None and not run_uuids
            and not segment_ids and not has_exclude
        ):
            return None
        mask = np.ones(self.num_rows, dtype=bool)
        if has_exclude:
            idx = np.asarray(exclude_rows, dtype=np.int64)
            mask[idx[(idx >= 0) & (idx < self.num_rows)]] = False
        if segment_ids:
            mask &= self.segment_mask(segment_ids)
        if vehicles:
            uniques = self._meta["vehicle_uniques"]
            codes = [i for i, v in enumerate(uniques) if v in vehicles]
            mask &= np.isin(self._meta["vehicle"], codes)
        if run_uuids:
            uniques = self._meta["run_uuid_uniques"]
            codes = [i for i, v in enumerate(uniques) if v in run_uuids]
            mask &= np.isin(self._meta["run_uuid"], codes)
        if date_range is not None:
            lo, hi = date_range
            starts = self._meta["chunk_start_unix"]
            if lo is not None:
                mask &= starts >= lo
            if hi is not None:
                mask &= starts < hi
        return mask

    # ---- ranking -------------------------------------------------------

    @staticmethod
    def _top_k(scores: np.ndarray, k: int) -> np.ndarray:
        """Indices of the k highest scores, best first.

        A plain `argpartition` over 34M scores costs ~875ms -- several times the
        scan that produced them. Sampling a cutoff first shrinks the candidate
        set to a few multiples of k, and the partition then runs over that.
        Falls back to the direct partition whenever the cutoff fails to narrow
        enough (or narrows too far), so the result is identical either way.
        """
        n = scores.size
        if k >= n:
            return np.argsort(-scores, kind="stable")
        cand: np.ndarray | None = None
        if n > 1_000_000:
            step = max(1, n // 200_000)
            sample = scores[::step]
            q = 1.0 - min(1.0, (8.0 * k) / n)
            cutoff = float(np.quantile(sample, q)) if q > 0.0 else -np.inf
            picked = np.nonzero(scores >= cutoff)[0]
            if k <= picked.size <= max(200_000, 100 * k):
                cand = picked
        if cand is None:
            part = np.argpartition(scores, -k)[-k:]
            return part[np.argsort(-scores[part], kind="stable")]
        sub = np.argpartition(scores[cand], -k)[-k:]
        idx = cand[sub]
        return idx[np.argsort(-scores[idx], kind="stable")]

    def search(
        self,
        query: np.ndarray,
        limit: int = 50,
        *,
        offset: int = 0,
        vehicles: "set[str] | None" = None,
        date_range: "tuple[int | None, int | None] | None" = None,
        run_uuids: "set[str] | None" = None,
        segment_ids: "set[str] | frozenset[str] | None" = None,
        exclude_rows: "list[int] | np.ndarray | None" = None,
    ) -> "tuple[list[Hit], int]":
        """`(hits, candidates)`: rows `offset`..`offset+limit`, best first.

        Paging selects the top `offset + limit` and returns the tail slice.
        Every score is already resident, so a deeper page costs one more
        selection pass rather than another scan -- there is no cursor to keep
        and no state between requests. `candidates` is how many rows survived
        the filters, which is what the caller needs to size a pager.
        """
        if limit <= 0:
            raise ValueError("limit must be positive")
        if offset < 0:
            raise ValueError("offset must not be negative")
        scores, err = self.score(query)
        mask = self.filter_mask(vehicles, date_range, run_uuids, segment_ids, exclude_rows)
        depth = offset + limit
        if mask is None:
            candidates = self.num_rows
            order = self._top_k(scores, min(depth, candidates))
        else:
            allowed = np.nonzero(mask)[0]
            candidates = int(allowed.size)
            if candidates == 0:
                return [], 0
            order = allowed[self._top_k(scores[allowed], min(depth, candidates))]
        page = order[offset:]
        return (
            [
                self._hit(int(i), offset + rank, float(scores[i]), err)
                for rank, i in enumerate(page, start=1)
            ],
            candidates,
        )

    # ---- export-scale selection ---------------------------------------

    def select(
        self,
        query: np.ndarray,
        *,
        k: int | None = None,
        tau: float | None = None,
        vehicles: "set[str] | None" = None,
        date_range: "tuple[int | None, int | None] | None" = None,
        run_uuids: "set[str] | None" = None,
        segment_ids: "set[str] | frozenset[str] | None" = None,
        exclude_rows: "list[int] | np.ndarray | None" = None,
        max_rows: int = 0,
    ) -> "Selection":
        """Rows matching a top-`k` cut or a `tau` threshold, as plain arrays.

        Deliberately NOT `search`: that builds one `Hit` object per result, which
        is right for a 200-row page and fatal for an export -- a 3.4M-row result
        costs well over a gigabyte of Python objects plus the GC time to walk
        them. Everything here stays in numpy, and the metadata is materialized
        columnwise only once, in `to_arrow`.

        Exactly one of `k` / `tau` is required. `k` is bounded by construction;
        `tau` is not, so `max_rows` (when non-zero) truncates by score and sets
        `truncated`, rather than letting a permissive threshold decide how much
        memory this process uses.
        """
        if (k is None) == (tau is None):
            raise ValueError("pass exactly one of k= or tau=")
        if k is not None and k <= 0:
            raise ValueError("k must be positive")
        scores, err = self.score(query)
        mask = self.filter_mask(vehicles, date_range, run_uuids, segment_ids, exclude_rows)
        if mask is None:
            allowed = None
            candidates = self.num_rows
        else:
            allowed = np.nonzero(mask)[0]
            candidates = int(allowed.size)
        if candidates == 0:
            empty = np.empty(0, dtype=np.int64)
            return Selection(empty, np.empty(0, dtype=np.float32), err, 0, 0.0, False)

        truncated = False
        if k is not None:
            depth = min(int(k), candidates)
            if allowed is None:
                rows = self._top_k(scores, depth)
            else:
                rows = allowed[self._top_k(scores[allowed], depth)]
            cut = float(scores[rows[-1]]) if rows.size else 0.0
        else:
            # Threshold over the SCREEN score. A row's exact PCA score is within
            # +/-err of this, so rows within err of tau are genuinely undecided;
            # `score_kind` on the result says so rather than implying otherwise.
            hits = scores >= np.float32(tau)
            if allowed is not None:
                sub = hits[allowed]
                rows = allowed[sub]
            else:
                rows = np.nonzero(hits)[0]
            matched = int(rows.size)
            if max_rows and matched > max_rows:
                rows = rows[self._top_k(scores[rows], max_rows)]
                truncated = True
            else:
                rows = rows[np.argsort(-scores[rows], kind="stable")]
            cut = float(tau)
            # What passed the filters AND the cutoff -- not what passed the
            # filters alone, which is what a top-k selection reports. A capped
            # export needs this number to say how much it left behind.
            candidates = matched
        return Selection(
            rows=np.asarray(rows, dtype=np.int64),
            scores=scores[rows].astype(np.float32),
            error_bound=err,
            candidates=candidates,
            cutoff=cut,
            truncated=truncated,
        )

    def tau_for_k(self, query: np.ndarray, k: int, **filters) -> float:
        """The screen score of the k-th best row: the `tau` that selects the same
        set `k` does. Every score is already resident, so this is one scan."""
        sel = self.select(query, k=k, **filters)
        return float(sel.cutoff)

    def exact_scores(
        self, rows: np.ndarray, query: np.ndarray, *, deadline: float | None = None
    ) -> np.ndarray:
        """768-d cosine for `rows`, in the order given, fetched in chunks.

        `rescore` returns a dict keyed by row and verifies every chunk_id, which
        is right for a page on screen. At export size the dict and the per-row
        check cost more than the arithmetic, so this returns a bare array and
        spot-checks instead (see `_verify_alignment`).

        `deadline` is a `time.monotonic()` value: the fetch stops between chunks
        once it is passed. The caller estimates the cost before starting, but
        that estimate uses a fixed per-row constant, and a slow object store makes
        it optimistic. Overrunning silently means the request is killed mid-write
        and the caller loses both the artifact and their download, so this fails
        early and says how far it got.
        """
        import time as _time
        if self.dataset is None:
            raise RuntimeError("corpus has no dataset handle; exact scores unavailable")
        rows = np.asarray(rows, dtype=np.int64)
        if rows.size == 0:
            return np.empty(0, dtype=np.float64)
        q = np.asarray(query, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(q))
        if norm == 0.0:
            raise ValueError("query vector is all zeros")
        q = q / norm
        col = vector_full_column(CORPUS_MODEL)
        # take() addresses by canonical position, so the fetch must be ordered;
        # results are scattered back to the caller's order at the end.
        order = np.argsort(rows, kind="stable")
        ordered = rows[order]
        out = np.empty(rows.size, dtype=np.float64)
        for at in range(0, ordered.size, _EXACT_CHUNK_ROWS):
            if deadline is not None and at and _time.monotonic() > deadline:
                raise TimeoutError(
                    f"exact scoring exceeded its time budget after {at:,} of "
                    f"{ordered.size:,} rows; narrow the selection or export with "
                    "exact=false"
                )
            part = ordered[at : at + _EXACT_CHUNK_ROWS]
            tbl = self.dataset.take(part, columns=[col])
            flat = tbl.column(col).combine_chunks().values.to_numpy(
                zero_copy_only=False
            )
            mat = np.asarray(flat, dtype=np.float32).reshape(part.size, -1)
            out[order[at : at + _EXACT_CHUNK_ROWS]] = np.einsum(
                "ij,j->i", mat, q, dtype=np.float64
            )
        return out

    def window_rows(
        self, *, run_uuid: str = "", segment_id: str = "",
        start_unix: int = 0, end_unix: int = 0,
    ) -> np.ndarray:
        """Rows of the clips already embedded for a drive/segment and time window.

        The query-by-example counterpart of `search_engine.window_query`'s front
        half, over the resident metadata: run_uuid and segment_id are matched on
        dictionary codes and the Arrow column rather than row-by-row in Python,
        which is the difference between milliseconds and a minute at 34M rows.
        """
        import pyarrow as pa
        import pyarrow.compute as pc

        run_uuid = (run_uuid or "").strip()
        segment_id = (segment_id or "").strip()
        if not run_uuid and not segment_id:
            raise ValueError("window_rows needs a run_uuid or a segment_id")
        m = self._meta
        mask = np.ones(self.num_rows, dtype=bool)
        if run_uuid:
            try:
                code = m["run_uuid_uniques"].index(run_uuid)
            except ValueError:
                return np.empty(0, dtype=np.int64)
            mask &= m["run_uuid"] == code
        if segment_id:
            seg = m["segment_id"]
            if seg is None:
                return np.empty(0, dtype=np.int64)
            hit = pc.equal(seg, pa.scalar(segment_id, type=pa.string()))
            mask &= pc.fill_null(hit, False).to_numpy(zero_copy_only=False).astype(bool)
        # Half-open overlap: chunk [cs, ce) overlaps [start, end) iff cs < end and
        # ce > start. A 0 bound leaves that side open.
        starts = m["chunk_start_unix"]
        ends = np.where(m["chunk_end_unix"] < 0, starts + _DEFAULT_WINDOW_S, m["chunk_end_unix"])
        if start_unix:
            mask &= ends > int(start_unix)
        if end_unix:
            mask &= starts < int(end_unix)
        rows = np.nonzero(mask)[0]
        return rows[np.argsort(starts[rows], kind="stable")]

    def probe_vector_fp(self, sample: int = 64) -> dict:
        """Is `vector_fp` the true pre-quantization projection, or a fallback?

        `lance_writer` populates that column from a real fp32 PCA projection when
        one was supplied at write time, and otherwise from the int8 dequantized
        back to fp32. The two are indistinguishable by schema -- no flag records
        which -- but not by content: the fallback equals `i8 * scale / 127`
        exactly, because that is how it was produced.

        This matters because the whole point of the middle cascade tier is to
        resolve quantization error. If it IS the quantization, refining through
        it resolves nothing and the tier is dead weight dressed as precision.
        """
        if self.dataset is None:
            return {"available": False, "reason": "no dataset handle"}
        col = vector_fp_column(CORPUS_MODEL)
        if col not in self.dataset.schema.names:
            return {"available": False, "reason": f"no column {col!r}"}
        rows = np.unique(
            np.linspace(0, self.num_rows - 1, num=min(sample, self.num_rows))
            .astype(np.int64)
        )
        try:
            tbl = self.dataset.take(rows, columns=[col])
        except Exception as exc:  # noqa: BLE001 - a probe must never break load
            return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}
        flat = tbl.column(col).combine_chunks().values.to_numpy(zero_copy_only=False)
        fp = np.asarray(flat, dtype=np.float32).reshape(rows.size, -1)
        dequant = self.corpus_i8[rows].astype(np.float32) * (
            self.scale.astype(np.float32) / np.float32(127.0)
        )
        dev = float(np.abs(fp - dequant).max())
        # The dequantized value is exactly representable, so a genuine fallback
        # matches to float32 round-off. A real projection differs by the
        # quantization error it was quantized FROM -- orders of magnitude larger.
        quant_step = float((self.scale / 127.0).max())
        is_fallback = dev < quant_step * 1e-3
        return {
            "available": True,
            "is_fallback": is_fallback,
            "max_deviation": dev,
            "quant_step": quant_step,
            "rows_probed": int(rows.size),
        }

    def vectors_for(self, rows: "list[int] | np.ndarray") -> np.ndarray:
        """The original 768-d embeddings for `rows`, in the order given.

        Relevance feedback needs the actual vectors of the clips someone marked,
        and this corpus holds only the int8/PCA screen. The marked set is a
        handful of rows, so one `take()` is cheaper than holding 105GB of fp32
        resident to avoid it.
        """
        if self.dataset is None:
            raise RuntimeError("corpus has no dataset handle; vectors unavailable")
        rows = np.asarray(rows, dtype=np.int64)
        if rows.size == 0:
            return np.empty((0, 0), dtype=np.float32)
        col = vector_full_column(CORPUS_MODEL)
        order = np.argsort(rows, kind="stable")
        tbl = self.dataset.take(rows[order], columns=[col])
        flat = tbl.column(col).combine_chunks().values.to_numpy(zero_copy_only=False)
        mat = np.asarray(flat, dtype=np.float32).reshape(rows.size, -1)
        out = np.empty_like(mat)
        out[order] = mat
        return out

    def _verify_alignment(self, rows: np.ndarray, sample: int = 32) -> None:
        """Raise if `rows` no longer resolve to the clips this corpus holds.

        Row positions are only valid against the loaded dataset version. A full
        per-row check is what `rescore` does; at export size that is another pass
        over the table, so this checks a spread-out sample. It catches a shifted
        snapshot (which moves every row) without pretending to catch a single
        altered row.
        """
        if self.dataset is None or rows.size == 0:
            return
        probe = np.unique(
            np.linspace(0, rows.size - 1, num=min(sample, rows.size)).astype(np.int64)
        )
        idx = np.sort(rows[probe])
        got = self.dataset.take(idx, columns=["chunk_id"]).column("chunk_id").to_pylist()
        m = self._meta
        for pos, row in enumerate(idx):
            expect = (
                f"{m['run_uuid_uniques'][m['run_uuid'][row]]}"
                f"#t{int(m['chunk_start_unix'][row])}"
            )
            if got[pos] != expect:
                raise RuntimeError(
                    f"row {int(row)} resolved to {got[pos]!r}, expected {expect!r}: "
                    "the dataset changed under the loaded snapshot"
                )

    def to_arrow(
        self,
        rows: np.ndarray,
        scores: np.ndarray,
        *,
        tag: str = "",
        exact: np.ndarray | None = None,
    ) -> object:
        """Export rows as an Arrow table, built column-at-a-time from the
        resident arrays. No per-row Python objects: the dictionary-coded columns
        are expanded by numpy fancy-indexing into the uniques list, which is what
        keeps a multi-million-row export inside a sane memory budget.
        """
        import pyarrow as pa

        rows = np.asarray(rows, dtype=np.int64)
        m = self._meta
        starts = m["chunk_start_unix"][rows]
        ends = m["chunk_end_unix"][rows]
        run_codes = m["run_uuid"][rows]
        runs = np.asarray(m["run_uuid_uniques"], dtype=object)[run_codes]
        dt_codes = m["dt"][rows]
        dts = np.where(
            dt_codes >= 0, np.asarray(m["dt_uniques"], dtype=object)[dt_codes], ""
        )
        veh_codes = m["vehicle"][rows]
        veh_uniques = np.asarray(m["vehicle_uniques"] + [None], dtype=object)
        vehicles = veh_uniques[np.where(veh_codes >= 0, veh_codes, len(m["vehicle_uniques"]))]
        seg = (
            m["segment_id"].take(pa.array(rows)).to_pylist()
            if m["segment_id"] is not None
            else [None] * rows.size
        )
        chunk_ids = [f"{r}#t{int(s)}" for r, s in zip(runs, starts)]
        media = [
            _MEDIA_URI_TEMPLATE.format(dt=d, run_uuid=r, chunk_start_unix=int(s))
            for d, r, s in zip(dts, runs, starts)
        ]
        cols = {
            "rank": np.arange(1, rows.size + 1, dtype=np.int64),
            "score": np.asarray(scores, dtype=np.float64),
            "segment_id": seg,
            "chunk_id": chunk_ids,
            "run_uuid": runs.tolist(),
            "start_timestamp_ns": starts * 1_000_000_000,
            "end_timestamp_ns": np.where(ends < 0, -1, ends * 1_000_000_000),
            "source_media_uri": media,
            "vehicle": vehicles.tolist(),
            "tag": [tag] * rows.size,
            "corpus_row": rows,
        }
        if exact is not None:
            cols["exact_score"] = np.asarray(exact, dtype=np.float64)
        return pa.table(cols)

    def drive_groups(
        self, rows: np.ndarray, scores: np.ndarray
    ) -> "list[tuple[str, np.ndarray, np.ndarray, np.ndarray]]":
        """`(run_uuid, cell_score, cell_center, cell_peak_row)` per drive, the
        input `interval_core._merge_drive` expects.

        `search_engine._build_drives` does the same job but groups by comparing
        run_uuid STRINGS: it materializes an object array and lexsorts it, which
        at export size spends most of its time in Python string compares. The
        resident corpus already holds run_uuid dictionary-coded, so this sorts
        int32 codes instead and reads each drive's name once.
        """
        import interval_core

        rows = np.asarray(rows, dtype=np.int64)
        if rows.size == 0:
            return []
        m = self._meta
        codes = m["run_uuid"][rows]
        starts = m["chunk_start_unix"][rows]
        ends = m["chunk_end_unix"][rows]
        ends = np.where(ends < 0, starts + _DEFAULT_WINDOW_S, ends)
        g = np.lexsort((starts, codes))
        rows_s, codes_s = rows[g], codes[g]
        starts_s, ends_s = starts[g], ends[g]
        scores_s = np.asarray(scores, dtype=np.float64)[g]
        cuts = np.nonzero(codes_s[1:] != codes_s[:-1])[0] + 1
        out = []
        for grp in np.split(np.arange(rows_s.size), cuts):
            if grp.size == 0:
                continue
            cs, cc, cpr, _d = interval_core._drive_cells(
                starts_s[grp], ends_s[grp], scores_s[grp], rows_s[grp]
            )
            out.append((m["run_uuid_uniques"][int(codes_s[grp[0]])], cs, cc, cpr))
        return out

    def intervals(
        self, rows: np.ndarray, scores: np.ndarray, *, tau: float
    ) -> "list":
        """Merge selected clips into per-drive intervals above `tau`."""
        import interval_core

        merged = []
        for run_uuid, cs, cc, cpr in self.drive_groups(rows, scores):
            merged.extend(interval_core._merge_drive(run_uuid, cs, cc, cpr, tau))
        merged.sort(key=lambda iv: iv.peak_score, reverse=True)
        return merged

    def rescore(self, rows: "list[int]", query: np.ndarray) -> "dict[int, float]":
        """True 768-d cosine for `rows`, read from the original embeddings.

        The screen answers with a quantized, PCA-reduced score. This reads the
        column those were derived from, so the number is the same one the app's
        float corpus produces -- which is what makes a threshold calibrated
        elsewhere mean anything here.

        Deliberately NOT part of `search`: this is a single S3 round trip with a
        ~1.65s floor regardless of row count, so folding it in would take a 190ms
        search to nearly two seconds. Callers fetch a page fast, then sharpen the
        rows on screen.

        `chunk_id` is fetched alongside and checked against the resident
        metadata. Row positions are only valid against the loaded dataset
        version, so a drift shows up as an error rather than as a plausible score
        attached to the wrong clip.
        """
        if self.dataset is None:
            raise RuntimeError("corpus has no dataset handle; rescore unavailable")
        idx = np.asarray(sorted(set(int(r) for r in rows)), dtype=np.int64)
        if idx.size == 0:
            return {}
        q = np.asarray(query, dtype=np.float32).reshape(-1)
        if q.shape[0] != self.dim:
            raise ValueError(f"query must be {self.dim}-d, got {q.shape[0]}")
        norm = float(np.linalg.norm(q))
        if norm == 0.0:
            raise ValueError("query vector is all zeros")
        q = q / norm

        col = vector_full_column(CORPUS_MODEL)
        tbl = self.dataset.take(idx, columns=[col, "chunk_id"])
        got = tbl.column("chunk_id").to_pylist()
        for pos, row in enumerate(idx):
            expect = f"{self._meta['run_uuid_uniques'][self._meta['run_uuid'][row]]}" \
                     f"#t{int(self._meta['chunk_start_unix'][row])}"
            if got[pos] != expect:
                raise RuntimeError(
                    f"row {row} resolved to {got[pos]!r}, expected {expect!r}: the "
                    "dataset changed under the loaded snapshot"
                )
        flat = tbl.column(col).combine_chunks().values.to_numpy(zero_copy_only=False)
        mat = np.asarray(flat, dtype=np.float32).reshape(idx.size, -1)
        scores = np.einsum("ij,j->i", mat, q, dtype=np.float64)
        return {int(r): float(sc) for r, sc in zip(idx, scores)}

    def _hit(self, i: int, rank: int, score: float, err: float) -> Hit:
        m = self._meta
        run_uuid = m["run_uuid_uniques"][m["run_uuid"][i]]
        start = int(m["chunk_start_unix"][i])
        dt_code = m["dt"][i]
        dt = m["dt_uniques"][dt_code] if dt_code >= 0 else ""
        veh_code = m["vehicle"][i]
        end = m["chunk_end_unix"][i]
        dx = m["dx_internal_id"][i]
        seg = m["segment_id"][i].as_py() if m["segment_id"] is not None else None
        return Hit(
            row=i,
            rank=rank,
            score=score,
            score_error_bound=err,
            segment_id=seg,
            run_uuid=run_uuid,
            chunk_id=f"{run_uuid}#t{start}",
            source_media_uri=_MEDIA_URI_TEMPLATE.format(
                dt=dt, run_uuid=run_uuid, chunk_start_unix=start
            ),
            chunk_start_unix=start,
            chunk_end_unix=None if end < 0 else int(end),
            vehicle=m["vehicle_uniques"][veh_code] if veh_code >= 0 else None,
            dx_internal_id=None if dx < 0 else int(dx),
        )


def load(*, model: str = CORPUS_MODEL) -> FullCorpus:
    """Read the pinned corpus into memory. ~2 minutes, ~12GB resident.

    `scan_in_order=True` keeps the screen matrix and the metadata arrays on the
    same canonical row order, so a position in one indexes the other.
    """
    so = oci_s3.lance_storage_options()
    ds = lance.dataset(DEFAULT_CORPUS_TABLE_URI, storage_options=so)
    i8_col = embedding_column(model)
    if i8_col not in ds.schema.names:
        raise ValueError(
            f"{DEFAULT_CORPUS_TABLE_URI} has no column {i8_col!r} "
            f"(available: {sorted(n for n in ds.schema.names if 'embedding' in n)})"
        )
    pca, scale, model_id = _read_field_pca(ds, model)

    # Decoded batch by batch into one preallocated array rather than via
    # to_table. Materializing the whole column as Arrow and then copying it to
    # numpy holds both at once: ~18GB peak for an 8.8GB result, measured. This
    # process already holds the resident browse matrix (~6GB) and the model
    # (~5GB), so that peak does not fit in the container and the load is killed
    # partway through. Streaming keeps the extra cost to a single batch.
    t0 = time.perf_counter()
    n_rows = ds.count_rows()
    dim = ds.schema.field(i8_col).type.list_size
    corpus_i8 = np.empty((n_rows, dim), dtype=np.int8)
    nulls = 0
    at = 0
    for batch in ds.to_batches(columns=[i8_col], scan_in_order=True):
        child = batch.column(i8_col).values
        # A null embedding value decodes to NaN, and casting NaN to int8 is
        # undefined -- the row would score on garbage rather than fail. Zero
        # scores exactly 0 against any query, so the row is inert instead.
        if child.null_count:
            nulls += child.null_count
            child = child.fill_null(0)
        block = np.asarray(child.to_numpy(zero_copy_only=False), dtype=np.int8)
        rows = batch.num_rows
        corpus_i8[at : at + rows] = block.reshape(rows, dim)
        at += rows
    if at != n_rows:
        raise ValueError(f"decoded {at} rows, expected {n_rows}")
    if nulls:
        LOGGER.warning(
            "%s: %d null values filled with 0 (those rows score 0)", i8_col, nulls
        )
    LOGGER.info(
        "full corpus screen: %d rows x %d dim (%.2fGB) in %.1fs",
        corpus_i8.shape[0], corpus_i8.shape[1],
        corpus_i8.nbytes / 1e9, time.perf_counter() - t0,
    )

    # One column at a time, for the same reason as the screen above: reading all
    # seven together holds every raw string buffer (run_uuid, segment_id, dt) in
    # memory alongside the compact arrays replacing them. Each column is released
    # before the next is read.
    def _column(name):
        return ds.to_table(columns=[name], scan_in_order=True).column(name)

    veh_codes, veh_uniques = _dictionary_codes(_column("vehicle"))
    run_codes, run_uniques = _dictionary_codes(_column("run_uuid"))
    dt_codes, dt_uniques = _dictionary_codes(_column("dt"))

    class _Lazy:
        """Reads a column on attribute access so the dict below stays one-at-a-time."""

        @staticmethod
        def column(name):
            return _column(name)

    meta_table = _Lazy()
    meta = {
        "vehicle": veh_codes, "vehicle_uniques": veh_uniques,
        "run_uuid": run_codes, "run_uuid_uniques": run_uniques,
        "dt": dt_codes, "dt_uniques": dt_uniques,
        "chunk_start_unix": meta_table.column("chunk_start_unix")
        .combine_chunks().to_numpy(zero_copy_only=False).astype(np.int64),
        # -1 stands in for NULL in both nullable int columns so the resident
        # arrays stay plain int64 (no mask array, no object dtype).
        "chunk_end_unix": meta_table.column("chunk_end_unix")
        .combine_chunks().to_numpy(zero_copy_only=False).astype(np.int64),
        "dx_internal_id": meta_table.column("dx_internal_id")
        .combine_chunks().fill_null(-1).to_numpy(zero_copy_only=False).astype(np.int64),
        "segment_id": meta_table.column("segment_id").combine_chunks(),
    }
    del meta_table
    LOGGER.info("full corpus metadata: %d vehicles, %d runs, %d dates",
                len(veh_uniques), len(run_uniques), len(dt_uniques))
    corpus = FullCorpus(corpus_i8, pca, scale, meta)
    LOGGER.info(
        "pca basis from %s field metadata: %s, encoder %s",
        vector_full_column(model), "x".join(str(d) for d in pca.shape), model_id or "?",
    )
    corpus.model_id = model_id
    corpus.dataset = ds
    corpus.corpus_uri = DEFAULT_CORPUS_TABLE_URI
    corpus.dataset_version = getattr(ds, "version", None)
    corpus.loaded_at = time.time()
    probe = corpus.probe_vector_fp()
    corpus.vector_fp_usable = bool(probe.get("available") and not probe.get("is_fallback"))
    if not probe.get("available"):
        LOGGER.info("vector_fp unavailable (%s); cascade is int8 -> 768d",
                    probe.get("reason"))
    elif probe["is_fallback"]:
        LOGGER.warning(
            "vector_fp is the int8 dequantized (max dev %.3g vs quant step %.3g): "
            "refining through it would resolve nothing, so it is not used",
            probe["max_deviation"], probe["quant_step"],
        )
    else:
        LOGGER.info(
            "vector_fp is a real pre-quantization projection (max dev %.3g vs "
            "quant step %.3g); usable as the middle cascade tier",
            probe["max_deviation"], probe["quant_step"],
        )
    corpus.warm()
    return corpus
