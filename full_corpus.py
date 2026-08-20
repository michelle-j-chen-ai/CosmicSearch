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
active one is `CORPUS_MODEL`. The PCA basis + quantization scales live in the
schema metadata of a sibling carrier table (`PCA_BASIS_URI`) rather than on the
corpus itself, so they are read from there.
"""

from __future__ import annotations

import dataclasses
import logging
import time

import lance
import numpy as np

import eps_bound
import lance_writer
import oci_s3

LOGGER = logging.getLogger(__name__)

# Pinned, NOT a caller/workflow input -- see the module docstring.
CORPUS_TABLE_URI = (
    "s3://neuron-prod-data-intelligence-exploratory/vlm/corpus/video_embeddings.lance"
)
PCA_BASIS_URI = (
    "s3://neuron-prod-data-intelligence-exploratory/vlm/corpus/pca_basis.lance"
)
# Encoder whose column family this module scores against. Must match the model
# the app encodes queries with, or every score is meaningless.
CORPUS_MODEL = "black_dwarf"

# Rebuilt per hit rather than held resident: 34.4M source_media_uri strings are
# ~4.5GB, and the value is a pure function of (dt, run_uuid, chunk_start_unix).
# Verified against the corpus: reconstruction matched on every sampled row.
_MEDIA_URI_TEMPLATE = (
    "s3://neuron-prod-data-intelligence-exploratory/vlm/chunks_mp4_v2/"
    "dt={dt}/{run_uuid}_t{chunk_start_unix}.mp4"
)

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
        self.corpus_uri = CORPUS_TABLE_URI
        self.dataset_version = None
        self.loaded_at = 0.0
        # Held open from load time. Row positions address THIS version; the table
        # grows daily, so reopening it per request would let positions drift onto
        # different clips with nothing to signal it.
        self.dataset = None

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

    def filter_mask(
        self,
        vehicles: "set[str] | None" = None,
        date_range: "tuple[int | None, int | None] | None" = None,
        run_uuids: "set[str] | None" = None,
    ) -> np.ndarray | None:
        """AND of the given filters; None when nothing was asked for.

        Unlike `threshold_search._filter_mask`, `vehicles` is a SET: the app's
        vehicle box accepts a list, and a single-value filter silently dropped
        every vehicle but one.
        """
        if not vehicles and date_range is None and not run_uuids:
            return None
        mask = np.ones(self.num_rows, dtype=bool)
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
        mask = self.filter_mask(vehicles, date_range, run_uuids)
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
    ds = lance.dataset(CORPUS_TABLE_URI, storage_options=so)
    i8_col = embedding_column(model)
    if i8_col not in ds.schema.names:
        raise ValueError(
            f"{CORPUS_TABLE_URI} has no column {i8_col!r} "
            f"(available: {sorted(n for n in ds.schema.names if 'embedding' in n)})"
        )
    pca, scale = lance_writer.read_pca_metadata(
        lance.dataset(PCA_BASIS_URI, storage_options=so)
    )

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
    corpus.dataset = ds
    corpus.corpus_uri = CORPUS_TABLE_URI
    corpus.dataset_version = getattr(ds, "version", None)
    corpus.loaded_at = time.time()
    corpus.warm()
    return corpus
