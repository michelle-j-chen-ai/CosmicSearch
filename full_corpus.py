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


@dataclasses.dataclass(frozen=True)
class Hit:
    """One ranked result. `score` is the int8 screening score: the exact PCA
    score lies within +/- `score_error_bound` of it."""

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
        vehicles: "set[str] | None" = None,
        date_range: "tuple[int | None, int | None] | None" = None,
        run_uuids: "set[str] | None" = None,
    ) -> list[Hit]:
        """Top-`limit` rows for `query`, best first. No S3 access."""
        if limit <= 0:
            raise ValueError("limit must be positive")
        scores, err = self.score(query)
        mask = self.filter_mask(vehicles, date_range, run_uuids)
        if mask is None:
            order = self._top_k(scores, min(limit, self.num_rows))
        else:
            allowed = np.nonzero(mask)[0]
            if allowed.size == 0:
                return []
            sub_scores = scores[allowed]
            order = allowed[self._top_k(sub_scores, min(limit, allowed.size))]
        return [self._hit(int(i), rank, float(scores[i]), err)
                for rank, i in enumerate(order, start=1)]

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

    t0 = time.perf_counter()
    i8_table = ds.to_table(columns=[i8_col], scan_in_order=True)
    child = i8_table.column(i8_col).combine_chunks().values
    # A null embedding value decodes to NaN via to_numpy, and casting NaN to int8
    # is undefined -- the row would score on garbage rather than fail. Fill with
    # zero instead: a zero vector scores exactly 0 against any query, so such a
    # row is inert and can never outrank a real match.
    if child.null_count:
        LOGGER.warning(
            "%s: %d null values filled with 0 (those rows score 0)",
            i8_col, child.null_count,
        )
        child = child.fill_null(0)
    flat = np.asarray(child.to_numpy(zero_copy_only=False), dtype=np.int8)
    corpus_i8 = np.ascontiguousarray(flat.reshape(i8_table.num_rows, -1))
    del i8_table, child, flat
    LOGGER.info(
        "full corpus screen: %d rows x %d dim (%.2fGB) in %.1fs",
        corpus_i8.shape[0], corpus_i8.shape[1],
        corpus_i8.nbytes / 1e9, time.perf_counter() - t0,
    )

    meta_table = ds.to_table(columns=list(_METADATA_COLUMNS), scan_in_order=True)
    veh_codes, veh_uniques = _dictionary_codes(meta_table.column("vehicle"))
    run_codes, run_uniques = _dictionary_codes(meta_table.column("run_uuid"))
    dt_codes, dt_uniques = _dictionary_codes(meta_table.column("dt"))
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
    # Drop the Arrow table now the resident arrays exist: it still holds the
    # raw run_uuid/dt/vehicle string buffers (several GB at this row count),
    # which the dictionary codes above have replaced. segment_id is kept, so
    # its buffer survives via the combined array referenced in `meta`.
    del meta_table
    LOGGER.info("full corpus metadata: %d vehicles, %d runs, %d dates",
                len(veh_uniques), len(run_uniques), len(dt_uniques))
    return FullCorpus(corpus_i8, pca, scale, meta)
