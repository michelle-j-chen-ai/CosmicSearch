"""Brute-force natural-language video retrieval over Cosmos-Embed embeddings.

The search core is intentionally self-contained (no dependency on the
`finetuning` package, which pulls in repo-internal modules that cannot ship in
a slim container). The encode + rank logic mirrors
`text_query_search._encode_texts` and `_rank_top_k`, but loads the model on CPU
and keeps the corpus matrix resident in memory.

At 1M x 768 the corpus is ~1.5GB in fp16 and a single query is a matrix-vector
product (~50-200ms on CPU, memory-bandwidth bound). An ANN index only becomes
worthwhile past ~10M rows; see README "Scaling" for the migration path.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import shutil
import tempfile
import time
from pathlib import Path

import local_cache
import oci_s3
import torch
from transformers import AutoModel, AutoProcessor

import numpy as np
from config import (
    BASE_MODEL_REVISION,
    BASE_MODEL_URI,
    OUTPUT_TABLE_NAME,
)

# Interval projection + arrow extraction live in a dependency-light shared module
# so the offline Spark scan workflow produces byte-for-byte the same intervals.
# Re-exported here so existing `search_engine.<name>` references keep working.
from interval_core import (  # noqa: F401  (re-exported for back-compat)
    _DEFAULT_STRIDE_S,
    _DEFAULT_WINDOW_S,
    _VECTOR_COLUMNS,
    ScoredInterval,
    _chunk_ends_from_arrow,
    _chunk_starts_from_arrow,
    _drive_cells,
    _interval_threshold,
    _merge_drive,
    _vector_column_name,
    _vectors_from_arrow,
)

LOGGER = logging.getLogger(__name__)


def _resolve_model_source(model_artifact_uri: str) -> tuple[str, str | None]:
    """Return (local-or-hub source, revision) for transformers.from_pretrained.

    - Empty artifact URI -> pinned base HF model.
    - s3:// artifact URI -> download the merged snapshot to the disk cache.
    - Anything else      -> treat as a local path / hub id as-is.
    """
    if not model_artifact_uri:
        return BASE_MODEL_URI, BASE_MODEL_REVISION
    if model_artifact_uri.startswith(("s3://", "s3a://")):
        local = local_cache.ensure_model_local(model_artifact_uri, oci_s3.s3_client())
        return str(local), None
    return model_artifact_uri, None


@dataclasses.dataclass(frozen=True)
class RankedHit:
    # 1-based position in the (optionally date-filtered) descending-score order.
    # Carried on the hit so a windowed page that starts past the top still shows
    # the true global rank, not its offset within the page.
    rank: int
    # Row index into the corpus matrix, so a reviewed hit can be mapped back to
    # its embedding vector for relevance-feedback refinement (see centroid_query).
    index: int
    chunk_id: str
    run_uuid: str
    chunk_start_unix: int
    source_media_uri: str
    # Original 30s source segment_id (empty if the corpus metadata lacks it).
    segment_id: str
    score: float
    # Chunk END time (epoch seconds); None for corpora that carry only a start.
    chunk_end_unix: int | None = None


@dataclasses.dataclass
class Corpus:
    # (n, dim) row-L2-normalized embedding matrix, fp16 or fp32.
    matrix: np.ndarray
    chunk_id: list[str]
    run_uuid: list[str]
    chunk_start_unix: list[int]
    source_media_uri: list[str]
    # Original 30s source segment_ids (empty strings if the metadata lacks them).
    segment_id: list[str]
    # DORA global internal segment counter per row, for roaring-bitmap segment-set
    # filtering. None when the corpus metadata lacks the column (older corpora);
    # the segment filter then falls back to the external_id path.
    dx_internal_id: list[int] | None = None
    # Chunk END time (epoch seconds) per row; None for corpora that carry only a
    # start (e.g. the older npy metadata). Date filtering uses the start; the end
    # is carried for display + export.
    chunk_end_unix: list[int] | None = None
    # Vehicle id per row (e.g. truck-808 / a car vehicle_name), for the vehicle
    # filter. None when the corpus metadata lacks a vehicle column.
    vehicle: list[str] | None = None

    @property
    def num_rows(self) -> int:
        return self.matrix.shape[0]

    @property
    def dim(self) -> int:
        return self.matrix.shape[1] if self.matrix.ndim == 2 else 0

    def chunk_start_array(self) -> np.ndarray:
        """chunk_start_unix as a cached int64 array, for vectorized date masking.

        Built once and memoized on the instance (the corpus is cached for the
        process lifetime, so the conversion is paid a single time per corpus).
        """
        arr = self.__dict__.get("_chunk_start_arr")
        if arr is None:
            arr = np.asarray(self.chunk_start_unix, dtype=np.int64)
            self.__dict__["_chunk_start_arr"] = arr
        return arr

    def segment_id_array(self) -> np.ndarray:
        """segment_id as a cached string array, for vectorized segment-set masking.

        Memoized on the instance like ``chunk_start_array``. Rows whose metadata
        lacked a segment_id hold "" (see the loaders), so callers can detect a
        corpus with no usable segment_id via ``.any()`` on a non-empty mask.
        """
        arr = self.__dict__.get("_segment_id_arr")
        if arr is None:
            arr = np.asarray(self.segment_id, dtype=object)
            self.__dict__["_segment_id_arr"] = arr
        return arr

    def chunk_end_array(self) -> np.ndarray:
        """chunk_end_unix as a cached int64 array, for vectorized time-window
        overlap. Falls back to chunk_start (a zero-length span) when the corpus
        metadata carries no end, so callers can always treat it as [start, end]."""
        arr = self.__dict__.get("_chunk_end_arr")
        if arr is None:
            if self.chunk_end_unix is None:
                arr = self.chunk_start_array()
            else:
                arr = np.asarray(self.chunk_end_unix, dtype=np.int64)
            self.__dict__["_chunk_end_arr"] = arr
        return arr

    def run_uuid_array(self) -> np.ndarray:
        """run_uuid as a cached string array, for vectorized drive masking.
        Memoized like ``segment_id_array``."""
        arr = self.__dict__.get("_run_uuid_arr")
        if arr is None:
            arr = np.asarray(self.run_uuid, dtype=object)
            self.__dict__["_run_uuid_arr"] = arr
        return arr

    def vehicle_array(self) -> np.ndarray | None:
        """vehicle as a cached string array for vectorized vehicle masking, or
        None when the corpus has no vehicle column. Memoized like the others."""
        if self.vehicle is None:
            return None
        arr = self.__dict__.get("_vehicle_arr")
        if arr is None:
            arr = np.asarray(self.vehicle, dtype=object)
            self.__dict__["_vehicle_arr"] = arr
        return arr

    def has_vehicle(self) -> bool:
        return self.vehicle is not None

    def has_internal_ids(self) -> bool:
        """True when the corpus carries DORA internal segment counters, enabling
        the roaring-bitmap segment-set filter (one ``include_bitmap`` call vs.
        paginating the set's external_ids)."""
        return self.dx_internal_id is not None

    def dx_internal_id_array(self) -> np.ndarray:
        """dx_internal_id as a cached int64 array, for roaring-bitmap masking.

        Memoized like ``segment_id_array``. Only valid when ``has_internal_ids``;
        rows whose metadata lacked an id hold -1 (never a real DORA counter).
        """
        arr = self.__dict__.get("_dx_internal_id_arr")
        if arr is None:
            src = self.dx_internal_id or []
            arr = np.fromiter(
                (-1 if v is None else int(v) for v in src),
                dtype=np.int64,
                count=len(src),
            )
            self.__dict__["_dx_internal_id_arr"] = arr
        return arr

    def time_span(self) -> tuple[int, int]:
        """(min, max) chunk_start_unix over the corpus; (0, 0) if empty."""
        starts = self.chunk_start_array()
        if starts.size == 0:
            return (0, 0)
        return (int(starts.min()), int(starts.max()))


def load_model(model_artifact_uri: str, device: str) -> tuple[object, object]:
    """Load the Cosmos-Embed processor + model onto the configured device.

    A fine-tuned snapshot is given as an s3:// merged-model URI and is
    downloaded to the disk cache on first load; an empty URI loads the pinned
    base model from the HF hub.
    """
    source, revision = _resolve_model_source(model_artifact_uri)
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    LOGGER.info("loading model %s (device=%s, dtype=%s)", source, device, dtype)
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(
        source, revision=revision, trust_remote_code=True
    )
    model = AutoModel.from_pretrained(
        source, revision=revision, trust_remote_code=True, torch_dtype=dtype
    ).to(device)
    model.eval()
    LOGGER.info("model loaded in %.1fs", time.time() - t0)
    return processor, model


def encode_query(
    text: str, processor: object, model: object, device: str
) -> np.ndarray:
    """Encode one query string into the joint video/text space (L2-normalized).

    Mirrors text_query_search._encode_texts but device-agnostic: fp32 on CPU,
    bf16 on GPU.
    """
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    inputs = processor(text=[text])
    moved: dict[str, torch.Tensor] = {}
    for key, value in inputs.items():
        if not isinstance(value, torch.Tensor):
            continue
        moved[key] = (
            value.to(device, dtype=dtype)
            if value.is_floating_point()
            else value.to(device)
        )
    with torch.inference_mode():
        output = model.get_text_embeddings(**moved)
    vector = torch.nn.functional.normalize(output.text_proj.float(), dim=-1)
    return vector.detach().cpu().numpy().astype("float32")[0]


def load_corpus(embeddings_uri: str, matrix_dtype: str) -> Corpus:
    """Download the corpus to the disk cache (once) and load it into a resident
    matrix.

    Dispatches on the on-disk format that local_cache materialized: the fast
    `embeddings.npy` (+ metadata.parquet) is loaded by a single contiguous read
    (_load_corpus_npy); otherwise the Lance `rank=NNNNN/` shards are read and
    converted (_load_corpus_lance). The download is cached and lock-guarded, so
    repeat users of the same URI never re-pay the transfer.
    """
    local_dir = local_cache.ensure_corpus_local(embeddings_uri, oci_s3.s3_client())
    import gpu_corpus

    if gpu_corpus.is_gpu_artifact(local_dir):
        # int8 PCA corpus (very large sets) -> compressed backend that owns its
        # own scoring/sort (gpu_score / gpu_argsort), dispatched below.
        return gpu_corpus.load_gpu_corpus(local_dir, os.environ.get("NLS_DEVICE", "cpu"))
    if (local_dir / local_cache.NPY_MATRIX_FILE).exists():
        return _load_corpus_npy(local_dir, matrix_dtype)
    if embeddings_uri.rstrip("/").endswith(".lance"):
        return _load_corpus_lance_dataset(local_dir, matrix_dtype)
    return _load_corpus_lance(local_dir, embeddings_uri, matrix_dtype)


def _internal_ids_from_arrow(arrow_table: object) -> list[int] | None:
    """The dx_internal_id column as a list, or None when the corpus lacks it."""
    if "dx_internal_id" not in arrow_table.column_names:
        return None
    return arrow_table.column("dx_internal_id").to_pylist()


# Candidate vehicle-id column names in the corpus metadata (build-time join from
# the Ursa runs table, or parsed from segment_id). First present wins.
_VEHICLE_COLUMNS = ("vehicle", "vehicle_name", "vehicle_id")


def _vehicle_from_arrow(arrow_table: object) -> list[str] | None:
    """The vehicle-id column as a list of strings, or None when absent."""
    names = arrow_table.column_names
    col = next((c for c in _VEHICLE_COLUMNS if c in names), None)
    if col is None:
        return None
    LOGGER.info("vehicle column: %s", col)
    return [("" if v is None else str(v)) for v in arrow_table.column(col).to_pylist()]


def _corpus_from_arrow(arrow_table: object, matrix_dtype: str) -> Corpus:
    """Build a Corpus from a single Arrow table with the standard columns."""
    n = arrow_table.num_rows
    seg = (
        arrow_table.column("segment_id").to_pylist()
        if "segment_id" in arrow_table.column_names
        else [""] * n
    )
    vec_col = _vector_column_name(arrow_table)
    return Corpus(
        matrix=_vectors_from_arrow(arrow_table, vec_col).astype(
            matrix_dtype, copy=False
        ),
        chunk_id=arrow_table.column("chunk_id").to_pylist(),
        run_uuid=arrow_table.column("run_uuid").to_pylist(),
        chunk_start_unix=_chunk_starts_from_arrow(arrow_table),
        source_media_uri=arrow_table.column("source_media_uri").to_pylist(),
        segment_id=seg,
        dx_internal_id=_internal_ids_from_arrow(arrow_table),
        chunk_end_unix=_chunk_ends_from_arrow(arrow_table),
        vehicle=_vehicle_from_arrow(arrow_table),
    )


def _load_corpus_lance_dataset(local_dir: Path, matrix_dtype: str) -> Corpus:
    """Load a single direct Lance dataset (e.g. a `.../chunks.lance` URI)."""
    import lance

    t0 = time.time()
    ds = lance.dataset(str(local_dir))
    corpus = _corpus_from_arrow(ds.to_table(), matrix_dtype)
    LOGGER.info(
        "corpus ready (lance dataset): %d rows x %d dim (%s) in %.1fs",
        corpus.matrix.shape[0],
        corpus.dim,
        corpus.matrix.dtype,
        time.time() - t0,
    )
    return corpus


def _fast_local_copy(path: Path) -> Path | None:
    """Stage a (gcs-fuse-backed) file to local instance storage via one sequential
    streaming copy, returning the local Path -- or None if staging failed.

    np.load / parquet reads against the gcs-fuse mount issue their reads in a
    pattern FUSE serves pathologically slowly (observed ~2.6 MB/s -> 40+ min for the
    6.5GB corpus matrix on a cold instance). A single large-buffer sequential copy
    streams the bytes far faster; we then load from the local copy. On any OS error
    (e.g. no local space) we return None and the caller reads the original path.
    """
    try:
        staged = Path(tempfile.gettempdir()) / f"nls_{os.getpid()}_{path.name}"
        t = time.time()
        with open(path, "rb") as src, open(staged, "wb") as dst:
            shutil.copyfileobj(src, dst, length=64 * 1024 * 1024)
        LOGGER.info(
            "staged %s locally (%.0f MB) in %.1fs",
            path.name,
            staged.stat().st_size / 1e6,
            time.time() - t,
        )
        return staged
    except OSError as exc:
        LOGGER.warning(
            "local staging of %s failed (%s); reading directly", path.name, exc
        )
        return None


def _load_corpus_npy(local_dir: Path, matrix_dtype: str) -> Corpus:
    """Load the fast corpus format: a contiguous embeddings.npy + metadata.parquet.

    The matrix is one contiguous read (no Lance/Arrow conversion, no per-row
    fragment fetch). Both files are staged to local storage first (see
    _fast_local_copy) because reading them directly off the shared gcs-fuse mount is
    pathologically slow on a cold instance.
    """
    import pyarrow.parquet as pq

    t0 = time.time()
    mpath = local_dir / local_cache.NPY_MATRIX_FILE
    staged = _fast_local_copy(mpath)
    try:
        matrix = np.load(staged or mpath)
    finally:
        if staged is not None:
            staged.unlink(missing_ok=True)
    if str(matrix.dtype) != matrix_dtype:
        matrix = matrix.astype(matrix_dtype)
    ppath = local_dir / local_cache.NPY_METADATA_FILE
    pstaged = _fast_local_copy(ppath)
    try:
        meta = pq.read_table(pstaged or ppath)
    finally:
        if pstaged is not None:
            pstaged.unlink(missing_ok=True)
    n = matrix.shape[0]
    segment_id = (
        meta.column("segment_id").to_pylist()
        if "segment_id" in meta.column_names
        else [""] * n
    )
    corpus = Corpus(
        matrix=matrix,
        chunk_id=meta.column("chunk_id").to_pylist(),
        run_uuid=meta.column("run_uuid").to_pylist(),
        chunk_start_unix=_chunk_starts_from_arrow(meta),
        source_media_uri=meta.column("source_media_uri").to_pylist(),
        segment_id=segment_id,
        dx_internal_id=_internal_ids_from_arrow(meta),
        chunk_end_unix=_chunk_ends_from_arrow(meta),
        vehicle=_vehicle_from_arrow(meta),
    )
    LOGGER.info(
        "corpus ready (npy): %d rows x %d dim (%s) in %.1fs",
        matrix.shape[0],
        matrix.shape[1],
        matrix.dtype,
        time.time() - t0,
    )
    return corpus


def _load_corpus_lance(
    local_dir: Path, embeddings_uri: str, matrix_dtype: str
) -> Corpus:
    """Read the rank=NNNNN/ Lance shards into one resident matrix."""
    import lancedb

    rank_dirs = sorted(
        d for d in local_dir.iterdir() if d.is_dir() and d.name.startswith("rank=")
    )
    if not rank_dirs:
        raise FileNotFoundError(f"no rank=NNNNN/ dirs in cached download {local_dir}")
    LOGGER.info("loading %d cached rank shards from %s", len(rank_dirs), local_dir)

    matrices: list[np.ndarray] = []
    chunk_id: list[str] = []
    run_uuid: list[str] = []
    chunk_start_unix: list[int] = []
    source_media_uri: list[str] = []
    segment_id: list[str] = []
    internal_ids: list[int] = []
    has_internal = True  # cleared if any shard lacks the dx_internal_id column
    for rank_dir in rank_dirs:
        db = lancedb.connect(str(rank_dir))
        table = db.open_table(OUTPUT_TABLE_NAME)
        arrow_table = table.to_arrow()
        if arrow_table.num_rows == 0:
            continue
        matrices.append(_vectors_from_arrow(arrow_table).astype(matrix_dtype))
        chunk_id.extend(arrow_table.column("chunk_id").to_pylist())
        run_uuid.extend(arrow_table.column("run_uuid").to_pylist())
        chunk_start_unix.extend(
            int(v) for v in arrow_table.column("chunk_start_unix").to_pylist()
        )
        source_media_uri.extend(arrow_table.column("source_media_uri").to_pylist())
        if "segment_id" in arrow_table.column_names:
            segment_id.extend(arrow_table.column("segment_id").to_pylist())
        else:
            segment_id.extend([""] * arrow_table.num_rows)
        if has_internal and "dx_internal_id" in arrow_table.column_names:
            internal_ids.extend(arrow_table.column("dx_internal_id").to_pylist())
        else:
            has_internal = False
        LOGGER.info("  %s: %d rows", rank_dir.name, arrow_table.num_rows)

    if not matrices:
        raise ValueError(f"all shards under {embeddings_uri} were empty")
    matrix = np.concatenate(matrices, axis=0)
    LOGGER.info(
        "corpus ready: %d rows x %d dim (%s)",
        matrix.shape[0],
        matrix.shape[1],
        matrix.dtype,
    )
    return Corpus(
        matrix=matrix,
        chunk_id=chunk_id,
        run_uuid=run_uuid,
        chunk_start_unix=chunk_start_unix,
        source_media_uri=source_media_uri,
        segment_id=segment_id,
        dx_internal_id=internal_ids if has_internal else None,
    )


def _hit(corpus: Corpus, scores: np.ndarray, index: int, rank: int) -> RankedHit:
    return RankedHit(
        rank=rank,
        index=int(index),
        chunk_id=corpus.chunk_id[index],
        run_uuid=corpus.run_uuid[index],
        chunk_start_unix=corpus.chunk_start_unix[index],
        source_media_uri=corpus.source_media_uri[index],
        segment_id=corpus.segment_id[index],
        score=float(scores[index]),
        chunk_end_unix=(
            corpus.chunk_end_unix[index] if corpus.chunk_end_unix is not None else None
        ),
    )


def rank_top_k(query_vector: np.ndarray, corpus: Corpus, top_k: int) -> list[RankedHit]:
    """Score the corpus by cosine similarity and return the top-k hits."""
    if corpus.num_rows == 0:
        return []
    if hasattr(corpus, "gpu_score"):
        scores = corpus.gpu_score(query_vector)
        order = corpus.gpu_argsort(scores, None)
        top = order[: min(top_k, scores.shape[0])]
        return [_hit(corpus, scores, int(i), r) for r, i in enumerate(top, start=1)]
    # Match query dtype to the matrix so an fp32 corpus hits the BLAS gemv
    # path (~58ms at 1M x 768). scores stay fp32-safe via float() below.
    scores = corpus.matrix @ query_vector.astype(corpus.matrix.dtype)
    k = min(top_k, scores.shape[0])
    top = np.argpartition(-scores, k - 1)[:k]
    top = top[np.argsort(-scores[top])]
    return [_hit(corpus, scores, int(i), rank) for rank, i in enumerate(top, start=1)]


def score_corpus(query_vector: np.ndarray, corpus: Corpus) -> np.ndarray:
    """Cosine similarity of the query against every corpus row (1-D, fp-matrix dtype)."""
    if hasattr(corpus, "gpu_score"):
        # int8 PCA backend: project + dequant-fold + blocked int8 matmul.
        return corpus.gpu_score(query_vector)
    return corpus.matrix @ query_vector.astype(corpus.matrix.dtype)


def segment_mask(
    corpus: Corpus, allowed_segment_ids: frozenset[str] | set[str]
) -> np.ndarray:
    """Boolean mask over corpus rows: True where the row's segment_id is allowed.

    Uses O(1) set membership over the corpus's ~10^5 rows, so the cost scales
    with the corpus, NOT the segment set (which can be millions of ids) -- and
    there is no per-call sort of the id set. The result is immutable for a given
    (corpus, set), so callers cache it and reuse it across searches/pages.
    """
    seg = corpus.segment_id_array()
    return np.fromiter(
        (sid in allowed_segment_ids for sid in seg), dtype=bool, count=len(seg)
    )


def vehicle_mask(
    corpus: Corpus, allowed_vehicles: frozenset[str] | set[str]
) -> np.ndarray | None:
    """Boolean mask over corpus rows: True where the row's vehicle is allowed.

    O(corpus) set membership over the vehicle column, mirroring ``segment_mask``.
    Returns None when the corpus has no vehicle column (filter inert) so callers
    can skip it without special-casing. Matching is exact on the vehicle id.
    """
    veh = corpus.vehicle_array()
    if veh is None:
        return None
    return np.fromiter(
        (v in allowed_vehicles for v in veh), dtype=bool, count=len(veh)
    )


def run_mask(
    corpus: Corpus, allowed_runs: frozenset[str] | set[str]
) -> np.ndarray:
    """Boolean mask over corpus rows: True where the row's run_uuid (drive id) is
    allowed. run_uuid is always present, so (unlike ``vehicle_mask``) this never
    returns None. Exact match on the full run_uuid; O(corpus) set membership."""
    runs = corpus.run_uuid
    return np.fromiter(
        (r in allowed_runs for r in runs), dtype=bool, count=len(runs)
    )


def segment_mask_from_bitmap(corpus: Corpus, set_bitmap) -> np.ndarray:
    """Boolean mask over corpus rows via DORA's roaring bitmap of the segment set.

    ``set_bitmap`` is a ``pyroaring.BitMap`` of the set's global internal segment
    counters (one ``DescribeDataSet(include_bitmap=True)`` call -- no pagination).
    Membership is O(1) per row, so cost scales with the corpus, not the set.
    Requires ``corpus.has_internal_ids()``.
    """
    ids = corpus.dx_internal_id_array()
    # Rows without a DORA internal id hold -1 (see dx_internal_id_array). pyroaring's
    # membership test only accepts uint32, so a negative sentinel raises OverflowError.
    # A -1 row is never in the set, so short-circuit it to False before the lookup.
    return np.fromiter(
        (i >= 0 and int(i) in set_bitmap for i in ids), dtype=bool, count=len(ids)
    )


# Columns a downsample dataset may key on, in priority order, mapped to the
# corpus column they cross-reference against. ``scenario_id`` is an alias for
# ``segment_id`` (same id space, different producer). ``run_uuid`` is a curated
# run list (every corpus also carries run_uuid). We use the FIRST column the
# dataset has and intersect it against the corpus's mapped column.
_FILTER_KEY_COLUMNS: tuple[tuple[str, str], ...] = (
    ("dx_internal_id", "dx_internal_id"),
    ("segment_id", "segment_id"),
    ("scenario_id", "segment_id"),
    ("run_uuid", "run_uuid"),
)

# Offline-scan downsample priority: segment_id-first, NO dx_internal_id. The Lilypad scan
# worker matches on the segment_id string space (from chunks_metadata / the npy corpus), not
# the corpus dx-internal-id bitmap, so resolve a string key the worker can use.
_SCAN_FILTER_KEY_COLUMNS: tuple[tuple[str, str], ...] = (
    ("segment_id", "segment_id"),
    ("scenario_id", "segment_id"),
    ("run_uuid", "run_uuid"),
)


def read_filter_ids(
    local_dir: Path, is_lance: bool, key_columns: tuple[tuple[str, str], ...] | None = None
) -> tuple[str, frozenset[str]]:
    """``(corpus_key, ids)`` for a downsample dataset (lance dir or parquet files).

    Backs the "downsample by a Lance/parquet dataset" filter: the user points at
    an arbitrary dataset whose ``segment_id`` / ``scenario_id`` (segment id space)
    or ``run_uuid`` column defines what to keep; we intersect it against the
    corpus's mapped column, reusing the ``value_mask`` membership primitive.
    Returns the CORPUS key (``segment_id`` or ``run_uuid``) so callers mask the
    right column. Raises ``ValueError`` if the dataset has no usable key column.
    ``key_columns`` overrides the dataset-column -> corpus-key priority (default
    ``_FILTER_KEY_COLUMNS``; the offline scan passes ``_SCAN_FILTER_KEY_COLUMNS``).
    """
    cols = key_columns or _FILTER_KEY_COLUMNS
    if is_lance:
        import lance

        names = lance.dataset(str(local_dir)).schema.names
    else:
        import pyarrow.dataset as pads

        files = [str(p) for p in sorted(local_dir.rglob("*.parquet"))]
        if not files:
            raise ValueError("no .parquet files in downsample dataset")
        names = pads.dataset(files, format="parquet").schema.names

    match = next((p for p in cols if p[0] in names), None)
    if match is None:
        wanted = " / ".join(c for c, _ in cols)
        raise ValueError(
            f"downsample dataset has no {wanted} column to cross-reference "
            f"(has {sorted(names)})"
        )
    dataset_col, corpus_key = match
    if is_lance:
        import lance

        col = (
            lance.dataset(str(local_dir))
            .to_table(columns=[dataset_col])
            .column(dataset_col)
            .to_pylist()
        )
    else:
        import pyarrow.dataset as pads

        files = [str(p) for p in sorted(local_dir.rglob("*.parquet"))]
        col = (
            pads.dataset(files, format="parquet")
            .to_table(columns=[dataset_col])
            .column(dataset_col)
            .to_pylist()
        )
    if corpus_key == "dx_internal_id":
        # int roaring-bitmap key -- keep as ints, not strings.
        return corpus_key, frozenset(int(s) for s in col if s is not None)
    return corpus_key, frozenset(str(s) for s in col if s is not None and s != "")


def value_mask(values: list[str], allowed: frozenset[str] | set[str]) -> np.ndarray:
    """Boolean mask over ``values`` (a corpus column) -- True where allowed.

    Generalizes ``segment_mask`` to any string column (e.g. ``run_uuid``), so the
    downsample-dataset filter can cross-reference whichever key the dataset has.
    """
    return np.fromiter((v in allowed for v in values), dtype=bool, count=len(values))


def ranked_order(
    scores: np.ndarray,
    corpus: Corpus,
    start_unix: int | None = None,
    end_unix: int | None = None,
    allowed_segment_ids: frozenset[str] | set[str] | None = None,
    allowed_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Corpus row indices sorted by descending score, optionally filtered.

    Rows are kept only if ``start_unix <= chunk_start_unix < end_unix`` (either
    bound may be ``None`` to leave that side open) AND, when
    ``allowed_segment_ids`` is given, the row's ``segment_id`` is in that set
    (the Data Explorer segment-set downsample). A full argsort over ~10^5 rows
    is a few ms, so this runs fresh on every rerun rather than being cached.
    """
    n = scores.shape[0]
    has_seg = allowed_mask is not None or allowed_segment_ids is not None
    if start_unix is None and end_unix is None and not has_seg:
        idx = np.arange(n)
    else:
        mask = np.ones(n, dtype=bool)
        if start_unix is not None or end_unix is not None:
            starts = corpus.chunk_start_array()
            if start_unix is not None:
                mask &= starts >= start_unix
            if end_unix is not None:
                mask &= starts < end_unix
        if allowed_mask is not None:
            mask &= allowed_mask
        elif allowed_segment_ids is not None:
            mask &= segment_mask(corpus, allowed_segment_ids)
        idx = np.nonzero(mask)[0]
    # int8 backend: sort on its own device/kernel (the CPU path caps the
    # unfiltered case for speed). None lets it reuse the resident score tensor.
    if hasattr(corpus, "gpu_argsort"):
        filtered = start_unix is not None or end_unix is not None or has_seg
        return corpus.gpu_argsort(scores, idx if filtered else None)
    # Stable sort so equal scores keep corpus order (deterministic paging).
    return idx[np.argsort(-scores[idx], kind="stable")]


def filter_funnel(
    corpus: Corpus,
    start_unix: int | None = None,
    end_unix: int | None = None,
    allowed_segment_ids: frozenset[str] | set[str] | None = None,
    allowed_mask: np.ndarray | None = None,
) -> dict:
    """How many corpus clips survive each filter independently + combined.

    Returns counts so the UI can show the funnel from the full corpus down to the
    matched set (e.g. 100k corpus -> 70k in date range -> 9.5k in segment set ->
    16 in both). The query is a ranker, not a filter, so it does not appear here.
    """
    n = corpus.num_rows
    date_mask = None
    if start_unix is not None or end_unix is not None:
        starts = corpus.chunk_start_array()
        date_mask = np.ones(n, dtype=bool)
        if start_unix is not None:
            date_mask &= starts >= start_unix
        if end_unix is not None:
            date_mask &= starts < end_unix
    seg_mask = None
    if allowed_mask is not None:
        seg_mask = allowed_mask
    elif allowed_segment_ids is not None:
        seg_mask = segment_mask(corpus, allowed_segment_ids)

    combined = np.ones(n, dtype=bool)
    if date_mask is not None:
        combined &= date_mask
    if seg_mask is not None:
        combined &= seg_mask
    return {
        "corpus_total": int(n),
        "in_date_range": int(date_mask.sum()) if date_mask is not None else int(n),
        "in_segment_set": int(seg_mask.sum()) if seg_mask is not None else None,
        "matched": int(combined.sum()),
        "date_filtered": date_mask is not None,
        "segment_filtered": seg_mask is not None,
    }


def start_index_for_score(
    scores: np.ndarray, order: np.ndarray, score_threshold: float
) -> int:
    """First position in `order` whose score is <= score_threshold.

    `order` is already sorted by descending score, so this is a binary search:
    everything before the returned offset scores strictly higher than the
    threshold. Returns len(order) if every score exceeds the threshold.
    """
    if order.size == 0:
        return 0
    sorted_desc = scores[order]
    # -sorted_desc is ascending; first element >= -threshold is the first hit
    # whose own score is <= threshold.
    return int(np.searchsorted(-sorted_desc, -float(score_threshold), side="left"))


def hits_from_order(
    corpus: Corpus, scores: np.ndarray, order: np.ndarray, start: int, count: int
) -> list[RankedHit]:
    """Materialize a window of `count` hits from `order` beginning at `start`.

    `start` is a 0-based offset into the ranking; each hit's `rank` is its
    1-based global position (start + 1, start + 2, ...).
    """
    window = order[start : start + count]
    return [
        _hit(corpus, scores, int(i), start + 1 + offset)
        for offset, i in enumerate(window)
    ]


def _build_drives(scores, corpus, allowed_mask):
    """Group the allowed rows by drive and build each drive's 4s-cell signal.

    Returns a list of ``(run_uuid, cell_score, cell_center, cell_peak_row)``,
    one per drive (the shared front half of interval projection)."""
    n = corpus.num_rows
    if n == 0:
        return []
    idx = np.nonzero(allowed_mask)[0] if allowed_mask is not None else np.arange(n)
    if idx.size == 0:
        return []
    starts_all = corpus.chunk_start_array()
    if corpus.chunk_end_unix is not None:
        ends_all = np.asarray(corpus.chunk_end_unix, dtype=np.int64)
    else:
        ends_all = starts_all + _DEFAULT_WINDOW_S
    runs = np.asarray([corpus.run_uuid[i] for i in idx], dtype=object)
    g = np.lexsort((starts_all[idx], runs))  # sort by (run, start)
    idx_s = idx[g]
    runs_s = runs[g]
    boundaries = np.nonzero(runs_s[1:] != runs_s[:-1])[0] + 1
    out = []
    for grp in np.split(np.arange(idx_s.size), boundaries):
        rows = idx_s[grp]
        cs, cc, cpr, _d = _drive_cells(
            starts_all[rows], ends_all[rows], scores[rows].astype(np.float64), rows
        )
        out.append((corpus.run_uuid[int(rows[0])], cs, cc, cpr))
    return out


def project_intervals(
    scores: np.ndarray,
    corpus: Corpus,
    allowed_mask: np.ndarray | None,
    *,
    mode: str = "k",
    k: int = 100,
    score_cutoff: float | None = None,
) -> tuple[list[ScoredInterval], float]:
    """Merge the per-clip scores into variable-length intervals per drive.

    ``allowed_mask`` is the final boolean keep-mask over corpus rows (date +
    segment-set + lance + vehicle), or None for the whole corpus. ``mode`` is
    ``"k"`` (threshold = the k-th largest grid-cell score, so ~k cells survive)
    or ``"score"`` (threshold = ``score_cutoff``). Returns
    ``(intervals_sorted_by_peak_desc, threshold_used)``.

    The grid cell math is in ``_drive_cells``; here we threshold, walk each
    drive's cells, and linearly interpolate the two boundary crossings.
    """
    per_drive = _build_drives(scores, corpus, allowed_mask)
    if not per_drive:
        return [], 0.0
    tau = _interval_threshold(per_drive, mode, k, score_cutoff)
    if tau is None:
        return [], 0.0
    intervals: list[ScoredInterval] = []
    for run_uuid, cs, cc, cpr in per_drive:
        intervals.extend(_merge_drive(run_uuid, cs, cc, cpr, tau))
    intervals.sort(key=lambda iv: iv.peak_score, reverse=True)
    return intervals, tau


def interval_threshold(
    scores: np.ndarray,
    corpus: Corpus,
    allowed_mask: np.ndarray | None,
    *,
    mode: str = "k",
    k: int = 100,
    score_cutoff: float | None = None,
) -> float | None:
    """The cell-score threshold tau that ``project_intervals`` would use for
    these scores + filters, WITHOUT building the intervals (used by the
    similarity-distribution endpoint). ``"score"`` mode returns ``score_cutoff``;
    ``"k"`` mode returns
    the k-th largest 4s-cell score pooled across drives (None if no finite cells).
    """
    per_drive = _build_drives(scores, corpus, allowed_mask)
    if not per_drive:
        return None
    return _interval_threshold(per_drive, mode, k, score_cutoff)


def _unit(vector: np.ndarray) -> np.ndarray:
    """L2-normalize to a unit fp32 vector; raise if it has zero norm."""
    norm = np.linalg.norm(vector)
    if norm == 0.0:
        raise ValueError("vector has zero norm; cannot normalize")
    return (vector / norm).astype("float32")


def refine_query(
    corpus: Corpus,
    positive_indices: list[int],
    negative_indices: list[int] | None = None,
    text_vector: np.ndarray | None = None,
    negative_weight: float = 0.5,
    text_weight: float = 0.0,
) -> np.ndarray:
    """Build a refined query direction from reviewed examples (relevance feedback).

    This is a *prototype* (Rocchio) objective, deliberately not a max-margin
    classifier. Retrieval wants a vector close to the confirmed positives and
    away from the negatives -- not the boundary that best separates them. With
    only a handful of marks in a ~768-d space a discriminative separator is
    badly underdetermined: it latches onto whichever spurious plane happens to
    split the current marks and gets *worse* as you iterate. The prototype
    direction instead stays anchored to the positive centroid (a low-variance
    estimate), so it degrades gracefully and converges toward the true cluster.

    Maximizing ``alignment(w, positives) - negative_weight * alignment(w,
    negatives)`` over unit ``w`` has the closed form::

        w = unit( mean(positives) - negative_weight * mean(negatives) )

    ``negative_weight`` (gamma) scales how hard to reject the negatives; 0
    ignores them (pure positive centroid). ``text_weight > 0`` then blends the
    original text query back in (Rocchio's query term) as an extra anchor:
    ``unit(w + text_weight * unit(text_vector))``. The result is L2-normalized,
    so it drops straight into `rank_top_k` like any query vector.
    """
    if not positive_indices:
        raise ValueError("refine_query needs at least one positive example")
    direction = corpus.matrix[positive_indices].astype("float32").mean(axis=0)
    if negative_indices:
        neg_mean = corpus.matrix[negative_indices].astype("float32").mean(axis=0)
        direction = direction - float(negative_weight) * neg_mean

    direction = _unit(direction)
    if text_vector is not None and text_weight > 0.0:
        direction = _unit(direction + float(text_weight) * _unit(text_vector))
    return direction


@dataclasses.dataclass(frozen=True)
class WindowMatch:
    """The in-corpus chunks that make up a video-window query.

    ``indices`` are corpus row indices of every mini-segment overlapping the
    requested [start, end] window of the requested drive/segment; ``vector`` is
    their L2-normalized mean (the query-by-example direction, drop-in for
    ``rank_top_k`` / ``score_corpus``). ``preview`` is an evenly-sampled handful
    of those indices for the UI filmstrip (count scales with the window length).
    """

    vector: np.ndarray
    indices: np.ndarray
    preview: list[int]
    span_seconds: int


def window_query(
    corpus: Corpus,
    *,
    run_uuid: str = "",
    segment_id: str = "",
    start_unix: int = 0,
    end_unix: int = 0,
    max_preview: int = 12,
) -> WindowMatch:
    """Resolve a video-window query against the *pre-embedded* corpus.

    Given a drive (``run_uuid``) or 30s source segment (``segment_id``) and an
    optional [start_unix, end_unix] time window (unix seconds), select the
    mini-segment chunks already embedded in the corpus that overlap it, and mean-
    pool their embeddings into one query vector (same prototype direction as
    ``refine_query``). No model inference or MP4 decode -- the query clip's
    embeddings are read straight out of the corpus.

    Raises ``ValueError`` if neither key is given or no chunk matches (e.g. the
    drive was never embedded, or the window falls outside its coverage).
    """
    run_uuid = (run_uuid or "").strip()
    segment_id = (segment_id or "").strip()
    if not run_uuid and not segment_id:
        raise ValueError("window_query needs a run_uuid or a segment_id")

    mask = np.ones(corpus.num_rows, dtype=bool)
    if run_uuid:
        mask &= corpus.run_uuid_array() == run_uuid
    if segment_id:
        mask &= corpus.segment_id_array() == segment_id
    # Half-open overlap test: a chunk [cs, ce) overlaps [start, end) iff
    # cs < end and ce > start. Either bound 0/unset opens that side.
    if start_unix:
        mask &= corpus.chunk_end_array() > int(start_unix)
    if end_unix:
        mask &= corpus.chunk_start_array() < int(end_unix)

    idx = np.nonzero(mask)[0]
    if idx.size == 0:
        key = run_uuid or segment_id
        raise ValueError(
            f"no embedded chunks match {key!r} in the requested window -- the "
            "drive/segment may not be in this corpus, or the window is outside "
            "its coverage"
        )

    # Order the matched chunks chronologically so the preview filmstrip reads in
    # time order and the mean is over the window as the user sees it.
    starts = corpus.chunk_start_array()[idx]
    idx = idx[np.argsort(starts, kind="stable")]
    # Same prototype direction as relevance feedback: mean of the matched rows,
    # L2-normalized (centroid_query -> refine_query, positives only).
    vector = centroid_query(corpus, idx.tolist())

    # Show a handful, count scaling with the window length (~one per 8s chunk),
    # evenly sampled across the matched chunks and capped at ``max_preview``.
    span = int(corpus.chunk_end_array()[idx][-1] - corpus.chunk_start_array()[idx][0])
    n_preview = int(min(max_preview, max(1, idx.size)))
    sample = np.linspace(0, idx.size - 1, num=n_preview).round().astype(int)
    preview = [int(idx[s]) for s in np.unique(sample)]
    return WindowMatch(
        vector=vector, indices=idx, preview=preview, span_seconds=max(span, 0)
    )


# Multi-cluster-positive thresholds (tunable; defaults are conservative so we
# only split genuinely diverse 👍 sets — over-clustering a coherent set is the
# documented failure mode). All are cosine similarities of unit vectors.
_POS_TIGHT_SIM = 0.5    # if MEAN pairwise cos among 👍 >= this -> one coherent prototype
_POS_LINK_SIM = 0.5     # greedy grouping: join a cluster if cos to its running mean >= this
_POS_MAX_CLUSTERS = 3   # more clusters than this -> treat as noise, fall back to one prototype


def _positive_clusters(vecs: np.ndarray) -> list[list[int]]:
    """Group unit positive vectors into 1..``_POS_MAX_CLUSTERS`` clusters.

    Returns a list of member-index lists (indices into ``vecs``). Falls back to a
    SINGLE cluster (all members) when the positives are cohesive (high mean
    pairwise similarity) or when greedy grouping fragments into too many pieces --
    so multi-prototype only engages for genuinely multi-modal 👍 sets. Diversity
    gate first (mean pairwise cosine), then a single-pass greedy single-link
    grouping on cosine. Mean-pairwise tolerates one outlier mark, unlike the
    min-to-centroid which a balanced bimodal set also passes.
    """
    m = vecs.shape[0]
    if m <= 2:
        return [list(range(m))]  # too few marks to cluster reliably
    sims = vecs @ vecs.T
    mean_pair = float((sims.sum() - m) / (m * (m - 1)))  # exclude the unit diagonal
    if mean_pair >= _POS_TIGHT_SIM:
        return [list(range(m))]  # cohesive -> one prototype (== single-centroid behavior)
    members: list[list[int]] = []
    means: list[np.ndarray] = []
    for i in range(m):
        v = vecs[i]
        best, best_sim = -1, _POS_LINK_SIM
        for ci, mean in enumerate(means):
            s = float(v @ mean)
            if s >= best_sim:
                best, best_sim = ci, s
        if best < 0:
            members.append([i])
            means.append(v.copy())
        else:
            members[best].append(i)
            acc = vecs[members[best]].mean(axis=0)
            nn = float(np.linalg.norm(acc))
            means[best] = acc / nn if nn else acc
    if len(members) <= 1 or len(members) > _POS_MAX_CLUSTERS:
        return [list(range(m))]  # not actually multi-modal / too fragmented
    return members


def refine_scores(
    corpus: Corpus,
    positive_indices: list[int],
    negative_indices: list[int] | None = None,
    text_vector: np.ndarray | None = None,
    negative_weight: float = 0.5,
    text_weight: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Relevance-feedback scores with MULTI-CLUSTER positives and a PER-NEGATIVE
    penalty::

        score(x) = max_c cos(x, w_c)  -  negative_weight * max_j cos(x, neg_j)

    Positives are grouped into 1..3 cluster prototypes ``w_c`` (each a centroid +
    the optional text-query anchor). A coherent 👍 set yields ONE prototype (the
    classic single-centroid behavior); a diverse set yields several, and each row
    is scored by its NEAREST positive cluster (the ``max_c``) -- so two distinct
    👍 themes both rank high instead of averaging into a meaningless midpoint.
    The negative term demotes any row close to ANY specific rejected example.

    Returns ``(scores, w_pos)``: ``w_pos`` is the overall positive centroid (all
    👍 averaged + anchor) -- the single vector persisted for export / the offline
    scan. Neither the multi-cluster max nor the negative penalty is expressible as
    one vector, so export/scan use this representative centroid.
    """
    if not positive_indices:
        raise ValueError("refine_scores needs at least one positive example")

    def _prototype(local_idxs: list[int]) -> np.ndarray:
        # Reuse refine_query for the unit(mean(..)) + text-anchor logic exactly.
        global_idxs = [positive_indices[i] for i in local_idxs]
        return refine_query(
            corpus, global_idxs, negative_indices=None,
            text_vector=text_vector, negative_weight=0.0, text_weight=text_weight,
        )

    pos = corpus.matrix[positive_indices].astype("float32")  # unit rows
    clusters = _positive_clusters(pos)
    protos = [_prototype(c) for c in clusters]
    # Rank by the NEAREST positive cluster prototype.
    scores = np.maximum.reduce([score_corpus(w, corpus) for w in protos])
    # Representative single vector to persist (export / offline scan).
    w_pos = protos[0] if len(clusters) == 1 else _prototype(list(range(pos.shape[0])))

    if negative_indices and negative_weight > 0.0:
        # max over negatives: one score_corpus pass per 👎 (reuses the app's exact
        # dtype handling), then element-wise max -> closeness to the nearest 👎.
        penalty = np.maximum.reduce(
            [score_corpus(corpus.matrix[j], corpus) for j in negative_indices]
        )
        scores = scores - float(negative_weight) * penalty.astype(scores.dtype)
    return scores, w_pos


def centroid_query(corpus: Corpus, indices: list[int]) -> np.ndarray:
    """Nearest-centroid relevance feedback over the given positive rows.

    Thin wrapper over `refine_query` (positives only) for the simple case and
    existing callers.
    """
    return refine_query(corpus, positive_indices=indices)


# ---------------------------------------------------------------------------
# Threshold search: choose a per-tag score cutoff from labeled 👍/👎 examples.
#
# Picking a cosine cutoff from labeled positives/negatives is binary-classifier
# operating-point selection on a 1-D score. We sweep EVERY candidate threshold
# (each observed score) -- exact and O(n log n), no random subsampling -- and
# pick the one that optimizes an explicit objective. The companion sampler feeds
# an active-labeling loop so the cutoff can be found from a handful of clicks
# instead of assuming a large pre-existing label set.
# ---------------------------------------------------------------------------


def _pr_sweep(
    pos_scores: np.ndarray, neg_scores: np.ndarray
) -> dict[str, np.ndarray]:
    """Precision/recall/TPR/FPR at every distinct candidate threshold.

    A hit is ``score >= tau``. Sorting the pooled labeled scores descending and
    walking the prefix gives, at each cut, ``TP = #pos >= tau`` and
    ``FP = #neg >= tau``. We only evaluate at the LAST index of each equal-score
    run so a threshold never splits identical scores. Returns arrays aligned by
    candidate (``tau`` descending: high-precision/low-recall first).
    """
    n_pos = int(pos_scores.size)
    n_neg = int(neg_scores.size)
    scores = np.concatenate([pos_scores, neg_scores]).astype(np.float64)
    labels = np.concatenate(
        [np.ones(n_pos, dtype=np.int64), np.zeros(n_neg, dtype=np.int64)]
    )
    order = np.argsort(-scores, kind="mergesort")  # stable, score-descending
    s = scores[order]
    y = labels[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    # Keep only the last position of each run of equal scores: a valid tau sits
    # between distinct values, so identical scores must fall on the same side.
    keep = np.ones(s.size, dtype=bool)
    keep[:-1] = s[1:] != s[:-1]
    tau = s[keep]
    tp = tp[keep].astype(np.float64)
    fp = fp[keep].astype(np.float64)
    precision = np.divide(tp, tp + fp, out=np.ones_like(tp), where=(tp + fp) > 0)
    recall = tp / n_pos if n_pos else np.zeros_like(tp)  # == TPR
    fpr = fp / n_neg if n_neg else np.zeros_like(fp)
    return {
        "tau": tau,
        "precision": precision,
        "recall": recall,
        "fpr": fpr,
        "n_pos": np.int64(n_pos),
        "n_neg": np.int64(n_neg),
    }


def _metrics_at(
    pos_scores: np.ndarray, neg_scores: np.ndarray, tau: float, beta: float
) -> dict[str, float]:
    """Precision/recall/F-beta of the cut ``score >= tau`` on a labeled set."""
    tp = float(np.count_nonzero(pos_scores >= tau))
    fn = float(pos_scores.size) - tp
    fp = float(np.count_nonzero(neg_scores >= tau))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    b2 = beta * beta
    denom = b2 * precision + recall
    fbeta = (1 + b2) * precision * recall / denom if denom else 0.0
    return {"precision": precision, "recall": recall, "f1": fbeta}


def _pick_tau(sweep: dict[str, np.ndarray], objective: str, beta: float,
              min_precision: float) -> tuple[float, bool]:
    """Index the sweep for the best tau under ``objective``.

    Returns ``(tau, ok)`` where ``ok`` is False only for the precision-floor
    objective when no threshold reaches ``min_precision`` (caller can surface a
    warning); in that case we fall back to the highest-precision threshold.
    """
    p, r, tau = sweep["precision"], sweep["recall"], sweep["tau"]
    if objective == "youden":
        return float(tau[int(np.argmax(r - sweep["fpr"]))]), True
    if objective == "precision":
        ok_mask = p >= min_precision
        if ok_mask.any():
            # Highest recall among thresholds that clear the precision floor;
            # ties -> the higher tau (fewer, cleaner hits).
            cand = np.nonzero(ok_mask)[0]
            best = cand[int(np.argmax(r[cand]))]
            return float(tau[best]), True
        return float(tau[int(np.argmax(p))]), False  # nothing clears the floor
    # default: maximize F-beta
    b2 = beta * beta
    denom = b2 * p + r
    fbeta = np.divide((1 + b2) * p * r, denom, out=np.zeros_like(p), where=denom > 0)
    return float(tau[int(np.argmax(fbeta))]), True


def fit_threshold(
    pos_scores: np.ndarray,
    neg_scores: np.ndarray,
    *,
    objective: str = "f1",
    beta: float = 1.0,
    min_precision: float = 0.9,
    val_fraction: float = 0.0,
    max_curve_points: int = 200,
    seed: int = 0,
) -> dict:
    """Choose a score cutoff separating labeled positives from negatives.

    ``objective`` selects what "best" means:
      - ``"f1"``:        maximize F-beta (default beta=1; the usual retrieval default).
      - ``"youden"``:    maximize TPR - FPR (Youden's J, the ROC-classic pick).
      - ``"precision"``: max recall subject to precision >= ``min_precision``
                         (operational: "don't waste reviewer time").

    When ``val_fraction`` > 0 the threshold is chosen on a train split and the
    reported precision/recall/f1 are measured on a held-out split, so the numbers
    are not optimistically biased by fitting-and-scoring the same labels. The full
    PR curve (subsampled to ``max_curve_points``) is returned for plotting.

    Raises ``ValueError`` if either class is empty (no threshold is meaningful).
    """
    pos = np.asarray(pos_scores, dtype=np.float64).ravel()
    neg = np.asarray(neg_scores, dtype=np.float64).ravel()
    if pos.size == 0 or neg.size == 0:
        raise ValueError("fit_threshold needs at least one positive and one negative")

    fit_pos, fit_neg, eval_pos, eval_neg = pos, neg, pos, neg
    held_out = False
    if val_fraction and 0.0 < val_fraction < 1.0:
        rng = np.random.default_rng(seed)

        def _split(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            idx = rng.permutation(a.size)
            cut = max(1, int(round(a.size * (1.0 - val_fraction))))
            return a[idx[:cut]], a[idx[cut:]]

        tp_, vp_ = _split(pos)
        tn_, vn_ = _split(neg)
        # Only honor the split if BOTH val classes are non-empty; else fit==eval.
        if vp_.size and vn_.size:
            fit_pos, fit_neg, eval_pos, eval_neg = tp_, tn_, vp_, vn_
            held_out = True

    sweep = _pr_sweep(fit_pos, fit_neg)
    tau, ok = _pick_tau(sweep, objective, beta, min_precision)
    metrics = _metrics_at(eval_pos, eval_neg, tau, beta)

    # Subsample the PR curve for transport (keep endpoints).
    tau_c, p_c, r_c = sweep["tau"], sweep["precision"], sweep["recall"]
    if tau_c.size > max_curve_points:
        sel = np.linspace(0, tau_c.size - 1, max_curve_points).round().astype(int)
        sel = np.unique(sel)
        tau_c, p_c, r_c = tau_c[sel], p_c[sel], r_c[sel]
    # Average precision (area under PR, recall-weighted) as a single quality number.
    ap = float(np.sum(np.diff(np.concatenate([[0.0], sweep["recall"]])) * sweep["precision"]))

    return {
        "threshold": tau,
        "objective": objective,
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "average_precision": ap,
        "n_pos": int(pos.size),
        "n_neg": int(neg.size),
        "held_out": held_out,
        "precision_floor_met": bool(ok),
        "curve": {
            "tau": tau_c.tolist(),
            "precision": p_c.tolist(),
            "recall": r_c.tolist(),
        },
    }


def stratified_boundary_sample(
    scores: np.ndarray,
    candidate_mask: np.ndarray | None,
    labeled: "set[int] | np.ndarray | None",
    n: int,
    *,
    tau: float | None = None,
    band: float = 0.08,
    seed: int = 0,
) -> list[int]:
    """Pick up to ``n`` UNLABELED rows to hand the user for active labeling.

    Uniform-random sampling wastes the budget on the rare-positive corpus's flood
    of obvious negatives, none of which sit near the cutoff. Instead we bias toward
    the decision boundary (uncertainty sampling): about half the budget is drawn
    from the uncertain band ``[tau-band, tau+band]`` and the rest is spread across
    score quantiles -- so the high tail (positives, for recall) and low tail (clear
    negatives) still appear. Already-``labeled`` rows are excluded. With ``tau``
    None this degrades to pure quantile stratification.

    Returns global row indices (Python ints), highest-score first for a tidy UI.
    """
    if n <= 0:
        return []
    total = int(np.asarray(scores).shape[0])
    eligible = (
        np.asarray(candidate_mask, dtype=bool)
        if candidate_mask is not None
        else np.ones(total, dtype=bool)
    )
    if labeled is not None:
        lab = np.fromiter((int(i) for i in labeled), dtype=np.int64)
        lab = lab[(lab >= 0) & (lab < total)]
        eligible = eligible.copy()
        eligible[lab] = False
    idx = np.nonzero(eligible)[0]
    if idx.size == 0:
        return []
    sc = np.asarray(scores, dtype=np.float64)[idx]
    rng = np.random.default_rng(seed)
    picks: list[int] = []

    def _take(pool: np.ndarray, count: int) -> None:
        pool = np.setdiff1d(pool, np.asarray(picks, dtype=np.int64), assume_unique=False)
        if pool.size == 0 or count <= 0:
            return
        chosen = rng.choice(pool, size=min(count, pool.size), replace=False)
        picks.extend(int(i) for i in chosen)

    if tau is not None:
        near = idx[np.abs(sc - float(tau)) <= band]
        _take(near, n // 2)

    # Quantile stratification over the remaining budget: one row per score bin.
    remaining = n - len(picks)
    if remaining > 0:
        nbins = min(remaining, max(1, idx.size))
        # Rank eligible rows by score, split into ~nbins contiguous groups, take one.
        order = idx[np.argsort(-sc, kind="mergesort")]
        groups = np.array_split(order, nbins)
        for g in groups:
            if len(picks) >= n:
                break
            _take(g, 1)

    # Top-up from anywhere eligible if bins were exhausted (small corpora).
    if len(picks) < n:
        _take(idx, n - len(picks))

    picks = list(dict.fromkeys(picks))[:n]  # dedupe, cap
    picks.sort(key=lambda i: -float(np.asarray(scores)[i]))  # high score first
    return picks


# ---------------------------------------------------------------------------
# First-pass threshold policy: a label-free heuristic cutoff from the query's
# own score distribution, plus the feature vector used to log training episodes
# (so a learned linear policy can be fit later; see scripts/fit_threshold_policy).
# ---------------------------------------------------------------------------


def score_stats(scores: np.ndarray) -> dict:
    """Summary statistics of a query's corpus similarity scores, used both as the
    heuristic-threshold input and as the logged feature vector for a future policy.

    ``top_gap`` = how many std devs the extreme right tail sits above the mean —
    large when a distinct set of matches detaches from the bulk (a clear concept),
    small when scores are diffuse (a vague query). Returns finite floats only.
    """
    s = np.asarray(scores, dtype=np.float64)
    s = s[np.isfinite(s)]
    if s.size == 0:
        return {"mean": 0.0, "std": 0.0, "p50": 0.0, "p90": 0.0, "p99": 0.0,
                "p99_9": 0.0, "max": 0.0, "top_gap": 0.0, "n": 0}
    mean = float(s.mean())
    std = float(s.std())
    p50, p90, p99, p99_9 = (float(np.percentile(s, q)) for q in (50, 90, 99, 99.9))
    mx = float(s.max())
    top_gap = (p99_9 - mean) / std if std > 1e-9 else 0.0
    return {"mean": round(mean, 6), "std": round(std, 6), "p50": round(p50, 6),
            "p90": round(p90, 6), "p99": round(p99, 6), "p99_9": round(p99_9, 6),
            "max": round(mx, 6), "top_gap": round(top_gap, 6), "n": int(s.size)}


def heuristic_threshold(stats: dict, k: float = 3.0) -> float:
    """Label-free cutoff = mean + k*std of the query's score distribution, clamped
    to [0.05, 0.9]. Per-query by construction (each query's own mean/std), so it
    calibrates across concepts without any labels — the first-pass policy that
    gives every tag a sensible starting tau. ``k`` ~= the mean+3*std pattern seen
    across hand-tuned tags; it's the single knob a learned policy later replaces.
    """
    tau = float(stats.get("mean", 0.0)) + k * float(stats.get("std", 0.0))
    return float(min(max(tau, 0.05), 0.9))


# ---------------------------------------------------------------------------
# Learned threshold policy: a ridge regression tau = b + w . features fitted from
# the logged tuning episodes of ONE embedding space. It replaces the fixed
# `mean + 3*std` heuristic's single knob with data-driven weights over the whole
# score-distribution shape. Fitting is offline/background; serving is a dot product.
# ---------------------------------------------------------------------------

# Feature columns the policy regresses on (a subset of score_stats). Order defines
# the design-matrix / weight-vector column order and must stay stable across fit+serve.
POLICY_FEATURES = ["mean", "std", "p90", "p99", "p99_9", "top_gap"]


def fit_threshold_policy(
    episodes: "list[dict]", *, lam: float = 1e-2, min_episodes: int = 20,
    val_frac: float = 0.25, seed: int = 0,
) -> dict | None:
    """Fit ridge ``tau = b + w . POLICY_FEATURES`` from labeled tuning episodes.

    Each episode contributes a row (its ``features`` = score_stats) and target
    (its ``fit_tau`` = the F1-optimal cut found from that tune's labels). Returns
    ``{feature_names, weights:[bias, w...], n_episodes, mae_policy, mae_heuristic}``
    or ``None`` when there aren't yet ``min_episodes`` usable rows (caller keeps the
    heuristic live). ``mae_*`` are held-out mean-abs-errors vs the target tau, so a
    caller can see whether the policy actually beats the heuristic before trusting it.
    """
    xs, ys = [], []
    for ep in episodes or []:
        feats = ep.get("features") or {}
        tau = ep.get("fit_tau")
        if tau is None or not all(f in feats for f in POLICY_FEATURES):
            continue
        xs.append([float(feats[f]) for f in POLICY_FEATURES])
        ys.append(float(tau))
    if len(xs) < min_episodes:
        return None
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)

    def _ridge(xt, yt):
        xb = np.hstack([np.ones((xt.shape[0], 1)), xt])
        reg = lam * np.eye(xb.shape[1]); reg[0, 0] = 0.0  # don't penalize intercept
        return np.linalg.solve(xb.T @ xb + reg, xb.T @ yt)

    # Held-out MAE (policy vs heuristic) on a random split, for a trust signal.
    rng = np.random.default_rng(seed)
    perm = rng.permutation(x.shape[0])
    cut = max(1, int(round(x.shape[0] * (1.0 - val_frac))))
    tr, va = perm[:cut], perm[min(cut, x.shape[0] - 1):] if x.shape[0] > 1 else (perm, perm)
    mae_policy = mae_heur = None
    if len(va):
        w_tr = _ridge(x[tr], y[tr])
        pred = np.hstack([np.ones((x[va].shape[0], 1)), x[va]]) @ w_tr
        heur = np.array([heuristic_threshold(
            {"mean": r[POLICY_FEATURES.index("mean")], "std": r[POLICY_FEATURES.index("std")]}) for r in x[va]])
        mae_policy = float(np.mean(np.abs(pred - y[va])))
        mae_heur = float(np.mean(np.abs(heur - y[va])))

    w = _ridge(x, y)  # final fit on all usable rows
    return {
        "feature_names": list(POLICY_FEATURES),
        "weights": [float(v) for v in w],   # [bias, w1, ...]
        "n_episodes": int(x.shape[0]),
        "mae_policy": mae_policy,
        "mae_heuristic": mae_heur,
    }


def predict_threshold(stats: dict, policy: dict) -> float:
    """Apply a fitted policy: tau = bias + sum(w_i * feature_i), clamped like the
    heuristic. Falls back to the heuristic if the policy is malformed."""
    try:
        names = policy["feature_names"]
        w = policy["weights"]
        tau = float(w[0]) + sum(float(w[i + 1]) * float(stats.get(n, 0.0)) for i, n in enumerate(names))
        return float(min(max(tau, 0.05), 0.9))
    except (KeyError, IndexError, TypeError, ValueError):
        return heuristic_threshold(stats)
