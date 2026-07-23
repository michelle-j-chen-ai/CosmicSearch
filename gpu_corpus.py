"""GPU-resident int8 PCA-256 corpus backend for very large embedding sets.

The CPU resident-matrix path (search_engine.Corpus) keeps an (N, 768) fp16/fp32
matrix in RAM; at ~48M rows that is 73GB fp16 and does not fit any Cloud Run CPU
instance, and a full numpy argsort per query is seconds. This backend instead
holds the corpus on a single NVIDIA L4 (24GB) in a compressed form and does both
scoring and ranking on the GPU:

  * dimensionality reduction: the Cosmos-Embed embeddings are intrinsically
    ~rank-256, so an *uncentered* truncated-SVD projection to 256 dims is
    score-lossless (the projected dot product equals the original cosine; the
    basis rows are orthonormal so norms and inner products in the data subspace
    are preserved exactly). 768 -> 256 is a 3x shrink for free.
  * quantization: the projected vectors are stored int8 with a per-dimension
    symmetric scale (dequant value = int8 * scale / 127). Per-dim scaling is
    essential -- a single global scale starves the low-variance PCA components.
    Measured score correlation vs fp32-768 is 0.99995.

Net: 47.8M x 256 int8 = 12.2GB resident on the GPU, ~12GB headroom on an L4.

The artifact (built offline, see build_gpu_corpus.py) is four files under one
prefix:
  pca_components.npy  (D, 768) fp32   projection basis P (rows orthonormal)
  quant_scales.npy    (D,)     fp32   per-dim int8 scale
  corpus_int8.npy     (N, D)   int8   quantized projected corpus
  metadata.parquet    chunk_id, run_uuid, chunk_start_unix, source_media_uri,
                      segment_id (row-aligned with corpus_int8)

GpuCorpus is duck-type compatible with search_engine.Corpus for everything the
app touches (metadata accessors, time_span, segment_id_array, a `matrix` proxy
for relevance feedback), plus a `gpu_score` method that search_engine dispatches
to. Query vectors flowing through the app stay in the original 768-d model space
(encode_query is unchanged); the projection to 256-d happens inside gpu_score, and
the `matrix` proxy reconstructs 768-d rows on demand so refine_query is unchanged.
"""

from __future__ import annotations

import datetime as _dt
import logging
import os
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

LOGGER = logging.getLogger(__name__)

# Artifact filenames (must match build_gpu_corpus.py).
PCA_FILE = "pca_components.npy"
SCALE_FILE = "quant_scales.npy"
CORPUS_INT8_FILE = "corpus_int8.npy"
METADATA_FILE = "metadata.parquet"

# Row-block size for the int8 -> fp16 score matmul. Bounds the transient fp16
# tile (BLOCK x D x 2 bytes) so we never materialize the whole corpus in fp16.
# 4M (~2GB tile at D=256) suits an L4's ~11GB free VRAM; smaller cards (or a
# fuller GPU) need a smaller tile -- override with NLS_GPU_SCORE_BLOCK.
_SCORE_BLOCK_ROWS = int(os.environ.get("NLS_GPU_SCORE_BLOCK", "4000000"))

# CPU fast path: the same int8 corpus runs on a plain CPU instance (NLS_DEVICE=
# cpu) when no GPU is available. A naive int8->fp32 cast + BLAS is ~11s/query at
# 48M rows (the cast churns 49GB); a numba-fused kernel reads the 12.2GB int8
# once and accumulates fp32 with no intermediate, ~0.25s/query (14 threads). The
# JIT is compiled once, lazily, so a GPU deploy never imports numba.
_CPU_KERNEL = None


def _cpu_score_kernel():
    global _CPU_KERNEL
    if _CPU_KERNEL is None:
        from numba import njit, prange  # lazy: only the CPU path needs numba

        @njit(parallel=True, fastmath=True, cache=True)
        def _score_i8(ci, w, out):  # ci:(N,D) int8, w:(D,) fp32, out:(N,) fp32
            n, d = ci.shape
            for i in prange(n):
                acc = np.float32(0.0)
                row = ci[i]
                for j in range(d):
                    acc += np.float32(row[j]) * w[j]
                out[i] = acc

        _CPU_KERNEL = _score_i8
    return _CPU_KERNEL


def is_gpu_artifact(local_dir: Path) -> bool:
    """True if `local_dir` holds an int8 PCA GPU corpus (vs npy/Lance)."""
    return (local_dir / CORPUS_INT8_FILE).exists() and (local_dir / PCA_FILE).exists()


class _ArrowStrCol:
    """List-like read-only view over an Arrow string column.

    Lets GpuCorpus expose `chunk_id[i]` / `source_media_uri[i]` etc. without
    materializing 48M Python strings up front (corpus.<col>[index] is only ever
    called for the few ranked hits actually shown/exported). `.to_pylist()` is
    available for the rare full-column consumer.
    """

    __slots__ = ("_arr",)

    def __init__(self, arrow_array: object) -> None:
        # Keep the column chunked: combine_chunks() overflows arrow's int32
        # string offsets once a column exceeds 2GB (e.g. 6.45GB source_media_uri
        # at 48M rows). ChunkedArray supports [i] and to_pylist directly.
        self._arr = arrow_array

    def __len__(self) -> int:
        return len(self._arr)

    def __getitem__(self, i: int) -> str:
        v = self._arr[int(i)].as_py()
        return v if v is not None else ""

    def to_pylist(self) -> list[str]:
        return [v if v is not None else "" for v in self._arr.to_pylist()]


# The inference writes every chunk's MP4 to a fixed, derivable path, so the
# source_media_uri column (6.45GB at 48M rows -- the single biggest resident
# cost) is NOT loaded; it is reconstructed per-hit from run_uuid + chunk_start.
# Verified row-exact: dt is the UTC date of chunk_start_unix.
_MP4_PREFIX = os.environ.get(
    "NLS_MP4_PREFIX",
    "s3://neuron-prod-data-intelligence-exploratory/vlm/chunks_mp4_v2/",
)


class _ReconUriCol:
    """source_media_uri view that reconstructs each URI on access (no 6.45GB)."""

    __slots__ = ("_run_uuid", "_chunk_start")

    def __init__(self, run_uuid: _ArrowStrCol, chunk_start: np.ndarray) -> None:
        self._run_uuid = run_uuid
        self._chunk_start = chunk_start

    def __len__(self) -> int:
        return len(self._chunk_start)

    def __getitem__(self, i: int) -> str:
        i = int(i)
        c = int(self._chunk_start[i])
        d = _dt.datetime.fromtimestamp(c, _dt.timezone.utc).strftime("%Y-%m-%d")
        return f"{_MP4_PREFIX}dt={d}/{self._run_uuid[i]}_t{c}.mp4"

    def to_pylist(self) -> list[str]:  # rare (full-column export); built lazily
        return [self[i] for i in range(len(self))]


class _ReconChunkIdCol:
    """chunk_id view reconstructed from run_uuid + chunk_start (no resident column).

    chunk_id is deterministically `<run_uuid>#t<chunk_start_unix>` (verified
    row-exact against the corpus metadata), so storing it resident is ~2.4GB of
    arrow strings at 48M rows for nothing -- reconstruct it per-access instead.
    """

    __slots__ = ("_run_uuid", "_chunk_start")

    def __init__(self, run_uuid: _ArrowStrCol, chunk_start: np.ndarray) -> None:
        self._run_uuid = run_uuid
        self._chunk_start = chunk_start

    def __len__(self) -> int:
        return len(self._chunk_start)

    def __getitem__(self, i: int) -> str:
        i = int(i)
        return f"{self._run_uuid[i]}#t{int(self._chunk_start[i])}"

    def to_pylist(self) -> list[str]:  # rare (full-column export); built lazily
        return [self[i] for i in range(len(self))]


class _MatrixProxy:
    """Stand-in for Corpus.matrix that reconstructs 768-d rows on indexing.

    Only small index sets are ever requested (relevance-feedback marks, the
    one-row load benchmark), so this gathers those rows on the GPU, dequantizes
    int8 -> fp32, and projects back to the original 768-d space (P rows are
    orthonormal, so P.T @ proj reconstructs the in-subspace 768-d vector). Never
    materializes the full matrix. `score_corpus`/`rank_top_k` are dispatched to
    `gpu_score` *before* touching `.matrix`, so the O(N) `matrix @ q` path is
    never hit for a GpuCorpus.
    """

    __slots__ = ("_owner",)
    dtype = np.dtype("float32")
    ndim = 2

    def __init__(self, owner: "GpuCorpus") -> None:
        self._owner = owner

    @property
    def shape(self) -> tuple[int, int]:
        return (self._owner.num_rows, self._owner.model_dim)

    def __getitem__(self, idx) -> np.ndarray:
        o = self._owner
        if np.isscalar(idx):
            idx = [int(idx)]
        idx = np.asarray(idx, dtype=np.int64)
        if o.is_cpu:
            deq = o.corpus_int8[idx].astype("float32") * (o.scale / 127.0)  # (k, D)
            return (deq @ o.pca).astype("float32")  # (k, 768) back-projected
        idx_t = torch.as_tensor(idx, device=o.device)
        rows_i8 = o.corpus_int8.index_select(0, idx_t).to(torch.float32)  # (k, D)
        deq = rows_i8 * (o.scale / 127.0)  # (k, D) PCA-space fp32
        recon = deq @ o.pca.to(torch.float32)  # (k, 768) back-projected
        return recon.detach().cpu().numpy().astype("float32")


class GpuCorpus:
    """A GPU-resident int8 PCA corpus, duck-type compatible with Corpus."""

    def __init__(
        self,
        corpus_int8: torch.Tensor,  # (N, D) int8 on `device`
        pca: torch.Tensor,  # (D, 768) fp16 on `device`
        scale: torch.Tensor,  # (D,) fp32 on `device`
        chunk_id: _ArrowStrCol,
        run_uuid: _ArrowStrCol,
        chunk_start_unix: np.ndarray,  # (N,) int64
        source_media_uri: _ArrowStrCol,
        segment_id: _ArrowStrCol,
        device: str,
        vehicle: _ArrowStrCol | None = None,
    ) -> None:
        self.corpus_int8 = corpus_int8
        self.pca = pca
        self.scale = scale
        self.device = device
        self.is_cpu = device == "cpu"
        self.chunk_id = chunk_id
        self.run_uuid = run_uuid
        self._chunk_start = chunk_start_unix
        self.chunk_start_unix = chunk_start_unix  # int64 ndarray supports [i]
        self.source_media_uri = source_media_uri
        self.segment_id = segment_id
        self.vehicle = vehicle
        self.matrix = _MatrixProxy(self)
        self._segment_id_arr: np.ndarray | None = None
        self._vehicle_arr: np.ndarray | None = None
        # The embedding metadata carries no per-chunk end time or Data Explorer
        # internal id, so those Corpus features are simply unavailable here.
        self.chunk_end_unix = None
        self.dx_internal_id = None

    # --- shape ---------------------------------------------------------------
    @property
    def num_rows(self) -> int:
        return int(self.corpus_int8.shape[0])

    @property
    def pca_dim(self) -> int:
        return int(self.corpus_int8.shape[1])

    @property
    def model_dim(self) -> int:
        return int(self.pca.shape[1])

    @property
    def dim(self) -> int:
        # Report the model (search) space dim, matching what the encoder emits.
        return self.model_dim

    # --- metadata accessors (mirror Corpus) ----------------------------------
    def chunk_start_array(self) -> np.ndarray:
        return self._chunk_start

    def segment_id_array(self) -> np.ndarray:
        if self._segment_id_arr is None:
            self._segment_id_arr = np.asarray(self.segment_id.to_pylist(), dtype=object)
        return self._segment_id_arr

    def vehicle_array(self) -> np.ndarray | None:
        if self.vehicle is None:
            return None
        if self._vehicle_arr is None:
            self._vehicle_arr = np.asarray(self.vehicle.to_pylist(), dtype=object)
        return self._vehicle_arr

    def has_vehicle(self) -> bool:
        return self.vehicle is not None

    def time_span(self) -> tuple[int, int]:
        if self._chunk_start.size == 0:
            return (0, 0)
        return (int(self._chunk_start.min()), int(self._chunk_start.max()))

    def has_internal_ids(self) -> bool:
        return self.dx_internal_id is not None

    def dx_internal_id_array(self) -> np.ndarray:
        # Only valid when has_internal_ids(); this corpus has none.
        return np.empty(0, dtype=np.int64)

    # --- GPU scoring ---------------------------------------------------------
    def gpu_score(
        self, query_vector: np.ndarray, subset_idx: np.ndarray | None = None
    ) -> np.ndarray:
        """Exact cosine of a 768-d query against every row (1-D fp32 numpy).

        Project the query into PCA space, fold the per-dim dequant scale into it
        (so the corpus stays int8), then stream the int8 corpus through a
        blocked int8->fp16 matmul. The full score vector is returned (the app's
        threshold paging and exports need every row's score), but the heavy sort
        is done on-GPU by `gpu_argsort`.

        When ``subset_idx`` is given, only those rows are scored (the int8 rows
        are gathered first) and every other row is ``-inf``, so a downsample
        search costs a matmul over the selected rows instead of all ~10^7.
        """
        t0 = time.time()
        n = self.num_rows
        if self.is_cpu:
            # CPU: project with numpy, fold the dequant scale into the query,
            # then a numba-fused int8 dot (reads the int8 corpus once).
            qp = self.pca @ np.ascontiguousarray(query_vector, dtype=np.float32)
            w = (qp * (self.scale / 127.0)).astype("float32")
            if subset_idx is None:
                scores = np.empty(n, dtype="float32")
                _cpu_score_kernel()(self.corpus_int8, w, scores)
            else:
                sub = np.ascontiguousarray(self.corpus_int8[subset_idx])
                sub_scores = np.empty(sub.shape[0], dtype="float32")
                _cpu_score_kernel()(sub, w, sub_scores)
                scores = np.full(n, -np.inf, dtype="float32")
                scores[subset_idx] = sub_scores
            LOGGER.info(
                "cpu scored %d rows (int8 PCA-%d) in %.0fms",
                n if subset_idx is None else len(subset_idx),
                self.pca_dim,
                (time.time() - t0) * 1000,
            )
            return scores
        q = torch.from_numpy(np.ascontiguousarray(query_vector, dtype=np.float32)).to(
            self.device
        )
        qp = self.pca.to(torch.float32) @ q  # (D,) project: P (D,768) @ q (768,)
        w = (qp * (self.scale / 127.0)).to(torch.float16)  # fold dequant scale
        out = torch.full((n,), float("-inf"), dtype=torch.float32, device=self.device)
        if subset_idx is None:
            rows = range(0, n, _SCORE_BLOCK_ROWS)
            for s in rows:
                e = min(s + _SCORE_BLOCK_ROWS, n)
                blk = self.corpus_int8[s:e].to(torch.float16)  # transient fp16 tile
                out[s:e] = (blk @ w).to(torch.float32)
        else:
            sel = torch.from_numpy(
                np.ascontiguousarray(subset_idx, dtype=np.int64)
            ).to(self.device)
            gathered = self.corpus_int8.index_select(0, sel)  # (len_subset, D) int8
            m = gathered.shape[0]
            sub_out = torch.empty(m, dtype=torch.float32, device=self.device)
            for s in range(0, m, _SCORE_BLOCK_ROWS):
                e = min(s + _SCORE_BLOCK_ROWS, m)
                blk = gathered[s:e].to(torch.float16)
                sub_out[s:e] = (blk @ w).to(torch.float32)
            out.index_copy_(0, sel, sub_out)
        scores = out.detach().cpu().numpy()
        LOGGER.info(
            "gpu scored %d rows (int8 PCA-%d) in %.0fms",
            n if subset_idx is None else len(subset_idx),
            self.pca_dim,
            (time.time() - t0) * 1000,
        )
        # Stash on the instance so ranked_order can sort on-GPU without a
        # second host->device copy of the score vector.
        self._last_scores_gpu = out
        return scores

    def gpu_argsort(self, scores: np.ndarray, idx: np.ndarray | None) -> np.ndarray:
        """Descending-score argsort on the GPU; returns int64 numpy indices.

        If `idx` is given, sorts only that subset (post date/segment filtering);
        otherwise sorts all rows. Sorting ~48M floats on the L4 is ~100-200ms vs
        ~3-4s for numpy on the CPU instance.
        """
        if self.is_cpu:
            # A full numpy argsort of ~48M is ~3-4s. The app only ever pages/
            # exports the top of the ranking, so cap the unfiltered case to the
            # top NLS_CPU_ORDER_CAP via argpartition (~0.25s) and sort only those.
            # A date/segment-filtered subset is small enough to sort in full.
            cap = int(os.environ.get("NLS_CPU_ORDER_CAP", "200000"))
            if idx is None:
                if scores.shape[0] <= cap:
                    return np.argsort(-scores, kind="stable")
                top = np.argpartition(-scores, cap)[:cap]
                return top[np.argsort(-scores[top], kind="stable")]
            idx = np.asarray(idx, dtype=np.int64)
            sub = scores[idx]
            if sub.shape[0] <= cap:
                return idx[np.argsort(-sub, kind="stable")]
            top = np.argpartition(-sub, cap)[:cap]
            return idx[top[np.argsort(-sub[top], kind="stable")]]
        cached = getattr(self, "_last_scores_gpu", None)
        if idx is None:
            s_gpu = (
                cached
                if cached is not None and cached.shape[0] == scores.shape[0]
                else torch.from_numpy(scores).to(self.device)
            )
            order = torch.argsort(s_gpu, descending=True, stable=True)
            return order.detach().cpu().numpy()
        idx = np.asarray(idx, dtype=np.int64)
        s_sub = torch.from_numpy(scores[idx]).to(self.device)
        order = torch.argsort(s_sub, descending=True, stable=True)
        return idx[order.detach().cpu().numpy()]


def load_gpu_corpus(local_dir: Path, device: str = "cuda") -> GpuCorpus:
    """Load the int8 PCA artifact for the GPU (cuda) or CPU (cpu) backend.

    cuda: the int8 corpus + fp16 basis live on the GPU. cpu: the int8 corpus
    stays a resident numpy array (the numba scorer reads it directly) and the
    basis/scale stay fp32 numpy -- no torch device tensors.
    """
    t0 = time.time()
    pca = np.load(local_dir / PCA_FILE).astype("float32")  # (D, 768)
    scale = np.load(local_dir / SCALE_FILE).astype("float32")  # (D,)
    if device == "cpu":
        # Two traps on Cloud Run with the gcs-fuse-mounted corpus:
        #   1. mmap'ing the gcs-fuse file directly returns ZEROS to the in-place numba
        #      reads (FUSE mmap doesn't back in-place page reads with content) -> all
        #      similarity scores come out 0.
        #   2. a resident np.load of the gcs-fuse file holds 12.2GB in an anonymous
        #      array WHILE gcs-fuse caches the same 12.2GB in its (unlimited) file
        #      cache -> ~24GB -> OOM on the 32Gi instance.
        # Fix: copy the file to local tmpfs (/tmp, real RAM that supports mmap), then
        # mmap THAT. tmpfs mmap serves correct bytes (scores non-zero) and the mmap is
        # a view, so the corpus is held exactly once (~12.2GB in tmpfs) -- no
        # anonymous-array + gcs-fuse-cache doubling. Total ~22GB, fits with headroom.
        src = local_dir / CORPUS_INT8_FILE
        staged = Path(tempfile.gettempdir()) / "nls_corpus_int8.npy"
        if not staged.exists() or staged.stat().st_size != src.stat().st_size:
            t = time.time()
            shutil.copyfile(src, staged)
            LOGGER.info(
                "staged int8 corpus to tmpfs (%.1fGB) in %.1fs",
                staged.stat().st_size / 1e9,
                time.time() - t,
            )
        corpus = np.load(staged, mmap_mode="r")  # mmap the tmpfs copy (correct + single)
        n, d = corpus.shape
        LOGGER.info("int8 corpus %d x %d mmapped from tmpfs (correct, single copy)", n, d)
        corpus_t, pca_t, scale_t = corpus, pca, scale
    else:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "NLS GPU backend requires CUDA but torch.cuda is unavailable"
            )
        # mmap so it never fully materializes in host RAM: np.load writes it
        # C-contiguous, so torch.from_numpy is a zero-copy view and .to(device)
        # streams the pages straight to the GPU.
        corpus_np = np.load(local_dir / CORPUS_INT8_FILE, mmap_mode="r")
        n, d = corpus_np.shape
        LOGGER.info("loading int8 corpus %d x %d onto %s", n, d, device)
        corpus_t = torch.from_numpy(corpus_np).to(device)
        pca_t = torch.from_numpy(pca).to(torch.float16).to(device)
        scale_t = torch.from_numpy(scale).to(torch.float32).to(device)

    # source_media_uri is the single biggest resident cost (6.45GB of arrow
    # strings at 48M rows) and is deterministically derivable from run_uuid +
    # chunk_start_unix, so we never read it -- saving that 6.45GB keeps the
    # full corpus inside Cloud Run's 32Gi cap (8 CPU). It is reconstructed
    # per-hit by _ReconUriCol below.
    available = set(pq.ParquetFile(local_dir / METADATA_FILE).schema.names)
    # First present vehicle-id column (build-time join from the Ursa runs table,
    # or parsed from segment_id) -> enables the vehicle filter; None = inert.
    veh_name = next(
        (c for c in ("vehicle", "vehicle_name", "vehicle_id") if c in available), None
    )
    # chunk_id is NOT loaded -- it is reconstructed from run_uuid + chunk_start by
    # _ReconChunkIdCol below (saves ~2.4GB resident at 48M rows), same as the URI.
    want = [c for c in ("run_uuid", "chunk_start_unix", "segment_id", veh_name)
            if c and c in available]
    meta = pq.read_table(local_dir / METADATA_FILE, columns=want)
    starts = np.asarray(
        meta.column("chunk_start_unix").combine_chunks().to_numpy(zero_copy_only=False),
        dtype=np.int64,
    )
    cols = set(meta.column_names)

    def col(name: str) -> _ArrowStrCol:
        import pyarrow as pa

        if name in cols:
            return _ArrowStrCol(meta.column(name))
        return _ArrowStrCol(pa.chunked_array([pa.array([""] * n)]))

    run_uuid_col = col("run_uuid")
    vehicle_col = col(veh_name) if (veh_name and veh_name in cols) else None
    LOGGER.info("vehicle column: %s", veh_name or "(none)")
    corpus = GpuCorpus(
        corpus_int8=corpus_t,
        pca=pca_t,
        scale=scale_t,
        chunk_id=_ReconChunkIdCol(run_uuid_col, starts),
        run_uuid=run_uuid_col,
        chunk_start_unix=starts,
        source_media_uri=_ReconUriCol(run_uuid_col, starts),
        segment_id=col("segment_id"),
        device=device,
        vehicle=vehicle_col,
    )
    nbytes = corpus_t.nbytes if device == "cpu" else (
        corpus_t.element_size() * corpus_t.nelement()
    )
    LOGGER.info(
        "int8 corpus ready: %d rows x %d (%.1fGB on %s) in %.1fs",
        n,
        d,
        nbytes / 1e9,
        device,
        time.time() - t0,
    )
    return corpus
