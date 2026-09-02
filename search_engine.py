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
import time
from pathlib import Path

import local_cache
import oci_s3
import torch
from transformers import AutoModel, AutoProcessor

# Pin torch's intra-op thread pool to the allocated vCPUs (mirrors the BLAS env caps set
# in web_server before import). Without this torch defaults to the HOST core count on
# Cloud Run and oversubscribes the 8 allocated vCPUs -- the ViT video-encode forward
# thrashes (a single image encode was taking ~60-97s).
_torch_threads = int(os.environ.get("NLS_NUM_THREADS", "8"))
if _torch_threads > 0:
    torch.set_num_threads(_torch_threads)

import numpy as np
from config import (
    BASE_MODEL_REVISION,
    BASE_MODEL_URI,
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


# Video-encode geometry -- MUST match the corpus builder (fine_tuned_embed_inference):
# 8 frames at 448x448 through processor(videos=...) -> get_video_embeddings.visual_proj.
_ENCODE_NUM_FRAMES = 8
_ENCODE_RESOLUTION = 448


def encode_video_frames(
    frames_btchw: object, processor: object, model: object, device: str
) -> np.ndarray:
    """Encode a batch of video frames ``[1, T, C, H, W]`` (uint8) into the joint
    video/text space (L2-normalized), via the SAME ``processor(videos=...)`` +
    ``get_video_embeddings`` path that built the corpus. Returns one 768-d vector.
    """
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    inputs = processor(
        text="",
        videos=frames_btchw,
        resolution=_ENCODE_RESOLUTION,
        num_video_frames=_ENCODE_NUM_FRAMES,
    )
    videos = inputs["videos"].to(device, dtype=dtype)
    with torch.inference_mode():
        output = model.get_video_embeddings(videos=videos)
    vector = torch.nn.functional.normalize(output.visual_proj.float(), dim=-1)
    return vector.detach().cpu().numpy().astype("float32")[0]


def _frame_from_bytes(data: bytes):
    """One image's bytes -> a 448x448 uint8 C,H,W tensor. Requires Pillow."""
    import io as _io

    from PIL import Image

    img = Image.open(_io.BytesIO(data)).convert("RGB").resize(
        (_ENCODE_RESOLUTION, _ENCODE_RESOLUTION)
    )
    arr = np.asarray(img, dtype=np.uint8)  # H, W, C
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # C, H, W


def encode_frames_list(
    frames_bytes: list, processor: object, model: object, device: str
) -> np.ndarray:
    """Encode a list of frame images (each raw bytes) into the corpus's video/text
    space. Resamples the frames to the model's fixed 8-frame input (evenly spaced;
    a single frame -> replicated). Used by both image drag-drop (1 frame) and video
    drag-drop (the browser extracts ~8 frames from a capped window). Requires Pillow.
    """
    imgs = [_frame_from_bytes(b) for b in frames_bytes if b]
    if not imgs:
        raise ValueError("no decodable frames")
    n = len(imgs)
    if n == 1:
        idx = [0] * _ENCODE_NUM_FRAMES
    else:  # evenly-spaced indices across whatever the client sent
        idx = [round(i * (n - 1) / (_ENCODE_NUM_FRAMES - 1)) for i in range(_ENCODE_NUM_FRAMES)]
    frames_tchw = torch.stack([imgs[j] for j in idx])  # T, C, H, W
    return encode_video_frames(frames_tchw.unsqueeze(0), processor, model, device)


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
# First-pass threshold: a label-free heuristic cutoff from the query's own score
# distribution.
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

