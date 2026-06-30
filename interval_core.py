"""Shared, dependency-light primitives (numpy + pyarrow only) used by BOTH the
in-app search core (`search_engine.py`) and the offline full-corpus Spark scan
workflow (`nls_interval_scan_spark_workflow`).

Single-sourcing this logic guarantees the offline scan produces byte-for-byte
the same intervals as the in-app export. It deliberately imports nothing from the
app (no torch / lancedb / config) so a Spark executor can import it cheaply.

Contents:
  * Arrow extraction: embedding matrix + chunk start/end columns from a Lance/
    parquet Arrow table.
  * Interval projection math: project a drive's overlapping 8s clips onto a 4s
    grid, threshold, and merge contiguous above-threshold cells into
    variable-length intervals with linearly-interpolated boundaries.
"""

from __future__ import annotations

import dataclasses

import numpy as np

# Default chunk geometry, used only as a fallback when a drive has too few clips
# to infer them from the data: 8s mini-segment window, 4s stride (50% overlap).
_DEFAULT_STRIDE_S = 4
_DEFAULT_WINDOW_S = 8


# ---------------------------------------------------------------------------
# Arrow extraction (embedding matrix + chunk times)
# ---------------------------------------------------------------------------
_VECTOR_COLUMNS = ("vector", "embedding_vec", "embedding", "vec")


def _vector_column_name(arrow_table: object) -> str:
    """The embedding column name in a corpus table (varies by producer)."""
    names = set(arrow_table.column_names)
    for cand in _VECTOR_COLUMNS:
        if cand in names:
            return cand
    raise ValueError(
        f"no embedding column found; tried {_VECTOR_COLUMNS}, have {sorted(names)}"
    )


def _vectors_from_arrow(arrow_table: object, column_name: str = "vector") -> np.ndarray:
    """Extract the embedding list-column as a contiguous (n, dim) fp32 array.

    Flattening the Arrow ListArray child avoids the per-row Python list
    materialization that text_query_search uses (fine at 1e5, wasteful at 1e6).
    """
    column = arrow_table.column(column_name).combine_chunks()
    # Single materialization: to_numpy gives one copy of the flat buffer; only
    # astype when the dtype actually differs (copy=False reuses it otherwise). On a
    # ~2M x 768 corpus each avoided copy is ~7GB, so the redundant casts that used
    # to run here risked OOM during load.
    flat = column.values.to_numpy(zero_copy_only=False)
    if flat.dtype != np.float32:
        flat = flat.astype("float32", copy=False)
    num_rows = len(column)
    if num_rows == 0:
        return np.empty((0, 0), dtype="float32")
    dim = flat.shape[0] // num_rows
    return flat.reshape(num_rows, dim)


def _chunk_starts_from_arrow(arrow_table: object) -> list[int]:
    """Per-row chunk START (epoch seconds).

    Prefers an explicit ``chunk_start_unix`` column, then ``start_timestamp_ns``
    (the chunk's own start in ns -> //1e9), and finally the ``chunk_id``
    ``<run_uuid>#t<unix>`` suffix.
    """
    cols = arrow_table.column_names
    if "chunk_start_unix" in cols:
        return [int(v) for v in arrow_table.column("chunk_start_unix").to_pylist()]
    if "start_timestamp_ns" in cols:
        return [
            int(v) // 1_000_000_000
            for v in arrow_table.column("start_timestamp_ns").to_pylist()
        ]
    out: list[int] = []
    for cid in arrow_table.column("chunk_id").to_pylist():
        try:
            out.append(int(str(cid).rsplit("#t", 1)[1]))
        except (IndexError, ValueError):
            out.append(0)
    return out


def _chunk_ends_from_arrow(arrow_table: object) -> list[int] | None:
    """Per-row chunk END (epoch seconds), or None when no end column is present."""
    cols = arrow_table.column_names
    if "chunk_end_unix" in cols:
        return [int(v) for v in arrow_table.column("chunk_end_unix").to_pylist()]
    if "end_timestamp_ns" in cols:
        return [
            int(v) // 1_000_000_000
            for v in arrow_table.column("end_timestamp_ns").to_pylist()
        ]
    return None


# ---------------------------------------------------------------------------
# Interval projection
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class ScoredInterval:
    """A merged, variable-length high-similarity span within one drive.

    Produced by ``project_intervals``: the 8s mini-segment scores are projected
    onto a regular grid (the chunk stride, ~4s), thresholded, and contiguous
    above-threshold cells are merged. ``start_unix``/``end_unix`` are FRACTIONAL
    epoch seconds — the linearly-interpolated threshold crossings, so an interval
    is not snapped to the 8s/4s grid.
    """

    run_uuid: str
    start_unix: float
    end_unix: float
    peak_score: float
    mean_score: float
    num_cells: int
    # Corpus row of the highest-scoring 8s clip in the span, so the export can
    # carry that clip's chunk_id / segment_id / source_media_uri for preview.
    peak_index: int


def _drive_cells(
    starts: np.ndarray, ends: np.ndarray, scores: np.ndarray, rows: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Project one drive's overlapping 8s clips onto a regular stride grid.

    ``starts``/``ends``/``scores``/``rows`` describe the clips of a single drive
    (epoch seconds; ``rows`` are corpus indices). Returns, for the grid cells:
    ``(cell_score, cell_center, cell_peak_row, stride)``. A cell's score is the
    MEAN of the clips overlapping it; cells with no covering clip are ``-inf``
    (a gap, which breaks intervals). ``cell_peak_row`` is the corpus row of the
    highest-scoring clip touching the cell (-1 for a gap).

    Geometry: clips sit on a stride-d grid (d = the modal positive gap between
    consecutive starts, typically 4). A clip at grid position j spans cells j and
    j+1 (window = 2d), so cell m = mean of clips at positions m-1 and m.
    """
    o = np.argsort(starts, kind="stable")
    starts, ends, scores, rows = starts[o], ends[o], scores[o], rows[o]
    base = int(starts[0])
    diffs = np.diff(np.unique(starts))
    diffs = diffs[diffs > 0]
    d = int(diffs.min()) if diffs.size else _DEFAULT_STRIDE_S
    if d <= 0:
        d = _DEFAULT_STRIDE_S
    pos = np.rint((starts - base) / d).astype(np.int64)  # clip grid positions
    p_max = int(pos.max())
    # Per-position clip score + row (last write wins on the rare dup position).
    y = np.full(p_max + 1, np.nan, dtype=np.float64)
    yrow = np.full(p_max + 1, -1, dtype=np.int64)
    y[pos] = scores
    yrow[pos] = rows
    # Cell m (0..p_max+1) averages clip positions m-1 and m.
    left = np.concatenate(([np.nan], y))  # left[m]  = y[m-1]
    right = np.concatenate((y, [np.nan]))  # right[m] = y[m]
    both = np.vstack([left, right])
    finite = ~np.all(np.isnan(both), axis=0)
    cell_score = np.full(left.shape[0], -np.inf, dtype=np.float64)
    # nanmean over the two covering clips, only where at least one exists.
    cell_score[finite] = np.nanmean(both[:, finite], axis=0)
    # Per-cell peak clip row = the covering position with the higher score.
    lrow = np.concatenate(([-1], yrow))
    rrow = np.concatenate((yrow, [-1]))
    lval = np.where(np.isnan(left), -np.inf, left)
    rval = np.where(np.isnan(right), -np.inf, right)
    cell_peak_row = np.where(lval >= rval, lrow, rrow)
    cell_center = base + (np.arange(left.shape[0]) + 0.5) * d
    return cell_score, cell_center, cell_peak_row, d


def _interval_threshold(per_drive, mode, k, score_cutoff):
    """The cutoff τ: a direct score, or the k-th largest finite cell score
    pooled across all drives. Returns None when k-mode has no finite cells.

    ``per_drive`` is a list of ``(run_uuid, cell_score, cell_center, cell_peak_row)``.
    """
    if mode == "score":
        return float(score_cutoff if score_cutoff is not None else 0.0)
    pooled = [cs[np.isfinite(cs)] for _r, cs, _c, _p in per_drive]
    allv = np.concatenate(pooled) if pooled else np.array([], dtype=np.float64)
    if allv.size == 0:
        return None
    kk = max(1, min(int(k), allv.size))
    return float(np.partition(allv, allv.size - kk)[allv.size - kk])


def _merge_drive(run_uuid, cell_scores, cell_centers, cell_peak_rows, tau):
    """Merge one drive's above-tau cells into intervals, interpolating boundaries.

    ``cell_scores`` / ``cell_centers`` / ``cell_peak_rows`` are the per-4s-cell
    arrays from ``_drive_cells`` (a NaN score marks a gap with no clip). Each
    contiguous run of cells with score >= ``tau`` becomes one ``ScoredInterval``;
    its start/end time is linearly interpolated to the exact tau crossing when the
    neighbouring cell is a real below-tau cell, otherwise extended half a cell out.
    """
    above_tau = np.isfinite(cell_scores) & (cell_scores >= tau)
    if not above_tau.any():
        return []

    # Find contiguous above-tau runs: a +1 transition opens a run, -1 closes it.
    # When the first/last cell is already above tau there is no transition to
    # detect, so clamp those open ends explicitly.
    transitions = np.diff(above_tau.astype(np.int8))
    run_starts = list(np.nonzero(transitions == 1)[0] + 1)
    run_ends = list(np.nonzero(transitions == -1)[0])
    if above_tau[0]:
        run_starts.insert(0, 0)
    if above_tau[-1]:
        run_ends.append(above_tau.size - 1)

    half_cell = (
        (cell_centers[1] - cell_centers[0]) / 2.0
        if cell_centers.size > 1
        else float(_DEFAULT_STRIDE_S) / 2.0
    )

    intervals = []
    for first, last in zip(run_starts, run_ends):
        # Leading edge: interpolate the tau crossing if a real below-tau cell sits
        # just before the run, else extend the boundary half a cell earlier.
        if first - 1 >= 0 and np.isfinite(cell_scores[first - 1]) and cell_scores[first - 1] < tau:
            below, above = cell_scores[first - 1], cell_scores[first]
            frac = (tau - below) / (above - below) if above != below else 0.0
            start_unix = cell_centers[first - 1] + frac * (cell_centers[first] - cell_centers[first - 1])
        else:
            start_unix = cell_centers[first] - half_cell
        # Trailing edge: same, using the cell just after the run.
        if last + 1 < cell_scores.size and np.isfinite(cell_scores[last + 1]) and cell_scores[last + 1] < tau:
            above, below = cell_scores[last], cell_scores[last + 1]
            frac = (above - tau) / (above - below) if above != below else 0.0
            end_unix = cell_centers[last] + frac * (cell_centers[last + 1] - cell_centers[last])
        else:
            end_unix = cell_centers[last] + half_cell

        run_scores = cell_scores[first : last + 1]
        peak_row = int(cell_peak_rows[first + int(np.argmax(run_scores))])
        if peak_row < 0:  # peak cell was an interpolated gap; fall back to any real clip
            present = cell_peak_rows[first : last + 1][cell_peak_rows[first : last + 1] >= 0]
            peak_row = int(present[0]) if present.size else -1
        intervals.append(
            ScoredInterval(
                run_uuid=run_uuid,
                start_unix=float(start_unix),
                end_unix=float(end_unix),
                peak_score=float(run_scores.max()),
                mean_score=float(run_scores.mean()),
                num_cells=int(last - first + 1),
                peak_index=peak_row,
            )
        )
    return intervals
