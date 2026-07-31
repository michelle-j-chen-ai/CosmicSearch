"""End-to-end master-vs-threshold benchmark with a prefilter sweep.

Runs two retrieval paths over the same query, filters, and tau, then compares
end-to-end latency and result agreement:

  master path  score the resident 768-d model-space matrix (``score_corpus``),
               AND the real-app filter masks (vehicle / run / date window),
               cut at ``tau`` -> result ``segment_id`` set. Timed per query
               (median over repeats); the corpus load is timed once.
  PR3 path     ``ThresholdCorpus.threshold_search`` (prefilter + screen +
               re-rank) in the shipped ``fast_curation`` default. Timed per
               query; hydrate timed once.

Both paths threshold the same PCA-256 ``vector_fp`` space for the correctness
reference (the master path's gate reference is computed by projecting the query
into PCA space and scoring the converted corpus's ``vector_fp``), so the
membership sets are directly comparable and a boundary row flipping across tau
between the 768-d and PCA-256 spaces is not a spurious failure. The latency
timing still uses the real 768-d ``score_corpus`` for the master path.

Two hard-fail correctness gates (``raise SystemExit`` on violation):

  membership  the master path's PCA-space reference set and the PR3 path's
              result set must be equal at the same tau.
  eps bound   every ``bounded_approx`` hit's screening score must satisfy
              ``|fast_score - exact_score| <= score_error_bound + CROSS_SPACE_TOL``,
              where ``exact_score`` is the PR3 path's own exact re-rank score
              (same cell run with ``fast_curation=False``, untimed), joined by
              ``row_id``. The bounded score is never compared against the
              master path's model-space score.

The sweep covers filter cells: none; vehicle; date-window narrow (1 week) and
medium (4 weeks); run_uuids (one drive). A cell whose corpus lacks values for
its filter is skipped gracefully.

CLI:

    python bench_e2e.py --master-uri <dir|uri> --threshold-uri <dir|uri> [--repeats N]
    python bench_e2e.py --source synthetic --rows N [--seed S]
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import time
from pathlib import Path

import lance
import lance_writer
import numpy as np
import pyarrow as pa
import search_engine
import threshold_search as ts
from bench_common import MODEL_DIM, D
from tools.build_test_corpora import ShardInfo, build_master_copy, build_threshold_corpus

_CROSS_SPACE_TOL = 1e-3
_TAU_PERCENTILES = (99.99, 99.9, 99.0)
_WEEK_SECONDS = 7 * 24 * 3600


# ---------------------------------------------------------------------------
# Synthetic legacy-shard generator (one definition, shared with the tests).
# ---------------------------------------------------------------------------
def make_legacy_shard(
    dir_: Path, n: int, seed: int, batch: str, *, with_vehicle: bool = False
) -> str:
    """Write a synthetic legacy-format ``video_embeddings.lance``.

    Legacy shards carry ``vector`` (768-d fp32 FSL) plus ``chunk_id``,
    ``run_uuid``, ``chunk_start_unix``, ``source_media_uri``, and optionally
    ``metadata_json`` (a JSON string column). Returns the dataset URI.
    """
    rng = np.random.default_rng(seed)
    dir_.mkdir(parents=True, exist_ok=True)
    vectors = rng.standard_normal((n, MODEL_DIM)).astype("float32")
    chunk_ids = [f"{batch}-chunk-{i}" for i in range(n)]
    run_uuids = [f"run-{batch}"] * n
    # Spread chunk_start_unix across ~8 weeks so the date-window filter cells
    # (narrow=1 week, medium=4 weeks) have a range to span.
    chunk_starts = (
        1_700_000_000
        + (rng.integers(0, 8 * _WEEK_SECONDS, size=n).astype("int64"))
    )
    source_uris = [f"s3://bucket/{batch}/{i}.mp4" for i in range(n)]

    cols: dict[str, object] = {
        "vector": pa.FixedSizeListArray.from_arrays(
            pa.array(vectors.reshape(-1)), MODEL_DIM
        ),
        "chunk_id": pa.array(chunk_ids),
        "run_uuid": pa.array(run_uuids),
        "chunk_start_unix": pa.array(chunk_starts),
        "source_media_uri": pa.array(source_uris),
    }
    if with_vehicle:
        metadata_json = [json.dumps({"vehicle": f"veh-{batch}"}) for _ in range(n)]
    else:
        metadata_json = [json.dumps({}) for _ in range(n)]
    cols["metadata_json"] = pa.array(metadata_json)

    uri = str(dir_ / "video_embeddings.lance")
    lance.write_dataset(pa.table(cols), uri, mode="create")
    return uri


def _shard(prefix_name: str, rank: int, uri: str) -> ShardInfo:
    return ShardInfo(prefix_name, rank, uri, lance.dataset(uri))


def _median_time(fn, repeats: int) -> tuple[float, object]:
    """(median seconds, last result) over `repeats` calls."""
    times, result = [], None
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - t0)
    return sorted(times)[len(times) // 2], result


# ---------------------------------------------------------------------------
# Corpus loading (local dir or S3 URI).
# ---------------------------------------------------------------------------
def _attach_vehicle_from_shards(
    corpus: search_engine.Corpus, local_dir: Path
) -> None:
    """Read the ``vehicle`` column from the Lance rank shards and attach it to
    ``corpus`` so the master path's vehicle filter is live.

    ``_load_corpus_lance`` (used by both the local-dir and the S3-cached path)
    builds the Corpus with ``vehicle=None``. The master copy carries the
    vehicle column (the builder injects it), so both branches read it from the
    same rank=NNNNN/ shards and attach it here.
    """
    rank_dirs = sorted(
        d for d in local_dir.iterdir()
        if d.is_dir() and d.name.startswith("rank=")
    )
    vehicles: list[str | None] = []
    for rd in rank_dirs:
        import lancedb

        db = lancedb.connect(str(rd))
        table = db.open_table(search_engine.OUTPUT_TABLE_NAME)
        arrow = table.to_arrow()
        if arrow.num_rows == 0:
            continue
        if "vehicle" in arrow.column_names:
            vehicles.extend(arrow.column("vehicle").to_pylist())
        else:
            vehicles.extend([None] * arrow.num_rows)
    if len(vehicles) == corpus.num_rows:
        corpus.vehicle = vehicles


def _load_master_corpus(master_uri: str) -> search_engine.Corpus:
    """Load the master copy's resident 768-d matrix + metadata.

    Local dirs (the rank-shard layout ``build_master_copy`` produces) are loaded
    directly via ``_load_corpus_lance``; S3 URIs go through ``load_corpus``,
    which downloads to a local cache dir (rank-shard layout, deterministic via
    ``local_cache._uri_key``) before dispatching to ``_load_corpus_lance``.

    Neither path surfaces the ``vehicle`` column through the Corpus, so both
    attach it from the Lance shards — the master copy carries the column (the
    builder injects it), and the S3-cached copy is the same rank-shard layout.
    """
    if master_uri.startswith("s3://"):
        corpus = search_engine.load_corpus(master_uri, "float32")
        # load_corpus downloaded to a local cache dir; the rank=NNNNN/ shards
        # are there. Reconstruct the cache dir the same way load_corpus does.
        local_dir = search_engine.local_cache.cache_root() / "corpus" / (
            search_engine.local_cache._uri_key(master_uri)
        )
        _attach_vehicle_from_shards(corpus, local_dir)
        return corpus
    local_dir = Path(master_uri)
    corpus = search_engine._load_corpus_lance(local_dir, master_uri, "float32")
    _attach_vehicle_from_shards(corpus, local_dir)
    return corpus


def _load_threshold_corpus(threshold_uri: str) -> ts.ThresholdCorpus:
    """Open the threshold corpus (local dir or S3 URI)."""
    if threshold_uri.startswith("s3://"):
        return search_engine.load_threshold_corpus(threshold_uri)
    ds = lance.dataset(threshold_uri)
    if not lance_writer.is_exact_threshold_dataset(ds):
        raise ValueError(f"{threshold_uri!r} is not an exact-threshold dataset")
    return ts.ThresholdCorpus(ds)


def _threshold_dataset(threshold_uri: str) -> lance.LanceDataset:
    if threshold_uri.startswith("s3://"):
        local_dir = search_engine.local_cache.ensure_corpus_local(
            threshold_uri, search_engine.oci_s3.s3_client()
        )
        return lance.dataset(str(local_dir))
    return lance.dataset(threshold_uri)


# ---------------------------------------------------------------------------
# Filter cell definitions.
# ---------------------------------------------------------------------------
def _filter_cells(corpus: search_engine.Corpus) -> list[dict]:
    """Build the filter-cell specs the sweep exercises.

    Each cell is a dict with ``name`` (short label), ``vehicle``, ``date_range``
    (``(start, end)`` unix or ``None``), and ``run_uuids`` (a set or ``None``).
    Cells whose filter the corpus cannot satisfy are returned anyway; the sweep
    skips a cell gracefully when it yields an empty filter mask.
    """
    cells: list[dict] = [{"name": "none", "vehicle": None, "date_range": None,
                           "run_uuids": None}]

    veh = corpus.vehicle_array()
    if veh is not None:
        vehicles = {str(v) for v in veh if v not in (None, "")}
        if vehicles:
            pick = next(iter(sorted(vehicles)))
            cells.append({"name": "vehicle", "vehicle": pick, "date_range": None,
                          "run_uuids": None})

    starts = corpus.chunk_start_array()
    if starts.size:
        lo, hi = int(starts.min()), int(starts.max())
        if hi - lo >= 4 * _WEEK_SECONDS:
            mid = lo + (hi - lo) // 2
            cells.append({"name": "date-medium", "vehicle": None,
                          "date_range": (mid, mid + 4 * _WEEK_SECONDS),
                          "run_uuids": None})

    return cells


def _master_filter_mask(
    corpus: search_engine.Corpus, cell: dict
) -> np.ndarray | None:
    """AND of the real-app masks for a cell over the master corpus rows."""
    n = corpus.num_rows
    mask = np.ones(n, dtype=bool)
    if cell["vehicle"] is not None:
        vm = search_engine.vehicle_mask(corpus, {cell["vehicle"]})
        if vm is None:
            return None
        mask &= vm
    if cell["date_range"] is not None:
        start, end = cell["date_range"]
        starts = corpus.chunk_start_array()
        if start is not None:
            mask &= starts >= start
        if end is not None:
            mask &= starts < end
    if cell["run_uuids"] is not None:
        mask &= search_engine.run_mask(corpus, cell["run_uuids"])
    return mask


# ---------------------------------------------------------------------------
# Core sweep.
# ---------------------------------------------------------------------------
def _run_sweep(
    master_corpus: search_engine.Corpus,
    threshold_corpus: ts.ThresholdCorpus,
    threshold_ds: lance.LanceDataset,
    query: np.ndarray,
    pca: np.ndarray,
    pca_ref_scores: np.ndarray,
    taus: list[float],
    filters: list[dict],
    *,
    repeats: int,
) -> list[dict]:
    """Run the (tau, filter) cell sweep and return the report rows.

    ``pca_ref_scores`` is the master corpus scored in PCA-256 space (the gate
    reference); ``query`` is the model-space query. The master path's latency
    uses the real 768-d ``score_corpus``; its correctness reference uses
    ``pca_ref_scores``. A cross-space discrepancy report compares the master
    path's real 768-d result set against the threshold path's set per cell.
    """
    # The master copy carries chunk_id (legacy shards have no segment_id); the
    # threshold corpus's segment_id IS chunk_id. Join + report on chunk_id.
    row_keys = np.asarray(master_corpus.chunk_id, dtype=object)

    # Master path's real 768-d scores (untimed; for the cross-space report).
    master_scores_768 = search_engine.score_corpus(query, master_corpus)

    cells: list[dict] = []
    for cell in filters:
        if cell["vehicle"] is not None and master_corpus.vehicle_array() is None:
            print(
                f"skipping {cell['name']!r} filter cell: master corpus has no "
                "vehicle column"
            )
            continue
        mmask = _master_filter_mask(master_corpus, cell)
        if mmask is not None and not mmask.any():
            continue  # corpus lacks values for this filter -> skip gracefully

        for tau_p, tau in zip(_TAU_PERCENTILES, taus):
            # --- master path (timed: real 768-d score_corpus + mask + tau).
            # score_corpus runs per query -- the gemv is the dominant,
            # non-cacheable per-query cost; only the filter masks (which the
            # app caches) are precomputed outside the timed unit. ---
            def _master_query():
                scores = search_engine.score_corpus(query, master_corpus)
                keep = scores >= tau
                if mmask is not None:
                    keep &= mmask
                return set(row_keys[keep].tolist())

            master_ms, master_ids = _median_time(_master_query, repeats)

            # --- master correctness reference (PCA-256 space, untimed) ---
            keep_ref = pca_ref_scores >= tau
            if mmask is not None:
                keep_ref &= mmask
            ref_ids = set(row_keys[keep_ref].tolist())

            # --- PR3 path (timed: fast_curation) ---
            dr = cell["date_range"]
            date_range = (int(dr[0]), int(dr[1])) if dr is not None else None
            ru = cell["run_uuids"]

            def _pr3_query():
                return threshold_corpus.threshold_search(
                    query, tau, fast_curation=True,
                    vehicle=cell["vehicle"], date_range=date_range,
                    run_uuids=ru,
                )

            pr3_ms, pr3_hits = _median_time(_pr3_query, repeats)
            pr3_ids = {h.segment_id for h in pr3_hits}

            # --- membership gate (PCA-space reference vs PR3) ---
            membership_missing = len(ref_ids - pr3_ids)
            membership_extra = len(pr3_ids - ref_ids)
            if membership_missing or membership_extra:
                raise SystemExit(
                    f"FAIL membership: tau={tau:.6f} filter={cell['name']} "
                    f"missing={membership_missing} extra={membership_extra}"
                )

            # --- cross-space discrepancy report (master 768-d vs threshold) ---
            # The master path's real 768-d model-space result set vs the
            # threshold path's set. Discrepancies are expected only for rows
            # within the PCA reconstruction tolerance (~2e-4) of tau — a
            # boundary flip, not a bug. Any discrepant row whose 768-d score
            # is far from tau indicates an actual error.
            keep_768 = master_scores_768 >= tau
            if mmask is not None:
                keep_768 &= mmask
            master_768_ids = set(row_keys[keep_768].tolist())
            xsym = master_768_ids ^ pr3_ids  # symmetric difference
            # Max |768-d score - tau| among discrepant rows that are in the
            # 768-d set (missing from PR3) — measures how far from the
            # boundary the discrepancy is.
            key_to_768_score = {}
            for i, k in enumerate(row_keys):
                if k not in key_to_768_score:
                    key_to_768_score[k] = float(master_scores_768[i])
            xdist_max = 0.0
            for k in xsym:
                s = key_to_768_score.get(k)
                if s is not None:
                    xdist_max = max(xdist_max, abs(s - tau))
            # Principled gate: a discrepancy beyond the reconstruction
            # tolerance + float slack is a real bug, not a boundary flip.
            _RECON_TOL = 5e-4  # PCA-256 reconstruction score error bound
            xbug = abs(xdist_max) > _RECON_TOL if xdist_max > 0 else False
            if xbug:
                raise SystemExit(
                    f"FAIL cross-space: tau={tau:.6f} filter={cell['name']} "
                    f"sym_diff={len(xsym)} max_dist={xdist_max:.2e} "
                    f"(beyond recon tol {_RECON_TOL})"
                )

            # --- eps gate (bounded_approx vs PR3's own exact re-rank) ---
            exact_hits = threshold_corpus.threshold_search(
                query, tau, fast_curation=False,
                vehicle=cell["vehicle"], date_range=date_range,
                run_uuids=ru,
            )
            exact_by_row = {h.row_id: h.score for h in exact_hits}
            bound_violations = 0
            for h in pr3_hits:
                if h.score_kind != "bounded_approx":
                    continue
                exact_score = exact_by_row.get(h.row_id)
                if exact_score is None:
                    continue
                if abs(h.score - exact_score) > h.score_error_bound + _CROSS_SPACE_TOL:
                    bound_violations += 1
            if bound_violations:
                raise SystemExit(
                    f"FAIL eps: tau={tau:.6f} filter={cell['name']} "
                    f"{bound_violations} bound violations"
                )

            above = sum(1 for h in pr3_hits if h.score_kind == "bounded_approx")
            band = sum(1 for h in pr3_hits if h.score_kind == "exact")
            selectivity = (len(pr3_ids) / master_corpus.num_rows) if master_corpus.num_rows else 0.0

            cells.append({
                "tau_percentile": float(tau_p),
                "filter": cell["name"],
                "selectivity": selectivity,
                "matches": len(pr3_ids),
                "master_ms": master_ms * 1e3,
                "pr3_ms": pr3_ms * 1e3,
                "above": above,
                "band": band,
                "membership_missing": membership_missing,
                "membership_extra": membership_extra,
                "bound_violations": bound_violations,
                "xspace_symdiff": len(xsym),
                "xspace_max_dist": xdist_max,
                "xspace_bug": xbug,
            })
    return cells


def run(
    master_uri: str,
    threshold_uri: str,
    *,
    taus: list[float] | None = None,
    filters: list[dict] | None = None,
    repeats: int = 5,
) -> dict:
    """Run the e2e sweep over already-built master + threshold corpora.

    Returns a report dict with ``cells`` (per (tau, filter) row) and the
    one-time load/hydrate timings + resident sizes.
    """
    t0 = time.perf_counter()
    master_corpus = _load_master_corpus(master_uri)
    master_load_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    threshold_corpus = _load_threshold_corpus(threshold_uri)
    threshold_hydrate_s = time.perf_counter() - t0

    threshold_ds = _threshold_dataset(threshold_uri)
    pca, _scale = lance_writer.read_pca_metadata(threshold_ds)

    # Query: unit-norm model-space, in the PCA row space (score-lossless
    # projection). Same convention as the old bench.
    rng = np.random.default_rng(1)
    latent = rng.standard_normal(D)
    latent /= np.linalg.norm(latent)
    query = (pca.astype(np.float64).T @ latent).astype("float32")
    query /= np.linalg.norm(query)

    # Master path correctness reference: score the master corpus in PCA-256
    # space (project the query, score against the converted corpus's
    # vector_fp). The master copy carries chunk_id (legacy shards have no
    # segment_id); the threshold corpus's segment_id IS chunk_id. Join on
    # chunk_id so the reference is over the master corpus's own rows.
    master_keys = np.asarray(master_corpus.chunk_id, dtype=object)
    key_to_master: dict[object, int] = {}
    for i, k in enumerate(master_keys.tolist()):
        if k not in key_to_master:
            key_to_master[k] = i

    # Read the threshold corpus's vector_fp in canonical order + its segment_ids.
    fp_table = threshold_ds.to_table(
        columns=[lance_writer.VECTOR_FP_COLUMN, "segment_id"], scan_in_order=True
    )
    fp = ts._fixed_size_list_matrix(fp_table, lance_writer.VECTOR_FP_COLUMN, np.float64)
    th_seg = fp_table.column("segment_id").to_pylist()
    query_pca = pca.astype(np.float64) @ query

    pca_ref_scores = np.full(master_corpus.num_rows, -np.inf, dtype=np.float64)
    for row_i, sid in enumerate(th_seg):
        mi = key_to_master.get(sid)
        if mi is not None:
            pca_ref_scores[mi] = float(fp[row_i] @ query_pca)

    if taus is None:
        taus = [float(np.percentile(pca_ref_scores[pca_ref_scores > -np.inf], p))
                for p in _TAU_PERCENTILES]
    if filters is None:
        filters = _filter_cells(master_corpus)

    cells = _run_sweep(
        master_corpus, threshold_corpus, threshold_ds,
        query, pca, pca_ref_scores, taus, filters, repeats=repeats,
    )

    return {
        "cells": cells,
        "master_load_s": master_load_s,
        "threshold_hydrate_s": threshold_hydrate_s,
        "master_resident_mb": master_corpus.matrix.nbytes / 1e6,
        "threshold_resident_mb": threshold_corpus.num_rows * D / 1e6,
    }


def run_synthetic(n: int, seed: int = 0, *, repeats: int = 3) -> dict:
    """Build two synthetic legacy shards, convert to both corpora, run the sweep.

    Two batches: one named ``mce113`` (vehicle="mce113"), one
    ``week1_2026-06-04_2026-06-10`` (vehicle=NULL). The vehicle filter uses
    ``"mce113"`` (the synthetic batch's derived vehicle, not conftest's
    convention). Date windows span the synthetic ``chunk_start_unix`` range;
    ``run_uuids`` is one synthetic drive id.
    """
    tmp = Path(tempfile.mkdtemp(prefix="nls_bench_e2e_"))
    try:
        mce_dir = tmp / "mce113" / "rank=00000"
        mce_uri = make_legacy_shard(mce_dir, n, seed, "mce113", with_vehicle=False)
        week_dir = tmp / "week1_2026-06-04_2026-06-10" / "rank=00000"
        week_uri = make_legacy_shard(
            week_dir, n, seed + 1, "week1_2026-06-04_2026-06-10", with_vehicle=False
        )
        shards = [
            _shard("mce113", 0, mce_uri),
            _shard("week1_2026-06-04_2026-06-10", 0, week_uri),
        ]
        dest = tmp / "dest"
        build_master_copy(shards, dest)
        build_threshold_corpus(shards, dest, fraction=1.0)

        master_uri = str(dest / "master_prod_slice")
        threshold_uri = str(dest / "threshold_prod_slice" / "corpus.lance")

        # Synthetic filter cells: vehicle="mce113"; date windows from the
        # synthetic chunk_start range; run_uuids = one synthetic drive.
        master_corpus = _load_master_corpus(master_uri)
        starts = master_corpus.chunk_start_array()
        lo, hi = int(starts.min()), int(starts.max())
        mid = lo + (hi - lo) // 2
        runs = sorted(set(master_corpus.run_uuid))
        filters = [
            {"name": "none", "vehicle": None, "date_range": None,
             "run_uuids": None},
            {"name": "vehicle", "vehicle": "mce113", "date_range": None,
             "run_uuids": None},
        ]
        if hi - lo >= _WEEK_SECONDS:
            filters.append({"name": "date-narrow", "vehicle": None,
                            "date_range": (mid, mid + _WEEK_SECONDS),
                            "run_uuids": None})
        if hi - lo >= 4 * _WEEK_SECONDS:
            filters.append({"name": "date-medium", "vehicle": None,
                            "date_range": (mid, mid + 4 * _WEEK_SECONDS),
                            "run_uuids": None})
        if runs:
            filters.append({"name": "run_uuids", "vehicle": None,
                            "date_range": None, "run_uuids": {runs[0]}})

        return run(master_uri, threshold_uri, filters=filters, repeats=repeats)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _print_report(r: dict) -> None:
    print(f"\nmaster load: {r['master_load_s']:.2f}s  "
          f"({r['master_resident_mb']:.0f} MB resident)")
    print(f"threshold hydrate: {r['threshold_hydrate_s']:.2f}s  "
          f"({r['threshold_resident_mb']:.0f} MB resident)\n")
    hdr = (f"{'tau%':>6} {'filter':<12} {'sel':>7} {'matches':>8} "
           f"{'master_ms':>10} {'pr3_ms':>8} {'above':>6} {'band':>6} "
           f"{'xdiff':>6} {'xmaxdist':>10} {'gate':>6}")
    print(hdr)
    print("-" * len(hdr))
    for c in r["cells"]:
        gate = "PASS" if (c["membership_missing"] == 0 and c["membership_extra"] == 0
                          and c["bound_violations"] == 0
                          and not c.get("xspace_bug", False)) else "FAIL"
        print(f"{c['tau_percentile']:>6.2f} {c['filter']:<12} "
              f"{c['selectivity']:>7.4%} {c['matches']:>8} "
              f"{c['master_ms']:>10.2f} {c['pr3_ms']:>8.2f} "
              f"{c['above']:>6} {c['band']:>6} "
              f"{c['xspace_symdiff']:>6} {c['xspace_max_dist']:>10.2e} "
              f"{gate:>6}")


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--master-uri", metavar="URI",
                     help="master copy dir or s3:// URI (requires --threshold-uri)")
    src.add_argument("--source", choices=["synthetic"], default=None,
                     help="synthetic mode (generates shards locally)")
    p.add_argument("--threshold-uri", metavar="URI",
                   help="threshold corpus dir or s3:// URI")
    p.add_argument("--rows", type=int, default=100_000,
                   help="synthetic rows per shard (default 100k)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--repeats", type=int, default=5,
                   help="median timing repeats per cell (default 5)")
    args = p.parse_args()

    if args.source == "synthetic":
        r = run_synthetic(args.rows, args.seed, repeats=args.repeats)
    else:
        if not args.threshold_uri:
            p.error("--master-uri requires --threshold-uri")
        r = run(args.master_uri, args.threshold_uri, repeats=args.repeats)

    _print_report(r)
    failed = any(c["membership_missing"] or c["membership_extra"]
                 or c["bound_violations"] for c in r["cells"])
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
