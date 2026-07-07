"""Fast offline checks for the search core: no model download, no network.

Run from this directory:
    python smoke_test.py
"""

from __future__ import annotations

import os

import local_cache
import numpy as np
import oci_s3
import pyarrow as pa
import search_engine


def test_vectors_from_arrow_roundtrip() -> None:
    vectors = [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]
    table = pa.table({"vector": pa.array(vectors, type=pa.list_(pa.float32()))})
    out = search_engine._vectors_from_arrow(table)
    assert out.shape == (3, 3), out.shape
    np.testing.assert_allclose(out, np.array(vectors, dtype="float32"))


def test_rank_top_k_orders_by_cosine() -> None:
    # Three unit rows; query aligns most with row 1, then 0, then 2.
    matrix = np.array(
        [[0.8, 0.6, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype="float32"
    )
    corpus = search_engine.Corpus(
        matrix=matrix,
        chunk_id=["a", "b", "c"],
        run_uuid=["r", "r", "r"],
        chunk_start_unix=[1, 2, 3],
        source_media_uri=["s3://x/a.mp4", "s3://x/b.mp4", "s3://x/c.mp4"],
        segment_id=["seg_a", "seg_b", "seg_c"],
    )
    query = np.array([1.0, 0.0, 0.0], dtype="float32")
    hits = search_engine.rank_top_k(query, corpus, top_k=2)
    assert [h.chunk_id for h in hits] == ["b", "a"], hits
    assert hits[0].score > hits[1].score
    # index points back at the source matrix row (b is row 1, a is row 0).
    assert [h.index for h in hits] == [1, 0], hits


def _toy_corpus() -> "search_engine.Corpus":
    matrix = np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.6, 0.8, 0.0]],
        dtype="float32",
    )
    return search_engine.Corpus(
        matrix=matrix,
        chunk_id=["a", "b", "c", "d"],
        run_uuid=["r"] * 4,
        chunk_start_unix=[1, 2, 3, 4],
        source_media_uri=[f"s3://x/{c}.mp4" for c in "abcd"],
        segment_id=[f"seg_{c}" for c in "abcd"],
    )


def test_interval_core_standalone_merge() -> None:
    # interval_core must be importable + usable WITHOUT search_engine/torch (the
    # offline Spark workflow imports it standalone). Same bump as the
    # project_intervals test: cell 3 (=0.55) is the only >=0.45 cell; its merged
    # interval interpolates to [112.0, 116.667] (hand-computed, not derived).
    import interval_core as ic

    starts = np.array([100, 104, 108, 112, 116, 120], dtype=np.int64)
    ends = starts + 8
    scores = np.array([0.1, 0.2, 0.5, 0.6, 0.2, 0.1], dtype=np.float64)
    rows = np.arange(6)
    cs, cc, cpr, stride = ic._drive_cells(starts, ends, scores, rows)
    assert stride == 4, stride
    ivs = ic._merge_drive("r", cs, cc, cpr, 0.45)
    assert len(ivs) == 1, ivs
    assert abs(ivs[0].start_unix - 112.0) < 1e-6, ivs[0].start_unix
    assert abs(ivs[0].end_unix - 116.6667) < 1e-3, ivs[0].end_unix
    assert ivs[0].peak_index == 3, ivs[0].peak_index


def test_run_mask_filters_by_drive() -> None:
    corpus = _toy_corpus()  # run_uuid = ["r","r","r","r"]
    m = search_engine.run_mask(corpus, frozenset({"r"}))
    assert m.tolist() == [True, True, True, True], m
    m2 = search_engine.run_mask(corpus, frozenset({"other"}))
    assert m2.tolist() == [False, False, False, False], m2


def _interval_corpus(starts, scores, run="r"):
    """A 1-D corpus whose only-row dot product with [1.0] equals the given score,
    so score_corpus(corpus) reproduces `scores` exactly for interval tests."""
    n = len(starts)
    return search_engine.Corpus(
        matrix=np.array(scores, dtype="float32").reshape(n, 1),
        chunk_id=[f"{run}#t{s}" for s in starts],
        run_uuid=[run] * n,
        chunk_start_unix=list(starts),
        source_media_uri=[f"s3://x/{i}.mp4" for i in range(n)],
        segment_id=[f"seg_{i}" for i in range(n)],
    )


def test_project_intervals_interpolates_boundaries() -> None:
    # 8s clips, 4s stride at starts 100..120. The 4s-cell scores peak at cell 3
    # (=0.55, between clips 0.5 and 0.6). With a 0.45 cutoff only cell 3 survives;
    # its boundaries interpolate to 112.0 (between centers 110@0.35 and 114@0.55)
    # and 116.667 (between 114@0.55 and 118@0.40). Hand-computed, not derived.
    starts = [100, 104, 108, 112, 116, 120]
    scores = np.array([0.1, 0.2, 0.5, 0.6, 0.2, 0.1], dtype="float32")
    corpus = _interval_corpus(starts, scores)
    ivs, tau = search_engine.project_intervals(
        scores, corpus, None, mode="score", score_cutoff=0.45
    )
    assert tau == 0.45, tau
    assert len(ivs) == 1, ivs
    assert abs(ivs[0].start_unix - 112.0) < 1e-6, ivs[0].start_unix
    assert abs(ivs[0].end_unix - 116.6667) < 1e-3, ivs[0].end_unix
    assert ivs[0].peak_index == 3, ivs[0].peak_index  # clip @112 (score 0.6)


def test_project_intervals_gap_splits_into_two() -> None:
    # Two score bumps separated by a >stride gap (108 -> 200) must not merge.
    starts = [100, 104, 108, 200, 204, 208]
    scores = np.array([0.6, 0.6, 0.6, 0.7, 0.7, 0.7], dtype="float32")
    corpus = _interval_corpus(starts, scores)
    ivs, _ = search_engine.project_intervals(
        scores, corpus, None, mode="score", score_cutoff=0.5
    )
    assert len(ivs) == 2, ivs


def test_project_intervals_mask_filters_rows() -> None:
    starts = [100, 104, 108, 200, 204, 208]
    scores = np.array([0.6, 0.6, 0.6, 0.7, 0.7, 0.7], dtype="float32")
    corpus = _interval_corpus(starts, scores)
    mask = np.array([True, True, True, False, False, False])
    ivs, _ = search_engine.project_intervals(
        scores, corpus, mask, mode="score", score_cutoff=0.5
    )
    assert len(ivs) == 1, ivs  # only the first drive-half survives the mask


def test_centroid_query_averages_and_normalizes() -> None:
    corpus = _toy_corpus()
    # Mean of rows 0 ([1,0,0]) and 1 ([0,1,0]) is [.5,.5,0]; normalized that is
    # [1/sqrt2, 1/sqrt2, 0].
    out = search_engine.centroid_query(corpus, [0, 1])
    np.testing.assert_allclose(out, [0.70710677, 0.70710677, 0.0], rtol=1e-5)
    np.testing.assert_allclose(np.linalg.norm(out), 1.0, rtol=1e-6)


def test_centroid_query_refines_toward_selected_cluster() -> None:
    # Selecting rows 0 and 3 (both in the x-y plane) builds a centroid that
    # ranks the x-y rows above the pure-z row c, which a single-axis query
    # might have missed.
    corpus = _toy_corpus()
    centroid = search_engine.centroid_query(corpus, [0, 3])
    hits = search_engine.rank_top_k(centroid, corpus, top_k=4)
    assert hits[-1].chunk_id == "c", hits  # the orthogonal row ranks last


def test_centroid_query_empty_raises() -> None:
    corpus = _toy_corpus()
    try:
        search_engine.centroid_query(corpus, [])
    except ValueError:
        return
    raise AssertionError("expected ValueError for empty selection")


def _window_corpus() -> "search_engine.Corpus":
    # Drive A: 3 chunks t=[100,108),[108,116),[116,124); drive B: 1 chunk.
    matrix = np.array(
        [[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype="float32"
    )
    return search_engine.Corpus(
        matrix=matrix,
        chunk_id=["a0", "a1", "a2", "b0"],
        run_uuid=["A", "A", "A", "B"],
        chunk_start_unix=[100, 108, 116, 100],
        source_media_uri=[f"s3://x/{c}.mp4" for c in ["a0", "a1", "a2", "b0"]],
        segment_id=["segA", "segA", "segA", "segB"],
        chunk_end_unix=[108, 116, 124, 108],
    )


def test_window_query_overlap_and_mean() -> None:
    corpus = _window_corpus()
    wm = search_engine.window_query(corpus, run_uuid="A", start_unix=104, end_unix=120)
    assert wm.indices.tolist() == [0, 1, 2], wm.indices
    # mean([1,0,0],[1,0,0],[0,1,0]) = [2,1,0]/3 -> unit
    np.testing.assert_allclose(
        wm.vector, search_engine._unit(np.array([2, 1, 0], dtype="float32")), rtol=1e-5
    )
    assert wm.span_seconds == 24, wm.span_seconds
    assert wm.preview, wm.preview


def test_window_query_by_segment_id() -> None:
    corpus = _window_corpus()
    wm = search_engine.window_query(corpus, segment_id="segA")
    assert wm.indices.tolist() == [0, 1, 2], wm.indices


def test_window_query_no_match_raises() -> None:
    corpus = _window_corpus()
    try:
        search_engine.window_query(corpus, run_uuid="A", start_unix=500, end_unix=600)
    except ValueError:
        return
    raise AssertionError("expected ValueError when no chunk overlaps the window")


def test_window_query_needs_a_key() -> None:
    corpus = _window_corpus()
    try:
        search_engine.window_query(corpus)
    except ValueError:
        return
    raise AssertionError("expected ValueError when neither run_uuid nor segment_id given")


def test_rank_top_k_sets_global_rank() -> None:
    corpus = _toy_corpus()
    query = np.array([1.0, 0.0, 0.0], dtype="float32")
    hits = search_engine.rank_top_k(query, corpus, top_k=4)
    assert [h.rank for h in hits] == [1, 2, 3, 4], hits


def test_time_span_min_max() -> None:
    corpus = _toy_corpus()  # chunk_start_unix = [1, 2, 3, 4]
    assert corpus.time_span() == (1, 4)


def test_ranked_order_descends_by_score() -> None:
    # rows: a=[1,0,0], b=[0,1,0], c=[0,0,1], d=[.6,.8,0]; query on x-axis.
    corpus = _toy_corpus()
    query = np.array([1.0, 0.0, 0.0], dtype="float32")
    scores = search_engine.score_corpus(query, corpus)
    order = search_engine.ranked_order(scores, corpus)
    # a (1.0) then d (0.6); b and c (0.0) follow.
    assert order[0] == 0 and order[1] == 3, order.tolist()


def test_ranked_order_filters_by_date() -> None:
    corpus = _toy_corpus()  # times 1,2,3,4
    query = np.array([1.0, 0.0, 0.0], dtype="float32")
    scores = search_engine.score_corpus(query, corpus)
    # keep chunk_start_unix in [2, 4): rows b (2) and c (3) -> indices 1, 2.
    order = search_engine.ranked_order(scores, corpus, start_unix=2, end_unix=4)
    assert sorted(order.tolist()) == [1, 2], order.tolist()


def test_hits_from_order_windows_with_global_rank() -> None:
    corpus = _toy_corpus()
    query = np.array([1.0, 0.0, 0.0], dtype="float32")
    scores = search_engine.score_corpus(query, corpus)
    order = search_engine.ranked_order(scores, corpus)
    hits = search_engine.hits_from_order(corpus, scores, order, start=1, count=2)
    assert [h.rank for h in hits] == [2, 3], hits
    assert hits[0].index == int(order[1]), hits


def test_start_index_for_score_threshold() -> None:
    corpus = _toy_corpus()
    query = np.array([1.0, 0.0, 0.0], dtype="float32")
    scores = search_engine.score_corpus(query, corpus)
    order = search_engine.ranked_order(scores, corpus)
    # descending scores: 1.0, 0.6, 0.0, 0.0.
    assert search_engine.start_index_for_score(scores, order, 0.7) == 1  # <=0.7: d
    assert search_engine.start_index_for_score(scores, order, 0.5) == 2  # <=0.5: zeros
    assert search_engine.start_index_for_score(scores, order, 2.0) == 0  # all qualify


def test_parse_s3_uri() -> None:
    assert oci_s3.parse_s3_uri("s3://bucket/a/b.mp4") == ("bucket", "a/b.mp4")
    assert oci_s3.parse_s3_uri("s3a://bucket/a/b.mp4") == ("bucket", "a/b.mp4")
    for bad in ("http://x/y", "bucket/key", ""):
        try:
            oci_s3.parse_s3_uri(bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {bad!r}")


def test_lance_storage_options_from_env() -> None:
    os.environ["AWS_ACCESS_KEY_ID"] = "AK"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "SK"
    os.environ["AWS_ENDPOINT_URL_S3"] = (
        "https://example.compat.objectstorage.oraclecloud.com"
    )
    opts = oci_s3.lance_storage_options()
    assert opts["aws_access_key_id"] == "AK"
    assert opts["aws_endpoint"].startswith("https://")
    assert opts["aws_allow_http"] == "false"


def test_uri_key_stable_and_safe() -> None:
    uri = "s3://bucket/sibogeng/eval_pipeline/embeddings/main_bal_2k-ckpt-1200/"
    key = local_cache._uri_key(uri)
    # trailing slash must not change the key, and it must be filesystem-safe
    assert key == local_cache._uri_key(uri.rstrip("/")), "trailing slash changed key"
    assert "/" not in key and " " not in key, key
    assert local_cache._uri_key("s3://b/x") != local_cache._uri_key("s3://b/y")


def test_cache_root_env_override() -> None:
    os.environ["NLS_CACHE_ROOT"] = "/tmp/custom_nls"
    try:
        assert str(local_cache.cache_root()) == "/tmp/custom_nls"
    finally:
        del os.environ["NLS_CACHE_ROOT"]


def test_file_lock_is_reentrant_across_calls(
    tmp: str = "/tmp/nls_lock_test.lock",
) -> None:
    from pathlib import Path

    p = Path(tmp)
    with local_cache._file_lock(p):
        pass
    with local_cache._file_lock(p):  # second acquisition must succeed
        pass


def _unit_rows(rows) -> np.ndarray:
    a = np.asarray(rows, dtype=np.float64)
    return a / np.linalg.norm(a, axis=-1, keepdims=True)


def test_positive_clusters_cohesive_set_is_single() -> None:
    # A coherent 👍 set (all mutually similar) stays one prototype.
    tight = _unit_rows([[1, 0.05, 0], [1, 0, 0.04], [0.98, 0.02, 0.01], [1, 0.03, 0.02]])
    assert search_engine._positive_clusters(tight) == [[0, 1, 2, 3]]


def test_positive_clusters_bimodal_splits_into_two() -> None:
    # Two distinct themes -> two clusters (the multi-prototype case).
    bi = _unit_rows([[1, 0, 0], [0.98, 0.03, 0], [0, 1, 0], [0.02, 0.97, 0], [0.01, 1, 0.02]])
    clusters = search_engine._positive_clusters(bi)
    assert sorted(sorted(c) for c in clusters) == [[0, 1], [2, 3, 4]]


def test_positive_clusters_tolerates_outlier_and_fragmentation() -> None:
    # One mild outlier among cohesive marks -> still single (mean-pairwise gate).
    outlier = _unit_rows([[1, 0, 0], [0.97, 0.05, 0], [0.95, 0.1, 0], [0.6, 0.5, 0.1]])
    assert search_engine._positive_clusters(outlier) == [[0, 1, 2, 3]]
    # Fully fragmented (4 orthogonal marks) -> too fragmented -> fall back to one.
    assert search_engine._positive_clusters(_unit_rows(np.eye(4))) == [[0, 1, 2, 3]]


def test_fit_threshold_separable_maxf1() -> None:
    # Cleanly separable: positives ~0.8, negatives ~0.2. The F1-optimal cut sits
    # between the clusters and recovers precision=recall=1.
    pos = np.array([0.75, 0.80, 0.85, 0.90])
    neg = np.array([0.10, 0.15, 0.20, 0.25])
    out = search_engine.fit_threshold(pos, neg, objective="f1")
    assert 0.25 < out["threshold"] <= 0.75, out["threshold"]
    assert out["precision"] == 1.0 and out["recall"] == 1.0, out
    assert out["n_pos"] == 4 and out["n_neg"] == 4
    assert len(out["curve"]["tau"]) == len(out["curve"]["precision"]) > 0


def test_fit_threshold_precision_floor() -> None:
    # Overlapping: one negative (0.72) sits above a positive (0.70). A precision
    # floor of 1.0 must exclude that negative -> threshold above 0.72.
    pos = np.array([0.70, 0.78, 0.86, 0.94])
    neg = np.array([0.30, 0.50, 0.72])
    out = search_engine.fit_threshold(pos, neg, objective="precision", min_precision=1.0)
    assert out["threshold"] > 0.72, out["threshold"]
    assert out["precision"] == 1.0, out
    assert out["precision_floor_met"] is True


def test_fit_threshold_precision_floor_unreachable_flags() -> None:
    # A positive (0.40) buried below a negative (0.60): precision 1.0 is impossible
    # at any useful recall -> flag it rather than silently lying.
    pos = np.array([0.40, 0.55])
    neg = np.array([0.60, 0.62])
    out = search_engine.fit_threshold(pos, neg, objective="precision", min_precision=1.0)
    assert out["precision_floor_met"] is False, out


def test_fit_threshold_requires_both_classes() -> None:
    try:
        search_engine.fit_threshold(np.array([0.5]), np.array([]))
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError with an empty negative set")


def test_stratified_boundary_sample_excludes_labeled_and_targets_band() -> None:
    scores = np.linspace(0.0, 1.0, 100)
    labeled = {10, 20, 30}
    picks = search_engine.stratified_boundary_sample(
        scores, None, labeled, n=10, tau=0.5, band=0.05, seed=1
    )
    assert len(picks) == 10 and len(set(picks)) == 10, picks
    assert not (set(picks) & labeled), "labeled rows must be excluded"
    # Half the budget targets the [0.45, 0.55] band (rows ~45..55).
    near = [i for i in picks if abs(scores[i] - 0.5) <= 0.05]
    assert len(near) >= 4, (near, picks)
    # Sorted high-score first for the UI.
    assert picks == sorted(picks, key=lambda i: -scores[i])


def test_stratified_boundary_sample_respects_candidate_mask() -> None:
    scores = np.linspace(0.0, 1.0, 50)
    mask = np.zeros(50, dtype=bool)
    mask[:20] = True  # only the low-score half is eligible
    picks = search_engine.stratified_boundary_sample(scores, mask, None, n=8, seed=2)
    assert picks and all(i < 20 for i in picks), picks


def test_score_stats_basic() -> None:
    s = np.array([0.0, 0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    st = search_engine.score_stats(s)
    assert st["n"] == 5
    assert abs(st["mean"] - 0.2) < 1e-6, st
    assert abs(st["max"] - 0.4) < 1e-6, st
    assert st["std"] > 0
    # NaNs are dropped, not counted.
    st2 = search_engine.score_stats(np.array([0.1, np.nan, 0.3], dtype=np.float64))
    assert st2["n"] == 2, st2


def test_heuristic_threshold_monotone_and_clamped() -> None:
    tight = search_engine.score_stats(np.array([0.10, 0.11, 0.12, 0.13]))
    wide = search_engine.score_stats(np.array([0.10, 0.20, 0.30, 0.40]))
    # Wider spread -> higher cutoff (mean+k*std).
    assert search_engine.heuristic_threshold(wide) > search_engine.heuristic_threshold(tight)
    # Clamped into [0.05, 0.9].
    hi = search_engine.heuristic_threshold({"mean": 5.0, "std": 5.0})
    lo = search_engine.heuristic_threshold({"mean": -1.0, "std": 0.0})
    assert hi == 0.9 and lo == 0.05, (hi, lo)


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL {name}: {exc}")
    raise SystemExit(1 if failures else 0)
