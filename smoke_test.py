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


def test_time_span_min_max() -> None:
    corpus = _toy_corpus()  # chunk_start_unix = [1, 2, 3, 4]
    assert corpus.time_span() == (1, 4)


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


def _toy_lance_rows() -> pa.Table:
    """Minimal Arrow schema matching production corpus columns."""
    return pa.table(
        {
            "vector": pa.array(
                [[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]],
                type=pa.list_(pa.float32()),
            ),
            "chunk_id": ["a", "b", "c"],
            "run_uuid": ["r", "r", "r"],
            "chunk_start_unix": [100, 200, 300],
            "source_media_uri": [
                "s3://x/a.mp4",
                "s3://x/b.mp4",
                "s3://x/c.mp4",
            ],
            "segment_id": ["seg_a", "seg_b", "seg_c"],
            "dx_internal_id": [11, 22, 33],
        }
    )


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
