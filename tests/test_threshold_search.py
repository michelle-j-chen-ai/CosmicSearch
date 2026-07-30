"""Offline checks for threshold_search: synthetic artifacts, no network, no model.

Run from the repo root:
    python -m pytest tests/test_threshold_search.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import conftest
import eps_bound
import lance
import lance_writer
import numpy as np
import pytest
import threshold_search as ts


def _taus(scores: np.ndarray) -> list[float]:
    """Thresholds spanning tight to broad match sets."""
    return [float(np.percentile(scores, p)) for p in (99.9, 99.0, 90.0, 50.0)]


@pytest.mark.parametrize("n, seed", [(5_000, 2), (100_000, 1)])
def test_matches_a_brute_force_oracle_across_thresholds(n: int, seed: int) -> None:
    """The core property, against a full-column brute-force reference: the
    returned set is exactly the rows scoring >= tau, deduplicated, carrying
    the oracle's own scores, sorted descending -- at several thresholds and
    at a corpus size large enough to span many pages of the screen.
    """
    with tempfile.TemporaryDirectory() as tmp:
        ds = conftest.build_corpus(Path(tmp), n=n, seed=seed)
        query = conftest.unit_query(seed=seed + 100)
        scores = conftest.exact_scores(ds, query)

        for tau in _taus(scores):
            hits = ts.threshold_search(query, tau, ds, fast_curation=False)
            expected = set(np.nonzero(scores >= tau)[0].tolist())
            got = [h.row_id for h in hits]
            assert set(got) == expected, f"tau={tau}: symmetric diff {set(got) ^ expected}"
            assert len(got) == len(set(got)), "returned duplicate rows"
            for h in hits:
                assert np.isclose(h.score, scores[h.row_id], rtol=1e-9, atol=1e-9)
            assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


def test_zero_false_negatives_against_true_pre_quantization_score() -> None:
    """The claimed guarantee, checked against a reference that is independent
    of the int8 screen.

    Every other oracle here reads `vector_fp`, which without a
    `pca_projection_fp32.npy` artifact IS the dequantized int8 -- the same
    value the screen computes, so those checks prove plumbing
    self-consistency but could never catch a quantization-induced false
    negative. This fixture supplies a genuine pre-quantization projection and
    an orthonormal basis, so the reference score is the true one and the eps
    bound has real quantization error to absorb.
    """
    with tempfile.TemporaryDirectory() as tmp:
        ds = conftest.build_corpus(
            Path(tmp), n=20_000, seed=31, orthonormal_pca=True, pre_quant_fp32=True
        )
        pca, _scale = lance_writer.read_pca_metadata(ds)
        query = conftest.unit_query(seed=131, pca=pca)
        assert abs(float(np.linalg.norm(query)) - 1.0) < 1e-4

        true_scores = conftest.exact_scores(ds, query)
        for tau in _taus(true_scores):
            got = {h.row_id for h in ts.threshold_search(query, tau, ds)}
            missing = set(np.nonzero(true_scores >= tau)[0].tolist()) - got
            assert not missing, (
                f"tau={tau}: dropped {len(missing)} row(s) whose TRUE "
                f"pre-quantization score >= tau: {sorted(missing)[:10]}"
            )


def test_eps_is_what_prevents_false_negatives() -> None:
    """Mutation check: plant tau between a row's screening score and its exact
    score, so the row is only found because eps widens the screen. Forcing
    eps=0 must drop it -- otherwise the oracle comparisons above would pass
    even with a broken bound and prove nothing.
    """
    with tempfile.TemporaryDirectory() as tmp:
        n = 2_000
        ds = conftest.build_corpus(Path(tmp), n=n, seed=21)
        query = conftest.unit_query(seed=121)
        pca, scale = lance_writer.read_pca_metadata(ds)

        corpus_i8 = ts._fixed_size_list_matrix(
            ds.to_table(columns=[lance_writer.EMBEDDING_I8_COLUMN], scan_in_order=True),
            lance_writer.EMBEDDING_I8_COLUMN,
            np.int8,
        )
        query_pca = ts._project_query(query, pca)
        screening = np.empty(n, dtype=np.float32)
        ts.gpu_corpus._cpu_score_kernel()(
            np.ascontiguousarray(corpus_i8),
            (query_pca * (scale.astype(np.float32) / 127.0)).astype(np.float32),
            screening,
        )
        exact = conftest.exact_scores(ds, query)

        gap = exact - screening.astype(np.float64)
        row = int(np.argmax(gap))
        assert gap[row] > 0, "no row where the screen underestimates the exact score"
        tau = (float(screening[row]) + float(exact[row])) / 2.0

        assert row in {h.row_id for h in ts.threshold_search(query, tau, ds)}

        original = eps_bound.eps_cauchy_schwarz
        eps_bound.eps_cauchy_schwarz = lambda scale: 0.0
        try:
            assert row not in {h.row_id for h in ts.threshold_search(query, tau, ds)}, (
                "eps=0 still found the row -- the oracle comparisons would not "
                "catch a broken eps bound"
            )
        finally:
            eps_bound.eps_cauchy_schwarz = original


def test_vector_fp_is_only_ever_read_by_take_on_a_small_row_set() -> None:
    """The cost model: `vector_fp` must never be materialized by a scan, only
    fetched by `take()` for the match set. A regression here turns every query
    into a full fp32 column read.
    """
    with tempfile.TemporaryDirectory() as tmp:
        n = 20_000
        ds = conftest.build_corpus(Path(tmp), n=n, seed=11)
        query = conftest.unit_query(seed=111)
        tau = _taus(conftest.exact_scores(ds, query))[1]

        scanned: list[object] = []
        taken: list[tuple[int, object]] = []
        orig_to_table, orig_take = lance.LanceDataset.to_table, lance.LanceDataset.take

        def spy_to_table(self, *args, **kwargs):
            scanned.append(kwargs.get("columns", args[0] if args else None))
            return orig_to_table(self, *args, **kwargs)

        def spy_take(self, indices, columns=None):
            taken.append((len(indices), columns))
            return orig_take(self, indices, columns)

        lance.LanceDataset.to_table, lance.LanceDataset.take = spy_to_table, spy_take
        try:
            hits = ts.threshold_search(query, tau, ds)
        finally:
            lance.LanceDataset.to_table, lance.LanceDataset.take = orig_to_table, orig_take

        assert hits, "test tau should produce at least one match"
        for columns in scanned:
            assert columns and lance_writer.VECTOR_FP_COLUMN not in columns, columns
        fp_takes = [c for c, cols in taken if cols and lance_writer.VECTOR_FP_COLUMN in cols]
        assert fp_takes, "expected at least one take() for vector_fp"
        assert all(count < n * 0.2 for count in fp_takes), (fp_takes, n)


def test_scan_order_pinned_survives_many_fragments() -> None:
    """Screening derives row positions from a scan, then `take()`s at those
    positions -- correct only while the scan returns canonical order. Built
    without compaction so a dropped `scan_in_order` pin has many fragments to
    actually reorder.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        artifact_dir = conftest.write_artifact(tmp_path / "artifact", n=5_000, seed=51)
        uri = str(tmp_path / "out.lance")
        lance.write_dataset(
            lance_writer.build_table(artifact_dir),
            uri,
            mode="create",
            data_storage_version=lance_writer.DATA_STORAGE_VERSION,
            max_rows_per_file=200,
        )
        ds = lance.dataset(uri)
        assert len(ds.get_fragments()) > 1, "fixture bug: expected multiple fragments"

        query = conftest.unit_query(seed=151)
        scores = conftest.exact_scores(ds, query)
        tau = _taus(scores)[1]
        got = {h.row_id for h in ts.threshold_search(query, tau, ds)}
        assert got == set(np.nonzero(scores >= tau)[0].tolist())


def test_threshold_corpus_decodes_embedding_i8_once_and_rejects_bad_queries() -> None:
    # Residency: the screen column is decoded at construction and reused, not
    # re-read per query. And a non-unit-norm query invalidates the eps bound's
    # precondition, so it must raise rather than silently return wrong rows.
    with tempfile.TemporaryDirectory() as tmp:
        ds = conftest.build_corpus(Path(tmp), n=2_000, seed=61)
        query = conftest.unit_query(seed=161)
        tau = _taus(conftest.exact_scores(ds, query))[1]

        decodes: list[object] = []
        orig_to_table = lance.LanceDataset.to_table

        def spy_to_table(self, *args, **kwargs):
            columns = kwargs.get("columns", args[0] if args else None)
            if columns and lance_writer.EMBEDDING_I8_COLUMN in columns:
                decodes.append(columns)
            return orig_to_table(self, *args, **kwargs)

        lance.LanceDataset.to_table = spy_to_table
        try:
            corpus = ts.ThresholdCorpus(ds)
            for _ in range(3):
                corpus.threshold_search(query, tau)
            assert len(decodes) == 1, f"embedding_i8 decoded {len(decodes)}x, expected once"
        finally:
            lance.LanceDataset.to_table = orig_to_table

        with pytest.raises(ValueError, match="unit-norm"):
            corpus.threshold_search(query * 3.0, tau)


def test_eps_bounds_the_screen_error_even_when_the_basis_amplifies() -> None:
    """The eps window is scaled by ||query_pca||, not left at the corpus
    constant, so the bound survives a basis whose rows are not orthonormal.

    A unit-norm model-space query can project to a much longer PCA-space
    vector through such a basis, and the screening error scales with it. This
    fixture is built to defeat the unscaled constant, so the assertions below
    fail if that scaling is ever dropped.
    """
    with tempfile.TemporaryDirectory() as tmp:
        ds = conftest.build_corpus(
            Path(tmp), n=4_000, seed=13, orthonormal_pca=False, pre_quant_fp32=True
        )
        pca, scale = lance_writer.read_pca_metadata(ds)
        query = conftest.unit_query(seed=113)
        query_pca = ts._project_query(query, pca)
        amplification = float(np.linalg.norm(query_pca))
        assert amplification > 1.5, f"fixture should amplify the projection, got {amplification}"

        corpus_i8 = ts._fixed_size_list_matrix(
            ds.to_table(columns=[lance_writer.EMBEDDING_I8_COLUMN], scan_in_order=True),
            lance_writer.EMBEDDING_I8_COLUMN,
            np.int8,
        )
        screen = np.empty(corpus_i8.shape[0], dtype=np.float32)
        ts.gpu_corpus._cpu_score_kernel()(
            np.ascontiguousarray(corpus_i8),
            (query_pca * (scale.astype(np.float32) / 127.0)).astype(np.float32),
            screen,
        )
        exact = conftest.exact_scores(ds, query)
        corpus_eps = eps_bound.eps_cauchy_schwarz(scale)

        # Plant tau where the UNSCALED corpus constant would exclude a row the
        # exact score keeps: screen[row] < tau - corpus_eps <= exact[row]. Only
        # a window scaled by the projection's amplification still covers it.
        gap = exact - screen.astype(np.float64)
        row = int(np.argmax(gap))
        assert gap[row] > corpus_eps, (
            f"fixture is not adversarial: largest screen error {gap[row]} is "
            f"already inside the unscaled eps {corpus_eps}"
        )
        tau = float(screen[row]) + corpus_eps + (gap[row] - corpus_eps) / 2.0
        assert float(screen[row]) < tau - corpus_eps <= exact[row]

        hits = ts.threshold_search(query, tau, ds)
        assert row in {h.row_id for h in hits}, (
            "dropped a row whose exact score is above tau -- the eps window was "
            "not scaled by ||query_pca||, so it is too narrow for this basis"
        )
        missing = set(np.nonzero(exact >= tau)[0].tolist()) - {h.row_id for h in hits}
        assert not missing, f"dropped {len(missing)} row(s) above tau"


def test_no_false_negatives_on_the_real_corpus_sample() -> None:
    """The guarantee on a real score distribution, offline.

    Synthetic corpora are shaped to resemble the production one but cannot
    reproduce its exact spectrum or per-dim quantization scales, which are
    what eps is sized against. This runs against a committed 10k-row sample
    of the real corpus (no network) whose `vector_fp` is a genuine
    pre-quantization projection, so the oracle is independent of the screen.
    """
    import bench_common as bench

    with tempfile.TemporaryDirectory() as tmp:
        embeddings, metadata, pca = bench.fixture_embeddings()
        ds = lance.dataset(bench.build_corpus(embeddings, metadata, Path(tmp), pca))
        query = conftest.unit_query(seed=7, pca=pca)
        scores = conftest.exact_scores(ds, query)

        for tau in _taus(scores):
            got = {h.row_id for h in ts.threshold_search(query, tau, ds)}
            missing = set(np.nonzero(scores >= tau)[0].tolist()) - got
            assert not missing, f"tau={tau}: dropped {len(missing)} real row(s)"


def test_threshold_hit_defaults_to_exact_score_kind() -> None:
    """The new fields default so existing callers (exact re-rank) are unchanged."""
    h = ts.ThresholdHit(
        row_id=0, score=0.5, run_uuid="r", segment_id="s",
        chunk_start_unix=1, chunk_end_unix=2, vehicle="v",
    )
    assert h.score_kind == "exact"
    assert h.score_error_bound == 0.0


def test_fast_curation_same_membership_mixed_score_kinds() -> None:
    """fast_curation returns the same rows as the exact path, but ABOVE rows
    carry bounded screening scores (no take()) and BAND rows carry exact
    scores. The eps bound must hold for every bounded row."""
    with tempfile.TemporaryDirectory() as tmp:
        ds = conftest.build_corpus(Path(tmp), n=2_000, seed=3, pre_quant_fp32=True)
        query = conftest.unit_query(seed=3)
        scores = conftest.exact_scores(ds, query)

        # A tau that leaves a non-trivial BAND: pick a percentile where the
        # int8 screen straddles tau within eps for some rows.
        tau = float(np.percentile(scores, 99.0))

        exact = ts.threshold_search(query, tau, ds, fast_curation=False)
        fast = ts.threshold_search(query, tau, ds, fast_curation=True)

        # Same membership (the core property).
        exact_ids = {h.row_id for h in exact}
        fast_ids = {h.row_id for h in fast}
        assert fast_ids == exact_ids, f"symmetric diff {fast_ids ^ exact_ids}"

        # Score-kind tagging: every hit is exact or bounded_approx.
        kinds = {h.score_kind for h in fast}
        assert kinds <= {"exact", "bounded_approx"}, kinds
        assert "bounded_approx" in kinds or len(fast) == 0, "expected ABOVE rows to be bounded"
        assert "exact" in kinds or len(fast) == 0, "expected BAND rows to be exact"

        # Every bounded row's screen score is within its error bound of the
        # true score; every exact row matches the true score.
        for h in fast:
            true = scores[h.row_id]
            if h.score_kind == "bounded_approx":
                assert abs(h.score - true) <= h.score_error_bound + 1e-6, (
                    f"row {h.row_id}: |{h.score} - {true}| = {abs(h.score - true)} "
                    f"> bound {h.score_error_bound}"
                )
            else:
                assert np.isclose(h.score, true, rtol=1e-9, atol=1e-9)


def test_fast_curation_still_filters_band_by_exact_score() -> None:
    """fast_curation must still re-rank BAND rows and drop those below tau.
    A BAND row whose true score is below tau must NOT appear."""
    with tempfile.TemporaryDirectory() as tmp:
        ds = conftest.build_corpus(Path(tmp), n=2_000, seed=7, pre_quant_fp32=True)
        query = conftest.unit_query(seed=7)
        scores = conftest.exact_scores(ds, query)
        tau = float(np.percentile(scores, 99.0))

        fast = ts.threshold_search(query, tau, ds, fast_curation=True)
        for h in fast:
            # Every returned row's TRUE score is >= tau. Bounded rows are
            # ABOVE (provably >= tau by the bound); exact rows are BAND that
            # passed the exact >= tau filter. Either way, no row below tau.
            assert scores[h.row_id] >= tau - 1e-9, (
                f"row {h.row_id} true score {scores[h.row_id]} < tau {tau}"
            )


def test_default_mode_is_fast_curation() -> None:
    """Calling threshold_search WITHOUT the fast_curation kwarg must default to
    fast_curation=True: ABOVE rows carry a bounded_approx screening score.
    If the default were ever flipped back to False, this test catches it."""
    with tempfile.TemporaryDirectory() as tmp:
        ds = conftest.build_corpus(Path(tmp), n=2000, seed=3, pre_quant_fp32=True)
        query = conftest.unit_query(seed=3)
        scores = conftest.exact_scores(ds, query)
        tau = float(np.percentile(scores, 99.0))

        hits = ts.threshold_search(query, tau, ds)
        assert hits, "expected at least one match at p99"
        kinds = {h.score_kind for h in hits}
        assert "bounded_approx" in kinds, (
            f"default mode produced no bounded_approx hits; kinds={kinds} "
            "-- the default is not fast_curation"
        )


def test_bench_e2e_offline_smoke() -> None:
    """Offline e2e bench: builds synthetic master+threshold corpora, runs the
    sweep, membership + eps gates pass, both paths timed."""
    import bench_e2e

    r = bench_e2e.run_synthetic(n=2_000, seed=5)
    assert r["cells"], "no cells produced"
    for cell in r["cells"]:
        assert cell["membership_missing"] == 0 and cell["membership_extra"] == 0
        assert cell["bound_violations"] == 0
        assert cell["master_ms"] > 0 and cell["pr3_ms"] > 0
    names = {c["filter"] for c in r["cells"]}
    assert "none" in names and "vehicle" in names, f"expected none + vehicle cells, got {names}"


def test_prefilter_matches_brute_force_filtered_reference() -> None:
    """Filtered search returns exactly the rows passing BOTH the filter and
    the tau cut — vehicle, date, run_uuid, each alone and combined."""
    with tempfile.TemporaryDirectory() as tmp:
        ds = conftest.build_corpus(Path(tmp), n=2_000, seed=11, pre_quant_fp32=True)
        query = conftest.unit_query(seed=11)
        scores = conftest.exact_scores(ds, query)
        tau = float(np.percentile(scores, 95.0))
        corpus = ts.ThresholdCorpus(ds)

        meta = ds.to_table(
            columns=["vehicle", "chunk_start_unix", "run_uuid"], scan_in_order=True
        )
        veh = np.array(meta.column("vehicle").to_pylist())
        csu = np.array(meta.column("chunk_start_unix").to_pylist())
        run = np.array(meta.column("run_uuid").to_pylist())

        lo, hi = int(np.percentile(csu, 25)), int(np.percentile(csu, 75))
        cases = [
            ({"vehicle": "veh_a"}, veh == "veh_a"),
            ({"date_range": (lo, hi)}, (csu >= lo) & (csu < hi)),
            ({"run_uuids": {"run-1", "run-3"}}, np.isin(run, ["run-1", "run-3"])),
            (
                {"vehicle": "veh_b", "date_range": (lo, None)},
                (veh == "veh_b") & (csu >= lo),
            ),
        ]
        for kwargs, fmask in cases:
            hits = corpus.threshold_search(query, tau, **kwargs)
            expected = set(np.nonzero(fmask & (scores >= tau))[0].tolist())
            got = {h.row_id for h in hits}
            assert got == expected, f"{kwargs}: diff {got ^ expected}"


def test_prefilter_empty_and_none_cases() -> None:
    """Zero-match filter returns [] cleanly; no-filter equals unfiltered."""
    with tempfile.TemporaryDirectory() as tmp:
        ds = conftest.build_corpus(Path(tmp), n=1_000, seed=12, pre_quant_fp32=True)
        query = conftest.unit_query(seed=12)
        scores = conftest.exact_scores(ds, query)
        tau = float(np.percentile(scores, 99.0))
        corpus = ts.ThresholdCorpus(ds)
        assert corpus.threshold_search(query, tau, vehicle="no_such_vehicle") == []
        unfiltered = {h.row_id for h in corpus.threshold_search(query, tau)}
        nofilter_kwargs = {h.row_id for h in corpus.threshold_search(
            query, tau, vehicle=None, date_range=None, run_uuids=None)}
        assert unfiltered == nofilter_kwargs


def test_prefilter_preserves_fast_curation_semantics() -> None:
    """Under a filter, ABOVE rows are still bounded_approx within their eps
    bound and BAND rows exact — same contract as unfiltered."""
    with tempfile.TemporaryDirectory() as tmp:
        ds = conftest.build_corpus(Path(tmp), n=2_000, seed=13, pre_quant_fp32=True)
        query = conftest.unit_query(seed=13)
        scores = conftest.exact_scores(ds, query)
        tau = float(np.percentile(scores, 95.0))
        corpus = ts.ThresholdCorpus(ds)
        hits = corpus.threshold_search(query, tau, vehicle="veh_a")
        assert hits, "fixture bug: veh_a filter matched nothing at p95"
        kinds = {h.score_kind for h in hits}
        assert kinds <= {"exact", "bounded_approx"}
        for h in hits:
            if h.score_kind == "bounded_approx":
                assert abs(h.score - scores[h.row_id]) <= h.score_error_bound + 1e-6
            else:
                assert np.isclose(h.score, scores[h.row_id], rtol=1e-9, atol=1e-9)
