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
    the oracle's own scores -- at several thresholds and at a corpus size
    large enough to span many pages of the screen.
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


def test_vector_fp_is_only_ever_read_for_the_needed_rows() -> None:
    """The cost model: `vector_fp` must never be materialized by an unbounded
    scan -- only fetched for the needed rows, by `take()` or by a range read
    bounded by their span. A regression here turns every query into a full
    fp32 column read.
    """
    with tempfile.TemporaryDirectory() as tmp:
        n = 20_000
        ds = conftest.build_corpus(Path(tmp), n=n, seed=11)
        query = conftest.unit_query(seed=111)
        tau = _taus(conftest.exact_scores(ds, query))[1]

        scanned: list[tuple[object, object]] = []
        taken: list[tuple[int, object]] = []
        orig_scanner, orig_take = lance.LanceDataset.scanner, lance.LanceDataset.take

        def spy_scanner(self, *args, **kwargs):
            cols = kwargs.get("columns") or (args[0] if args else None)
            scanned.append((cols, kwargs.get("limit")))
            return orig_scanner(self, *args, **kwargs)

        def spy_take(self, indices, columns=None):
            taken.append((len(indices), columns))
            return orig_take(self, indices, columns)

        lance.LanceDataset.scanner, lance.LanceDataset.take = spy_scanner, spy_take
        try:
            hits = ts.threshold_search(query, tau, ds)
        finally:
            lance.LanceDataset.scanner, lance.LanceDataset.take = orig_scanner, orig_take

        assert hits, "test tau should produce at least one match"
        for columns, limit in scanned:
            if columns and lance_writer.VECTOR_FP_COLUMN in columns:
                assert limit is not None and limit < n, (
                    f"unbounded vector_fp scan: columns={columns} limit={limit}"
                )
        fp_takes = [c for c, cols in taken if cols and lance_writer.VECTOR_FP_COLUMN in cols]
        fp_scans = [
            lim for cols, lim in scanned if cols and lance_writer.VECTOR_FP_COLUMN in cols
        ]
        assert fp_takes or fp_scans, "expected a vector_fp fetch"
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


# ---------------------------------------------------------------------------
# Hybrid prefilter path: Lance scalar-index predicate pushdown -> matching row
# ids -> resident int8 screen subset. The screen + re-rank cascade is shared
# with the resident path; only the source of `sub_idx` differs, so every test
# here asserts hybrid == resident on the same dataset/filters.
# ---------------------------------------------------------------------------


def _build_null_vehicle_corpus(tmp: Path, *, n: int, seed: int) -> lance.LanceDataset:
    """A threshold corpus whose `vehicle` column contains NULL rows.

    `conftest.write_artifact` never emits NULL vehicles, so this builds the
    artifact in place to pin Lance's SQL NULL semantics against the numpy
    object-array mask empirically.
    """
    rng = np.random.default_rng(seed)
    basis, _ = np.linalg.qr(rng.standard_normal((conftest.MODEL_DIM, conftest.D)))
    pca = np.ascontiguousarray(basis.T.astype("float32"))
    true_fp32 = (rng.standard_normal((n, conftest.D)) * rng.uniform(0.5, 3.0, size=conftest.D)).astype("float32")
    scale = np.abs(true_fp32).max(axis=0).astype("float32")
    corpus_i8 = np.clip(np.round(true_fp32 * 127.0 / scale), -127, 127).astype("int8")
    artifact = tmp / "artifact"
    artifact.mkdir(parents=True, exist_ok=True)
    np.save(artifact / lance_writer.PCA_FILE, pca)
    np.save(artifact / lance_writer.SCALE_FILE, scale)
    np.save(artifact / lance_writer.CORPUS_INT8_FILE, corpus_i8)
    np.save(artifact / lance_writer.PRE_QUANT_FP32_FILE, true_fp32)
    chunk_start = rng.integers(1_700_000_000, 1_700_100_000, size=n).astype("int64")
    vehicles = rng.choice(["veh_a", "veh_b"], size=n).tolist()
    for i in range(0, n, 7):  # inject NULL vehicles (~1/7 of rows)
        vehicles[i] = None
    import pyarrow as pa
    import pyarrow.parquet as pq
    pq.write_table(
        pa.table(
            {
                "run_uuid": [f"run-{i % 5}" for i in range(n)],
                "chunk_start_unix": chunk_start,
                "chunk_end_unix": chunk_start + 5,
                "segment_id": [f"seg-{i}" for i in range(n)],
                "vehicle": pa.array(vehicles, type=pa.string()),
            }
        ),
        artifact / lance_writer.METADATA_FILE,
    )
    return lance_writer.build_dataset(artifact, str(tmp / "out.lance"))


def test_hybrid_row_id_map_is_position_correct_multi_fragment(tmp_path: Path) -> None:
    """A genuinely multi-fragment dataset's resident row_ids are ascending, and
    a hybrid vehicle filter returns positions that index the resident screen
    correctly (equal to the resident path's `sub_idx`).

    Built with a small `max_rows_per_file` (no compaction) so the fragment-id
    encoding `(fragment_id << 32) | offset` is actually exercised -- the row-id
    map is NOT an identity here, which is the whole point of the searchsorted
    bridge. (`build_dataset` compacts to 1M-row fragments, which would make
    this a single-fragment identity map and pass even for a broken impl that
    assumed row_id == position.)
    """
    artifact_dir = conftest.write_artifact(tmp_path / "artifact", n=4_000, seed=21, pre_quant_fp32=True)
    uri = str(tmp_path / "out.lance")
    lance.write_dataset(
        lance_writer.build_table(artifact_dir), uri,
        mode="create", data_storage_version=lance_writer.DATA_STORAGE_VERSION,
        max_rows_per_file=200,
    )
    ds = lance.dataset(uri)
    assert len(ds.get_fragments()) > 1, "fixture bug: expected multiple fragments"

    hybrid = ts.ThresholdCorpus(ds, prefilter_mode="hybrid")
    row_ids = hybrid._resident.row_ids
    assert row_ids is not None
    assert np.all(np.diff(row_ids) > 0), "row_ids must be ascending at hydrate"

    # resident path sub_idx for veh_a (a resident-mode corpus holds the
    # metadata arrays; the hybrid corpus dropped them).
    resident = ts.ThresholdCorpus(ds)
    allowed = ts._filter_mask(resident._resident, "veh_a", None, None)
    assert allowed is not None
    resident_sub = np.nonzero(allowed)[0]

    sub = ts._hybrid_sub_idx(ds, row_ids, "veh_a", None, None)
    assert sub is not None
    np.testing.assert_array_equal(sub, resident_sub)


def test_hydrate_ascending_guard_raises_on_non_ascending_row_ids() -> None:
    """The hydrate guard turns a non-ascending row-id array into a hard error
    rather than letting searchsorted silently corrupt positions."""
    bad = np.array([0, 1, 5, 3, 4], dtype=np.int64)
    with pytest.raises(ValueError, match="ascending"):
        ts._assert_ascending_row_ids(bad)
    good = np.array([0, 1, 2, 3, 4], dtype=np.int64)
    ts._assert_ascending_row_ids(good)  # no raise


def test_hybrid_membership_equals_resident_across_filters(tmp_path: Path) -> None:
    """For vehicle / date_range (both bounds + open bounds) / run_uuids / none,
    hybrid membership equals the resident path equals a brute-force filtered
    reference. Fixture includes NULL-vehicle rows to pin Lance NULL semantics."""
    ds = _build_null_vehicle_corpus(tmp_path, n=3_000, seed=22)
    query = conftest.unit_query(seed=22)
    scores = conftest.exact_scores(ds, query)
    tau = float(np.percentile(scores, 90.0))
    resident = ts.ThresholdCorpus(ds)
    hybrid = ts.ThresholdCorpus(ds, prefilter_mode="hybrid")

    meta = ds.to_table(columns=["vehicle", "chunk_start_unix", "run_uuid"], scan_in_order=True)
    veh = np.array(meta.column("vehicle").to_pylist(), dtype=object)
    csu = np.array(meta.column("chunk_start_unix").to_pylist())
    run = np.array(meta.column("run_uuid").to_pylist(), dtype=object)
    lo, hi = int(np.percentile(csu, 25)), int(np.percentile(csu, 75))

    cases = [
        ("none", {}),
        ("vehicle", {"vehicle": "veh_a"}),
        ("date-range", {"date_range": (lo, hi)}),
        ("date-open-lo", {"date_range": (lo, None)}),
        ("date-open-hi", {"date_range": (None, hi)}),
        ("run_uuids", {"run_uuids": {"run-1", "run-3"}}),
    ]
    for name, kw in cases:
        ref_mask = np.ones(len(scores), dtype=bool)
        if "vehicle" in kw:
            ref_mask &= veh == kw["vehicle"]
        if "date_range" in kw:
            lo_, hi_ = kw["date_range"]
            if lo_ is not None:
                ref_mask &= csu >= lo_
            if hi_ is not None:
                ref_mask &= csu < hi_
        if "run_uuids" in kw:
            ref_mask &= np.isin(run, list(kw["run_uuids"]))
        ref = set(np.nonzero(ref_mask & (scores >= tau))[0].tolist())

        res_hits = {h.row_id for h in resident.threshold_search(query, tau, **kw)}
        hyb_hits = {h.row_id for h in hybrid.threshold_search(query, tau, **kw)}
        assert res_hits == ref, f"{name}: resident != brute-force ref, diff {res_hits ^ ref}"
        assert hyb_hits == ref, f"{name}: hybrid != brute-force ref, diff {hyb_hits ^ ref}"


def test_hybrid_no_filter_identical_to_resident_no_filter(tmp_path: Path) -> None:
    """With no filter, hybrid and resident both screen the whole corpus, so the
    result sets (row_ids + scores) are byte-identical."""
    ds = conftest.build_corpus(tmp_path, n=2_000, seed=23, pre_quant_fp32=True)
    query = conftest.unit_query(seed=23)
    scores = conftest.exact_scores(ds, query)
    tau = float(np.percentile(scores, 95.0))
    resident = ts.ThresholdCorpus(ds)
    hybrid = ts.ThresholdCorpus(ds, prefilter_mode="hybrid")
    res = resident.threshold_search(query, tau)
    hyb = hybrid.threshold_search(query, tau)
    assert [h.row_id for h in hyb] == [h.row_id for h in res]
    for hr, hb in zip(res, hyb):
        assert np.isclose(hr.score, hb.score, rtol=1e-9, atol=1e-9)
        assert hr.score_kind == hb.score_kind


def test_lance_filter_sql_assembles_and_escapes() -> None:
    """`_lance_filter_sql` AND-joins clauses, escapes quotes, and omits
    None bounds. Pure unit test (no dataset)."""
    # all None -> no filter
    assert ts._lance_filter_sql(None, None, None) is None
    # date_range=(None, None) -> no actual predicate -> None (not "")
    assert ts._lance_filter_sql(None, (None, None), None) is None
    # empty run_uuids set -> matches nothing -> literal false (not `IN ()`,
    # and not None which would mean "no filter / screen all").
    assert ts._lance_filter_sql(None, None, set()) == "false"
    # a vehicle clause alongside an empty set still matches nothing.
    assert ts._lance_filter_sql("veh_a", None, set()) == "false"
    # vehicle with an apostrophe + both date bounds + run set
    sql = ts._lance_filter_sql("o'brien", (100, 200), {"a", "b"})
    assert "o''brien" in sql
    assert "chunk_start_unix >= 100" in sql
    assert "chunk_start_unix < 200" in sql
    assert "run_uuid IN ('a', 'b')" in sql or "run_uuid IN ('b', 'a')" in sql
    # open bounds: only the present bound appears
    sql_lo = ts._lance_filter_sql(None, (100, None), None)
    assert "chunk_start_unix >= 100" in sql_lo and "chunk_start_unix <" not in sql_lo
    sql_hi = ts._lance_filter_sql(None, (None, 200), None)
    assert "chunk_start_unix < 200" in sql_hi and "chunk_start_unix >=" not in sql_hi


def test_hybrid_empty_set_filter_matches_nothing(tmp_path: Path) -> None:
    """An empty run_uuids set matches nothing in both modes (parity), neither
    raising nor screening the whole corpus."""
    ds = conftest.build_corpus(tmp_path, n=1_500, seed=24, pre_quant_fp32=True)
    query = conftest.unit_query(seed=24)
    tau = float(np.percentile(conftest.exact_scores(ds, query), 90.0))
    resident = ts.ThresholdCorpus(ds)
    hybrid = ts.ThresholdCorpus(ds, prefilter_mode="hybrid")
    assert resident.threshold_search(query, tau, run_uuids=set()) == []
    assert hybrid.threshold_search(query, tau, run_uuids=set()) == []
    # date_range=(None, None) is a no-op in both modes
    assert {h.row_id for h in hybrid.threshold_search(query, tau, date_range=(None, None))} == \
           {h.row_id for h in resident.threshold_search(query, tau)}
    # combined vehicle + empty run_uuids still matches nothing (the vehicle
    # clause must not resurrect the empty set into a screen-all).
    assert hybrid.threshold_search(query, tau, vehicle="veh_a", run_uuids=set()) == []
    assert resident.threshold_search(query, tau, vehicle="veh_a", run_uuids=set()) == []


# ---------------------------------------------------------------------------
# Read-path cost model: single union retrieval + take-vs-window-scan crossover.
# ---------------------------------------------------------------------------


def _install_fetch_spy(records: list) -> tuple:
    """Record every `take()` and `scanner()` call as (kind, size, columns, offset).

    `to_table` routes through `scanner`, so patching `take` + `scanner` sees
    every dataset read. Callers filter to row fetches via `_row_fetches`.
    """
    orig_take, orig_scanner = lance.LanceDataset.take, lance.LanceDataset.scanner

    def spy_take(self, indices, columns=None):
        records.append(("take", len(indices), tuple(columns or ()), None))
        return orig_take(self, indices, columns)

    def spy_scanner(self, *args, **kwargs):
        cols = kwargs.get("columns") or (args[0] if args else None)
        records.append(("scan", kwargs.get("limit"), tuple(cols or ()), kwargs.get("offset")))
        return orig_scanner(self, *args, **kwargs)

    lance.LanceDataset.take, lance.LanceDataset.scanner = spy_take, spy_scanner
    return orig_take, orig_scanner


def _restore_fetch_spy(originals: tuple) -> None:
    lance.LanceDataset.take, lance.LanceDataset.scanner = originals


def _row_fetches(records: list) -> list:
    """The subset of spy records that fetch row data (vector_fp / metadata),
    as opposed to filter-only scans (empty projection) or screen hydrate."""
    row_cols = {lance_writer.VECTOR_FP_COLUMN, *ts._METADATA_COLUMNS}
    return [r for r in records if r[2] and (set(r[2]) & row_cols)]


def test_resident_mode_builds_metadata_without_a_metadata_take() -> None:
    """Resident mode holds all five metadata columns from hydrate, so a query
    fetches only `vector_fp` -- and the hit metadata must still be exactly the
    dataset's rows."""
    with tempfile.TemporaryDirectory() as tmp:
        ds = conftest.build_corpus(Path(tmp), n=2_000, seed=3, pre_quant_fp32=True)
        query = conftest.unit_query(seed=3)
        scores = conftest.exact_scores(ds, query)
        tau = float(np.percentile(scores, 99.0))
        corpus = ts.ThresholdCorpus(ds)

        records: list = []
        originals = _install_fetch_spy(records)
        try:
            hits = corpus.threshold_search(query, tau)
        finally:
            _restore_fetch_spy(originals)

        assert hits, "expected matches at p99"
        for r in _row_fetches(records):
            assert set(r[2]) == {lance_writer.VECTOR_FP_COLUMN}, (
                f"resident-mode query fetched metadata columns from the dataset: {r}"
            )
        meta = ds.to_table(columns=list(ts._METADATA_COLUMNS), scan_in_order=True)
        cols = {name: meta.column(name).to_pylist() for name in ts._METADATA_COLUMNS}
        for h in hits:
            assert h.run_uuid == cols["run_uuid"][h.row_id]
            assert h.segment_id == cols["segment_id"][h.row_id]
            assert h.chunk_start_unix == cols["chunk_start_unix"][h.row_id]
            assert h.chunk_end_unix == cols["chunk_end_unix"][h.row_id]
            assert h.vehicle == cols["vehicle"][h.row_id]


def test_single_row_retrieval_per_query_across_modes() -> None:
    """Every mode combination issues AT MOST one row retrieval per query
    (resident metadata / union fetch), with both ABOVE and BAND nonempty so
    the union actually merges two populations."""
    with tempfile.TemporaryDirectory() as tmp:
        ds = conftest.build_corpus(Path(tmp), n=2_000, seed=3, pre_quant_fp32=True)
        query = conftest.unit_query(seed=3)
        scores = conftest.exact_scores(ds, query)
        tau = float(np.percentile(scores, 99.0))
        resident = ts.ThresholdCorpus(ds)
        hybrid = ts.ThresholdCorpus(ds, prefilter_mode="hybrid")

        kinds = {h.score_kind for h in resident.threshold_search(query, tau)}
        assert kinds == {"bounded_approx", "exact"}, (
            f"fixture must produce ABOVE and BAND at this tau, got {kinds}"
        )

        cases = [
            (resident, {"fast_curation": True}),
            (resident, {"fast_curation": False}),
            (hybrid, {"fast_curation": True}),
            (hybrid, {"fast_curation": False}),
            (hybrid, {"fast_curation": True, "vehicle": "veh_a"}),
        ]
        for corpus, kwargs in cases:
            records: list = []
            originals = _install_fetch_spy(records)
            try:
                hits = corpus.threshold_search(query, tau, **kwargs)
            finally:
                _restore_fetch_spy(originals)
            assert hits, kwargs
            fetches = _row_fetches(records)
            assert len(fetches) == 1, (kwargs, fetches)


def test_no_retrieval_when_meta_resident_and_band_empty() -> None:
    """A tau below every screening score minus eps makes every row ABOVE:
    resident fast_curation then needs nothing from the dataset at query time."""
    with tempfile.TemporaryDirectory() as tmp:
        n = 500
        ds = conftest.build_corpus(Path(tmp), n=n, seed=21, pre_quant_fp32=True)
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
        eps = eps_bound.eps_cauchy_schwarz(scale) * float(np.linalg.norm(query_pca))
        tau = float(screening.min()) - 2.0 * eps

        corpus = ts.ThresholdCorpus(ds)
        records: list = []
        originals = _install_fetch_spy(records)
        try:
            hits = corpus.threshold_search(query, tau)
        finally:
            _restore_fetch_spy(originals)

        assert len(hits) == n
        assert all(h.score_kind == "bounded_approx" for h in hits)
        assert _row_fetches(records) == [], _row_fetches(records)


def test_window_scan_arm_returns_identical_hits(tmp_path: Path, monkeypatch) -> None:
    """Forcing the crossover to the window-scan arm must return hits identical
    to the take arm -- scores bitwise, metadata equal -- on a genuinely
    multi-fragment dataset (the offset window crosses file boundaries), in
    both prefilter modes and both curation modes."""
    import dataclasses

    artifact = conftest.write_artifact(tmp_path / "a", n=1_000, seed=9, pre_quant_fp32=True)
    ds = lance_writer.build_dataset(artifact, str(tmp_path / "out.lance"), max_rows_per_file=200)
    query = conftest.unit_query(seed=9)
    scores = conftest.exact_scores(ds, query)
    tau = float(np.percentile(scores, 90.0))

    for mode in ("resident", "hybrid"):
        corpus = ts.ThresholdCorpus(ds, prefilter_mode=mode)
        for fast in (True, False):
            base = corpus.threshold_search(query, tau, fast_curation=fast)
            assert base, (mode, fast)

            records: list = []
            originals = _install_fetch_spy(records)
            monkeypatch.setattr(ts, "_TAKE_FLOOR_S", 1e9)
            try:
                forced = corpus.threshold_search(query, tau, fast_curation=fast)
            finally:
                _restore_fetch_spy(originals)
                monkeypatch.undo()

            scans = [r for r in _row_fetches(records) if r[0] == "scan"]
            assert scans and all(r[1] is not None for r in scans), (
                f"{mode}/fast={fast}: expected a bounded window scan, got {records}"
            )
            assert [dataclasses.astuple(h) for h in forced] == [
                dataclasses.astuple(h) for h in base
            ], f"{mode}/fast={fast}: window-scan arm diverged from take arm"


# ---------------------------------------------------------------------------
# Batched multi-query search.
# ---------------------------------------------------------------------------


def test_threshold_search_batch_matches_per_query_results() -> None:
    """A batch call returns exactly the per-query results, hit for hit --
    across prefilter modes, curation modes, and a shared filter, with
    per-query taus. The batch shares one screen pass and one retrieval, so
    equality here is the whole correctness contract."""
    import dataclasses

    with tempfile.TemporaryDirectory() as tmp:
        ds = conftest.build_corpus(Path(tmp), n=2_000, seed=3, pre_quant_fp32=True)
        queries = [conftest.unit_query(seed=s) for s in (3, 17, 29, 41)]
        base_scores = conftest.exact_scores(ds, queries[0])
        taus = [
            float(np.percentile(conftest.exact_scores(ds, q), p))
            for q, p in zip(queries, (99.0, 99.9, 95.0, 99.0))
        ]
        assert base_scores.size  # fixture sanity

        for mode in ("resident", "hybrid"):
            corpus = ts.ThresholdCorpus(ds, prefilter_mode=mode)
            for fast in (True, False):
                for kwargs in ({}, {"vehicle": "veh_a"}):
                    batch = corpus.threshold_search_batch(
                        queries, taus, fast_curation=fast, **kwargs
                    )
                    single = [
                        corpus.threshold_search(q, t, fast_curation=fast, **kwargs)
                        for q, t in zip(queries, taus)
                    ]
                    assert len(batch) == len(single) == len(queries)
                    for got, want in zip(batch, single):
                        assert [dataclasses.astuple(h) for h in got] == [
                            dataclasses.astuple(h) for h in want
                        ], (mode, fast, kwargs)


def test_threshold_search_batch_single_retrieval_and_validation() -> None:
    """The whole batch issues at most ONE row retrieval (shared union fetch),
    and mismatched queries/taus lengths are rejected."""
    with tempfile.TemporaryDirectory() as tmp:
        ds = conftest.build_corpus(Path(tmp), n=2_000, seed=3, pre_quant_fp32=True)
        queries = [conftest.unit_query(seed=s) for s in (3, 17, 29)]
        taus = [
            float(np.percentile(conftest.exact_scores(ds, q), 99.0)) for q in queries
        ]
        corpus = ts.ThresholdCorpus(ds)

        records: list = []
        originals = _install_fetch_spy(records)
        try:
            batch = corpus.threshold_search_batch(queries, taus)
        finally:
            _restore_fetch_spy(originals)
        assert any(batch), "expected matches in at least one query"
        assert len(_row_fetches(records)) <= 1, _row_fetches(records)

        with pytest.raises(ValueError, match="length"):
            corpus.threshold_search_batch(queries, taus[:2])
        with pytest.raises(ValueError, match="empty"):
            corpus.threshold_search_batch([], [])
