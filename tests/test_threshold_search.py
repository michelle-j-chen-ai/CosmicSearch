"""Offline checks for threshold_search: synthetic artifacts, no network, no model.

Run from the repo root:
    python -m pytest tests/test_threshold_search.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import eps_bound
import gpu_corpus
import lance
import lance_writer
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import threshold_search as ts

_D = 256
_MODEL_DIM = 768


def _write_synthetic_artifacts(tmp_dir: Path, n: int, seed: int = 0) -> Path:
    """Write a gpu_corpus-style int8 PCA artifact under `tmp_dir` and return it.

    Same generator shape as test_lance_writer.py's, so threshold_search is
    exercised against the same builder fixture lance_writer's own tests use.
    """
    rng = np.random.default_rng(seed)
    artifact_dir = tmp_dir / "artifact"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    pca = rng.standard_normal((_D, _MODEL_DIM)).astype("float32")
    scale = rng.uniform(0.01, 0.5, size=_D).astype("float32")
    corpus_i8 = rng.integers(-127, 128, size=(n, _D), dtype=np.int8)

    np.save(artifact_dir / lance_writer.PCA_FILE, pca)
    np.save(artifact_dir / lance_writer.SCALE_FILE, scale)
    np.save(artifact_dir / lance_writer.CORPUS_INT8_FILE, corpus_i8)

    chunk_start = rng.integers(1_700_000_000, 1_700_100_000, size=n).astype("int64")
    vehicles = rng.choice(["veh_a", "veh_b", "veh_c"], size=n)
    table = pa.table(
        {
            "run_uuid": [f"run-{i % 5}" for i in range(n)],
            "chunk_start_unix": chunk_start,
            "chunk_end_unix": chunk_start + 5,
            "segment_id": [f"seg-{i}" for i in range(n)],
            "vehicle": vehicles,
        }
    )
    pq.write_table(table, artifact_dir / lance_writer.METADATA_FILE)
    return artifact_dir


def _build(tmp_path: Path, n: int, seed: int = 0) -> lance.LanceDataset:
    artifact_dir = _write_synthetic_artifacts(tmp_path, n=n, seed=seed)
    return lance_writer.build_dataset(artifact_dir, str(tmp_path / "out.lance"))


def _query(seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    q = rng.standard_normal(_MODEL_DIM).astype("float32")
    return q / np.linalg.norm(q)


def _brute_force_oracle(
    ds: lance.LanceDataset, query: np.ndarray, tau: float
) -> list[tuple[int, float]]:
    """Naive oracle: exact fp32 score for EVERY row, filtered to >= tau.

    Deliberately reads the whole `vector_fp` column via `to_table` -- the
    "slow path" threshold_search exists to avoid -- as the independent
    reference implementation. Uses the same `_project_query` fp32 projection
    as threshold_search so the two scores are bit-comparable, not just
    approximately equal.
    """
    pca, _scale = lance_writer.read_pca_metadata(ds)
    query_pca = ts._project_query(query, pca).astype(np.float64)
    table = ds.to_table(columns=[lance_writer.VECTOR_FP_COLUMN])
    fp = ts._fixed_size_list_matrix(table, lance_writer.VECTOR_FP_COLUMN, np.float64)
    scores = fp @ query_pca
    idx = np.nonzero(scores >= tau)[0]
    order = idx[np.argsort(-scores[idx], kind="stable")]
    return [(int(i), float(scores[i])) for i in order]


def _assert_matches_oracle(hits: list[ts.ThresholdHit], oracle: list[tuple[int, float]]) -> None:
    hit_ids = [h.row_id for h in hits]
    oracle_ids = [i for i, _ in oracle]
    assert set(hit_ids) == set(oracle_ids), (
        f"row-id set mismatch: threshold_search returned {len(hit_ids)} rows, "
        f"oracle {len(oracle_ids)}; symmetric diff "
        f"{set(hit_ids) ^ set(oracle_ids)}"
    )
    assert len(hit_ids) == len(set(hit_ids)), "threshold_search returned duplicate rows"
    hit_scores = {h.row_id: h.score for h in hits}
    oracle_scores = {i: s for i, s in oracle}
    for row_id, oracle_score in oracle_scores.items():
        assert np.isclose(hit_scores[row_id], oracle_score, rtol=1e-9, atol=1e-9), (
            row_id,
            hit_scores[row_id],
            oracle_score,
        )
    # Sorted descending by score.
    scores_in_order = [h.score for h in hits]
    assert scores_in_order == sorted(scores_in_order, reverse=True), scores_in_order


def _tau_candidates(ds: lance.LanceDataset, query: np.ndarray) -> list[float]:
    """A few tau values spanning the score distribution (tight to broad match)."""
    pca, _scale = lance_writer.read_pca_metadata(ds)
    query_pca = ts._project_query(query, pca).astype(np.float64)
    table = ds.to_table(columns=[lance_writer.VECTOR_FP_COLUMN])
    fp = ts._fixed_size_list_matrix(table, lance_writer.VECTOR_FP_COLUMN, np.float64)
    scores = fp @ query_pca
    return [
        float(np.percentile(scores, p)) for p in (99.9, 99.0, 90.0, 50.0, 10.0)
    ]


def _check_zero_false_negatives(n: int, seed: int) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        ds = _build(tmp_path, n=n, seed=seed)
        query = _query(seed=seed + 100)
        for tau in _tau_candidates(ds, query):
            hits = ts.threshold_search(query, tau, ds)
            oracle = _brute_force_oracle(ds, query, tau)
            _assert_matches_oracle(hits, oracle)


def test_zero_false_negatives_10k() -> None:
    _check_zero_false_negatives(n=10_000, seed=0)


def test_zero_false_negatives_100k() -> None:
    # A 100k-row synthetic dataset exercises the full screen -> take() ->
    # re-rank path with a realistic row count in well under a minute on a dev
    # machine; a 1M-row run is substantially slower and not needed to prove
    # the property.
    _check_zero_false_negatives(n=100_000, seed=1)


def test_zero_false_negatives_across_multiple_seeds() -> None:
    # A handful of independent corpora/queries, not just one lucky draw.
    for seed in (2, 3, 4):
        _check_zero_false_negatives(n=5_000, seed=seed)


def test_band_selectivity_is_small_on_uniform_data() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        n = 50_000
        ds = _build(tmp_path, n=n, seed=7)
        query = _query(seed=107)
        pca, scale = lance_writer.read_pca_metadata(ds)
        query_pca = ts._project_query(query, pca)

        i8_table = ds.to_table(columns=[lance_writer.EMBEDDING_I8_COLUMN])
        corpus_i8 = ts._fixed_size_list_matrix(
            i8_table, lance_writer.EMBEDDING_I8_COLUMN, np.int8
        )
        w = (query_pca * (scale.astype(np.float32) / 127.0)).astype(np.float32)
        screening_scores = np.empty(n, dtype=np.float32)
        gpu_corpus._cpu_score_kernel()(np.ascontiguousarray(corpus_i8), w, screening_scores)
        tau = float(np.percentile(screening_scores, 50.0))
        eps = eps_bound.eps_cauchy_schwarz(scale)
        above, band, below = eps_bound.classify(screening_scores, tau, eps)

        band_fraction = float(band.sum()) / n
        print(f"  band fraction on uniform random data (n={n}): {band_fraction:.6f}")
        assert above.sum() + band.sum() + below.sum() == n
        # Not a hard 1e-4 gate (uniform-random int8 corpora don't reproduce a
        # real embedding's score distribution), just a sanity check that the
        # band is a small sliver, not a large fraction, of the corpus.
        assert band_fraction < 0.05, band_fraction


def test_never_fully_materializes_vector_fp() -> None:
    """threshold_search must only ever touch vector_fp via take() on a small
    row-index set, never via to_table() (a full or near-full column read)."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        n = 20_000
        ds = _build(tmp_path, n=n, seed=11)
        query = _query(seed=111)
        tau = _tau_candidates(ds, query)[1]  # 99th percentile: a real, small band+above

        to_table_calls: list[object] = []
        take_calls: list[tuple[int, object]] = []
        orig_to_table = lance.LanceDataset.to_table
        orig_take = lance.LanceDataset.take

        def spy_to_table(self, *args, **kwargs):
            columns = kwargs.get("columns") if "columns" in kwargs else (
                args[0] if args else None
            )
            to_table_calls.append(columns)
            return orig_to_table(self, *args, **kwargs)

        def spy_take(self, indices, columns=None):
            take_calls.append((len(indices), columns))
            return orig_take(self, indices, columns)

        lance.LanceDataset.to_table = spy_to_table
        lance.LanceDataset.take = spy_take
        try:
            hits = ts.threshold_search(query, tau, ds)
        finally:
            lance.LanceDataset.to_table = orig_to_table
            lance.LanceDataset.take = orig_take

        for columns in to_table_calls:
            assert columns is not None and lance_writer.VECTOR_FP_COLUMN not in columns, (
                "to_table() must never request vector_fp (would fully materialize "
                f"it); got columns={columns}"
            )

        vector_fp_take_calls = [
            (count, columns)
            for count, columns in take_calls
            if columns and lance_writer.VECTOR_FP_COLUMN in columns
        ]
        assert vector_fp_take_calls, "expected at least one take() call for vector_fp"
        for count, _columns in vector_fp_take_calls:
            assert count < n, (count, n)
            # A real threshold match set is a small sliver of the corpus, not
            # "most of it" -- catches an accidental take() of every row.
            assert count < n * 0.2, (count, n)
        assert len(hits) > 0, "test tau should produce at least one match"


def test_mutation_eps_zero_causes_false_negative() -> None:
    """Proves the oracle comparison actually discriminates: with a real eps,
    a boundary row is found; forcing eps=0 (no screening safety margin) drops
    it, a genuine false negative. If this test failed to show a difference,
    the zero-false-negative tests above would not be trustworthy."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        n = 2_000
        ds = _build(tmp_path, n=n, seed=21)
        query = _query(seed=121)
        pca, scale = lance_writer.read_pca_metadata(ds)
        query_pca = ts._project_query(query, pca)

        i8_table = ds.to_table(columns=[lance_writer.EMBEDDING_I8_COLUMN])
        corpus_i8 = ts._fixed_size_list_matrix(
            i8_table, lance_writer.EMBEDDING_I8_COLUMN, np.int8
        )
        w = (query_pca * (scale.astype(np.float32) / 127.0)).astype(np.float32)
        screening_scores = np.empty(n, dtype=np.float32)
        gpu_corpus._cpu_score_kernel()(np.ascontiguousarray(corpus_i8), w, screening_scores)

        fp_table = ds.to_table(columns=[lance_writer.VECTOR_FP_COLUMN])
        fp = ts._fixed_size_list_matrix(fp_table, lance_writer.VECTOR_FP_COLUMN, np.float64)
        exact_scores = fp @ query_pca.astype(np.float64)

        # Find a row where the fastmath float32 screening kernel and the plain
        # float64 exact computation disagree, then plant tau exactly between
        # them so eps=0 puts the row on the wrong side of the screening cut
        # while its exact score is still >= tau.
        diff = exact_scores - screening_scores.astype(np.float64)
        candidates = np.nonzero(diff > 0)[0]
        assert candidates.size > 0, (
            "no row where the float32 screening kernel underestimates the "
            "float64 exact score -- cannot construct the mutation probe"
        )
        row = int(candidates[np.argmax(diff[candidates])])
        tau = (float(screening_scores[row]) + float(exact_scores[row])) / 2.0
        assert screening_scores[row] < tau <= exact_scores[row]

        real_hits = {h.row_id for h in ts.threshold_search(query, tau, ds)}
        assert row in real_hits, "row should be found with the real eps bound"

        original_eps_fn = eps_bound.eps_cauchy_schwarz
        eps_bound.eps_cauchy_schwarz = lambda scale: 0.0
        try:
            mutated_hits = {h.row_id for h in ts.threshold_search(query, tau, ds)}
        finally:
            eps_bound.eps_cauchy_schwarz = original_eps_fn

        assert row not in mutated_hits, (
            "mutation check failed to reproduce a false negative -- the oracle "
            "comparison would not actually catch a broken eps bound"
        )


def _orthonormal_rows(rng: np.random.Generator, d: int, model_dim: int) -> np.ndarray:
    """A (d, model_dim) matrix with orthonormal rows (P @ P.T == I_d).

    Matches the real system's score-lossless SVD basis property (see
    gpu_corpus.py's module docstring): a query/embedding that lies in this
    basis's row space has its norm and inner products preserved exactly by
    projection, which is what makes eps_cauchy_schwarz's ||query_pca||_2 == 1
    precondition hold for a unit-norm raw query -- unlike the plain
    `rng.standard_normal((_D, _MODEL_DIM))` fixture the rest of this file
    uses, which is NOT orthonormal and so does not preserve query norm under
    projection (harmless for those tests, which only ever compare
    threshold_search's screen against its own vector_fp in the same,
    consistently-projected space; see test_zero_false_negatives_against_true_
    pre_quantization_score's docstring for why it matters here).
    """
    q, _ = np.linalg.qr(rng.standard_normal((model_dim, d)))
    return q[:, :d].T.astype(np.float32)


def test_zero_false_negatives_against_true_pre_quantization_score() -> None:
    """The one test in this suite that checks the actual claimed guarantee:
    zero false negatives against the TRUE pre-quantization score, independent
    of int8/vector_fp.

    Every other zero-false-negative test in this file (and in
    test_real_corpus_integration.py / test_load_corpus_threshold.py) oracles
    against `vector_fp`, which -- absent a `pca_projection_fp32.npy` artifact
    -- IS the dequantized int8 (see lance_writer.py's module docstring), the
    same value threshold_search's screen already computes. Those tests
    validate self-consistency of the screen/take/re-rank plumbing; they
    cannot detect a quantization-induced false negative, because no
    independent fp32 reference exists in their fixtures.

    This test supplies a genuine `pca_projection_fp32.npy` (the true,
    pre-quantization PCA-256 projection) so lance_writer stores it verbatim as
    `vector_fp` (see test_lance_writer.py::
    test_vector_fp_uses_pre_quant_fp32_when_provided), and computes the
    reference score independently from that same true array -- never through
    int8 or dequant. The query is constructed to lie exactly in the PCA row
    space (see `_orthonormal_rows`) so the eps bound's unit-norm precondition
    holds, matching how a real embedding relates to a real (orthonormal-row)
    SVD basis.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        n = 20_000
        rng = np.random.default_rng(31)
        pca = _orthonormal_rows(rng, _D, _MODEL_DIM)

        true_fp32 = (
            rng.standard_normal((n, _D)) * rng.uniform(0.5, 3.0, size=_D)
        ).astype("float32")
        scale = np.max(np.abs(true_fp32), axis=0).astype("float32")
        corpus_i8 = np.clip(np.round(true_fp32 * 127.0 / scale), -127, 127).astype(np.int8)

        artifact_dir = tmp_path / "artifact"
        artifact_dir.mkdir()
        np.save(artifact_dir / lance_writer.PCA_FILE, pca)
        np.save(artifact_dir / lance_writer.SCALE_FILE, scale)
        np.save(artifact_dir / lance_writer.CORPUS_INT8_FILE, corpus_i8)
        np.save(artifact_dir / lance_writer.PRE_QUANT_FP32_FILE, true_fp32)
        chunk_start = rng.integers(1_700_000_000, 1_700_100_000, size=n).astype("int64")
        table = pa.table(
            {
                "run_uuid": [f"run-{i % 5}" for i in range(n)],
                "chunk_start_unix": chunk_start,
                "chunk_end_unix": chunk_start + 5,
                "segment_id": [f"seg-{i}" for i in range(n)],
                "vehicle": rng.choice(["veh_a", "veh_b", "veh_c"], size=n),
            }
        )
        pq.write_table(table, artifact_dir / lance_writer.METADATA_FILE)

        ds = lance_writer.build_dataset(artifact_dir, str(tmp_path / "out.lance"))

        query_pca_true = rng.standard_normal(_D).astype(np.float64)
        query_pca_true /= np.linalg.norm(query_pca_true)
        raw_query = (pca.astype(np.float64).T @ query_pca_true).astype(np.float32)
        assert abs(float(np.linalg.norm(raw_query)) - 1.0) < 1e-4, np.linalg.norm(raw_query)

        # build_table physically re-sorts rows by (chunk_start_unix, vehicle),
        # so `true_fp32`'s original (generation-order) row indices do NOT match
        # ThresholdHit.row_id, which addresses the WRITTEN (sorted) order. Read
        # vector_fp back from the built dataset instead: it equals true_fp32
        # exactly here (a real pca_projection_fp32.npy artifact was supplied;
        # see test_lance_writer.py::test_vector_fp_uses_pre_quant_fp32_when_provided),
        # just re-ordered consistently with row_id.
        written_vector_fp = ts._fixed_size_list_matrix(
            ds.to_table(columns=[lance_writer.VECTOR_FP_COLUMN]),
            lance_writer.VECTOR_FP_COLUMN,
            np.float64,
        )
        true_scores = written_vector_fp @ query_pca_true

        for percentile in (99.9, 99.0, 90.0, 50.0):
            tau = float(np.percentile(true_scores, percentile))
            expected_ids = set(np.nonzero(true_scores >= tau)[0].tolist())
            hits = ts.threshold_search(raw_query, tau, ds)
            got_ids = {h.row_id for h in hits}
            missing = expected_ids - got_ids
            # One-sided: strictly checks NO false negative against the true
            # score. Not asserting exact set equality here, since threshold_
            # search's internal query projection round-trips through float32
            # (_project_query) while this test's reference query_pca_true is
            # kept in float64 -- an unrelated, tiny (~1e-7) rounding
            # difference can move a handful of rows sitting exactly on a
            # percentile boundary across it, which is not the property this
            # test exists to check.
            assert not missing, (
                f"tau@{percentile}pct={tau}: dropped {len(missing)} row(s) "
                f"with TRUE pre-quantization score >= tau (a real false "
                f"negative against the claimed guarantee): {sorted(missing)[:10]}"
            )


def test_query_not_unit_norm_raises() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        ds = _build(tmp_path, n=500, seed=41)
        rng = np.random.default_rng(141)
        non_unit_query = rng.standard_normal(_MODEL_DIM).astype("float32") * 3.0

        import pytest

        with pytest.raises(ValueError, match="unit-norm"):
            ts.threshold_search(non_unit_query, tau=0.0, dataset=ds)


def test_scan_order_pinned_survives_many_fragments() -> None:
    """Regression guard for the to_table()/take() positional-addressing
    contract: screening derives row positions from a to_table() scan, then
    resolves exact scores via take() at those same positions. That is only
    correct if the scan returns rows in the dataset's canonical order --
    guaranteed by pinning scan_in_order=True (see
    threshold_search._load_resident_corpus's docstring). This test forces
    many small fragments (instead of the single post-compaction fragment
    every other test in this file uses) so a regression that dropped the
    scan_in_order pin would have fragments to actually reorder."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        n = 5_000
        artifact_dir = _write_synthetic_artifacts(tmp_path, n=n, seed=51)
        out_uri = str(tmp_path / "out.lance")
        table = lance_writer.build_table(artifact_dir)
        # Small max_rows_per_file, and skip build_dataset's own compact_files
        # call, so the dataset stays split across many fragments.
        lance.write_dataset(
            table,
            out_uri,
            mode="create",
            data_storage_version=lance_writer.DATA_STORAGE_VERSION,
            max_rows_per_file=200,
        )
        ds = lance.dataset(out_uri)
        assert len(ds.get_fragments()) > 1, "fixture bug: expected multiple fragments"

        query = _query(seed=151)
        oracle = _brute_force_oracle(ds, query, tau=_tau_candidates(ds, query)[1])
        # threshold_search itself requires PCA schema metadata + the format gate,
        # both of which build_table/write_dataset already produced; only
        # create_scalar_index / compact_files were skipped, which threshold_search
        # does not depend on.
        hits = ts.threshold_search(query, _tau_candidates(ds, query)[1], ds)
        _assert_matches_oracle(hits, oracle)


def test_threshold_corpus_decodes_embedding_i8_once() -> None:
    """ThresholdCorpus must decode the resident int8 matrix exactly once, at
    construction, and reuse it across queries -- not re-read embedding_i8 from
    the dataset on every .threshold_search() call (the design's "resident
    shard" architecture; see threshold_search.py's module docstring)."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        ds = _build(tmp_path, n=2_000, seed=61)
        query = _query(seed=161)
        tau = _tau_candidates(ds, query)[1]

        i8_to_table_calls = []
        orig_to_table = lance.LanceDataset.to_table

        def spy_to_table(self, *args, **kwargs):
            columns = kwargs.get("columns") if "columns" in kwargs else (
                args[0] if args else None
            )
            if columns and lance_writer.EMBEDDING_I8_COLUMN in columns:
                i8_to_table_calls.append(columns)
            return orig_to_table(self, *args, **kwargs)

        lance.LanceDataset.to_table = spy_to_table
        try:
            corpus = ts.ThresholdCorpus(ds)
            assert len(i8_to_table_calls) == 1, (
                "constructing ThresholdCorpus should decode embedding_i8 exactly once"
            )
            for _ in range(3):
                corpus.threshold_search(query, tau)
            assert len(i8_to_table_calls) == 1, (
                f"expected embedding_i8 decoded once total (residency), got "
                f"{len(i8_to_table_calls)} to_table() calls across construction + "
                f"3 queries"
            )
        finally:
            lance.LanceDataset.to_table = orig_to_table
