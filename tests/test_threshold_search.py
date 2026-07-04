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
