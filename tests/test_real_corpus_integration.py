"""Self-consistency check against a real ~50,000-row NLS embeddings sample.

Runs by default as part of `python -m pytest tests/`.  The fixture
(`tests/fixtures/nls_real_corpus_sample.lance/`) is a Lance 2.1 dataset
committed directly to this repo -- a 50,000-row random sample (fixed seed 42,
numpy.random.default_rng) drawn from the full ~902,827-row real NLS corpus.
It carries the same column layout and PCA / quant-scale metadata as the full
dataset, so `threshold_search` exercises the identical code path.

Scope: this fixture has no `pca_projection_fp32.npy`, so its `vector_fp` is
the int8 dequantized back to fp32 (see `lance_writer.py`'s module docstring),
NOT an independent pre-quantization reference. This test therefore proves
threshold_search's screen/take/re-rank plumbing is self-consistent on a real
score distribution at realistic scale -- it does NOT prove zero false
negatives against the true pre-quantization score (an eps bound too small for
real int8 quantization error would still pass this test). For that property,
see `test_threshold_search.py::test_zero_false_negatives_against_true_pre_quantization_score`,
which supplies a genuine independent fp32 reference.

The sample is small enough (~65 MB) to commit and fast enough (~sub-second per
`threshold_search` call) to run in the default suite without a CI budget concern.
No external data, no env var, no credentials required.

Run:
    python -m pytest tests/test_real_corpus_integration.py -v -s
    # or simply:
    python -m pytest tests/ -v
"""

from __future__ import annotations

import time
from pathlib import Path

import lance
import lance_writer
import numpy as np
import pytest
import threshold_search as ts

# Fixture path: committed in-repo alongside this test file
_FIXTURE_DIR = Path(__file__).parent / "fixtures" / "nls_real_corpus_sample.lance"
_EXPECTED_ROW_COUNT = 50_000


def _query(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    q = rng.standard_normal(768).astype("float32")
    return q / np.linalg.norm(q)


def _brute_force_oracle_scores(ds: lance.LanceDataset, query: np.ndarray) -> np.ndarray:
    """vector_fp-based reference score for every row (reads the whole
    vector_fp column once, deliberately -- this is the slow, independently
    re-derived path, not the fast path under test). On this fixture vector_fp
    is dequant(int8) (see module docstring), so this reference is NOT
    independent of the int8 corpus threshold_search screens."""
    pca, _scale = lance_writer.read_pca_metadata(ds)
    query_pca = ts._project_query(query, pca).astype(np.float64)
    table = ds.to_table(columns=[lance_writer.VECTOR_FP_COLUMN])
    fp = ts._fixed_size_list_matrix(table, lance_writer.VECTOR_FP_COLUMN, np.float64)
    return fp @ query_pca


@pytest.fixture(scope="module")
def real_dataset() -> lance.LanceDataset:
    assert _FIXTURE_DIR.exists(), (
        f"In-repo fixture not found at {_FIXTURE_DIR} -- "
        "check that the repository was checked out completely"
    )
    ds = lance.dataset(str(_FIXTURE_DIR))
    assert lance_writer.is_v21_dataset(ds), (
        f"{_FIXTURE_DIR} is not a Lance 2.1 dataset -- fixture may be corrupt"
    )
    assert ds.count_rows() == _EXPECTED_ROW_COUNT, (
        f"{_FIXTURE_DIR} has {ds.count_rows()} rows, expected {_EXPECTED_ROW_COUNT}"
    )
    return ds


def test_zero_false_negatives_real_corpus(real_dataset: lance.LanceDataset) -> None:
    """threshold_search finds every row with re-rank score >= tau across
    several tau values on the real 50,000-row corpus sample -- a
    self-consistency check on a real score distribution (see module
    docstring for why this is not an independent-oracle test). Band
    selectivity and timing are reported informally; they are not gated (real
    data, unknown distribution)."""
    query = _query(seed=42)
    scores = _brute_force_oracle_scores(real_dataset, query)

    for percentile in (99.9, 99.0, 90.0, 50.0):
        tau = float(np.percentile(scores, percentile))
        oracle_idx = set(np.nonzero(scores >= tau)[0].tolist())

        t0 = time.time()
        hits = ts.threshold_search(query, tau, real_dataset)
        elapsed = time.time() - t0

        hit_idx = {h.row_id for h in hits}
        missing = oracle_idx - hit_idx
        assert not missing, (
            f"tau@{percentile}pct={tau}: threshold_search dropped "
            f"{len(missing)} row(s) with exact score >= tau (false negatives): "
            f"{sorted(missing)[:10]}"
        )
        assert hit_idx == oracle_idx, (
            f"tau@{percentile}pct={tau}: extra rows returned beyond the oracle "
            f"set (should not happen once false negatives are ruled out): "
            f"{sorted(hit_idx - oracle_idx)[:10]}"
        )

        print(
            f"\n[real-corpus] tau@{percentile}pct={tau:.6f}: "
            f"{len(hits)} hits ({len(hits) / _EXPECTED_ROW_COUNT:.4%} of corpus), "
            f"threshold_search={elapsed:.3f}s"
        )


def test_zero_false_negatives_real_corpus_second_query(
    real_dataset: lance.LanceDataset,
) -> None:
    """A second, independent query at a tight tau (small match set, the
    realistic threshold-workload regime) -- not just one lucky draw."""
    query = _query(seed=4242)
    scores = _brute_force_oracle_scores(real_dataset, query)
    tau = float(np.percentile(scores, 99.5))
    oracle_idx = set(np.nonzero(scores >= tau)[0].tolist())

    t0 = time.time()
    hits = ts.threshold_search(query, tau, real_dataset)
    elapsed = time.time() - t0

    hit_idx = {h.row_id for h in hits}
    assert hit_idx == oracle_idx, (
        f"row-id set mismatch at tau@99.5pct={tau}: symmetric diff "
        f"{hit_idx ^ oracle_idx}"
    )
    print(
        f"\n[real-corpus] second query tau@99.5pct={tau:.6f}: "
        f"{len(hits)} hits ({len(hits) / _EXPECTED_ROW_COUNT:.4%} of corpus), "
        f"threshold_search={elapsed:.3f}s"
    )
