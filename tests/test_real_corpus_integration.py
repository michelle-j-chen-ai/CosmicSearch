"""Zero-false-negative check against the real ~902,827-row NLS embeddings corpus.

Opt-in / skipped by default: `python -m pytest tests/` does NOT run this file
(it is skipped, not collected-and-failed) unless `NLS_REAL_CORPUS_LANCE_DIR`
is set. This test loads an already-converted, already-built Lance 2.1 dataset
directly -- it does NOT download raw embeddings and does NOT fit SVD at test
time. Those one-time steps live in `tests/fixtures/build_1m_int8_fixture.py`
(see that module's docstring for the regeneration procedure); this file only
ever opens the pre-built dataset (`lance.dataset(...)`) and runs the same
zero-false-negative oracle checks `threshold_search` is proven against
everywhere else in this suite.

Fixture location: point `NLS_REAL_CORPUS_LANCE_DIR` at a pre-built Lance 2.1
dataset directory -- the output of running `tests/fixtures/build_1m_int8_fixture.py`
once against the real embeddings, then once through `lance_writer.build_dataset`
(e.g. `~/nls_fixtures/nls_real_corpus.lance`). That directory is not committed
to this repo: at ~1.2GB it is analogous to the raw source data living in OCI
object storage rather than repo-tracked test data, so each environment that
wants to run this opt-in test regenerates or copies it once and points the
env var at it.

Run:
    NLS_REAL_CORPUS_LANCE_DIR=/path/to/nls_real_corpus.lance \\
        python -m pytest tests/test_real_corpus_integration.py -v -s

Expected runtime: a few seconds total -- dataset open plus a handful of
`threshold_search` calls, each sub-second at this row count. No download, no
SVD fit.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import lance
import lance_writer
import numpy as np
import pytest
import threshold_search as ts

_LANCE_DIR_ENV = "NLS_REAL_CORPUS_LANCE_DIR"
_EXPECTED_ROW_COUNT = 902_827

pytestmark = pytest.mark.integration

_lance_dir_str = os.environ.get(_LANCE_DIR_ENV)
_lance_dir = Path(_lance_dir_str) if _lance_dir_str else None
_skip_reason = (
    f"set {_LANCE_DIR_ENV} to a pre-built Lance 2.1 dataset directory (see "
    "this file's module docstring, and tests/fixtures/build_1m_int8_fixture.py, "
    "for how to regenerate it) to run this opt-in real-corpus integration test"
)


def _query(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    q = rng.standard_normal(768).astype("float32")
    return q / np.linalg.norm(q)


def _brute_force_oracle_scores(ds: lance.LanceDataset, query: np.ndarray) -> np.ndarray:
    """Exact fp32 cosine score for every row (reads the whole vector_fp column
    once, deliberately -- this is the independent reference, not the fast
    path under test)."""
    pca, _scale = lance_writer.read_pca_metadata(ds)
    query_pca = ts._project_query(query, pca).astype(np.float64)
    table = ds.to_table(columns=[lance_writer.VECTOR_FP_COLUMN])
    fp = ts._fixed_size_list_matrix(table, lance_writer.VECTOR_FP_COLUMN, np.float64)
    return fp @ query_pca


@pytest.fixture(scope="module")
def real_dataset() -> lance.LanceDataset:
    if _lance_dir is None:
        pytest.skip(_skip_reason)
    if not _lance_dir.exists():
        pytest.skip(f"{_LANCE_DIR_ENV}={_lance_dir} does not exist")

    t0 = time.time()
    ds = lance.dataset(str(_lance_dir))
    print(f"\n[real-corpus] opened pre-built dataset: {time.time() - t0:.3f}s")

    if not lance_writer.is_v21_dataset(ds):
        pytest.skip(
            f"{_LANCE_DIR_ENV}={_lance_dir} is not a Lance 2.1 exact-threshold "
            "dataset (missing data_storage_version=2.1 or the embedding_i8/"
            "vector_fp columns) -- regenerate it via "
            "tests/fixtures/build_1m_int8_fixture.py + lance_writer.build_dataset"
        )
    assert ds.count_rows() == _EXPECTED_ROW_COUNT, (
        f"{_LANCE_DIR_ENV}={_lance_dir} has {ds.count_rows()} rows, expected "
        f"{_EXPECTED_ROW_COUNT} -- wrong or stale fixture"
    )
    return ds


def test_zero_false_negatives_real_corpus(real_dataset: lance.LanceDataset) -> None:
    """threshold_search finds every row with exact score >= tau, across
    several tau values, on the real ~902,827-row corpus -- an oracle
    property test, informational band-selectivity/timing reporting, not a
    strict pass/fail gate on those numbers (real data, unknown distribution)."""
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

        band_fraction = len(hit_idx) / _EXPECTED_ROW_COUNT
        print(
            f"[real-corpus] tau@{percentile}pct={tau:.6f}: "
            f"{len(hits)} hits ({band_fraction:.4%} of corpus), "
            f"threshold_search={elapsed:.3f}s"
        )


def test_zero_false_negatives_real_corpus_second_query(
    real_dataset: lance.LanceDataset,
) -> None:
    """A second, independent query -- not just one lucky draw -- at a tight
    tau (small match set, the realistic threshold-workload regime)."""
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
        f"[real-corpus] second query tau@99.5pct={tau:.6f}: "
        f"{len(hits)} hits ({len(hits) / _EXPECTED_ROW_COUNT:.4%} of corpus), "
        f"threshold_search={elapsed:.3f}s"
    )
