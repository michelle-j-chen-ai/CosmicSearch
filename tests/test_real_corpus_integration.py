"""Zero-false-negative check against the real ~902,827-row NLS embeddings corpus.

Opt-in / skipped by default: `python -m pytest tests/` does NOT run this file
(it is skipped, not collected-and-failed) unless `NLS_INTEGRATION_TEST_DATA_DIR`
is set. It needs a 2.77GB download and OCI object-storage credentials that are
not available in a normal CI/dev checkout, and this repo's `tests/` suite must
stay hermetic and fast.

Setup (once):
    mkdir -p /path/to/nls_real_data
    aws --profile oci.phx --region us-phoenix-1 \\
        --endpoint-url https://idskhu5vqvtl.compat.objectstorage.us-phoenix-1.oraclecloud.com \\
        s3 cp s3://neuron-prod-data-intelligence-exploratory/sibogeng/nls_search/embeddings/v3_lr_5e5-ckpt-6549_npy/embeddings.npy /path/to/nls_real_data/
    aws --profile oci.phx --region us-phoenix-1 \\
        --endpoint-url https://idskhu5vqvtl.compat.objectstorage.us-phoenix-1.oraclecloud.com \\
        s3 cp s3://neuron-prod-data-intelligence-exploratory/sibogeng/nls_search/embeddings/v3_lr_5e5-ckpt-6549_npy/metadata.parquet /path/to/nls_real_data/

`aws configure list --profile oci.phx` must show valid credentials (OCI's
S3-compatible interop endpoint, not plain AWS S3 -- see
`tests/fixtures/build_1m_int8_fixture.py`'s module docstring). Also needs
`scikit-learn` installed (only for this opt-in test; not a default dependency).

Run:
    NLS_INTEGRATION_TEST_DATA_DIR=/path/to/nls_real_data \\
        python -m pytest tests/test_real_corpus_integration.py -v -s

Expected runtime: ~30-60s (SVD fit ~25-30s on a modern multi-core CPU, Lance
dataset build ~5s, threshold_search itself sub-second per tau value).
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

from tests.fixtures.build_1m_int8_fixture import build_fixture

_DATA_DIR_ENV = "NLS_INTEGRATION_TEST_DATA_DIR"
_EXPECTED_ROW_COUNT = 902_827

pytestmark = pytest.mark.integration

_data_dir_str = os.environ.get(_DATA_DIR_ENV)
_data_dir = Path(_data_dir_str) if _data_dir_str else None
_skip_reason = (
    f"set {_DATA_DIR_ENV} to a directory containing embeddings.npy + "
    "metadata.parquet (see this file's module docstring for the download "
    "command) to run this opt-in real-corpus integration test"
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
def real_dataset(tmp_path_factory: pytest.TempPathFactory) -> lance.LanceDataset:
    if _data_dir is None:
        pytest.skip(_skip_reason)
    embeddings_npy = _data_dir / "embeddings.npy"
    metadata_parquet = _data_dir / "metadata.parquet"
    if not embeddings_npy.exists() or not metadata_parquet.exists():
        pytest.skip(
            f"{_DATA_DIR_ENV}={_data_dir} is missing embeddings.npy and/or "
            "metadata.parquet"
        )

    work_dir = tmp_path_factory.mktemp("nls_real_corpus")
    artifact_dir = work_dir / "artifact"
    t0 = time.time()
    build_fixture(embeddings_npy, metadata_parquet, artifact_dir)
    print(f"\n[real-corpus] fixture-gen (SVD + int8 quant): {time.time() - t0:.1f}s")

    t0 = time.time()
    ds = lance_writer.build_dataset(artifact_dir, str(work_dir / "corpus.lance"))
    print(f"[real-corpus] lance_writer.build_dataset: {time.time() - t0:.1f}s")

    assert ds.count_rows() == _EXPECTED_ROW_COUNT
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
