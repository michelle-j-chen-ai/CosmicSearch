"""End-to-end checks for search_engine's corpus dispatch.

`load_threshold_corpus` must open an exact-threshold dataset and search it;
`load_corpus` must refuse that dataset with an actionable error (it has no
resident matrix for rank_top_k/score_corpus) while still handling a legacy
`.lance` corpus exactly as before.

Run from the repo root:
    python -m pytest tests/test_load_corpus_threshold.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import conftest
import lance
import lance_writer
import numpy as np
import pyarrow as pa
import pytest
import search_engine
import threshold_search as ts

import local_cache


@pytest.fixture
def local_corpus(monkeypatch):
    """Point `ensure_corpus_local` at a local dir, bypassing the S3 download."""

    def _use(local_dir: Path) -> None:
        monkeypatch.setattr(
            local_cache, "ensure_corpus_local", lambda embeddings_uri, client: local_dir
        )

    return _use


def test_load_threshold_corpus_opens_and_searches_the_dataset(local_corpus) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        ds = conftest.build_corpus(tmp_path, n=5_000, seed=0)
        local_corpus(tmp_path / "out.lance")

        corpus = search_engine.load_threshold_corpus("s3://bucket/corpus.lance")
        assert isinstance(corpus, ts.ThresholdCorpus)

        query = conftest.unit_query()
        scores = conftest.exact_scores(ds, query)
        tau = float(np.percentile(scores, 90.0))

        hits = corpus.threshold_search(query, tau)
        assert {h.row_id for h in hits} == set(np.nonzero(scores >= tau)[0].tolist())
        for h in hits:
            assert h.score >= tau
            assert h.segment_id, "ThresholdHit should carry resolvable metadata"


def test_load_corpus_refuses_an_exact_threshold_dataset(local_corpus) -> None:
    # Returning one would hand web_server.py/app.py an object that breaks on
    # their first .matrix / .time_span() / score_corpus() call.
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        conftest.build_corpus(tmp_path, n=500, seed=1)
        local_corpus(tmp_path / "out.lance")

        with pytest.raises(ValueError, match="load_threshold_corpus"):
            search_engine.load_corpus("s3://bucket/corpus.lance", "float16")


def test_load_corpus_still_handles_a_legacy_lance_corpus(local_corpus) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        n = 20
        rng = np.random.default_rng(2)
        out_dir = Path(tmp) / "legacy.lance"
        lance.write_dataset(
            pa.table(
                {
                    "chunk_id": [f"run-0#t{1_700_000_000 + i}" for i in range(n)],
                    "run_uuid": ["run-0"] * n,
                    "chunk_start_unix": np.arange(
                        1_700_000_000, 1_700_000_000 + n, dtype="int64"
                    ),
                    "source_media_uri": [f"s3://bucket/{i}.mp4" for i in range(n)],
                    "segment_id": [f"seg-{i}" for i in range(n)],
                    "vector": pa.FixedSizeListArray.from_arrays(
                        pa.array(
                            rng.standard_normal(n * conftest.MODEL_DIM).astype("float32")
                        ),
                        conftest.MODEL_DIM,
                    ),
                }
            ),
            str(out_dir),
            mode="create",
        )
        assert not lance_writer.is_exact_threshold_dataset(lance.dataset(str(out_dir))), (
            "fixture bug: legacy dataset should not look like an exact-threshold dataset"
        )

        local_corpus(out_dir)
        corpus = search_engine.load_corpus("s3://bucket/legacy.lance", "float16")
        assert isinstance(corpus, search_engine.Corpus)
        assert corpus.matrix.shape == (n, conftest.MODEL_DIM)
