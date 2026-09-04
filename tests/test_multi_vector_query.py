"""MaxSim: a query may be several directions, and a row keeps its best match.

Mean-pooling a 30s source averages the few seconds that matter together with
the ordinary driving around them, and the resulting direction can match
neither. `pooling="individual"` keeps the source's chunks apart so a corpus row
scores as its best match against ANY of them -- "did this occur at all" rather
than "was this the average of the clip".

The property that matters is not that multi-vector scoring runs, but that it
equals the elementwise max of scoring each direction alone. These tests pin
that against the single-vector path they must agree with.
"""

from __future__ import annotations

import numpy as np
import pytest

import full_corpus


def _unit(rng, n, dim):
    v = rng.standard_normal((n, dim)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


# ----------------------------------------------------------- as_directions


def test_a_single_vector_becomes_one_direction():
    q = np.array([3.0, 4.0], dtype=np.float32)
    out = full_corpus.as_directions(q)
    assert out.shape == (1, 2)
    assert np.isclose(np.linalg.norm(out[0]), 1.0)


def test_every_direction_is_normalized_independently():
    q = np.array([[3.0, 4.0], [0.0, 2.0]], dtype=np.float32)
    out = full_corpus.as_directions(q)
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0)


def test_a_zero_direction_is_rejected_rather_than_scored():
    """A zero row has no direction; normalizing it silently yields NaN, which
    would propagate into every score as a comparison that is always False."""
    q = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    with pytest.raises(ValueError, match="all zeros"):
        full_corpus.as_directions(q)


def test_the_direction_count_is_capped():
    """Each direction costs a full corpus sweep, so an unbounded set turns one
    query into an unbounded scan."""
    q = np.ones((full_corpus.MAX_QUERY_DIRECTIONS + 1, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="at most"):
        full_corpus.as_directions(q)


def test_an_empty_query_is_rejected():
    with pytest.raises(ValueError):
        full_corpus.as_directions(np.empty((0, 8), dtype=np.float32))


# ------------------------------------------------------------- the max rule


def test_multi_direction_scoring_is_the_max_of_the_directions_alone():
    """The whole contract, stated against the arithmetic the scorers perform:
    scoring N directions at once equals scoring each and taking the max."""
    rng = np.random.default_rng(0)
    rows = _unit(rng, 200, 32)
    dirs = full_corpus.as_directions(_unit(rng, 5, 32))

    together = (rows @ dirs.T).max(axis=1)
    apart = np.max([rows @ d for d in dirs], axis=0)

    assert np.allclose(together, apart)


def test_one_direction_scores_exactly_as_the_single_vector_path_did():
    """N=1 must be a no-op: every tag written before this existed is stored as
    a flat vector and has to keep scoring identically."""
    rng = np.random.default_rng(1)
    rows = _unit(rng, 128, 16)
    q = _unit(rng, 1, 16)[0]

    single = rows @ q
    through_directions = (rows @ full_corpus.as_directions(q).T).max(axis=1)

    assert np.allclose(single, through_directions)


def test_max_never_scores_below_the_mean_pooled_direction_on_its_own_chunks():
    """Why `individual` exists. A source whose chunks point in different
    directions pools to something closer to neither, so each chunk matches its
    own best more strongly than it matches the average."""
    chunks = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    pooled = full_corpus.as_directions(chunks.mean(axis=0))
    dirs = full_corpus.as_directions(chunks)

    pooled_scores = (chunks @ pooled.T).max(axis=1)
    maxsim_scores = (chunks @ dirs.T).max(axis=1)

    assert np.all(maxsim_scores >= pooled_scores)
    assert np.all(maxsim_scores > pooled_scores + 0.2)


# ------------------------------------------------------- the stored form


def test_a_single_direction_is_stored_flat_and_reads_back_unchanged():
    """Storage is shape-preserving in both directions: existing flat tags stay
    flat, so nothing has to migrate and nothing has to branch on shape."""
    import api_v1

    v = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    stored = api_v1._vector_json(v)
    assert stored == [pytest.approx(x) for x in [0.1, 0.2, 0.3]]
    assert np.asarray(stored).ndim == 1


def test_several_directions_are_stored_nested():
    import api_v1

    v = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    stored = api_v1._vector_json(v)
    assert np.asarray(stored).shape == (2, 2)


# ------------------------------------------- through the real scoring kernel


def _corpus_with(rows_i8: np.ndarray):
    """A FullCorpus over `rows_i8`, identity PCA and unit scales, so the
    screening score is a plain dot product and the max rule is checkable."""
    import pyarrow as pa

    n, dim = rows_i8.shape
    meta = {
        "vehicle": np.zeros(n, dtype=np.int32), "vehicle_uniques": ["v1"],
        "run_uuid": np.zeros(n, dtype=np.int32), "run_uuid_uniques": ["run-a"],
        "dt": np.zeros(n, dtype=np.int32), "dt_uniques": ["2026-01-01"],
        "chunk_start_unix": np.arange(n, dtype=np.int64),
        "chunk_end_unix": np.arange(n, dtype=np.int64) + 8,
        "dx_internal_id": np.full(n, -1, dtype=np.int64),
        "segment_id": pa.array([f"s{i}" for i in range(n)], pa.large_string()),
    }
    return full_corpus.FullCorpus(
        rows_i8, np.eye(dim, dtype=np.float32), np.ones(dim, dtype=np.float32), meta
    )


def test_corpus_score_of_many_directions_is_the_max_of_scoring_each():
    """The end-to-end property, through the numba kernel the service runs:
    one call with N directions == N calls reduced by max."""
    rng = np.random.default_rng(7)
    rows = rng.integers(-127, 127, size=(64, 8), dtype=np.int8)
    corpus = _corpus_with(rows)
    dirs = _unit(rng, 4, 8)

    together, err_together = corpus.score(dirs)
    apart = np.max([corpus.score(d)[0] for d in dirs], axis=0)
    worst_err = max(corpus.score(d)[1] for d in dirs)

    assert np.allclose(together, apart, atol=1e-5)
    # The winning direction differs by row, so only the worst bound holds
    # everywhere -- a tighter one would claim accuracy the answer lacks.
    assert err_together == pytest.approx(worst_err)


def test_corpus_score_is_unchanged_for_a_single_direction():
    """Every existing tag goes through the new code path; it must be a no-op."""
    rng = np.random.default_rng(8)
    rows = rng.integers(-127, 127, size=(32, 8), dtype=np.int8)
    corpus = _corpus_with(rows)
    q = _unit(rng, 1, 8)[0]

    flat, err_flat = corpus.score(q)
    nested, err_nested = corpus.score(q.reshape(1, -1))

    assert np.array_equal(flat, nested)
    assert err_flat == err_nested
