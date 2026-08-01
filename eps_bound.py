"""Zero-false-negative eps bound + ABOVE/BAND/BELOW classifier for int8 screening.

The threshold-retrieval workload needs ALL rows with true fp32 cosine score
>= tau, not an approximate top-k. Screening happens on the resident int8 PCA
column (`gpu_corpus._score_i8`'s input layout), which is a lossy dequantized
approximation of the true fp32 score. This module computes a hard error bound
`eps` on that approximation and uses it to make `tau - eps` a *provable*
superset threshold: any row with true score >= tau is mathematically
guaranteed to have int8-projected score >= tau - eps, so screening at
`tau - eps` can only ever produce false positives (extra rows to re-rank),
never false negatives (missing rows). The corresponding false positives are
resolved by an exact fp32 re-rank of the BAND, done elsewhere against the
full-precision column.

Bound derivation: per-dim symmetric int8 quantization stores
`dequant_d = int8_d * scale_d / 127`, rounding each true PCA-space value to
the nearest representable point on a `scale_d / 127`-wide grid, so the
per-dim dequantization error is bounded by half that step:
`|e_d| <= scale_d / (2 * 127) = scale_d / 254`. For a unit-norm query `q`,
Cauchy-Schwarz gives `|q . e| <= ||q||_2 * ||e||_2 <= ||scale||_2 / 254` --
a bound that does not depend on q's direction, only its norm. A
per-dimension (Hoelder-style) bound that does not require q to be unit-norm
is also provided: `|q . e| <= sum_d |q_d| * scale_d / 254`. For a unit-norm
query, applying Cauchy-Schwarz to `(|q_d|)` and `(scale_d)` shows this
Hoelder bound is never looser than the one above -- `sum_d |q_d| * scale_d
<= ||q||_2 * ||scale||_2 == ||scale||_2` -- so it is at least as tight,
just query-dependent rather than reusable across queries.

This module is pure numpy: no Lance/dataset dependency, so it is testable in
isolation from the corpus storage layer.

Scope of the guarantee: the math above bounds |q . e| where `e` is the
per-row int8 quantization error against WHATEVER fp32 value the caller
re-ranks against. The guarantee is against the true pre-quantization score
only if that fp32 value genuinely is the pre-quantization projection.
`lance_writer.py`'s current artifact contract does not always have one --
see its module docstring -- in which case its `vector_fp` falls back to
`dequant(int8)`, and re-ranking against it only re-derives the int8 screening
score in higher precision; the zero-false-negative property this module
proves still holds (against `vector_fp`), but it is a materially weaker
guarantee than "against the true fp32-256 embedding" in that configuration.
"""

from __future__ import annotations

import numpy as np


def eps_cauchy_schwarz(scale: np.ndarray) -> float:
    """Query-independent error bound `||scale||_2 / 254` for a unit-norm query.

    Valid for ANY unit-norm query vector, in any orientation -- it uses only
    ||q||_2 == 1, not q's direction, so it can be computed once per corpus
    (from `scale` alone) and reused across every query.
    """
    scale = np.asarray(scale, dtype=np.float64)
    return float(np.linalg.norm(scale) / 254.0)


def eps_hoelder(query: np.ndarray, scale: np.ndarray) -> float:
    """Query-dependent error bound `sum_d |q_d| * scale_d / 254`.

    Bounds each dot-product term by `|q_d| * max|e_d|` and sums via the
    triangle inequality. Unlike `eps_cauchy_schwarz`, it does not require q
    to be unit-norm -- provided as a fallback for callers that only trust the
    per-dim query magnitudes. Simpler to reason about per-dimension, but it
    recomputes per query (`eps_cauchy_schwarz` depends only on `scale`, so it
    can be computed once per corpus and reused across every query).
    """
    query = np.asarray(query, dtype=np.float64)
    scale = np.asarray(scale, dtype=np.float64)
    return float(np.sum(np.abs(query) * scale) / 254.0)


def score_i8(corpus_i8: np.ndarray, query_pca: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Reference (pure numpy) int8 x fp32 dot product.

    Mirrors `gpu_corpus._score_i8` / `GpuCorpus.gpu_score`'s dequant-fold
    convention: the per-dim dequant scale is folded into the query
    (`w = query_pca * scale / 127`) so the corpus stays int8, then each row's
    int8 . w is the dequantized-corpus-vector . query score. Provided so
    tests can produce int8-space scores without importing the numba-jitted
    kernel in `gpu_corpus.py`.
    """
    corpus_i8 = np.asarray(corpus_i8, dtype=np.float64)
    query_pca = np.asarray(query_pca, dtype=np.float64)
    scale = np.asarray(scale, dtype=np.float64)
    w = query_pca * (scale / 127.0)
    return corpus_i8 @ w


def classify(
    scores: np.ndarray, tau: float, eps: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split int8-projected `scores` into (above, band, below) boolean masks.

    ABOVE (score >= tau + eps): guaranteed true score >= tau, no re-rank needed.
    BAND  ([tau - eps, tau + eps)): true score could be on either side of tau
        (the eps window absorbs quantization error both ways) -- mandatory
        exact fp32 re-rank against `vector_fp` decides membership.
    BELOW (score < tau - eps): provably true score < tau, safe to exclude.

    The BELOW exclusion is only sound because `eps` is a genuine upper bound
    on |true_score - int8_score| for every row -- see `eps_cauchy_schwarz`.
    Deliberately does NOT attempt any partial-sum pruning of the underlying
    dot product: inner-product terms can be negative, so a partial sum over a
    prefix of dimensions is not monotone in the number of dimensions summed
    (unlike, e.g., squared L2 distance, whose terms are all non-negative).
    Stopping early on a partial sum can both overshoot and undershoot the
    final full-dimension score, so it cannot be used to prove membership or
    exclusion -- see `tests/test_eps_bound.py`'s non-monotonicity regression guard.
    """
    scores = np.asarray(scores, dtype=np.float64)
    above = scores >= tau + eps
    below = scores < tau - eps
    band = ~above & ~below
    return above, band, below
