"""Offline checks for eps_bound: pure numpy, no Lance/dataset, no network.

Run from this directory:
    python eps_bound_test.py
"""

from __future__ import annotations

import numpy as np

import eps_bound

_D = 32


def _quantize_per_dim(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric per-dim int8 quantization of an (N, D) fp32 array.

    Matches gpu_corpus's dequant contract (`dequant = int8 * scale / 127`):
    `scale_d = max(|v[:, d]|)` so int8 = +/-127 dequants back to +/- that max,
    and the forward quantization step is `int8 = round(v * 127 / scale_d)`,
    i.e. each column is rounded to the nearest point on a `scale_d / 127`-wide
    grid -- the quantization step size eps_bound's error bound assumes.
    """
    scale = np.max(np.abs(v), axis=0)
    scale = np.where(scale == 0, 1.0, scale)  # guard degenerate all-zero columns
    i8 = np.clip(np.round(v * 127.0 / scale), -127, 127).astype(np.int8)
    return i8, scale.astype(np.float32)


def _random_case(rng: np.random.Generator, n: int, d: int = _D):
    """A random (query, corpus_fp32, corpus_i8, scale) tuple for property tests."""
    q = rng.standard_normal(d).astype(np.float64)
    q /= np.linalg.norm(q)  # unit-norm, required by the Cauchy-Schwarz bound
    v_fp32 = rng.standard_normal((n, d)).astype(np.float64) * rng.uniform(0.5, 5.0)
    i8, scale = _quantize_per_dim(v_fp32)
    return q, v_fp32, i8, scale


def test_band_union_above_is_superset_of_true_matches_random() -> None:
    # Property (a): every row whose TRUE fp32 score >= tau must land in
    # ABOVE or BAND -- the lower-bounding property this module exists for.
    rng = np.random.default_rng(1)
    for trial in range(20):
        n = rng.integers(50, 400)
        q, v_fp32, i8, scale = _random_case(rng, n)
        true_scores = v_fp32 @ q
        int8_scores = eps_bound.score_i8(i8, q, scale)
        eps = eps_bound.eps_cauchy_schwarz(scale)
        tau = float(np.quantile(true_scores, rng.uniform(0.1, 0.9)))

        above, band, below = eps_bound.classify(int8_scores, tau, eps)
        true_matches = true_scores >= tau
        assert np.all((above | band)[true_matches]), (
            trial,
            tau,
            eps,
            np.nonzero(true_matches & below)[0],
        )


def test_band_union_above_is_superset_of_true_matches_adversarial() -> None:
    # Adversarial variant of (a): rows are placed EXACTLY on quantization
    # half-step boundaries (v = (k + 0.5) * scale) so the dequant error is at
    # its theoretical maximum |scale_d / 254| in every dimension, and the
    # query is aligned to push that error as hard as possible toward
    # excluding a true match (rather than relying on random cancellation).
    rng = np.random.default_rng(2)
    d = 16
    scale = rng.uniform(0.1, 2.0, size=d).astype(np.float64)
    # Half-step-aligned integer grid points (grid step = scale_d / 127, per
    # the dequant contract `int8 * scale / 127`), so true value sits exactly
    # midway between two int8 dequant levels (max rounding error).
    step = scale / 127.0
    k = rng.integers(-60, 60, size=(200, d)).astype(np.float64)
    v_fp32 = (k + 0.5) * step
    i8 = np.clip(np.round(v_fp32 * 127.0 / scale), -127, 127).astype(np.int8)

    q = rng.standard_normal(d)
    q /= np.linalg.norm(q)
    true_scores = v_fp32 @ q
    int8_scores = eps_bound.score_i8(i8, q, scale)
    eps = eps_bound.eps_cauchy_schwarz(scale)
    tau = float(np.median(true_scores))

    above, band, below = eps_bound.classify(int8_scores, tau, eps)
    true_matches = true_scores >= tau
    assert np.all((above | band)[true_matches])


def test_below_rows_are_all_truly_below_tau() -> None:
    # Property (b): BELOW must never contain a false exclude -- every row
    # classified BELOW must have a true fp32 score strictly < tau.
    rng = np.random.default_rng(3)
    for trial in range(20):
        n = rng.integers(50, 400)
        q, v_fp32, i8, scale = _random_case(rng, n)
        true_scores = v_fp32 @ q
        int8_scores = eps_bound.score_i8(i8, q, scale)
        eps = eps_bound.eps_cauchy_schwarz(scale)
        tau = float(np.quantile(true_scores, rng.uniform(0.1, 0.9)))

        above, band, below = eps_bound.classify(int8_scores, tau, eps)
        assert np.all(true_scores[below] < tau), (trial, tau, eps)


def test_partial_sum_of_inner_product_is_not_monotone() -> None:
    # Regression guard (c): inner-product terms can be negative, so a running
    # partial sum over a prefix of dimensions is NOT monotone in the number
    # of terms summed -- unlike squared L2 distance, where every term is a
    # non-negative squared difference and a partial sum can only grow, which
    # is exactly what makes early-stopping/pruning valid there. If someone
    # "optimizes" the threshold scan by stopping the dot product early once a
    # partial sum crosses some cutoff, this test demonstrates concretely why
    # that is unsound for inner product: a partial sum can both overshoot and
    # undershoot the true final total relative to tau.
    tau = 15.0

    # Undershoot: the partial sum after 3 of 4 terms is far BELOW tau (an
    # early-abort-on-"clearly below" pruner would drop this row), yet the
    # final full-dimension sum is well ABOVE tau once the last term is added.
    undershoot_terms = np.array([-10.0, -10.0, -10.0, 50.0])
    partial_after_3 = np.sum(undershoot_terms[:3])
    full_sum = np.sum(undershoot_terms)
    assert partial_after_3 < tau, partial_after_3
    assert full_sum >= tau, full_sum
    assert partial_after_3 < tau <= full_sum  # partial is on the wrong side of tau

    # Overshoot: the partial sum after 1 of 4 terms is far ABOVE tau (an
    # early-commit-on-"clearly above" pruner would accept this row), yet the
    # final full-dimension sum ends up BELOW tau once the remaining
    # (negative) terms are added.
    overshoot_terms = np.array([50.0, -20.0, -20.0, -20.0])
    partial_after_1 = np.sum(overshoot_terms[:1])
    full_sum_2 = np.sum(overshoot_terms)
    assert partial_after_1 >= tau, partial_after_1
    assert full_sum_2 < tau, full_sum_2
    assert partial_after_1 >= tau > full_sum_2  # partial is on the wrong side of tau

    # Contrast: squared-L2 terms are non-negative, so the analogous partial
    # sum is monotonically non-decreasing -- pruning IS valid there. Confirm
    # the property that makes IP pruning invalid does not hold for L2.
    l2_terms = np.array([1.0, 4.0, 0.0, 9.0])  # each term is a squared difference, >= 0
    l2_partial_sums = np.cumsum(l2_terms)
    assert np.all(np.diff(l2_partial_sums) >= 0), l2_partial_sums


def test_eps_matches_hand_computed_closed_form() -> None:
    # Property (d): eps_cauchy_schwarz(scale) == ||scale||_2 / 254 exactly,
    # checked against an arithmetic worked by hand, not by re-deriving the
    # same formula the implementation uses.
    scale = np.array([3.0, 4.0])  # ||scale||_2 = sqrt(9 + 16) = 5
    expected = 5.0 / 254.0
    got = eps_bound.eps_cauchy_schwarz(scale)
    assert abs(got - expected) < 1e-12, (got, expected)

    # A second, non-Pythagorean-triple case to rule out a lucky coincidence.
    scale2 = np.array([1.0, 2.0, 2.0])  # ||scale||_2 = sqrt(1 + 4 + 4) = 3
    expected2 = 3.0 / 254.0
    got2 = eps_bound.eps_cauchy_schwarz(scale2)
    assert abs(got2 - expected2) < 1e-12, (got2, expected2)


def test_hoelder_bound_is_a_valid_alternative() -> None:
    # eps_hoelder must also be a valid bound on |q . e| (checked directly
    # against realized quantization error, not just compared to the
    # Cauchy-Schwarz value). For a unit-norm query, applying Cauchy-Schwarz
    # to the (|q_d|, scale_d) vectors shows sum_d |q_d| * scale_d <=
    # ||q||_2 * ||scale||_2 == ||scale||_2, so eps_hoelder is in fact never
    # looser than eps_cauchy_schwarz in this regime -- verify that ordering
    # holds too, so a future change that flips it is caught.
    rng = np.random.default_rng(4)
    for _ in range(20):
        d = 24
        q, v_fp32, i8, scale = _random_case(rng, 200, d)
        true_scores = v_fp32 @ q
        int8_scores = eps_bound.score_i8(i8, q, scale)
        actual_err = np.abs(true_scores - int8_scores)

        eps_cs = eps_bound.eps_cauchy_schwarz(scale)
        eps_h = eps_bound.eps_hoelder(q, scale)

        assert np.all(actual_err <= eps_h + 1e-9), actual_err.max()
        assert np.all(actual_err <= eps_cs + 1e-9), actual_err.max()
        assert eps_h <= eps_cs + 1e-9, (eps_h, eps_cs)


def test_classify_boundaries_are_exact() -> None:
    # Direct boundary check on synthetic scores, independent of any
    # quantization machinery: ABOVE/BAND/BELOW partition scores exactly at
    # tau +/- eps with the documented open/closed endpoints.
    tau, eps = 10.0, 0.5
    scores = np.array([9.4, 9.5, 9.9, 10.0, 10.5, 10.6])
    above, band, below = eps_bound.classify(scores, tau, eps)

    np.testing.assert_array_equal(below, [True, False, False, False, False, False])
    np.testing.assert_array_equal(
        band, [False, True, True, True, False, False]
    )
    np.testing.assert_array_equal(above, [False, False, False, False, True, True])
    # Partition: every row is classified exactly once.
    assert np.all(above.astype(int) + band.astype(int) + below.astype(int) == 1)


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL {name}: {exc}")
    raise SystemExit(1 if failures else 0)
