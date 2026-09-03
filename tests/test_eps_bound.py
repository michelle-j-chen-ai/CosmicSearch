"""Offline checks for eps_bound: pure numpy, no Lance/dataset, no network.

Run from the repo root:
    python -m pytest tests/test_eps_bound.py
"""

from __future__ import annotations

import eps_bound
import numpy as np
import pytest

_D = 32


def _case(rng: np.random.Generator, n: int, d: int = _D):
    """A (query, corpus_fp32, corpus_i8, scale) tuple quantized the production way.

    `scale_d = max|v[:, d]|` and `int8 = round(v * 127 / scale_d)`, matching
    the `dequant = int8 * scale / 127` contract -- each column is
    rounded onto a `scale_d / 127`-wide grid, the step eps_bound assumes.
    """
    q = rng.standard_normal(d)
    q /= np.linalg.norm(q)  # unit-norm, required by the Cauchy-Schwarz bound
    v_fp32 = rng.standard_normal((n, d)) * rng.uniform(0.5, 5.0)
    scale = np.max(np.abs(v_fp32), axis=0)
    i8 = np.clip(np.round(v_fp32 * 127.0 / scale), -127, 127).astype(np.int8)
    return q, v_fp32, i8, scale.astype(np.float32)


@pytest.mark.parametrize("adversarial", [False, True])
def test_classification_is_sound_in_both_directions(adversarial: bool) -> None:
    """The two properties the screen rests on, over many random corpora:

    ABOVE|BAND is a superset of the true matches (no false negative), and
    BELOW contains no row whose true score reaches tau (no false exclude).

    The adversarial variant puts every value exactly on a quantization
    half-step, so dequant error sits at its theoretical maximum
    `scale_d / 254` in every dimension instead of randomly cancelling.
    """
    rng = np.random.default_rng(1 if adversarial else 2)
    for _ in range(20):
        if adversarial:
            d = 16
            scale = rng.uniform(0.1, 2.0, size=d)
            step = scale / 127.0
            v_fp32 = (rng.integers(-60, 60, size=(200, d)) + 0.5) * step
            i8 = np.clip(np.round(v_fp32 * 127.0 / scale), -127, 127).astype(np.int8)
            q = rng.standard_normal(d)
            q /= np.linalg.norm(q)
            scale = scale.astype(np.float32)
        else:
            q, v_fp32, i8, scale = _case(rng, int(rng.integers(50, 400)))

        true_scores = v_fp32 @ q
        int8_scores = eps_bound.score_i8(i8, q, scale)
        eps = eps_bound.eps_cauchy_schwarz(scale)
        tau = float(np.quantile(true_scores, rng.uniform(0.1, 0.9)))

        above, band, below = eps_bound.classify(int8_scores, tau, eps)
        true_matches = true_scores >= tau
        assert np.all((above | band)[true_matches]), np.nonzero(true_matches & below)[0]
        assert np.all(true_scores[below] < tau)


def test_eps_matches_hand_computed_closed_form() -> None:
    # ||scale||_2 / 254, checked against arithmetic worked by hand rather than
    # by re-deriving the formula the implementation uses.
    assert abs(eps_bound.eps_cauchy_schwarz(np.array([3.0, 4.0])) - 5.0 / 254.0) < 1e-12
    assert abs(eps_bound.eps_cauchy_schwarz(np.array([1.0, 2.0, 2.0])) - 3.0 / 254.0) < 1e-12


def test_hoelder_bound_is_valid_and_never_looser() -> None:
    # eps_hoelder must bound the realized quantization error too, and for a
    # unit-norm query Cauchy-Schwarz on (|q_d|, scale_d) says it is never the
    # looser of the two -- so a change that flips that ordering is caught.
    rng = np.random.default_rng(4)
    for _ in range(20):
        q, v_fp32, i8, scale = _case(rng, 200, d=24)
        actual_err = np.abs((v_fp32 @ q) - eps_bound.score_i8(i8, q, scale))
        eps_cs = eps_bound.eps_cauchy_schwarz(scale)
        eps_h = eps_bound.eps_hoelder(q, scale)
        assert np.all(actual_err <= eps_h + 1e-9), actual_err.max()
        assert np.all(actual_err <= eps_cs + 1e-9), actual_err.max()
        assert eps_h <= eps_cs + 1e-9, (eps_h, eps_cs)


def test_classify_boundaries_are_exact() -> None:
    # ABOVE/BAND/BELOW partition scores exactly at tau +/- eps, with the
    # documented open/closed endpoints and no row classified twice.
    above, band, below = eps_bound.classify(
        np.array([9.4, 9.5, 9.9, 10.0, 10.5, 10.6]), tau=10.0, eps=0.5
    )
    np.testing.assert_array_equal(below, [True, False, False, False, False, False])
    np.testing.assert_array_equal(band, [False, True, True, True, False, False])
    np.testing.assert_array_equal(above, [False, False, False, False, True, True])
    assert np.all(above.astype(int) + band.astype(int) + below.astype(int) == 1)


def test_partial_sum_of_inner_product_is_not_monotone() -> None:
    # Guard against "optimizing" the scan by stopping a dot product early:
    # inner-product terms can be negative, so a partial sum lands on either
    # side of tau relative to the full sum. Squared-L2, where every term is
    # non-negative, is the contrast case where such pruning IS sound.
    tau = 15.0
    undershoot = np.array([-10.0, -10.0, -10.0, 50.0])  # partial below, total above
    assert undershoot[:3].sum() < tau <= undershoot.sum()
    overshoot = np.array([50.0, -20.0, -20.0, -20.0])  # partial above, total below
    assert overshoot[:1].sum() >= tau > overshoot.sum()
    l2_terms = np.array([1.0, 4.0, 0.0, 9.0])
    assert np.all(np.diff(np.cumsum(l2_terms)) >= 0)
