"""Fit the first-pass LEARNED threshold policy from logged tuning episodes.

The live app currently suggests a cutoff with the label-free heuristic
``search_engine.heuristic_threshold`` (tau = mean + k*std of the query's score
distribution). Every labeled tune is logged to the ``threshold_episodes`` table
as ``(score-distribution features, fitted tau, F1)`` (web_server.py
``/api/threshold_search`` -> ``db.insert_threshold_episode``). Once enough
episodes accumulate, this script fits a ridge regression ``tau = w . features``
and reports whether it beats the heuristic on held-out tags -- the point at
which it's worth wiring the learned weights into serving.

This is an OFFLINE analysis tool: it does not modify the app or write anything.

Usage (inside the app env / with exp-db reachable):
    python3 scripts/fit_threshold_policy.py [--min-episodes 30] [--ridge 1e-2]
"""

from __future__ import annotations

import argparse

import numpy as np

import db
import search_engine

# The feature columns the policy regresses on (a subset of score_stats). Order
# matters -- it's the column order of the design matrix and the printed weights.
_FEATURES = ["mean", "std", "p90", "p99", "p99_9", "top_gap"]


def _design_matrix(episodes: list[dict]) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Build (X, y, kept) from episodes that carry all features + a fitted tau."""
    xs, ys, kept = [], [], []
    for ep in episodes:
        feats = ep.get("features") or {}
        tau = ep.get("fit_tau")
        if tau is None or not all(f in feats for f in _FEATURES):
            continue
        xs.append([float(feats[f]) for f in _FEATURES])
        ys.append(float(tau))
        kept.append(ep)
    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64), kept


def _ridge_fit(x: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """Closed-form ridge with an intercept column. Returns weights [b, w...]."""
    xb = np.hstack([np.ones((x.shape[0], 1)), x])
    reg = lam * np.eye(xb.shape[1])
    reg[0, 0] = 0.0  # don't regularize the intercept
    return np.linalg.solve(xb.T @ xb + reg, xb.T @ y)


def _predict(w: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.hstack([np.ones((x.shape[0], 1)), x]) @ w


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-episodes", type=int, default=30)
    ap.add_argument("--ridge", type=float, default=1e-2)
    ap.add_argument("--val-frac", type=float, default=0.25)
    args = ap.parse_args()

    episodes = db.threshold_episodes()
    x, y, kept = _design_matrix(episodes)
    print(f"episodes: {len(episodes)} logged, {len(kept)} usable (features + fit_tau)")
    if len(kept) < args.min_episodes:
        print(f"Not enough episodes yet (need >= {args.min_episodes}). Keep tuning tags; "
              "the heuristic stays live until then.")
        return

    # Held-out split (by row; episodes are already newest-first).
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(kept))
    cut = int(round(len(kept) * (1.0 - args.val_frac)))
    tr, va = perm[:cut], perm[cut:]

    w = _ridge_fit(x[tr], y[tr], args.ridge)
    pred = _predict(w, x[va])
    # Heuristic baseline on the same held-out rows (mean + 3*std, clamped).
    heur = np.array([
        search_engine.heuristic_threshold({"mean": r[0], "std": r[1]}) for r in x[va]
    ])
    mae_policy = float(np.mean(np.abs(pred - y[va])))
    mae_heur = float(np.mean(np.abs(heur - y[va])))

    print("\nlearned weights (tau = b + w . features):")
    print(f"  intercept = {w[0]:+.4f}")
    for name, wi in zip(_FEATURES, w[1:]):
        print(f"  {name:>7} = {wi:+.4f}")
    print(f"\nheld-out MAE(|tau_pred - tau_fit|):  policy={mae_policy:.4f}  "
          f"heuristic={mae_heur:.4f}")
    verdict = ("policy BEATS heuristic -- consider wiring these weights into serving"
               if mae_policy < mae_heur else
               "heuristic still wins -- keep it live; gather more episodes")
    print(verdict)


if __name__ == "__main__":
    main()
