"""Conformalized quantile regression (CQR) for the P10 to P90 band.

Quantile GBMs give a band that widens on hot or volatile days but is not guaranteed to cover 80%. CQR fixes the
coverage with a single widening term per lead bucket: score = max(lo - y, y - hi) / p50 on past forecasts whose
target day is already metered, and the band becomes [lo - q * p50, hi + q * p50] with q the finite-sample
(1 - alpha) quantile of those scores. Only scores known at issue time are used.
"""

from __future__ import annotations

import numpy as np


def conformal_quantile(scores: np.ndarray, alpha: float = 0.2) -> float:
    s = np.sort(np.asarray(scores, float)[np.isfinite(scores)])
    n = s.size
    if n == 0:
        return 0.0
    k = int(np.ceil((n + 1) * (1 - alpha))) - 1
    return float(s[min(max(k, 0), n - 1)])


def scores(y, lo, hi, p50) -> np.ndarray:
    return np.maximum(lo - y, y - hi) / p50


def calibrate(
    issues: np.ndarray,
    y: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    p50: np.ndarray,
    ok: np.ndarray,
    groups: list[np.ndarray],
    window: int = 90,
    alpha: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    """Rolling CQR over a backtest. Arrays are (n_iss, J, 24, M) for issues `issues` (consecutive days) and leads 1..J.

    groups: node index arrays calibrated separately (zones pooled, system alone). Lead 1 and leads 2..J are
    separate buckets. Returns adjusted (lo, hi)."""
    n_iss, J = y.shape[:2]
    s = scores(y, lo, hi, p50)
    s = np.where(ok, s, np.nan)
    lo2, hi2 = lo.copy(), hi.copy()
    buckets = [np.array([0]), np.arange(1, J)]
    lead = np.arange(1, J + 1)
    for a in range(n_iss):
        # past forecasts whose target day (issue + lead) ended before this issue day
        past = (issues[a] - issues[:, None] - lead[None, :] >= 1) & (issues[a] - issues[:, None] <= window)
        for g in groups:
            for b in buckets:
                mask = past[:, b]
                if not mask.any():
                    continue
                sel = s[:, b][mask][..., g]
                q = conformal_quantile(sel.reshape(-1), alpha)
                for jb in b:
                    lo2[a, jb][:, g] = lo[a, jb][:, g] - q * p50[a, jb][:, g]
                    hi2[a, jb][:, g] = hi[a, jb][:, g] + q * p50[a, jb][:, g]
    return lo2, hi2


def live_q(score_pool: np.ndarray, alpha: float = 0.2) -> float:
    return conformal_quantile(score_pool.reshape(-1), alpha)
