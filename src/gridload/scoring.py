"""Forecast error metrics. Every function takes a weight array: real clock hours per cell (0 for the missing
spring hour, 2 for the repeated fall hour) times 0 or 1 for cells that may be scored."""

from __future__ import annotations

import numpy as np

QUANTILES = (0.1, 0.5, 0.9)


def _w(y, w):
    w = np.where(np.isfinite(y), w, 0.0)
    return np.nan_to_num(y), w


def mape(y, f, w) -> float:
    y, w = _w(y, w)
    return float(np.sum(w * np.abs(f - y) / np.where(y > 0, y, 1)) / np.sum(w))


def rmse(y, f, w) -> float:
    y, w = _w(y, w)
    return float(np.sqrt(np.sum(w * (f - y) ** 2) / np.sum(w)))


def bias(y, f, w) -> float:
    y, w = _w(y, w)
    return float(np.sum(w * (f - y)) / np.sum(w * y))


def pinball(y, f, q: float, w) -> float:
    """Mean pinball (quantile) loss: q * (y - f) when under, (1 - q) * (f - y) when over."""
    y, w = _w(y, w)
    d = y - f
    loss = np.maximum(q * d, (q - 1) * d)
    return float(np.sum(w * loss) / np.sum(w))


def mean_pinball(y, qf: dict[float, np.ndarray], w) -> float:
    return float(np.mean([pinball(y, qf[q], q, w) for q in QUANTILES]))


def coverage(y, lo, hi, w) -> float:
    y, w = _w(y, w)
    inside = (y >= lo) & (y <= hi)
    return float(np.sum(w * inside) / np.sum(w))


def peak_error(y, f, w) -> float:
    """Mean absolute percent error of the daily peak. y, f, w: (days, 24). A day counts if every hour is scorable."""
    y, w = _w(y, w)
    day_ok = (w > 0).sum(-1) >= 23
    ym = np.where(w > 0, y, -np.inf).max(-1)
    fm = np.where(w > 0, f, -np.inf).max(-1)
    return float(np.mean(np.abs(fm[day_ok] - ym[day_ok]) / ym[day_ok]))
