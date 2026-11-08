"""Peak alerts for demand response.

Each morning the operator decides whether to call a demand response (DR) event for tomorrow. A call costs money
(incentive payments to enrolled customers) whether or not the peak arrives. A system peak above the threshold
with no call sets a higher capacity obligation, which costs far more. With a calibrated probability p, calling is
worth it when p * miss_cost > call_cost, so the break-even probability is call_cost / miss_cost. The backtest
also picks the threshold that minimizes realized cost, which corrects for miscalibration.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AlertCosts:
    call_usd: float = 85_000.0  # incentive payments for one DR event
    miss_usd: float = 600_000.0  # capacity charge exposure when a peak above the threshold is not shaved


def exceed_probability(pred_peak: float, past_log_ratios: np.ndarray, threshold: float) -> float:
    """P(actual peak > threshold), with the actual peak modeled as pred_peak * exp(r) for past errors r."""
    r = np.asarray(past_log_ratios, float)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return float(pred_peak > threshold)
    return float(np.mean(pred_peak * np.exp(r) > threshold))


def realized_cost(prob: np.ndarray, exceeded: np.ndarray, cut: float, costs: AlertCosts) -> dict:
    call = np.asarray(prob) >= cut
    ex = np.asarray(exceeded, bool)
    tp, fp = int(np.sum(call & ex)), int(np.sum(call & ~ex))
    fn, tn = int(np.sum(~call & ex)), int(np.sum(~call & ~ex))
    cost = (tp + fp) * costs.call_usd + fn * costs.miss_usd
    return {
        "cut": float(cut),
        "calls": tp + fp,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "cost": float(cost),
    }


def cost_curve(prob, exceeded, costs: AlertCosts, grid: np.ndarray | None = None) -> list[dict]:
    grid = np.round(np.arange(0.0, 1.0001, 0.01), 2) if grid is None else grid
    return [realized_cost(prob, exceeded, c, costs) for c in grid]


def best_cut(curve: list[dict], costs: AlertCosts) -> dict:
    """Lowest realized cost; ties go to the cut closest to the theoretical break-even."""
    be = costs.call_usd / costs.miss_usd
    return min(curve, key=lambda r: (r["cost"], abs(r["cut"] - be)))


def break_even(costs: AlertCosts) -> float:
    return costs.call_usd / costs.miss_usd
