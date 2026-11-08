"""Imbalance cost: what forecast error costs in dollars.

The utility buys tomorrow's forecast in the day-ahead market. If actual load is lower (over-forecast), the surplus
was bought at the day-ahead price and is sold back in real time at a discount. If actual load is higher
(under-forecast), the shortfall is bought in real time at a premium that rises when the system is tight.
Cost is measured against a perfect forecast, so a perfect forecast costs zero.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Prices:
    base: float = 24.0  # $/MWh at light load
    slope: float = 70.0  # $/MWh added at the reference load (merit order gets steep near the top)
    ref_mw: float = 9000.0
    rt_sell: float = 0.70  # real-time resale price as a share of day-ahead
    rt_buy: float = 1.35  # real-time purchase price as a share of day-ahead at normal load
    scarcity: float = 4.0  # extra real-time multiple per unit of load above 90% of ref


DEFAULT = Prices()


def day_ahead_price(load_mw: np.ndarray, p: Prices | None = None) -> np.ndarray:
    p = p or DEFAULT
    return p.base + p.slope * (np.asarray(load_mw) / p.ref_mw) ** 4


def rt_buy_multiple(load_mw: np.ndarray, p: Prices | None = None) -> np.ndarray:
    p = p or DEFAULT
    tight = np.clip(np.asarray(load_mw) / p.ref_mw - 0.9, 0, None)
    return p.rt_buy + p.scarcity * tight


def imbalance_cost(forecast, actual, weight, p: Prices | None = None) -> dict:
    """Hourly MW forecasts and actuals; weight is real hours per cell (MWh per MW). Returns $ and MWh totals."""
    p = p or DEFAULT
    f, a = np.asarray(forecast, float), np.asarray(actual, float)
    w = np.where(np.isfinite(a), weight, 0.0)
    a = np.nan_to_num(a)
    da = day_ahead_price(a, p)
    over = np.clip(f - a, 0, None) * w
    under = np.clip(a - f, 0, None) * w
    over_cost = over * da * (1 - p.rt_sell)
    under_cost = under * da * (rt_buy_multiple(a, p) - 1)
    return {
        "over_mwh": float(over.sum()),
        "under_mwh": float(under.sum()),
        "over_usd": float(over_cost.sum()),
        "under_usd": float(under_cost.sum()),
        "total_usd": float(over_cost.sum() + under_cost.sum()),
    }
