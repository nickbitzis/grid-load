import numpy as np
import pytest

from gridload.alerts import AlertCosts, best_cut, break_even, cost_curve, exceed_probability, realized_cost

COSTS = AlertCosts(call_usd=10, miss_usd=100)
PROB = np.array([0.9, 0.2, 0.05, 0.6])
EXC = np.array([True, False, False, True])


def test_exceed_probability_from_past_errors():
    r = np.log(np.array([0.9, 1.0, 1.05, 1.1]))
    # predicted 1000: scenarios 900, 1000, 1050, 1100; two of four exceed 1020
    assert exceed_probability(1000, r, 1020) == 0.5
    assert exceed_probability(1000, np.array([]), 900) == 1.0


def test_realized_cost_by_hand():
    r = realized_cost(PROB, EXC, 0.5, COSTS)
    assert (r["calls"], r["tp"], r["fn"], r["cost"]) == (2, 2, 0, 20)
    assert r["precision"] == 1 and r["recall"] == 1
    r = realized_cost(PROB, EXC, 0.7, COSTS)
    assert (r["calls"], r["fn"], r["cost"]) == (1, 1, 110)
    assert realized_cost(PROB, EXC, 0.0, COSTS)["cost"] == 40


def test_never_calling_pays_every_miss():
    r = realized_cost(PROB, EXC, 1.01, COSTS)
    assert r["cost"] == 200 and r["recall"] == 0 and r["precision"] is None


def test_break_even():
    assert break_even(COSTS) == pytest.approx(0.1)


def test_best_cut_hand_worked():
    """Cuts in (0.2, 0.6] cost 20 (two calls, both hit). The tie goes to the cut nearest break-even (0.1): 0.21."""
    best = best_cut(cost_curve(PROB, EXC, COSTS), COSTS)
    assert best["cost"] == 20 and best["cut"] == pytest.approx(0.21)


def test_cost_curve_grid():
    c = cost_curve(PROB, EXC, COSTS)
    assert len(c) == 101 and c[0]["cut"] == 0 and c[-1]["cut"] == 1
