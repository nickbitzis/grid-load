import numpy as np
import pytest

from gridload.economics import Prices, day_ahead_price, imbalance_cost, rt_buy_multiple


def test_perfect_forecast_costs_nothing():
    a = np.array([5000.0, 7000.0])
    assert imbalance_cost(a, a, np.ones(2))["total_usd"] == 0


def test_over_forecast_by_hand():
    # actual 4500 MW: DA price 24 + 70 * 0.5^4 = 28.375; 100 MWh surplus resold at 70%
    c = imbalance_cost(np.array([4600.0]), np.array([4500.0]), np.ones(1))
    assert day_ahead_price(np.array([4500.0]))[0] == pytest.approx(28.375)
    assert c["over_mwh"] == 100 and c["over_usd"] == pytest.approx(100 * 28.375 * 0.3)


def test_under_forecast_pays_scarcity_by_hand():
    # actual 9000 MW = ref: DA 94, real-time multiple 1.35 + 4 * 0.1 = 1.75
    c = imbalance_cost(np.array([8900.0]), np.array([9000.0]), np.ones(1))
    assert rt_buy_multiple(np.array([9000.0]))[0] == pytest.approx(1.75)
    assert c["under_usd"] == pytest.approx(100 * 94 * 0.75)


def test_repeated_fall_hour_counts_twice_and_missing_actuals_drop():
    one = imbalance_cost(np.array([4600.0]), np.array([4500.0]), np.array([1.0]))["total_usd"]
    two = imbalance_cost(np.array([4600.0]), np.array([4500.0]), np.array([2.0]))["total_usd"]
    assert two == pytest.approx(2 * one)
    assert imbalance_cost(np.array([4600.0]), np.array([np.nan]), np.ones(1))["total_usd"] == 0


def test_shortfall_costs_more_than_surplus():
    p = Prices()
    over = imbalance_cost(np.array([6100.0]), np.array([6000.0]), np.ones(1), p)["total_usd"]
    under = imbalance_cost(np.array([5900.0]), np.array([6000.0]), np.ones(1), p)["total_usd"]
    assert under > over
