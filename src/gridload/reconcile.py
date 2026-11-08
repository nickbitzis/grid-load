"""Hierarchical reconciliation: five zone forecasts and a direct system forecast, made to add up.

The summing matrix S maps the 5 zones to the 6 nodes (zones, then system). Bottom-up keeps the zones and sums them.
WLS (MinT with a diagonal covariance) finds coherent forecasts closest to all six base forecasts, weighting each
node by the inverse of its error variance, so a node that forecasts well moves less.
"""

from __future__ import annotations

import numpy as np


def summing_matrix(n_zones: int) -> np.ndarray:
    return np.vstack([np.eye(n_zones), np.ones((1, n_zones))])


def projection(n_zones: int, variances: np.ndarray | None = None) -> np.ndarray:
    """P such that reconciled = S @ P @ base. variances: (n_zones + 1,) error variance per node."""
    S = summing_matrix(n_zones)
    w = np.ones(n_zones + 1) if variances is None else np.asarray(variances, float)
    Winv = np.diag(1.0 / w)
    return np.linalg.solve(S.T @ Winv @ S, S.T @ Winv)


def reconcile(base: np.ndarray, method: str = "wls", variances: np.ndarray | None = None) -> np.ndarray:
    """base: (..., n_zones + 1) with the system last. Returns coherent forecasts of the same shape."""
    n = base.shape[-1] - 1
    S = summing_matrix(n)
    if method == "bottom_up":
        bottom = base[..., :n]
    elif method == "ols":
        bottom = base @ projection(n).T
    elif method == "wls":
        bottom = base @ projection(n, variances).T
    else:
        raise ValueError(method)
    return bottom @ S.T
