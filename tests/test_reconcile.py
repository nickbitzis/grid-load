import numpy as np

from gridload.reconcile import projection, reconcile, summing_matrix


def test_summing_matrix():
    S = summing_matrix(3)
    assert S.shape == (4, 3) and np.array_equal(S[-1], [1, 1, 1])


def test_every_method_is_coherent():
    base = np.random.default_rng(0).uniform(100, 200, (10, 24, 6))
    for method in ("bottom_up", "ols", "wls"):
        r = reconcile(base, method, np.arange(1, 7.0))
        assert np.allclose(r[..., :5].sum(-1), r[..., 5])


def test_bottom_up_keeps_the_zones():
    base = np.array([10.0, 20, 30, 40, 50, 999])
    assert np.allclose(reconcile(base, "bottom_up"), [10, 20, 30, 40, 50, 150])


def test_coherent_input_is_unchanged():
    base = np.array([10.0, 20, 30, 40, 50, 150])
    assert np.allclose(reconcile(base, "wls", np.array([1, 2, 3, 4, 5, 6.0])), base)


def test_ols_splits_the_gap_evenly():
    base = np.array([10.0, 10, 10, 10, 10, 56])  # zones sum to 50, system says 56
    r = reconcile(base, "ols")
    assert np.allclose(r[:5], 11) and np.isclose(r[5], 55)


def test_wls_trusts_the_low_variance_node():
    base = np.array([10.0, 10, 10, 10, 10, 60])
    tight_system = reconcile(base, "wls", np.array([1, 1, 1, 1, 1, 1e-6]))
    tight_zones = reconcile(base, "wls", np.array([1e-6] * 5 + [1.0]))
    assert np.isclose(tight_system[5], 60, atol=1e-3) and np.isclose(tight_zones[5], 50, atol=1e-3)
    assert projection(5).shape == (5, 6)
