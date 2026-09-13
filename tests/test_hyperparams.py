"""Tests for the N-adaptive causal-forest hyperparameter rule.

Pure-Python arithmetic -- no econml/sklearn needed, unlike
test_estimate_cli.py / test_simulation_steps.py.
"""

from __future__ import annotations

from adsim.hyperparams import cf_hyperparams, e_model_leaf_grid, m_model_leaf_grid


# ---------------------------------------------------------------------------
# cf_hyperparams
# ---------------------------------------------------------------------------

def test_focal_arm_uncapped_below_ratio():
    # n_focal (20,000) is under 3x n_base (30,000) -> no capping.
    hp = cf_hyperparams(n_base=10_000, n_focal=20_000)
    assert hp.n_focal_used == 20_000
    assert hp.n_total == 30_000


def test_focal_arm_capped_above_ratio():
    # n_focal (2,600,000) is far above 3x n_base (300,000) -> capped.
    hp = cf_hyperparams(n_base=100_000, n_focal=2_600_000)
    assert hp.n_focal_used == 300_000
    assert hp.n_total == 400_000


def test_focal_arm_cap_ratio_is_configurable():
    hp = cf_hyperparams(n_base=10_000, n_focal=1_000_000, focal_cap_ratio=5.0)
    assert hp.n_focal_used == 50_000
    assert hp.n_total == 60_000


def test_max_samples_uses_n_over_2_under_the_cap():
    hp = cf_hyperparams(n_base=50_000, n_focal=50_000)  # n_total = 100,000
    assert hp.max_samples == 50_000


def test_max_samples_hits_the_absolute_cap():
    hp = cf_hyperparams(n_base=100_000, n_focal=2_600_000, max_samples_cap=100_000)
    # n_total = 400,000 after capping -> n_total // 2 = 200,000 > cap.
    assert hp.max_samples == 100_000


def test_max_samples_never_exceeds_n_total_over_2():
    # This is the constraint econml enforces under inference="blb": an
    # integer max_samples must not exceed n_samples // 2.
    for n_base, n_focal in [(50, 10), (12_000, 75), (100_000, 150), (100_000, 20_000)]:
        hp = cf_hyperparams(n_base, n_focal)
        assert hp.max_samples <= hp.n_total // 2


def test_min_samples_split_is_twice_the_smaller_leaf_candidate():
    hp = cf_hyperparams(n_base=100_000, n_focal=100_000)
    assert hp.min_samples_split == 2 * min(hp.min_samples_leaf_grid)


def test_leaf_grid_is_sorted_positive_and_at_most_two_values():
    for n_base, n_focal in [(50, 10), (12_000, 75), (100_000, 150), (100_000, 500_000)]:
        hp = cf_hyperparams(n_base, n_focal)
        assert 1 <= len(hp.min_samples_leaf_grid) <= 2
        assert all(v > 0 for v in hp.min_samples_leaf_grid)
        assert hp.min_samples_leaf_grid == sorted(hp.min_samples_leaf_grid)


def test_tiny_pair_does_not_crash_or_produce_nonpositive_values():
    hp = cf_hyperparams(n_base=12_000, n_focal=75)
    assert hp.n_total > 0
    assert hp.max_samples >= 2
    assert min(hp.min_samples_leaf_grid) >= 1


def test_fixed_fields_are_not_searched():
    hp = cf_hyperparams(n_base=100_000, n_focal=100_000, n_estimators=250, max_depth=4)
    assert hp.n_estimators == 250
    assert hp.max_depth == 4
    assert hp.min_var_fraction_leaf_grid == [0.01, 0.1]
    assert hp.min_balancedness_tol == 0.45


# ---------------------------------------------------------------------------
# m_model_leaf_grid / e_model_leaf_grid
# ---------------------------------------------------------------------------

def test_m_leaf_grid_rate_based_below_the_cap():
    # click_rate=0.001, N=400,000: 10/0.001=10,000 (cap N/40=10,000),
    # 100/0.001=100,000 (cap N/10=40,000) -> both candidates capped.
    grid = m_model_leaf_grid(n_total=400_000, click_rate=0.001)
    assert grid == [10_000, 40_000]


def test_m_leaf_grid_shrinks_for_a_smaller_pair():
    small = m_model_leaf_grid(n_total=50_000, click_rate=0.001)
    large = m_model_leaf_grid(n_total=400_000, click_rate=0.001)
    assert max(small) <= max(large)


def test_e_leaf_grid_uses_rare_arm_rate():
    # rare_arm_rate=0.25 (a 1:3 split), N=400,000:
    # 10/0.25=40 (cap N/100=4,000), 50/0.25=200 (cap N/20=20,000).
    grid = e_model_leaf_grid(n_total=400_000, rare_arm_rate=0.25)
    assert grid == [40, 200]


def test_e_leaf_grid_zero_rate_guard_does_not_raise():
    grid = e_model_leaf_grid(n_total=100_000, rare_arm_rate=0.0)
    assert all(v >= 1 for v in grid)


def test_m_leaf_grid_zero_click_rate_guard_does_not_raise():
    grid = m_model_leaf_grid(n_total=100_000, click_rate=0.0)
    assert all(v >= 1 for v in grid)


def test_leaf_grids_are_sorted_and_unique():
    grid = m_model_leaf_grid(n_total=100_000, click_rate=0.05)
    assert grid == sorted(set(grid))
