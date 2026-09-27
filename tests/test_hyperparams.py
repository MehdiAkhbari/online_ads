"""Tests for the causal-forest hyperparameter rule in adsim.hyperparams.

Pure arithmetic -- no econml/sklearn needed.
"""

from __future__ import annotations

import math

import pytest

from adsim.hyperparams import (
    MAX_SAMPLES_CAP,
    arm_sizes_after_cap,
    cf_hyperparams,
    e_model_leaf_grid,
    m_model_leaf_grid,
)


# ---------------------------------------------------------------------------
# arm_sizes_after_cap
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_base, n_focal, expected", [
    (100_000, 2_600_000, (100_000, 300_000)),   # large focal arm capped at 3x base
    (100_000, 150, (450, 150)),                 # large base arm capped at 3x focal
    (10_000, 20_000, (10_000, 20_000)),         # within 3:1 -> untouched
    (7, 7, (7, 7)),
])
def test_arm_cap_is_symmetric(n_base, n_focal, expected):
    assert arm_sizes_after_cap(n_base, n_focal, 3.0) == expected


def test_arm_cap_ratio_is_configurable():
    assert arm_sizes_after_cap(10_000, 1_000_000, 5.0) == (10_000, 50_000)


def test_arm_cap_below_one_is_rejected():
    with pytest.raises(ValueError):
        arm_sizes_after_cap(10, 10, 0.5)


# ---------------------------------------------------------------------------
# cf_hyperparams
# ---------------------------------------------------------------------------

def _plan(n_base, n_focal, rate_base=0.001, rate_focal=0.001, **kw):
    return cf_hyperparams(n_base, n_focal, rate_base, rate_focal, **kw)


@pytest.mark.parametrize("n_total", [2, 3, 41, 1_000, 49_999, 50_001, 60_000, 85_000, 99_999, 100_001,
                                     399_999, 400_000, 2_700_000])
def test_max_samples_never_exceeds_half_the_rows(n_total):
    # econml raises under inference="blb" if an integer max_samples exceeds
    # n_samples // 2 (and tune()/fit on a 70% split would fail above 0.7 N).
    plan = _plan(n_total // 4, n_total - n_total // 4)
    assert 1 <= plan.max_samples <= plan.n_total // 2


def test_max_samples_hits_the_cap_for_large_pairs():
    assert _plan(100_000, 300_000).max_samples == min(MAX_SAMPLES_CAP, 200_000)
    assert _plan(100_000, 300_000, max_samples_cap=50_000).max_samples == 50_000


def test_leaf_floor_holds_k_expected_clicks_from_each_arm():
    plan = _plan(100_000, 300_000, rate_base=0.0009, rate_focal=0.0012, min_clicks_per_arm=5)
    # base arm has fewer expected clicks (90 vs 360).
    assert plan.min_samples_leaf == math.ceil(5 * 400_000 / 90)
    leaf_clicks_base = plan.min_samples_leaf * plan.expected_clicks_base / plan.n_total
    leaf_clicks_focal = plan.min_samples_leaf * plan.expected_clicks_focal / plan.n_total
    assert min(leaf_clicks_base, leaf_clicks_focal) >= 5


def test_leaf_floor_is_scale_free_and_leaves_grow_with_n():
    # Same arm shares and click rates at two sample sizes: the leaf floor
    # is unchanged, but the larger pair can hold more leaves per tree.
    small = _plan(20_000, 60_000)
    large = _plan(40_000, 120_000)
    assert small.min_samples_leaf == pytest.approx(large.min_samples_leaf, abs=1)
    assert large.max_leaves_per_tree > small.max_leaves_per_tree


def test_rule_is_continuous_in_n():
    # No thresholds: adding one row never jumps max_samples or the leaf floor.
    prev = _plan(25_000, 75_000)
    for n_focal in range(75_001, 75_200):
        cur = _plan(25_000, n_focal)
        assert abs(cur.max_samples - prev.max_samples) <= 1
        assert abs(cur.min_samples_leaf - prev.min_samples_leaf) <= 1
        prev = cur


def test_zero_click_arm_gets_no_splits():
    plan = _plan(3_000, 500, rate_base=0.05, rate_focal=0.0)
    assert plan.min_samples_leaf == plan.n_total
    assert plan.max_leaves_per_tree < 2


def test_split_is_twice_the_leaf_and_fixed_fields_pass_through():
    plan = _plan(100_000, 100_000, n_estimators=250, min_var_fraction_leaf=0.2)
    assert plan.min_samples_split == 2 * plan.min_samples_leaf
    assert plan.n_estimators == 250
    assert plan.min_var_fraction_leaf == 0.2


def test_too_few_rows_is_rejected():
    with pytest.raises(ValueError):
        _plan(1, 0)


# ---------------------------------------------------------------------------
# m_model_leaf_grid / e_model_leaf_grid
# ---------------------------------------------------------------------------

def test_m_leaf_grid_rate_based_below_the_cap():
    # click_rate=0.001, N=400,000: 10/0.001=10,000 (cap N/40=10,000),
    # 100/0.001=100,000 (cap N/10=40,000) -> both candidates capped.
    assert m_model_leaf_grid(n_total=400_000, click_rate=0.001) == [10_000, 40_000]


def test_m_leaf_grid_shrinks_for_a_smaller_pair():
    small = m_model_leaf_grid(n_total=50_000, click_rate=0.001)
    large = m_model_leaf_grid(n_total=400_000, click_rate=0.001)
    assert max(small) <= max(large)


def test_e_leaf_grid_uses_rare_arm_rate():
    # rare_arm_rate=0.25 (a 1:3 split), N=400,000:
    # 10/0.25=40 (cap N/100=4,000), 50/0.25=200 (cap N/20=20,000).
    assert e_model_leaf_grid(n_total=400_000, rare_arm_rate=0.25) == [40, 200]


@pytest.mark.parametrize("grid_fn", [m_model_leaf_grid, e_model_leaf_grid])
def test_leaf_grids_zero_rate_guard(grid_fn):
    grid = grid_fn(100_000, 0.0)
    assert all(v >= 1 for v in grid)
    assert grid == sorted(set(grid))
