"""Tests for adsim.diagnostics.stats (pure numpy / scipy helpers)."""

from __future__ import annotations

import numpy as np
import pytest

from adsim.diagnostics.stats import (
    cluster_robust_wald,
    clustered_mean,
    crump_alpha,
    expected_calibration_error,
    mixed_cell_share,
    ovb_bounds,
    ovb_params,
    partialling_out_ate,
    permute_within,
    robustness_value,
    standardized_mean_difference,
    within_cell_smd,
)


def test_smd_matches_the_textbook_formula():
    x = np.array([1.0, 2.0, 3.0, 4.0, 6.0, 8.0])
    t = np.array([0, 0, 0, 1, 1, 1])
    expected = (6.0 - 2.0) / np.sqrt((x[t == 1].var(ddof=1) + x[t == 0].var(ddof=1)) / 2)
    assert standardized_mean_difference(x, t) == pytest.approx(expected)


def test_weighted_smd_keeps_the_unweighted_scale():
    x = np.array([0.0, 1.0, 0.0, 1.0])
    t = np.array([0, 0, 1, 1])
    assert standardized_mean_difference(x, t) == 0
    # Up-weighting the treated x=1 row shifts only the treated mean.
    w = np.array([1.0, 1.0, 1.0, 3.0])
    assert standardized_mean_difference(x, t, weights=w) == pytest.approx(0.25 / np.sqrt(0.5))


def test_within_cell_smd_removes_confounding_by_cell():
    rng = np.random.default_rng(0)
    cells = rng.integers(0, 50, 20_000)
    t = (rng.random(20_000) < 0.2 + 0.6 * (cells % 2)).astype(int)
    x = cells % 2 + rng.normal(size=20_000)
    assert abs(standardized_mean_difference(x, t)) > 0.4
    assert abs(within_cell_smd(x, t, cells)) < 0.05


def test_cluster_robust_wald_rejects_only_under_the_alternative():
    rng = np.random.default_rng(1)
    n, groups = 5_000, np.repeat(np.arange(1_000), 5)
    X = rng.normal(size=(n, 3))
    null = cluster_robust_wald(rng.normal(size=n), X, groups)
    alt = cluster_robust_wald(0.2 * X[:, 0] + rng.normal(size=n), X, groups)
    assert null["df_num"] == 3 and null["df_den"] == 999
    assert null["p_value"] > 0.01
    assert alt["p_value"] < 1e-6


def test_clustered_se_exceeds_naive_se_when_clusters_are_correlated():
    rng = np.random.default_rng(2)
    groups = np.repeat(np.arange(200), 20)
    values = rng.normal(size=200)[groups] + 0.1 * rng.normal(size=4_000)
    _, naive = clustered_mean(values)
    _, clustered = clustered_mean(values, groups)
    assert clustered > 3 * naive


def test_partialling_out_recovers_a_constant_effect():
    rng = np.random.default_rng(3)
    n = 50_000
    x = rng.normal(size=n)
    e = 1 / (1 + np.exp(-x))
    t = (rng.random(n) < e).astype(int)
    y = 0.5 * x + 0.3 * t + rng.normal(size=n)
    est = partialling_out_ate(y, t, 0.5 * x + 0.3 * e, e)
    assert est["theta"] == pytest.approx(0.3, abs=3 * est["se"])
    assert est["p_value"] < 1e-6


def test_crump_alpha_trims_only_when_propensities_are_extreme():
    assert crump_alpha(np.linspace(0.3, 0.7, 1_000)) == 0.0
    extreme = np.r_[np.linspace(0.3, 0.7, 900), np.full(100, 0.005)]
    alpha = crump_alpha(extreme)
    assert 0.005 < alpha < 0.3


def test_ece_is_small_for_calibrated_and_large_for_biased_forecasts():
    rng = np.random.default_rng(4)
    p = rng.uniform(0.01, 0.2, 200_000)
    y = (rng.random(len(p)) < p).astype(int)
    assert expected_calibration_error(p, y) < 0.005
    assert expected_calibration_error(2 * p, y) > 0.05


def _ovb_fixture():
    rng = np.random.default_rng(5)
    t_res = rng.normal(scale=0.5, size=10_000)
    y_res = 0.2 * t_res + rng.normal(size=10_000)
    return ovb_params(y_res, t_res)


def test_ovb_bounds_collapse_to_theta_without_confounding_and_widen_with_it():
    params = _ovb_fixture()
    none = ovb_bounds(params, 0.0, 0.0)
    assert none["theta_lo"] == pytest.approx(params["theta"]) == pytest.approx(none["theta_hi"])
    assert none["ci_lo"] < params["theta"] < none["ci_hi"]
    widths = [ovb_bounds(params, c, c)["theta_hi"] - ovb_bounds(params, c, c)["theta_lo"] for c in (0.01, 0.05, 0.1)]
    assert widths == sorted(widths) and widths[0] > 0


def test_robustness_value_is_where_the_bound_reaches_zero():
    params = _ovb_fixture()
    rv = robustness_value(params, interval="theta")
    assert 0 < rv < 1
    assert ovb_bounds(params, rv, rv)["theta_lo"] == pytest.approx(0, abs=1e-6)
    assert robustness_value(params, interval="ci") < rv


def test_permute_within_keeps_each_cells_values():
    rng = np.random.default_rng(6)
    cells = rng.integers(0, 20, 1_000)
    values = rng.integers(0, 2, 1_000)
    shuffled = permute_within(values, cells, rng)
    for c in range(20):
        assert sorted(shuffled[cells == c]) == sorted(values[cells == c])
    assert (shuffled != values).mean() > 0.2


def test_mixed_cell_share_counts_rows_in_cells_with_both_arms():
    t = np.array([0, 1, 0, 0, 1, 1])
    cells = np.array([0, 0, 1, 1, 2, 3])
    assert mixed_cell_share(t, cells) == pytest.approx(2 / 6)
