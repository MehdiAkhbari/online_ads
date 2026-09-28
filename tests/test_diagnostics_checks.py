"""Diagnostic checks on synthetic estimation data (runs econml / sklearn).

`make_estimation_data(selection=True)` shows ad 1 more often to users with
many past subject-1 visits, who also click more: a violation of Assumption 1
that the identification tests must flag, and must not flag without it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from adsim.diagnostics import identification, model_checks
from adsim.diagnostics.data import (
    AD_HISTORY_COLS,
    BROWSING_COLS,
    CONTEXT_COLS,
    FEATURE_COLS,
    PLACEBO_LAG_COL,
    RankData,
    Settings,
    add_lag_click,
    add_segments,
    diagnosable_ranks,
    top_ranks,
)
from adsim.simulation_steps import define_xyt, prepare_data

SETTINGS = Settings(n_jobs=2, refit_trees=8, n_permutations=9)


def _prepared(raw: pd.DataFrame) -> pd.DataFrame:
    data = raw.copy()
    prepare_data(data)
    add_lag_click(data, group_col="global_token_new", visit_col="user_visit_no")
    add_segments(data)
    return data


@pytest.fixture(scope="module")
def clean(make_estimation_data):
    return RankData.build(_prepared(make_estimation_data(selection=False)), 1, None, SETTINGS)


@pytest.fixture(scope="module")
def confounded(make_estimation_data):
    return RankData.build(_prepared(make_estimation_data(selection=True)), 1, None, SETTINGS)


# ----------------------------------------------------------------------
# Data layer
# ----------------------------------------------------------------------

def test_feature_blocks_partition_the_forest_features_in_order(make_pair):
    assert FEATURE_COLS == list(define_xyt(make_pair(2, 2))[0].columns)
    assert set(AD_HISTORY_COLS) | set(BROWSING_COLS) | set(CONTEXT_COLS) == set(FEATURE_COLS)


def test_ranks_exclude_the_base_ad_and_the_fringe(make_estimation_data):
    data = _prepared(make_estimation_data())
    assert diagnosable_ranks(data) == [1, 2, 3]
    assert set(top_ranks(data, 2)) < {1, 2, 3}


def test_lag_click_is_the_previous_impressions_click(make_estimation_data):
    data = _prepared(make_estimation_data(n_users=50))
    user = data[data["global_token_new"] == 0].sort_values("user_visit_no")
    assert np.isnan(user[PLACEBO_LAG_COL].iloc[0])
    assert user[PLACEBO_LAG_COL].iloc[1:].tolist() == user["is_clicked"].iloc[:-1].tolist()


def test_rank_data_holds_base_and_focal_rows(clean):
    assert set(clean.pair["advertiser_rank"]) == {0, 1}
    assert clean.groups is not None
    assert len(np.unique(clean.cells)) > 1


# ----------------------------------------------------------------------
# Identification tests
# ----------------------------------------------------------------------

def _row(table: pd.DataFrame, col: str, value) -> pd.Series:
    return table.set_index(col).loc[value]


def test_propensity_information_flags_selection_on_browsing_history(clean, confounded):
    ok = identification.propensity_information(clean)["propensity_info"]
    bad = identification.propensity_information(confounded)["propensity_info"]
    assert list(ok["feature_set"]) == ["context", "context+browsing", "context+ad_history", "all"]
    assert _row(ok, "feature_set", "context+browsing")["p_perm"] > 0.01
    assert _row(bad, "feature_set", "context+browsing")["z_perm"] > 5
    assert _row(bad, "feature_set", "context+browsing")["p_perm"] < 1e-3
    assert _row(bad, "feature_set", "context+browsing")["p_perm_empirical"] == pytest.approx(0.1)
    assert _row(bad, "feature_set", "context+browsing")["auc"] > _row(bad, "feature_set", "context")["auc"]


def test_permutation_null_absorbs_the_gain_from_an_irrelevant_block(make_estimation_data):
    # Without selection the raw gain is non-zero from overfitting alone; the
    # permuted refits reproduce it, so no history block is flagged.
    rd = RankData.build(_prepared(make_estimation_data(seed=3)), 1, None, SETTINGS)
    table = identification.propensity_information(rd)["propensity_info"].set_index("feature_set")
    assert table.loc["context", "gain"] == 0
    for name in ("context+browsing", "context+ad_history", "all"):
        assert table.loc[name, "null_gain_sd"] > 0
        assert table.loc[name, "p_perm"] > 0.01


def test_balance_flags_the_selected_covariate(clean, confounded):
    ok = identification.covariate_balance(clean)
    bad = identification.covariate_balance(confounded)
    assert len(ok["balance_smd"]) == len(BROWSING_COLS) + len(AD_HISTORY_COLS)
    assert ok["balance_smd"][["smd_raw", "smd_within_cell", "smd_ipw"]].abs().max().max() < 0.15
    assert abs(_row(bad["balance_smd"], "covariate", "visit_s1")["smd_ipw"]) > 0.5
    assert _row(ok["balance_omnibus"], "block", "browsing")["p_value"] > 0.01
    assert _row(bad["balance_omnibus"], "block", "browsing")["p_value"] < 1e-6
    assert 0 < ok["balance_omnibus"]["share_rows_in_mixed_cells"].iloc[0] <= 1


def test_placebo_outcomes_are_null_only_without_selection(clean, confounded):
    ok = identification.placebo_outcomes(clean)["placebo_outcomes"]
    bad = identification.placebo_outcomes(confounded)["placebo_outcomes"]
    assert set(ok["placebo_outcome"]) == {"any_prior_click", "lag_click"}
    assert (ok["p_value"] > 0.01).all()
    assert _row(bad, "placebo_outcome", "any_prior_click")["p_value"] < 1e-4
    assert _row(bad, "placebo_outcome", "lag_click")["theta"] > 0


def test_sensitivity_benchmark_reflects_selection(clean, confounded):
    ok = identification.ovb_sensitivity(clean)["sensitivity"]
    bad = identification.ovb_sensitivity(confounded)["sensitivity"]
    bench = "benchmark: browsing history"
    assert _row(ok, "scenario", bench)["c_t"] < 0.02
    assert _row(bad, "scenario", bench)["c_t"] > 0.05
    none = _row(ok, "scenario", "none")
    assert none["theta_lo"] == pytest.approx(none["theta"])
    assert 0 <= ok["rv_ci"].iloc[0] <= ok["rv_theta"].iloc[0] <= 1


def test_overlap_reports_trimming_and_histograms(clean):
    out = identification.overlap(clean)
    assert out["overlap_summary"]["crump_alpha"].iloc[0] >= 0
    assert list(out["overlap_trimming"]["trimming"]) == ["none", "crump", "0.1-0.9"]
    hist = out["overlap_histogram"]
    assert hist["base"].sum() == (clean.T == 0).sum() and hist["focal"].sum() == clean.T.sum()


def test_refuters_placebo_effect_vanishes_when_labels_move(make_estimation_data):
    # Coarse cells so that most labels actually move under the permutation.
    data = _prepared(make_estimation_data(n_users=2_500, seed=2))
    rd = RankData.build(data, 1, None, Settings(n_jobs=2, refit_trees=40, cell_cols=("hour",)))
    out = identification.refutation_tests(rd)["refuters"].set_index("test")
    assert list(out.index) == ["observed", "placebo_treatment", "random_common_cause", "data_subset"]
    assert out.loc["placebo_treatment", "share_T_changed"] > 0.3
    observed, placebo = out.loc["observed"], out.loc["placebo_treatment"]
    assert observed["ate_ci_lo"] > 0
    assert placebo["ate_ci_lo"] < 0 < placebo["ate_ci_hi"]
    assert abs(placebo["ate"]) < observed["ate"] / 2
    assert abs(out.loc["random_common_cause", "ate_minus_observed"]) < observed["ate_se"]


# ----------------------------------------------------------------------
# Model diagnostics
# ----------------------------------------------------------------------

def test_nuisance_scores_overall_and_by_segment(clean):
    out = model_checks.nuisance_scores(clean)
    assert list(out["nuisance_scores"]["model"]) == ["outcome m(X)", "propensity e(C)"]
    seg = out["nuisance_scores_by_segment"]
    assert set(seg["segment"]) == {"device", "user", "browsing"}
    assert seg.groupby(["model", "segment"])["n"].sum().eq(len(clean.pair)).all()


def test_cate_validation_scores_a_heterogeneous_forest(make_estimation_data):
    data = _prepared(make_estimation_data(n_users=2_500, seed=2))
    rd = RankData.build(data, 1, None, Settings(n_jobs=2, refit_trees=40))
    out = model_checks.cate_validation(rd, n_bootstrap=50)
    row = out["cate_validation"].iloc[0]
    assert row["n_train"] + row["n_val"] == len(rd.pair)
    assert np.isfinite(row["rscore"]) and np.isfinite(row["blp_slope"]) and np.isfinite(row["cal_r2"])
    assert {"gate", "g_cate"} <= set(out["cate_calibration_groups"].columns)
    assert out["pair_reliability"]["n"].sum() == row["n_val"]


def test_estimator_comparison_covers_every_estimator(clean):
    table = model_checks.estimator_comparison(clean)["estimators"]
    assert list(table["estimator"]) == [
        "causal_forest", "causal_forest_discrete_outcome", "forest_dr_learner",
        "linear_dr_learner", "t_learner_gbm", "logit_interactions",
    ]
    assert table["rscore"].notna().all()


def test_stability_splits_by_day(clean):
    row = model_checks.stability(clean)["stability"].iloc[0]
    assert row["n_first"] + row["n_second"] == len(clean.pair)
    assert np.isfinite(row["ate_diff_z"])


def test_proba_t_learner_keeps_levels_in_the_unit_interval():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2_000, 3))
    T = rng.integers(0, 2, 2_000)
    Y = (rng.random(2_000) < 0.1 + 0.1 * T).astype(int)
    from sklearn.linear_model import LogisticRegression
    est = model_checks.ProbaTLearner(LogisticRegression()).fit(Y, T, X=X)
    assert est.const_marginal_effect(X).shape == (2_000, 1)
    assert np.all((est.predict_arm(X, 1) >= 0) & (est.predict_arm(X, 1) <= 1))
    no_clicks = model_checks.ProbaTLearner(LogisticRegression()).fit(np.zeros(2_000), T, X=X)
    assert np.all(no_clicks.const_marginal_effect(X) == 0)


# ----------------------------------------------------------------------
# Pipeline checks (the simulation's y_r = y0 + tau_r)
# ----------------------------------------------------------------------

class LinearCate:
    """Stand-in forest: tau(x) = a + b * hour."""

    def __init__(self, a, b=0.0):
        self.a, self.b = a, b

    def const_marginal_effect(self, X):
        return (self.a + self.b * np.asarray(X["hour"], dtype=float)).reshape(-1, 1)


def _sample(n=2_000, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.random((n, len(FEATURE_COLS))), columns=FEATURE_COLS)
    X["hour"] = rng.integers(0, 24, n)
    X["ads_on_page"] = rng.integers(1, 4, n)
    X["advertiser_rank"] = rng.choice([0, 1, 2, 3], n)
    X["global_token_new"] = rng.integers(0, n // 4, n)
    return X


def test_out_of_range_counts_clipped_click_rates():
    sample = _sample()
    y0 = np.full(len(sample), 0.01)
    forests = {1: LinearCate(0.005), 2: LinearCate(-0.02), 3: LinearCate(-0.02, 0.001)}
    ctrs = model_checks.pipeline_ctrs(sample[FEATURE_COLS], forests, y0)
    table = model_checks.out_of_range(ctrs, y0)["out_of_range"].set_index("rank")
    assert table.loc[1, "share_below_0"] == 0
    assert table.loc[2, "share_below_0"] == 1
    assert table.loc[3, "share_below_0"] == pytest.approx(np.mean(sample["hour"] < 10))
    assert table.loc["all ranks", "n"] == 3 * len(sample)


def test_level_calibration_of_correct_predictions():
    rng = np.random.default_rng(1)
    sample = _sample(40_000)
    y0 = np.full(len(sample), 0.02)
    forests = {1: LinearCate(0.01), 2: LinearCate(0.0, 0.001), 3: LinearCate(-0.03)}
    support = pd.Series({1: "high", 2: "medium", 3: "low"})
    truth = model_checks.shown_ad_predictions(sample.assign(is_clicked=0), forests, y0, support)["p"]
    sample["is_clicked"] = (rng.random(len(sample)) < truth).astype(int)
    preds = model_checks.shown_ad_predictions(sample, forests, y0, support)
    assert (preds.loc[preds["advertiser_rank"] == 3, "p"] == 0).all()
    out = model_checks.level_calibration(preds)
    overall = out["calibration"].iloc[0]
    assert overall["predicted_over_observed"] == pytest.approx(1, abs=0.1)
    assert overall["share_raw_below_0"] == pytest.approx(np.mean(sample["advertiser_rank"] == 3))
    assert set(out["reliability"]["support"]) == {"all", "base ad", "high", "medium", "low"}


def test_placebo_monopoly_gain_is_zero_for_identical_forests_and_positive_for_noisy_ones():
    sample = _sample()
    X = sample[FEATURE_COLS]
    y0 = np.full(len(sample), 0.01)
    forests = {r: LinearCate(0.001 * r, 0.0002 * (2 - r)) for r in (1, 2, 3)}
    ctrs = model_checks.pipeline_ctrs(X, forests, y0)
    vals = pd.Series({1: 10.0, 2: 30.0, 3: 20.0})
    kw = {"valuations": vals, "groups": sample["global_token_new"].to_numpy()}

    same = model_checks.placebo_monopoly(sample, ctrs, y0, ctrs, y0, **kw)["placebo_monopoly"]
    assert list(same["criterion"]) == ["CTR", "revenue"]
    assert np.allclose(same["mechanical_gain"], 0) and np.allclose(same["share_slots_same_ad"], 1)

    rng = np.random.default_rng(0)
    noisy = ctrs + rng.normal(scale=0.004, size=ctrs.shape)
    out = model_checks.placebo_monopoly(sample, ctrs, y0, noisy, y0, **kw)["placebo_monopoly"].set_index("criterion")
    # Each criterion's own objective can only fall when a noisier model chooses.
    assert out.loc["CTR", "mechanical_gain"] > 0
    assert out.loc["CTR", "diff"] > 2 * out.loc["CTR", "diff_se"]
    assert out.loc["revenue", "value_gain"] > 0
    assert (out["n_slots"] == np.minimum(sample["ads_on_page"], 15).sum()).all()
