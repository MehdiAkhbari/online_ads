"""End-to-end fit_one_rank on small synthetic pairs (runs econml).

Covers the cases the hyperparameter rule must survive without an econml
error: the per-tree cap binding, N // 2 binding, an arm with no clicks,
a tiny arm, and data without a user-id column.
"""

from __future__ import annotations

import joblib
import numpy as np
import pytest

from adsim.estimate import fit_one_rank
from adsim.simulation_steps import define_xyt


def _max_tree_depth(cf) -> int:
    def trees(model):
        for t in getattr(model, "estimators_", []):
            yield from ([t] if hasattr(t, "get_depth") else trees(t))
    return max(t.get_depth() for t in trees(cf.model_cate))


def _fit(tmp_path, data, **kw):
    out = tmp_path / "cf.pkl"
    plan = fit_one_rank(rank=7, data=data, out_path=out, n_jobs=2, random_state=0, n_estimators=8, **kw)
    return plan, joblib.load(out)


@pytest.mark.parametrize("n_base, n_focal, cap", [
    (3_000, 3_000, 2_000),     # per-tree cap binds (N // 2 = 3,000)
    (1_500, 1_500, 10_000),    # N // 2 binds
    (3_000, 30_000, 1_000),    # focal arm capped at 3x base, then the per-tree cap
])
def test_fit_runs_and_respects_max_samples(tmp_path, make_pair, n_base, n_focal, cap):
    plan, cf = _fit(tmp_path, make_pair(n_base, n_focal), max_samples_cap=cap)
    assert plan.max_samples <= plan.n_total // 2
    assert cf.max_samples == plan.max_samples
    assert cf.adsim_plan_["max_samples"] == plan.max_samples
    tau = cf.const_marginal_effect(define_xyt(make_pair(20, 20))[0])
    assert np.isfinite(tau).all()


def test_forest_splits_when_the_click_floor_allows(tmp_path, make_pair):
    plan, cf = _fit(tmp_path, make_pair(3_000, 3_000), max_samples_cap=3_000, min_clicks_per_arm=5)
    assert plan.max_leaves_per_tree >= 2
    assert _max_tree_depth(cf) >= 1


def test_arm_without_clicks_gives_a_constant_effect(tmp_path, make_pair):
    plan, cf = _fit(tmp_path, make_pair(3_000, 500, rate_focal=0.0), max_samples_cap=2_000)
    assert plan.max_leaves_per_tree < 2
    assert _max_tree_depth(cf) == 0


def test_tiny_arm_does_not_raise(tmp_path, make_pair):
    plan, cf = _fit(tmp_path, make_pair(3_000, 4, n_users=300), max_samples_cap=2_000)
    assert plan.n_total == 16
    assert _max_tree_depth(cf) == 0


def test_too_few_users_per_arm_falls_back_to_row_folds(tmp_path, make_pair):
    plan, cf = _fit(tmp_path, make_pair(3_000, 4, n_users=2), max_samples_cap=2_000)
    assert cf.adsim_plan_["grouped_by"] is None


def test_missing_group_column_falls_back_to_rows(tmp_path, make_pair):
    data = make_pair(1_500, 1_500).drop(columns="global_token_new")
    plan, cf = _fit(tmp_path, data, max_samples_cap=1_000)
    assert cf.adsim_plan_["grouped_by"] is None


def test_group_column_is_recorded(tmp_path, make_pair):
    plan, cf = _fit(tmp_path, make_pair(1_500, 1_500), max_samples_cap=1_000)
    assert cf.adsim_plan_["grouped_by"] == "global_token_new"
