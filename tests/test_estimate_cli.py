"""Tests for the adsim.estimate CLI logic.

Targets the pure-function helpers (select_ranks, parse_args, build_pair,
plan_for_pair) and the scenario dispatch table. fit_one_rank() itself is
exercised on synthetic data in test_estimate_fit.py.
"""

from __future__ import annotations

import pytest

from adsim.estimate import SCENARIOS, build_pair, parse_args, plan_for_pair, select_ranks
from adsim.hyperparams import ARM_RATIO_CAP, MAX_SAMPLES_CAP, MIN_CLICKS_PER_ARM


# ---------------------------------------------------------------------------
# select_ranks
# ---------------------------------------------------------------------------

def test_select_ranks_returns_all_when_no_filter():
    assert select_ranks([1, 2, 3, 5, 10], explicit=None, expr=None, limit=None) == [1, 2, 3, 5, 10]


def test_select_ranks_explicit_filters_to_intersection():
    assert select_ranks([1, 2, 3, 5, 10], explicit=[2, 5, 99], expr=None, limit=None) == [2, 5]


def test_select_ranks_expr_filters_by_python_expression():
    # `rank > 3` keeps only ranks strictly above 3.
    assert select_ranks([1, 2, 3, 5, 10], explicit=None, expr="rank > 3", limit=None) == [5, 10]


def test_select_ranks_expr_handles_complex_expression():
    assert select_ranks(
        [1, 2, 3, 5, 7, 10, 12], explicit=None, expr="rank % 2 == 1 and rank > 1", limit=None,
    ) == [3, 5, 7]


def test_select_ranks_limit_caps_to_first_n():
    assert select_ranks([1, 2, 3, 5, 10], explicit=None, expr=None, limit=3) == [1, 2, 3]


def test_select_ranks_combines_filters():
    # explicit list + expression + limit applied in order.
    out = select_ranks(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        explicit=[2, 4, 6, 8, 10],   # keep evens
        expr="rank > 3",             # keep > 3
        limit=2,                     # take first 2
    )
    assert out == [4, 6]


def test_select_ranks_expr_blocks_builtins():
    # The eval'd expression has __builtins__ stripped; abusing it should fail.
    with pytest.raises(NameError):
        select_ranks([1, 2], explicit=None, expr="__import__('os').system('rm -rf /')", limit=None)


def test_select_ranks_preserves_order_of_discovered():
    # Order in `explicit=` doesn't reorder the discovered list.
    assert select_ranks([5, 3, 1], explicit=[1, 3, 5], expr=None, limit=None) == [5, 3, 1]


# ---------------------------------------------------------------------------
# parse_args / scenario validation
# ---------------------------------------------------------------------------

def test_parse_args_monopoly_no_split():
    args = parse_args(["--scenario", "monopoly"])
    assert args.scenario == "monopoly"
    assert args.split is None
    assert args.force is False
    assert args.dry_run is False


def test_parse_args_split_requires_split_flag():
    # SystemExit comes from argparse's p.error().
    with pytest.raises(SystemExit):
        parse_args(["--scenario", "split"])


def test_parse_args_split_with_split():
    args = parse_args(["--scenario", "split", "--split", "7"])
    assert args.scenario == "split"
    assert args.split == 7


def test_parse_args_split_root_n_requires_split():
    with pytest.raises(SystemExit):
        parse_args(["--scenario", "split-root-n"])


def test_parse_args_root_n_does_not_require_split():
    args = parse_args(["--scenario", "root-n", "--subsample-ratio", "0.5"])
    assert args.scenario == "root-n"
    assert args.subsample_ratio == 0.5


def test_parse_args_unknown_scenario_rejected():
    with pytest.raises(SystemExit):
        parse_args(["--scenario", "totally-made-up"])


def test_parse_args_force_and_dry_run():
    args = parse_args(["--scenario", "monopoly", "--force", "--dry-run"])
    assert args.force is True
    assert args.dry_run is True


def test_parse_args_explicit_ranks():
    args = parse_args(["--scenario", "monopoly", "--ranks", "1,2,3"])
    assert args.ranks == "1,2,3"


def test_parse_args_rule_defaults():
    args = parse_args(["--scenario", "monopoly"])
    assert args.arm_ratio_cap == ARM_RATIO_CAP
    assert args.max_samples_cap == MAX_SAMPLES_CAP
    assert args.min_clicks_per_arm == MIN_CLICKS_PER_ARM
    assert args.group_col == "global_token_new"
    assert args.cache_values is False


def test_parse_args_rule_overrides():
    args = parse_args(["--scenario", "monopoly", "--arm-ratio-cap", "5", "--max-samples-cap", "50000",
                       "--min-clicks-per-arm", "10", "--group-col", "", "--cache-values"])
    assert (args.arm_ratio_cap, args.max_samples_cap, args.min_clicks_per_arm) == (5.0, 50_000, 10.0)
    assert args.group_col == ""
    assert args.cache_values is True


@pytest.mark.parametrize("argv", [
    ["--n-estimators", "501"],      # not divisible by econml's BLB subforest size
    ["--arm-ratio-cap", "0.5"],
])
def test_parse_args_rejects_invalid_rule_settings(argv):
    with pytest.raises(SystemExit):
        parse_args(["--scenario", "monopoly", *argv])


# ---------------------------------------------------------------------------
# build_pair / plan_for_pair
# ---------------------------------------------------------------------------

def test_build_pair_keeps_only_base_and_focal_and_balances_arms(make_pair):
    data = make_pair(1_000, 10_000)
    data.loc[:99, "advertiser_rank"] = 3   # rows of another rank must be dropped
    pair = build_pair(data, 7, arm_ratio_cap=3.0, random_state=0)
    counts = pair["advertiser_rank"].value_counts().to_dict()
    assert counts == {0: 900, 7: 2_700}


def test_build_pair_is_deterministic_given_the_seed(make_pair):
    data = make_pair(1_000, 10_000)
    a = build_pair(data, 7, random_state=1)
    b = build_pair(data, 7, random_state=1)
    assert a.equals(b)


def test_plan_for_pair_uses_arm_click_rates(make_pair):
    data = make_pair(2_000, 6_000, rate_base=0.02, rate_focal=0.04)
    pair = build_pair(data, 7)
    plan = plan_for_pair(pair, max_samples_cap=10_000, min_clicks_per_arm=5)
    assert (plan.n_base, plan.n_focal) == (2_000, 6_000)
    assert plan.expected_clicks_base == pair.loc[pair.advertiser_rank == 0, "is_clicked"].sum()
    assert plan.max_samples == 4_000


# ---------------------------------------------------------------------------
# SCENARIOS table
# ---------------------------------------------------------------------------

def test_scenarios_dispatch_keys():
    # If anyone adds/removes a scenario, this is a clear signal.
    assert set(SCENARIOS) == {"monopoly", "split", "split-root-n", "root-n"}


def test_scenarios_split_paths_interpolate_split():
    s = SCENARIOS["split"]
    assert s.needs_split is True
    p = s.data_path(7)
    assert "Split 7" in str(p)
    assert p.suffix == ".dta"


def test_scenarios_root_n_output_dir_uses_subsample_ratio():
    s = SCENARIOS["root-n"]
    out = s.output_dir(split=None, subsample_ratio=0.8)
    assert "Subsampling Ratio = 0.8" in str(out)


def test_scenarios_monopoly_does_not_need_split():
    assert SCENARIOS["monopoly"].needs_split is False
