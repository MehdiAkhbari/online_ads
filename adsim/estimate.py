"""Fit per-rank causal forests for one scenario.

Replaces the family of one-off estimation*.py scripts. Designed to be
HPC-friendly: one scenario per job, auto-resumes by skipping ranks
whose forest pickle already exists, structured per-rank logs.

Usage:
    # Monopoly
    python -m adsim.estimate --scenario monopoly

    # Duopoly / split (one job per split)
    python -m adsim.estimate --scenario split --split 7
    python -m adsim.estimate --scenario split --split 8

    # Root-N split
    python -m adsim.estimate --scenario split-root-n --split 6

    # Sample-size scenario (subsampled monopoly data)
    python -m adsim.estimate --scenario root-n --subsample-ratio 0.8

Common flags:
    --ranks 1,2,3              Only fit these ranks (comma-separated).
    --ranks-filter "rank > 10" Eval'd against `rank`. Recovers the
                               estimation2.py "continuation" behavior.
    --limit N                  Cap to first N ranks (after filtering).
                               Useful for HPC walltime tuning.
    --force                    Re-fit even if the output pkl exists.
    --n-jobs N                 Cores for the nuisance models and the forest.
    --random-state SEED        Seed for arm subsampling, CV folds and the
                               forest (default 42).
    --dry-run                  Print each rank's hyperparameter plan, don't fit.

Hyperparameter rule (see adsim.hyperparams and docs/hyperparameter_choices.txt):
    --arm-ratio-cap R          Neither arm may exceed R x the other (default 3).
    --max-samples-cap N        Ceiling on rows per tree (default 200,000).
    --min-clicks-per-arm K     Expected clicks from each arm per leaf (default 5).
    --min-var-fraction-leaf F  econml leaf treatment-variance floor (default 0.1).
    --n-estimators N           Trees per forest (default 500).
    --group-col NAME           User-id column for grouped cross-fitting
                               (default global_token_new; '' disables).
    --cache-values             Keep econml's cached training data in the pickle
                               (needed for cf.summary(); ~120 MB per large pair).

Output:
    results/Full Model/<scenario_dir>/CF - Rank {r}.pkl
    Each saved forest carries its plan as `cf.adsim_plan_` (a dict).

Override RESULTS_DIR / DATA_DIR via the `ADSIM_RESULTS_DIR` /
`ADSIM_DATA_DIR` env vars.
"""

from __future__ import annotations

import argparse
import logging
import pickle
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from econml.dml import CausalForestDML

from adsim import config
from adsim.hyperparams import (
    ARM_RATIO_CAP,
    MAX_SAMPLES_CAP,
    MIN_CLICKS_PER_ARM,
    MIN_VAR_FRACTION_LEAF,
    N_ESTIMATORS,
    CFPlan,
    arm_sizes_after_cap,
    cf_hyperparams,
    e_model_best_estimator,
    e_model_leaf_grid,
    m_model_best_estimator,
    m_model_leaf_grid,
)
from adsim.paths import DATA_DIR, RESULTS_DIR
from adsim.simulation_steps import define_xyt, extract_ranks, prepare_data


log = logging.getLogger("adsim.estimate")

DEFAULT_GROUP_COL = "global_token_new"
# Below ~10 users in an arm, StratifiedGroupKFold (used by both the nuisance
# grid search and econml's cross-fitting) sometimes leaves a training fold
# with no rows of that arm, which econml rejects.
MIN_USERS_PER_ARM_FOR_GROUPS = 20


# ----------------------------------------------------------------------
# Scenario specs
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class Scenario:
    name: str                 # CLI key
    needs_split: bool         # whether --split is required
    data_relpath_template: str  # path under DATA_DIR, may interpolate {split}
    out_subdir_template: str    # path under RESULTS_DIR/Full Model

    def data_path(self, split: int | None) -> Path:
        return DATA_DIR / self.data_relpath_template.format(split=split)

    def output_dir(self, split: int | None, subsample_ratio: float) -> Path:
        return RESULTS_DIR / "Full Model" / self.out_subdir_template.format(
            split=split, subsample_ratio=subsample_ratio,
        )


SCENARIOS: dict[str, Scenario] = {
    "monopoly": Scenario(
        name="monopoly",
        needs_split=False,
        data_relpath_template="Full Model/Estimation Data - Full Model - Monopoly.dta",
        out_subdir_template="Monopoly",
    ),
    "split": Scenario(
        name="split",
        needs_split=True,
        data_relpath_template="Full Model/Estimation Data - Full Model - Split {split}.dta",
        out_subdir_template="Split {split}",
    ),
    "split-root-n": Scenario(
        name="split-root-n",
        needs_split=True,
        data_relpath_template="Full Model/Estimation Data - Full Model - Split {split} - Root N.dta",
        out_subdir_template="Split {split} - Root N",
    ),
    "root-n": Scenario(
        name="root-n",
        needs_split=False,
        data_relpath_template="Full Model/Estimation Data - Full Model - Monopoly.dta",
        out_subdir_template="Root N - Random/Subsampling Ratio = {subsample_ratio}",
    ),
}


# ----------------------------------------------------------------------
# Estimation step
# ----------------------------------------------------------------------

def build_pair(
    data: pd.DataFrame,
    rank: int,
    *,
    arm_ratio_cap: float = ARM_RATIO_CAP,
    random_state: int = 42,
) -> pd.DataFrame:
    """Rows of the base ad (rank 0) and `rank`, with the larger arm randomly
    subsampled so neither arm exceeds `arm_ratio_cap` x the other.

    `data` must already have gone through `prepare_data` (base ad coded 0).
    """
    base = data[data["advertiser_rank"] == 0]
    focal = data[data["advertiser_rank"] == rank]
    n_base, n_focal = arm_sizes_after_cap(len(base), len(focal), arm_ratio_cap)
    if n_base < len(base):
        base = base.sample(n=n_base, random_state=random_state)
    if n_focal < len(focal):
        focal = focal.sample(n=n_focal, random_state=random_state)
    return pd.concat([base, focal]).reset_index(drop=True)


def plan_for_pair(
    pair: pd.DataFrame,
    *,
    max_samples_cap: int = MAX_SAMPLES_CAP,
    min_clicks_per_arm: float = MIN_CLICKS_PER_ARM,
    min_var_fraction_leaf: float = MIN_VAR_FRACTION_LEAF,
    n_estimators: int = N_ESTIMATORS,
) -> CFPlan:
    is_base = pair["advertiser_rank"] == 0
    clicks = pair["is_clicked"]
    return cf_hyperparams(
        n_base=int(is_base.sum()),
        n_focal=int((~is_base).sum()),
        click_rate_base=float(clicks[is_base].mean()) if is_base.any() else 0.0,
        click_rate_focal=float(clicks[~is_base].mean()) if (~is_base).any() else 0.0,
        max_samples_cap=max_samples_cap,
        min_clicks_per_arm=min_clicks_per_arm,
        min_var_fraction_leaf=min_var_fraction_leaf,
        n_estimators=n_estimators,
    )


def pair_groups(pair: pd.DataFrame, T: pd.Series, group_col: str | None) -> np.ndarray | None:
    """User ids for grouped cross-fitting, or None when `group_col` is absent
    or an arm has fewer than MIN_USERS_PER_ARM_FOR_GROUPS users."""
    if not group_col or group_col not in pair.columns:
        return None
    users_per_arm = pair.groupby(T)[group_col].nunique()
    if len(users_per_arm) == 2 and users_per_arm.min() >= MIN_USERS_PER_ARM_FOR_GROUPS:
        return pair[group_col].to_numpy()
    return None


def build_forest(plan: CFPlan, model_y, model_t, *, n_jobs: int, random_state: int) -> CausalForestDML:
    """An unfitted CausalForestDML with `plan`'s settings (the only place the
    estimator is configured, so diagnostics refit exactly what is saved)."""
    return CausalForestDML(
        model_y=model_y,
        model_t=model_t,
        discrete_treatment=True,
        criterion="het",
        honest=True,
        n_estimators=plan.n_estimators,
        max_samples=plan.max_samples,
        min_samples_leaf=plan.min_samples_leaf,
        min_samples_split=plan.min_samples_split,
        min_var_fraction_leaf=plan.min_var_fraction_leaf,
        max_depth=None,
        n_jobs=n_jobs,
        random_state=random_state,
        verbose=0,
    )


def fit_one_rank(
    *,
    rank: int,
    data: pd.DataFrame,
    out_path: Path,
    n_jobs: int,
    random_state: int,
    n_estimators: int = N_ESTIMATORS,
    arm_ratio_cap: float = ARM_RATIO_CAP,
    max_samples_cap: int = MAX_SAMPLES_CAP,
    min_clicks_per_arm: float = MIN_CLICKS_PER_ARM,
    min_var_fraction_leaf: float = MIN_VAR_FRACTION_LEAF,
    group_col: str | None = DEFAULT_GROUP_COL,
    cache_values: bool = False,
) -> CFPlan:
    """Fit one CausalForestDML for (base ad, `rank`) and save it.

    All causal-forest settings come from `adsim.hyperparams.cf_hyperparams`
    applied to this pair's arm sizes and click rates. The nuisance models
    are grid-searched over leaf size only (cross-validated, grouped by user
    when `group_col` is present in `data`).
    """
    pair = build_pair(data, rank, arm_ratio_cap=arm_ratio_cap, random_state=random_state)
    plan = plan_for_pair(
        pair,
        max_samples_cap=max_samples_cap,
        min_clicks_per_arm=min_clicks_per_arm,
        min_var_fraction_leaf=min_var_fraction_leaf,
        n_estimators=n_estimators,
    )
    log.info(
        "rank=%d plan: base=%d focal=%d (exp. clicks %.1f / %.1f) max_samples=%d "
        "min_samples_leaf=%d -> up to %.1f leaves per honest half",
        rank, plan.n_base, plan.n_focal, plan.expected_clicks_base, plan.expected_clicks_focal,
        plan.max_samples, plan.min_samples_leaf, plan.max_leaves_per_tree,
    )

    X, Y, T = define_xyt(pair)
    T = (T != 0).astype(int)
    groups = pair_groups(pair, T, group_col)
    if groups is None and group_col and group_col in pair.columns:
        log.warning("rank=%d: an arm has fewer than %d users; cross-fitting by row",
                    rank, MIN_USERS_PER_ARM_FOR_GROUPS)

    rare_arm_rate = min(plan.n_base, plan.n_focal) / plan.n_total
    t0 = time.perf_counter()
    best_params_e, model_t = e_model_best_estimator(
        X, T, e_model_leaf_grid(plan.n_total, rare_arm_rate),
        n_jobs=n_jobs, groups=groups, random_state=random_state,
    )
    best_params_m, model_y = m_model_best_estimator(
        X, Y, m_model_leaf_grid(plan.n_total, float(Y.mean())),
        n_jobs=n_jobs, groups=groups, random_state=random_state,
    )
    log.info("rank=%d nuisance grid search done in %.1fs (e=%s, m=%s)",
             rank, time.perf_counter() - t0, best_params_e, best_params_m)

    cf = build_forest(plan, model_y, model_t, n_jobs=n_jobs, random_state=random_state)

    t0 = time.perf_counter()
    cf.fit(Y=Y, T=T, X=X, groups=groups, inference="blb", cache_values=cache_values)
    log.info("rank=%d fit done in %.1fs", rank, time.perf_counter() - t0)

    cf.adsim_plan_ = {
        **plan.as_dict(),
        "rank": rank,
        "arm_ratio_cap": arm_ratio_cap,
        "min_clicks_per_arm": min_clicks_per_arm,
        "random_state": random_state,
        "grouped_by": group_col if groups is not None else None,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(cf, out_path)
    log.info("rank=%d saved -> %s", rank, out_path)
    return plan


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------

def select_ranks(
    discovered: list[int],
    *,
    explicit: list[int] | None,
    expr: str | None,
    limit: int | None,
) -> list[int]:
    selected = list(discovered)
    if explicit is not None:
        selected = [r for r in selected if r in explicit]
    if expr:
        selected = [r for r in selected if eval(expr, {"__builtins__": {}}, {"rank": r})]
    if limit is not None:
        selected = selected[:limit]
    return selected


def load_data_and_ranks(scenario: Scenario, split: int | None) -> tuple[pd.DataFrame, list[int]]:
    data_path = scenario.data_path(split)
    log.info("loading data from %s", data_path)
    data = pd.read_stata(data_path)
    prepare_data(data, base_ad=50, max_ad=100)

    ranks = extract_ranks(data)

    # Persist the ranks_list seen in this dataset for downstream consumers.
    rl_path = RESULTS_DIR / "main_scenario" / "ranks_list.pickle"
    rl_path.parent.mkdir(parents=True, exist_ok=True)
    with open(rl_path, "wb") as f:
        pickle.dump(ranks, f)
    log.info("wrote %d ranks to %s", len(ranks), rl_path)

    # Drop the base ad (rank 0) and the >max-ad fringe entry for fitting.
    ranks_for_fit = list(ranks)
    if ranks_for_fit and ranks_for_fit[0] == 0:
        ranks_for_fit.pop(0)
    if ranks_for_fit:
        ranks_for_fit.pop(-1)

    return data, ranks_for_fit


def maybe_subsample(data: pd.DataFrame, scenario: Scenario, ratio: float, seed: int) -> pd.DataFrame:
    if scenario.name != "root-n":
        return data
    rng = np.random.RandomState(seed)
    data = data.copy()
    data["__rand"] = rng.uniform(size=len(data))
    return data[data["__rand"] < ratio].drop(columns="__rand")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="adsim.estimate", description=__doc__.split("\n\n")[0])
    p.add_argument("--scenario", required=True, choices=sorted(SCENARIOS))
    p.add_argument("--split", type=int, default=None)
    p.add_argument("--subsample-ratio", type=float, default=0.8,
                   help="Only used for --scenario root-n.")

    p.add_argument("--ranks", type=str, default=None,
                   help="Comma-separated explicit ranks, e.g. '1,2,3'.")
    p.add_argument("--ranks-filter", type=str, default=None,
                   help="Python expression on `rank`, e.g. 'rank > 10'.")
    p.add_argument("--limit", type=int, default=None)

    p.add_argument("--force", action="store_true",
                   help="Re-fit even if the output pkl exists.")
    p.add_argument("--n-jobs", type=int, default=config.n_jobs)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--subsample-seed", type=int, default=42,
                   help="Seed for the root-n subsample.")

    p.add_argument("--n-estimators", type=int, default=N_ESTIMATORS,
                   help="Trees per forest; must be divisible by 4 (econml's BLB subforest size).")
    p.add_argument("--arm-ratio-cap", type=float, default=ARM_RATIO_CAP,
                   help="Subsample the larger arm to at most this multiple of the smaller one.")
    p.add_argument("--max-samples-cap", type=int, default=MAX_SAMPLES_CAP,
                   help="Ceiling on rows per tree; the rule uses min(this, N // 2).")
    p.add_argument("--min-clicks-per-arm", type=float, default=MIN_CLICKS_PER_ARM,
                   help="Expected clicks from each arm that every leaf must hold.")
    p.add_argument("--min-var-fraction-leaf", type=float, default=MIN_VAR_FRACTION_LEAF,
                   help="econml min_var_fraction_leaf: leaf treatment variance floor, "
                        "as a share of the pair's.")
    p.add_argument("--group-col", type=str, default=DEFAULT_GROUP_COL,
                   help="User-id column for grouped cross-fitting; '' to disable.")
    p.add_argument("--cache-values", action="store_true",
                   help="Store econml's cached training data in each pickle (needed for "
                        "cf.summary(); large).")

    p.add_argument("--dry-run", action="store_true",
                   help="Log each rank's hyperparameter plan without fitting.")
    p.add_argument("-v", "--verbose", action="count", default=0)

    args = p.parse_args(argv)

    scenario = SCENARIOS[args.scenario]
    if scenario.needs_split and args.split is None:
        p.error(f"--scenario {args.scenario} requires --split")
    if not scenario.needs_split and args.split is not None:
        log.warning("--split %s ignored for scenario %s", args.split, args.scenario)
    if args.n_estimators % 4 != 0:
        p.error("--n-estimators must be divisible by 4 (econml BLB subforest size)")
    if args.arm_ratio_cap < 1:
        p.error("--arm-ratio-cap must be >= 1")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    scenario = SCENARIOS[args.scenario]
    out_dir = scenario.output_dir(args.split, args.subsample_ratio)

    log.info("scenario=%s split=%s out=%s", scenario.name, args.split, out_dir)

    data, discovered = load_data_and_ranks(scenario, args.split)
    data = maybe_subsample(data, scenario, args.subsample_ratio, args.subsample_seed)

    group_col = args.group_col or None
    if group_col and group_col not in data.columns:
        log.warning("group column %r not in data; cross-fitting will treat rows as independent", group_col)
        group_col = None

    explicit = (
        [int(x) for x in args.ranks.split(",")]
        if args.ranks else None
    )
    ranks = select_ranks(
        discovered, explicit=explicit, expr=args.ranks_filter, limit=args.limit,
    )
    log.info("selected %d ranks for fitting", len(ranks))

    out_dir.mkdir(parents=True, exist_ok=True)

    n_done = n_skipped = n_failed = 0
    overall_t0 = time.perf_counter()

    for rank in ranks:
        out_path = out_dir / f"CF - Rank {rank}.pkl"
        if out_path.is_file() and not args.force:
            log.info("rank=%d SKIP (already exists): %s", rank, out_path)
            n_skipped += 1
            continue
        if args.dry_run:
            pair = build_pair(data, rank, arm_ratio_cap=args.arm_ratio_cap, random_state=args.random_state)
            plan = plan_for_pair(
                pair,
                max_samples_cap=args.max_samples_cap,
                min_clicks_per_arm=args.min_clicks_per_arm,
                min_var_fraction_leaf=args.min_var_fraction_leaf,
                n_estimators=args.n_estimators,
            )
            log.info("rank=%d DRY-RUN -> %s | %s", rank, out_path, plan)
            continue
        try:
            t0 = time.perf_counter()
            fit_one_rank(
                rank=rank,
                data=data,
                out_path=out_path,
                n_jobs=args.n_jobs,
                random_state=args.random_state,
                n_estimators=args.n_estimators,
                arm_ratio_cap=args.arm_ratio_cap,
                max_samples_cap=args.max_samples_cap,
                min_clicks_per_arm=args.min_clicks_per_arm,
                min_var_fraction_leaf=args.min_var_fraction_leaf,
                group_col=group_col,
                cache_values=args.cache_values,
            )
            log.info("rank=%d DONE in %.1fs", rank, time.perf_counter() - t0)
            n_done += 1
        except Exception:
            log.exception("rank=%d FAILED", rank)
            n_failed += 1

    elapsed = time.perf_counter() - overall_t0
    log.info(
        "summary: scenario=%s done=%d skipped=%d failed=%d elapsed=%.1fs",
        scenario.name, n_done, n_skipped, n_failed, elapsed,
    )
    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
