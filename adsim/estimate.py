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
    --n-jobs N                 Cores for the M/E model grid search.
    --random-state SEED        Causal forest seed (default 42).
    --dry-run                  Print what would run, don't fit.
    --focal-cap-ratio R        Cap each rank's rows at R x the base ad's
                               rows before fitting (default 3.0).
    --max-samples-cap N        Ceiling on the per-tree sample count
                               (default 100_000).

Most causal-forest hyperparameters (max_samples, min_samples_split,
min_samples_leaf's starting value, max_depth) are derived per rank from
that rank's own sample size rather than fixed -- see
docs/hyperparameter_choices.txt and adsim.hyperparams.cf_hyperparams.

Output:
    results/Full Model/<scenario_dir>/CF - Rank {r}.pkl

Override RESULTS_DIR / DATA_DIR via the `ADSIM_RESULTS_DIR` /
`ADSIM_DATA_DIR` env vars.
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from econml.dml import CausalForestDML
from sklearn.base import clone

from adsim import config
from adsim.hyperparams import (
    cf_hyperparams,
    e_model_best_estimator,
    e_model_leaf_grid,
    m_model_best_estimator,
    m_model_leaf_grid,
)
from adsim.paths import DATA_DIR, RESULTS_DIR
from adsim.simulation_steps import define_xyt, extract_ranks, prepare_data


log = logging.getLogger("adsim.estimate")


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

def fit_one_rank(
    *,
    rank: int,
    data: pd.DataFrame,
    out_path: Path,
    n_jobs: int,
    random_state: int,
    n_estimators: int = 500,
    max_depth: int = 3,
    focal_cap_ratio: float = 3.0,
    max_samples_cap: int = 100_000,
) -> None:
    """Fit one CausalForestDML for one advertiser rank and save it.

    Every causal-forest hyperparameter except min_samples_leaf and
    min_var_fraction_leaf (the only two worth cross-validating) is derived
    from this pair's own (n_base, n_focal) via adsim.hyperparams.cf_hyperparams
    -- see docs/hyperparameter_choices.txt for the full derivation. The
    focal arm is also subsampled down to `focal_cap_ratio` x the base arm
    before fitting: past that ratio the majority arm barely reduces
    Var(tau_hat) but still consumes most of each tree's sample budget.
    """
    base_mask = data["advertiser_rank"] == 0
    focal_mask = data["advertiser_rank"] == rank
    n_base = int(base_mask.sum())
    n_focal = int(focal_mask.sum())

    hp = cf_hyperparams(
        n_base, n_focal,
        focal_cap_ratio=focal_cap_ratio,
        max_samples_cap=max_samples_cap,
        n_estimators=n_estimators,
        max_depth=max_depth,
    )

    focal_df = data[focal_mask]
    if hp.n_focal_used < n_focal:
        focal_df = focal_df.sample(n=hp.n_focal_used, random_state=random_state)

    df = pd.concat([data[base_mask], focal_df]).reset_index(drop=True)
    X, Y, T = define_xyt(df)
    T = T.apply(lambda x: 0 if x == 0 else 1)

    click_rate = float(Y.mean())
    rare_arm_rate = min(n_base, hp.n_focal_used) / hp.n_total
    m_leaf_grid = m_model_leaf_grid(hp.n_total, click_rate)
    e_leaf_grid = e_model_leaf_grid(hp.n_total, rare_arm_rate)

    t0 = time.perf_counter()
    best_params_e, best_estimator_e = e_model_best_estimator(X, T, e_leaf_grid, n_jobs=n_jobs)
    best_params_m, best_estimator_m = m_model_best_estimator(X, Y, m_leaf_grid, n_jobs=n_jobs)
    log.info(
        "rank=%d e/m grid search done in %.1fs (e_leaf=%s -> %s, m_leaf=%s -> %s)",
        rank, time.perf_counter() - t0, e_leaf_grid, best_params_e, m_leaf_grid, best_params_m,
    )

    cf = CausalForestDML(
        model_y=clone(best_estimator_m),
        model_t=clone(best_estimator_e),
        discrete_treatment=True,
        criterion="het",
        honest=True,
        n_jobs=n_jobs,
        n_estimators=hp.n_estimators,
        min_samples_split=hp.min_samples_split,
        min_samples_leaf=hp.min_samples_leaf_grid[0],
        min_var_fraction_leaf=None,
        max_depth=hp.max_depth,
        max_samples=hp.max_samples,
        min_balancedness_tol=hp.min_balancedness_tol,
        random_state=random_state,
        verbose=0,
    )

    t0 = time.perf_counter()
    cf.tune(Y=Y, T=T, X=X, params={
        "min_samples_leaf": hp.min_samples_leaf_grid,
        "min_var_fraction_leaf": hp.min_var_fraction_leaf_grid,
    })
    log.info("rank=%d tune done in %.1fs", rank, time.perf_counter() - t0)

    t0 = time.perf_counter()
    cf.fit(Y=Y, T=T, X=X, inference="blb", cache_values=True)
    log.info("rank=%d fit done in %.1fs", rank, time.perf_counter() - t0)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(cf, out_path)
    log.info("rank=%d saved -> %s", rank, out_path)


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

    p.add_argument("--n-estimators", type=int, default=500,
                   help="Trees in the final causal forest. See "
                        "docs/hyperparameter_choices.txt.")
    p.add_argument("--focal-cap-ratio", type=float, default=3.0,
                   help="Cap the focal rank's rows at this multiple of the "
                        "base ad's rows before fitting. See "
                        "docs/hyperparameter_choices.txt.")
    p.add_argument("--max-samples-cap", type=int, default=100_000,
                   help="Ceiling on the per-tree sample count "
                        "(max_samples = min(this, pair_N // 2)). See "
                        "docs/hyperparameter_choices.txt.")
    p.add_argument("--max-depth", type=int, default=3,
                   help="Safety cap only; min-samples-split/leaf are expected "
                        "to bind well before this depth is reached.")

    p.add_argument("--dry-run", action="store_true")
    p.add_argument("-v", "--verbose", action="count", default=0)

    args = p.parse_args(argv)

    scenario = SCENARIOS[args.scenario]
    if scenario.needs_split and args.split is None:
        p.error(f"--scenario {args.scenario} requires --split")
    if not scenario.needs_split and args.split is not None:
        log.warning("--split %s ignored for scenario %s", args.split, args.scenario)
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
            log.info("rank=%d DRY-RUN -> would write %s", rank, out_path)
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
                max_depth=args.max_depth,
                focal_cap_ratio=args.focal_cap_ratio,
                max_samples_cap=args.max_samples_cap,
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
