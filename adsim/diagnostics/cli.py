"""Referee diagnostics for the per-rank causal forests.

Runs the identification tests and model diagnostics of the revision plan
(sections 4.2 and 4.3, plus the item 2.2 placebo-monopoly check) for one
scenario and writes CSV + LaTeX tables and figures.

Usage:
    # Everything cheap-to-moderate for the 10 most-shown ads in the monopoly
    python -m adsim.diagnostics --scenario monopoly

    # Only the identification tests, for three ranks
    python -m adsim.diagnostics --scenario monopoly --checks identification --ranks 1,2,3

    # A split platform's forests
    python -m adsim.diagnostics --scenario split --split 7 --checks model

    # Winner's-curse check against a second monopoly fit (different seed)
    python -m adsim.diagnostics --scenario monopoly --checks placebo-monopoly \
        --placebo-forest-dir "/path/to/placebo/results/Full Model/Monopoly"

Checks (`--checks` takes names, groups `identification` / `model`, or `all`):
    identification  propensity-info, balance, placebo-outcome, refuters,
                    sensitivity, overlap
    model           nuisance, cate-validation, estimators, stability,
                    calibration, out-of-range, placebo-monopoly

Per-rank checks run on the (base ad, rank) pair each saved forest was fit
on and reuse its nuisance settings (default settings if the forest is
missing). `refuters`, `cate-validation`, `estimators` and `stability`
refit forests with `--refit-trees` trees. `calibration` and
`out-of-range` need every rank's forest plus m1.pkl / e1.pkl.

Output: RESULTS_DIR/Full Model/Diagnostics/<scenario dir>/<check>/<table>.{csv,tex}
and <figure>.{pdf,png}, plus manifest.json (settings, ranks, skipped and
failed checks). Paths follow ADSIM_DATA_DIR / ADSIM_RESULTS_DIR.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from warnings import filterwarnings

import joblib
import matplotlib
import numpy as np
import pandas as pd

from adsim import config
from adsim.diagnostics import identification, model_checks
from adsim.diagnostics.data import (
    DEFAULT_CELL_COLS,
    FEATURE_COLS,
    RankData,
    Settings,
    diagnosable_ranks,
    load_estimation_data,
    load_rank_forest,
    support_buckets,
    top_ranks,
)
from adsim.diagnostics.output import FIGURES, save_figure, write_table
from adsim.estimate import SCENARIOS, maybe_subsample
from adsim.paths import DATA_DIR, RESULTS_DIR

log = logging.getLogger("adsim.diagnostics")


@dataclass(frozen=True)
class Check:
    func: Callable
    group: str   # "identification" or "model"
    scope: str   # "rank" (one pair at a time) or "pipeline" (all forests at once)
    plan: str    # revision-plan reference


CHECKS: dict[str, Check] = {
    "propensity-info": Check(identification.propensity_information, "identification", "rank", "4.2.1"),
    "balance": Check(identification.covariate_balance, "identification", "rank", "4.2.2"),
    "placebo-outcome": Check(identification.placebo_outcomes, "identification", "rank", "4.2.3"),
    "refuters": Check(identification.refutation_tests, "identification", "rank", "4.2.4"),
    "sensitivity": Check(identification.ovb_sensitivity, "identification", "rank", "4.2.5"),
    "overlap": Check(identification.overlap, "identification", "rank", "4.2.6"),
    "nuisance": Check(model_checks.nuisance_scores, "model", "rank", "4.3"),
    "cate-validation": Check(model_checks.cate_validation, "model", "rank", "4.3"),
    "estimators": Check(model_checks.estimator_comparison, "model", "rank", "4.3"),
    "stability": Check(model_checks.stability, "model", "rank", "R3 data"),
    "calibration": Check(model_checks.level_calibration, "model", "pipeline", "4.3"),
    "out-of-range": Check(model_checks.out_of_range, "model", "pipeline", "4.3, 2.9"),
    "placebo-monopoly": Check(model_checks.placebo_monopoly, "model", "pipeline", "2.2"),
}


def select_checks(spec: str) -> list[str]:
    selected: list[str] = []
    for token in (t.strip() for t in spec.split(",") if t.strip()):
        if token == "all":
            selected += list(CHECKS)
        elif token in ("identification", "model"):
            selected += [name for name, c in CHECKS.items() if c.group == token]
        elif token in CHECKS:
            selected.append(token)
        else:
            raise ValueError(f"unknown check {token!r}; choose from {sorted(CHECKS)}, identification, model, all")
    return list(dict.fromkeys(selected))


class Results:
    """Collects tables per check, plus skipped / failed checks for the manifest."""

    def __init__(self):
        self.tables: dict[str, dict[str, list[pd.DataFrame]]] = {}
        self.skipped: list[dict] = []
        self.failed: list[dict] = []

    def add(self, check: str, tables: dict[str, pd.DataFrame], **columns) -> None:
        for name, df in tables.items():
            if df is None or df.empty:
                continue
            df = df.copy()
            for i, (col, value) in enumerate(columns.items()):
                df.insert(i, col, value)
            self.tables.setdefault(check, {}).setdefault(name, []).append(df)

    def write(self, out_dir: Path) -> list[Path]:
        written = []
        for check, tables in self.tables.items():
            for name, parts in tables.items():
                df = pd.concat(parts, ignore_index=True)
                written += write_table(df, out_dir / check, name)
                if name in FIGURES:
                    try:
                        written += save_figure(FIGURES[name](df), out_dir / check, name)
                    except Exception:
                        log.exception("figure %s failed", name)
        return written


# ----------------------------------------------------------------------
# Runners
# ----------------------------------------------------------------------

def run_rank_checks(checks: list[str], data: pd.DataFrame, ranks: list[int], forest_dir: Path,
                    settings: Settings, results: Results, forest_found: dict) -> None:
    support = support_buckets(data)
    for rank in ranks:
        rd = RankData.build(data, rank, forest_dir, settings)
        forest_found[rank] = rd.cf is not None
        if rd.cf is None:
            log.warning("rank=%d: no saved forest in %s; using default nuisance settings", rank, forest_dir)
        log.info("rank=%d: pair of %d rows (%d focal)", rank, len(rd.pair), int(rd.T.sum()))
        for name in checks:
            t0 = time.perf_counter()
            try:
                results.add(name, CHECKS[name].func(rd), rank=rank, support=support.get(rank, ""))
                log.info("rank=%d %s done in %.1fs", rank, name, time.perf_counter() - t0)
            except Exception as exc:
                log.exception("rank=%d %s FAILED", rank, name)
                results.failed.append({"check": name, "rank": rank, "error": repr(exc)})


def sample_rows(data: pd.DataFrame, max_rows: int, group_col: str | None, seed: int) -> pd.DataFrame:
    """At most ~`max_rows` rows, keeping whole users."""
    if len(data) <= max_rows:
        return data
    rng = np.random.default_rng(seed)
    if group_col and group_col in data:
        users = data[group_col].unique()
        keep = users[rng.random(len(users)) < max_rows / len(data)]
        return data[data[group_col].isin(keep)]
    return data.sample(n=max_rows, random_state=seed)


def load_forests(forest_dir: Path, ranks: list[int]) -> dict[int, object]:
    forests = {}
    for rank in ranks:
        cf = load_rank_forest(forest_dir, rank)
        if cf is not None:
            forests[rank] = cf
    return forests


def run_pipeline_checks(checks: list[str], args, data: pd.DataFrame, forest_dir: Path,
                        results: Results) -> None:
    """calibration / out-of-range / placebo-monopoly: all forests on a sample of impressions."""
    helpers_dir = args.helpers_dir or RESULTS_DIR / "Full Model"
    monopoly_dir = args.monopoly_forest_dir or SCENARIOS["monopoly"].output_dir(None, args.subsample_ratio)
    needed = [helpers_dir / "m1.pkl", helpers_dir / "e1.pkl", monopoly_dir / "CF - Rank 1.pkl"]
    missing = [str(p) for p in needed if not p.is_file()]
    if missing:
        for name in checks:
            results.skipped.append({"check": name, "reason": f"missing {missing}"})
        log.warning("skipping %s: missing %s", checks, missing)
        return

    eval_data = data
    if args.eval_data is not None:
        eval_data = load_estimation_data(args.eval_data, group_col=args.group_col or None, visit_col=None)
    sample = sample_rows(eval_data, args.max_eval_rows, args.group_col, args.random_state).reset_index(drop=True)
    X = sample[FEATURE_COLS]
    helpers = {"m1": joblib.load(helpers_dir / "m1.pkl"), "e1": joblib.load(helpers_dir / "e1.pkl")}
    y0 = model_checks.base_ad_ctr(X, load_rank_forest(monopoly_dir, 1), helpers)

    forests = load_forests(forest_dir, diagnosable_ranks(data))
    log.info("pipeline checks: %d forests, %d sampled impressions", len(forests), len(sample))
    ctrs = None
    if {"out-of-range", "placebo-monopoly"} & set(checks):
        ctrs = model_checks.pipeline_ctrs(X, forests, y0)

    for name in checks:
        t0 = time.perf_counter()
        try:
            if name == "calibration":
                preds = model_checks.shown_ad_predictions(sample, forests, y0, support_buckets(data))
                tables = model_checks.level_calibration(preds, n_bins=args.n_bins)
            elif name == "out-of-range":
                tables = model_checks.out_of_range(ctrs, y0)
            else:
                tables = _placebo_monopoly(args, sample, X, ctrs, y0, helpers, results)
                if tables is None:
                    continue
            results.add(name, tables)
            log.info("%s done in %.1fs", name, time.perf_counter() - t0)
        except Exception as exc:
            log.exception("%s FAILED", name)
            results.failed.append({"check": name, "rank": None, "error": repr(exc)})


def _placebo_monopoly(args, sample, X, ctrs, y0, helpers, results):
    if args.placebo_forest_dir is None:
        results.skipped.append({"check": "placebo-monopoly", "reason": "no --placebo-forest-dir"})
        log.warning("placebo-monopoly skipped: pass --placebo-forest-dir")
        return None
    placebo = load_forests(args.placebo_forest_dir, list(ctrs.columns))
    if 1 not in placebo:
        raise FileNotFoundError(f"no CF - Rank 1.pkl in {args.placebo_forest_dir}")
    placebo_y0 = model_checks.base_ad_ctr(X, placebo[1], helpers)
    vals_path = args.vals_data or DATA_DIR / "Full Model" / "Advertiser Valuations.dta"
    valuations = None
    if vals_path.is_file():
        vals = pd.read_stata(vals_path)
        valuations = vals.set_index("advertiser_rank")["advertiser_val_cents"]
    group_col = args.group_col or None
    groups = sample[group_col].to_numpy() if group_col and group_col in sample else None
    return model_checks.placebo_monopoly(
        sample, ctrs, y0, model_checks.pipeline_ctrs(X, placebo, placebo_y0), placebo_y0,
        valuations=valuations, groups=groups, max_ads=config.max_ads_per_page,
    )


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="adsim.diagnostics", description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog="\n\n".join(__doc__.split("\n\n")[1:]))
    p.add_argument("--scenario", required=True, choices=sorted(SCENARIOS))
    p.add_argument("--split", type=int, default=None)
    p.add_argument("--subsample-ratio", type=float, default=0.8, help="Only used for --scenario root-n.")
    p.add_argument("--subsample-seed", type=int, default=42,
                   help="Seed of the root-n subsample; must match the one used by adsim.estimate.")
    p.add_argument("--checks", default="all", help="Comma-separated checks or groups (default: all).")
    p.add_argument("--ranks", default=None, help="Comma-separated ranks (default: the --top-k most shown).")
    p.add_argument("--top-k", type=int, default=10, help="Diagnose the K most-shown ads (default 10).")

    p.add_argument("--data", type=Path, default=None, help="Estimation .dta (default: the scenario's).")
    p.add_argument("--forest-dir", type=Path, default=None, help="Per-rank forests (default: the scenario's).")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Default: RESULTS_DIR/Full Model/Diagnostics/<scenario dir>.")
    p.add_argument("--eval-data", type=Path, default=None,
                   help="Impressions for calibration / out-of-range / placebo-monopoly "
                        "(default: the estimation data, i.e. in-sample).")
    p.add_argument("--helpers-dir", type=Path, default=None, help="Directory with m1.pkl, e1.pkl.")
    p.add_argument("--monopoly-forest-dir", type=Path, default=None,
                   help="Forests whose rank-1 model gives y0, as in the simulation (default: monopoly).")
    p.add_argument("--placebo-forest-dir", type=Path, default=None,
                   help="A second monopoly fit (other seed or resample) for placebo-monopoly.")
    p.add_argument("--vals-data", type=Path, default=None, help="Advertiser Valuations.dta.")

    p.add_argument("--group-col", default="global_token_new", help="User id column ('' for none).")
    p.add_argument("--visit-col", default="user_visit_no",
                   help="Within-user visit order, for the lagged-click placebo ('' for none).")
    p.add_argument("--cell-cols", default=",".join(DEFAULT_CELL_COLS),
                   help="Columns defining page-time cells (default: website x day x hour).")
    p.add_argument("--n-folds", type=int, default=3)
    p.add_argument("--holdout-frac", type=float, default=0.3, help="Held-out user share for validation.")
    p.add_argument("--refit-trees", type=int, default=100, help="Trees per refit forest (divisible by 4).")
    p.add_argument("--n-permutations", type=int, default=10,
                   help="Permutation refits per history block in propensity-info (0 skips the test).")
    p.add_argument("--max-eval-rows", type=int, default=100_000,
                   help="Impressions sampled (by user) for the pipeline checks.")
    p.add_argument("--n-bins", type=int, default=10, help="Quantile bins for calibration.")
    p.add_argument("--n-jobs", type=int, default=config.n_jobs)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("-v", "--verbose", action="count", default=0)

    args = p.parse_args(argv)
    scenario = SCENARIOS[args.scenario]
    if scenario.needs_split and args.split is None:
        p.error(f"--scenario {args.scenario} requires --split")
    if args.refit_trees % 4 != 0:
        p.error("--refit-trees must be divisible by 4 (econml BLB subforest size)")
    try:
        args.check_names = select_checks(args.checks)
    except ValueError as exc:
        p.error(str(exc))
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    matplotlib.use("Agg")
    filterwarnings("ignore", category=UserWarning)
    filterwarnings("ignore", category=FutureWarning)

    scenario = SCENARIOS[args.scenario]
    forest_dir = args.forest_dir or scenario.output_dir(args.split, args.subsample_ratio)
    out_dir = args.out_dir or (RESULTS_DIR / "Full Model" / "Diagnostics"
                               / scenario.out_subdir_template.format(split=args.split,
                                                                     subsample_ratio=args.subsample_ratio))
    settings = Settings(
        group_col=args.group_col or None,
        cell_cols=tuple(c.strip() for c in args.cell_cols.split(",") if c.strip()),
        n_folds=args.n_folds, n_jobs=args.n_jobs, random_state=args.random_state,
        refit_trees=args.refit_trees, holdout_frac=args.holdout_frac, n_permutations=args.n_permutations,
    )

    t0 = time.perf_counter()
    data = load_estimation_data(args.data or scenario.data_path(args.split),
                                group_col=settings.group_col, visit_col=args.visit_col or None)
    data = maybe_subsample(data, scenario, args.subsample_ratio, args.subsample_seed)
    ranks = [int(r) for r in args.ranks.split(",")] if args.ranks else top_ranks(data, args.top_k)
    log.info("scenario=%s forests=%s out=%s ranks=%s checks=%s",
             scenario.name, forest_dir, out_dir, ranks, args.check_names)

    results, forest_found = Results(), {}
    rank_checks = [c for c in args.check_names if CHECKS[c].scope == "rank"]
    pipeline_checks = [c for c in args.check_names if CHECKS[c].scope == "pipeline"]
    if rank_checks:
        run_rank_checks(rank_checks, data, ranks, forest_dir, settings, results, forest_found)
    if pipeline_checks:
        run_pipeline_checks(pipeline_checks, args, data, forest_dir, results)

    written = results.write(out_dir)
    manifest = {
        "scenario": scenario.name, "split": args.split, "forest_dir": str(forest_dir),
        "ranks": ranks, "forest_found": forest_found, "checks": args.check_names,
        "plan_sections": {c: CHECKS[c].plan for c in args.check_names},
        "settings": {k: (list(v) if isinstance(v, tuple) else v) for k, v in vars(settings).items()},
        "skipped": results.skipped, "failed": results.failed,
        "elapsed_seconds": round(time.perf_counter() - t0, 1),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    log.info("wrote %d files to %s (%d failed, %d skipped)",
             len(written) + 1, out_dir, len(results.failed), len(results.skipped))
    return 0 if not results.failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
