"""Fit ONE joint multi-treatment causal forest per scenario.

This is a joint-estimation counterpart to `adsim.estimate`. Where
`estimate.py::fit_one_rank` loops over every advertiser rank r and, for
each one:

    1) filters the data down to rows where advertiser_rank in {0 (base
       ad), r} only,
    2) collapses T to binary (0 = base ad, 1 = ad r),
    3) grid-searches an INDEPENDENT propensity model e_r(X) and outcome
       model m_r(X),
    4) fits an INDEPENDENT two-arm CausalForestDML,

...this script instead fits everything ONCE:

    1) keeps the FULL data (all ranks being estimated, not filtered to
       base-ad/rank-r pairs),
    2) leaves T = advertiser_rank as a single multi-valued categorical
       (0 = base ad = control, plus every other rank),
    3) grid-searches ONE multi-class propensity model e(X) = P(A=a|X)
       and ONE outcome model m(X) = E(y|X),
    4) fits ONE CausalForestDML with discrete_treatment=True and
       categories=[0, rank_1, ..., rank_K], which jointly estimates the
       WHOLE CATE vector tau(X) in a single forest.

This is the literal multi-treatment formulation in the appendix
(vector A, vector e(X), vector tau(X); eq. A.1-A.8) rather than K
separate pairwise comparisons against the base ad, and it removes the
~100x redundant nuisance-model grid searches that `estimate.py` repeats
per rank.

Single output file
-------------------
This script writes exactly ONE file per scenario:

    results/Full Model/<scenario_dir>/Joint CF.pkl

That file holds the one fitted CausalForestDML plus the ordered list
of ranks (a `JointForestArtifact`). There are no per-rank pickles on
disk at all.

Compatibility with the rest of the pipeline
--------------------------------------------
`adsim.config` and `adsim.simulation_steps` expect
`config.forests[rank].const_marginal_effect(X)` to return THAT rank's
CATE from a per-rank object. The joint forest's
`const_marginal_effect(X)` instead returns an (n_samples, n_ranks)
matrix in one call. Rather than writing that back out to disk as one
file per rank, `forests_dict_from_joint()` below loads the single
`Joint CF.pkl` once and hands back `{rank: RankView}` IN MEMORY --
`RankView` is a tiny pointer + column index, not a model, and every
RankView for a scenario shares one process-wide cache of the
underlying joint forest (see `_load_joint`), so building all ~100 of
them does not load ~100 copies into memory or touch disk again.
`config.py`'s loaders have been updated to call this helper -- see the
companion diff. `simulation_steps.py` needs no changes at all.

Usage (mirrors `adsim.estimate`)
---------------------------------
    # Monopoly
    python -m adsim.estimate_joint --scenario monopoly

    # Duopoly / split (one job per split)
    python -m adsim.estimate_joint --scenario split --split 7
    python -m adsim.estimate_joint --scenario split --split 8

    # Root-N split
    python -m adsim.estimate_joint --scenario split-root-n --split 6

    # Sample-size scenario (subsampled monopoly data)
    python -m adsim.estimate_joint --scenario root-n --subsample-ratio 0.8

Common flags:
    --ranks 1,2,3         Only include these ranks as treatment ARMS in
                           the joint fit (comma-separated). Unlike
                           estimate.py, this does NOT mean "separate
                           jobs" -- it's still one fit, just over fewer
                           treatment categories.
    --ranks-filter EXPR    Eval'd against `rank`, e.g. "rank > 10".
    --limit N              Cap to first N ranks (after filtering).
    --max-samples FLOAT    Per-tree subsample fraction for
                           CausalForestDML (default 0.01, same default
                           as estimate.py).
                           ** IMPORTANT: because the joint fit uses the
                           FULL dataset instead of each rank's small
                           two-arm subset, the same numeric max_samples
                           now implies a much larger ABSOLUTE number of
                           rows per tree. If fit time / memory balloons
                           relative to estimate.py, lower this. **
    --force                Re-fit even if Joint CF.pkl already exists.
    --n-jobs N             Cores for the M/E model grid search + forest.
    --random-state SEED    Causal forest seed (default 42).
    --dry-run              Print what would run, don't fit.

Output:
    results/Full Model/<scenario_dir>/Joint CF.pkl   (the ONLY file written)

Override RESULTS_DIR / DATA_DIR via the `ADSIM_RESULTS_DIR` /
`ADSIM_DATA_DIR` env vars (same as estimate.py, via adsim.paths).
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from econml.dml import CausalForestDML
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import GridSearchCV

from adsim.estimate import (
    SCENARIOS,
    Scenario,
    load_data_and_ranks,
    maybe_subsample,
    select_ranks,
)
from adsim.propensity_model import PropensityModel
from adsim.simulation_steps import (
    cf_param_grid,
    define_xyt,
    m_model_best_estimator,
    param_grid,
)

log = logging.getLogger("adsim.estimate_joint")

# ----------------------------------------------------------------------
# Per-rank view into a single jointly-fit forest
# ----------------------------------------------------------------------

# Process-wide cache so that loading many RankView pickles that all point
# at the same joint forest only loads that forest into memory once.
_JOINT_CACHE: dict[str, "JointForestArtifact"] = {}


def _load_joint(joint_path: Path) -> "JointForestArtifact":
    key = str(Path(joint_path).resolve())
    if key not in _JOINT_CACHE:
        log.info("loading joint forest -> %s", joint_path)
        _JOINT_CACHE[key] = joblib.load(joint_path)
    return _JOINT_CACHE[key]


@dataclass
class JointForestArtifact:
    """The single jointly-fit forest plus enough metadata to slice it."""

    cf: CausalForestDML
    ranks: list[int]  # non-base ranks, in the column order of cf.const_marginal_effect
    base_ad: int = 0

    def rank_column(self, rank: int) -> int:
        return self.ranks.index(rank)


@dataclass
class RankView:
    """Drop-in stand-in for a per-rank CausalForestDML.

    This is NOT a model -- it is a pointer to one column of a single
    jointly-fit multi-treatment CausalForestDML. `config.forests[rank]
    .const_marginal_effect(X)` keeps working exactly as before, without
    re-fitting or re-storing a separate forest per rank.
    """

    joint_path: Path
    rank: int

    def _artifact(self) -> JointForestArtifact:
        return _load_joint(self.joint_path)

    def const_marginal_effect(self, X):
        art = self._artifact()
        col = art.rank_column(self.rank)
        effect = art.cf.const_marginal_effect(X)
        effect = np.asarray(effect)
        if effect.ndim == 1:
            # Only possible if this scenario ended up with a single
            # non-base rank; kept for robustness.
            return effect
        return effect[:, col]


def forests_dict_from_joint(
    joint_path: Path, ranks: Iterable[int] | None = None,
) -> dict[int, RankView]:
    """Build a `{rank: RankView}` dict from a single `Joint CF.pkl` file.

    This is the single-file equivalent of loading `CF - Rank {r}.pkl`
    for every rank off disk: it loads the one shared joint forest once
    (cached process-wide by `_load_joint`), and returns lightweight
    in-memory views into it, keyed by rank. Nothing is written to disk.
    Use this from `config.py`'s loaders in place of a per-rank
    `joblib.load` loop.
    """
    art = _load_joint(joint_path)
    selected = list(ranks) if ranks is not None else art.ranks
    return {rank: RankView(joint_path=joint_path, rank=rank) for rank in selected}


# ----------------------------------------------------------------------
# Multi-class propensity grid search
# ----------------------------------------------------------------------

def e_model_best_estimator_multiclass(X, T, param_grid, n_jobs: int):
    """Same idea as `simulation_steps.e_model_best_estimator`, but scored
    with macro-averaged F1 instead of binary F1.

    T is now multi-valued (one class per advertiser rank plus the base
    ad), not just {0, 1}, so plain `f1_score` (which defaults to
    average="binary") is not applicable. Macro-F1 is used instead of
    (e.g.) log-loss because CV folds can be missing some of the
    rarest/lowest-volume ad ranks entirely, which would otherwise
    require careful `labels=` bookkeeping to keep log-loss well defined
    across folds; macro-F1 has no such requirement.
    """
    start_time = time.perf_counter()
    e_model = PropensityModel()
    scorer = make_scorer(f1_score, average="macro")
    grid_search = GridSearchCV(estimator=e_model, param_grid=param_grid, scoring=scorer, cv=5)
    grid_search.fit(X, T)
    best_params = grid_search.best_params_
    best_estimator = grid_search.best_estimator_
    finish_time = time.perf_counter()
    log.info("finished tuning the (multiclass) E model in %.1fs", finish_time - start_time)
    return best_params, best_estimator


# ----------------------------------------------------------------------
# Estimation step
# ----------------------------------------------------------------------

def fit_joint(
    *,
    ranks: list[int],
    data: pd.DataFrame,
    out_dir: Path,
    n_jobs: int,
    random_state: int,
    max_samples: float = 0.01,
    base_ad_label: int = 0,
) -> None:
    """Fit ONE multi-treatment CausalForestDML across all `ranks` at once,
    then write a lightweight per-rank RankView pickle for each rank so
    the rest of the pipeline (config.py, simulation_steps.py) does not
    need to change.
    """
    ranks = sorted(int(r) for r in ranks)
    categories = [int(base_ad_label)] + ranks

    # Mirror fit_one_rank's implicit behavior of dropping any rows whose
    # advertiser_rank isn't one of the categories being estimated (e.g.
    # the ultra-long-tail rank dropped by load_data_and_ranks).
    keep_mask = data["advertiser_rank"].isin(categories)
    df = data.loc[keep_mask].reset_index(drop=True).copy()

    X, Y, T = define_xyt(df)
    T = T.astype(int)
    assert sorted(T.unique().tolist()) == categories, (
        "advertiser_rank categories present in the filtered data do not "
        "exactly match [base_ad] + ranks; check `ranks` / filtering."
    )

    t0 = time.perf_counter()
    best_params_e, _ = e_model_best_estimator_multiclass(X, T, param_grid, n_jobs=n_jobs)
    best_params_m, _ = m_model_best_estimator(X, Y, param_grid, n_jobs=n_jobs)
    log.info(
        "joint e/m grid search done in %.1fs (n_ranks=%d, n_rows=%d)",
        time.perf_counter() - t0, len(ranks), len(df),
    )

    cf = CausalForestDML(
        model_y=RandomForestRegressor(**best_params_m),
        model_t=PropensityModel(**best_params_e),
        discrete_treatment=True,  # NOTE: estimate.py passes the string
        # "True" here, which happens to work only because a non-empty
        # string is truthy in Python. Using the real boolean instead.
        categories=categories,  # categories[0] (= base_ad_label) is
        # treated as the control arm by econml; the rest define the
        # column order of const_marginal_effect's output.
        criterion="het",
        n_jobs=n_jobs,
        n_estimators=100,
        min_samples_split=1000,
        max_depth=20,
        max_samples=max_samples,
        random_state=random_state,
        verbose=0,
    )

    t0 = time.perf_counter()
    cf.tune(Y=Y, T=T, X=X, params=cf_param_grid)
    log.info("joint tune done in %.1fs", time.perf_counter() - t0)

    t0 = time.perf_counter()
    cf.fit(Y=Y, T=T, X=X, inference="blb", cache_values=True)
    log.info("joint fit done in %.1fs", time.perf_counter() - t0)

    out_dir.mkdir(parents=True, exist_ok=True)
    joint_path = out_dir / "Joint CF.pkl"
    artifact = JointForestArtifact(cf=cf, ranks=ranks, base_ad=base_ad_label)
    joblib.dump(artifact, joint_path)
    log.info(
        "joint forest saved -> %s (single file, %d ranks, no per-rank pickles)",
        joint_path, len(ranks),
    )


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="adsim.estimate_joint", description=__doc__.split("\n\n")[0]
    )
    p.add_argument("--scenario", required=True, choices=sorted(SCENARIOS))
    p.add_argument("--split", type=int, default=None)
    p.add_argument("--subsample-ratio", type=float, default=0.8,
                    help="Only used for --scenario root-n.")
    p.add_argument("--ranks", type=str, default=None,
                    help="Comma-separated explicit ranks to include as "
                         "treatment arms, e.g. '1,2,3'.")
    p.add_argument("--ranks-filter", type=str, default=None,
                    help="Python expression on `rank`, e.g. 'rank > 10'.")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--max-samples", type=float, default=0.01,
                    help="Per-tree subsample fraction for CausalForestDML. "
                         "See the max_samples warning in the module docstring.")
    p.add_argument("--force", action="store_true",
                    help="Re-fit even if Joint CF.pkl already exists.")
    p.add_argument("--n-jobs", type=int, default=30)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--subsample-seed", type=int, default=42,
                    help="Seed for the root-n subsample.")
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
    log.info("scenario=%s split=%s out=%s (JOINT fit)", scenario.name, args.split, out_dir)

    data, discovered = load_data_and_ranks(scenario, args.split)
    data = maybe_subsample(data, scenario, args.subsample_ratio, args.subsample_seed)

    explicit = (
        [int(x) for x in args.ranks.split(",")]
        if args.ranks else None
    )
    ranks = select_ranks(
        discovered, explicit=explicit, expr=args.ranks_filter, limit=args.limit,
    )
    log.info("selected %d ranks as joint treatment arms", len(ranks))

    out_dir.mkdir(parents=True, exist_ok=True)
    joint_path = out_dir / "Joint CF.pkl"

    if joint_path.is_file() and not args.force:
        log.info("SKIP (already exists): %s", joint_path)
        return 0

    if args.dry_run:
        log.info("DRY-RUN -> would jointly fit %d ranks -> %s", len(ranks), joint_path)
        return 0

    overall_t0 = time.perf_counter()
    try:
        fit_joint(
            ranks=ranks,
            data=data,
            out_dir=out_dir,
            n_jobs=args.n_jobs,
            random_state=args.random_state,
            max_samples=args.max_samples,
        )
    except Exception:
        log.exception("joint fit FAILED for scenario=%s", scenario.name)
        return 1

    elapsed = time.perf_counter() - overall_t0
    log.info(
        "summary: scenario=%s joint fit done. n_ranks=%d elapsed=%.1fs",
        scenario.name, len(ranks), elapsed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
