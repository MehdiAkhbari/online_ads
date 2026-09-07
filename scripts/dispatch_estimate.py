"""Launch several concurrent `adsim.estimate` worker processes, splitting
the ranks for one scenario across them so they run in parallel instead of
one long sequential loop.

Why this exists
----------------
`adsim.estimate`'s own grid-search step (`e_model_best_estimator` /
`m_model_best_estimator`) does not parallelize across ranks -- each rank
is fit one at a time in a single process. But ranks are fully
independent, so the natural way to use a many-core machine is to run N
separate `adsim.estimate` processes concurrently, each handling a
disjoint subset of ranks via `--ranks`.

Two things make this non-trivial:
  1. Rank fit cost scales roughly with (rows in that rank + rows in the
     base ad), and rows-per-rank varies by ~4 orders of magnitude
     (measured: a few hundred rows up to ~2.6M). Splitting ranks evenly
     BY COUNT would leave some workers with all the cheap ranks and
     others stuck with the few expensive ones. This script instead
     load-balances by an estimated per-rank COST, via longest-processing-
     time-first (LPT) bin packing across `--workers` bins.
  2. Each individual rank's own CausalForestDML/grid-search step only
     partially benefits from more threads (measured: cutting threads
     4x only slowed one fit down ~1.9x) -- so it is usually better
     throughput to run MORE workers with FEWER threads each
     (`--n-jobs`) than fewer workers hogging all cores. Tune
     `--workers` x `--n-jobs` to roughly equal your machine's vCPU
     count and try a couple of splits; there's no universally-best
     ratio.

The cost model splits each rank's predicted time into two additive
pieces, calibrated from real timed fits (109K and 483K rows) with the
current hyperparameters (see docs/hyperparameter_choices.txt):
  - grid search (e/m nuisance models): independent of n_estimators and
    of the causal-forest hyperparameters, scales as rows**GRID_EXPONENT.
  - tune+fit (the causal forest itself): scales as
    rows**TUNEFIT_EXPONENT, and roughly LINEARLY in n_estimators (each
    tree is independent), since the two calibration points were both
    measured at n_estimators=100.
This is more accurate than a single power law because grid search does
NOT get more expensive with more trees, so the right mix shifts as
--n-estimators changes. The absolute hour/cost estimates are still a
rough guide from a 2-point calibration on an 8-vCPU dev machine -- not
a guarantee, and worth re-checking against a real timed batch on the
target VM before trusting for capacity planning.

Usage:
    # Show the schedule without launching anything.
    python scripts/dispatch_estimate.py --scenario monopoly --workers 8 --n-jobs 4 --dry-run

    # Actually launch 8 concurrent workers, 4 threads each.
    python scripts/dispatch_estimate.py --scenario monopoly --workers 8 --n-jobs 4

    # Validate on a small subset first (e.g. the 10 largest ranks).
    python scripts/dispatch_estimate.py --scenario monopoly --workers 4 --n-jobs 4 --only-largest 10
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from adsim.estimate import SCENARIOS, load_data_and_ranks, select_ranks
from adsim.paths import RESULTS_DIR

# Calibrated on an 8-vCPU dev VM at n_estimators=100 with the current
# hyperparameters; see module docstring and docs/hyperparameter_choices.txt.
GRID_CONST, GRID_EXPONENT = 7.83811e-05, 1.2329
TUNEFIT_CONST, TUNEFIT_EXPONENT = 1.12328e-05, 1.2934
TUNEFIT_CALIBRATION_N_ESTIMATORS = 100
THREAD_SLOWDOWN = {8: 1.0, 4: 1.1379, 2: 1.9463}  # relative to n_jobs=8, measured under the old hyperparameters


def predicted_seconds(n_rows: int, n_jobs: int, n_estimators: int) -> float:
    slowdown = THREAD_SLOWDOWN.get(n_jobs, THREAD_SLOWDOWN[min(THREAD_SLOWDOWN, key=lambda k: abs(k - n_jobs))])
    grid = GRID_CONST * (n_rows ** GRID_EXPONENT)
    tunefit = TUNEFIT_CONST * (n_rows ** TUNEFIT_EXPONENT) * (n_estimators / TUNEFIT_CALIBRATION_N_ESTIMATORS)
    return (grid + tunefit) * slowdown


def lpt_schedule(weighted_ranks: list[tuple[int, float]], n_workers: int) -> list[list[int]]:
    """Longest-processing-time-first bin packing: sort ranks by descending
    weight, repeatedly drop the next-heaviest rank into the currently
    least-loaded bin. Not optimal but within ~4/3 of optimal, standard
    for this kind of scheduling."""
    bins: list[list[int]] = [[] for _ in range(n_workers)]
    bin_load = [0.0] * n_workers
    for rank, weight in sorted(weighted_ranks, key=lambda x: -x[1]):
        i = min(range(n_workers), key=lambda k: bin_load[k])
        bins[i].append(rank)
        bin_load[i] += weight
    return bins, bin_load


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--scenario", required=True, choices=sorted(SCENARIOS))
    p.add_argument("--split", type=int, default=None)
    p.add_argument("--subsample-ratio", type=float, default=0.8)
    p.add_argument("--workers", type=int, required=True, help="Number of concurrent worker processes.")
    p.add_argument("--n-jobs", type=int, default=4, help="Threads per worker (passed to each estimate.py --n-jobs).")
    p.add_argument("--n-estimators", type=int, default=500,
                    help="Passed through to each estimate.py worker's --n-estimators; "
                         "also used for this script's own runtime projection.")
    p.add_argument("--only-largest", type=int, default=None,
                    help="Debug/validation aid: only schedule the N largest ranks by row count.")
    p.add_argument("--only-ranks", type=str, default=None,
                    help="Debug/validation aid: comma-separated explicit ranks to schedule, e.g. '85,80'.")
    p.add_argument("--force", action="store_true")
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--dry-run", action="store_true", help="Print the schedule and estimated cost, then exit.")
    p.add_argument("--log-dir", type=Path, default=None,
                    help="Defaults to results/Full Model/<scenario>/dispatch_logs/")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    scenario = SCENARIOS[args.scenario]

    print(f"loading data + ranks for scenario={args.scenario} split={args.split} ...")
    data, discovered = load_data_and_ranks(scenario, args.split)
    ranks = select_ranks(discovered, explicit=None, expr=None, limit=None)

    counts = data["advertiser_rank"].value_counts()
    base_count = int(counts.get(0, 0))
    rank_rows = {r: base_count + int(counts.get(r, 0)) for r in ranks}
    del data  # free the full dataframe; workers each load their own copy

    if args.only_ranks is not None:
        wanted = {int(x) for x in args.only_ranks.split(",")}
        ranks = [r for r in ranks if r in wanted]
    elif args.only_largest is not None:
        ranks = sorted(ranks, key=lambda r: -rank_rows[r])[: args.only_largest]

    weighted = [(r, predicted_seconds(rank_rows[r], args.n_jobs, args.n_estimators)) for r in ranks]
    bins, bin_seconds = lpt_schedule(weighted, args.workers)

    print(f"\n{len(ranks)} ranks split across {args.workers} workers (n_jobs={args.n_jobs} each):")
    for i, (rank_list, secs) in enumerate(zip(bins, bin_seconds)):
        print(f"  worker {i}: {len(rank_list)} ranks, predicted {secs/3600:.2f}h -> {rank_list}")
    makespan_h = max(bin_seconds) / 3600.0
    print(f"\npredicted wall-clock (max over workers): {makespan_h:.2f}h")
    print("(this is a rough estimate from a small calibration sample -- validate on this VM before trusting it)")

    if args.dry_run:
        return 0

    log_dir = args.log_dir or (RESULTS_DIR / "Full Model" / scenario.out_subdir_template.format(
        split=args.split, subsample_ratio=args.subsample_ratio,
    ) / "dispatch_logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    procs = []
    for i, rank_list in enumerate(bins):
        if not rank_list:
            continue
        cmd = [
            sys.executable, "-m", "adsim.estimate",
            "--scenario", args.scenario,
            "--ranks", ",".join(str(r) for r in rank_list),
            "--n-jobs", str(args.n_jobs),
            "--random-state", str(args.random_state),
            "--n-estimators", str(args.n_estimators),
        ]
        if args.split is not None:
            cmd += ["--split", str(args.split)]
        if args.scenario == "root-n":
            cmd += ["--subsample-ratio", str(args.subsample_ratio)]
        if args.force:
            cmd.append("--force")

        log_path = log_dir / f"worker_{i}.log"
        print(f"launching worker {i} -> {log_path} ({len(rank_list)} ranks)")
        log_f = open(log_path, "w")
        proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, cwd=Path(__file__).resolve().parent.parent)
        procs.append((i, proc, log_f))

    t0 = time.perf_counter()
    exit_codes = {}
    for i, proc, log_f in procs:
        exit_codes[i] = proc.wait()
        log_f.close()
    elapsed = time.perf_counter() - t0

    n_failed = sum(1 for c in exit_codes.values() if c != 0)
    print(f"\nall workers finished in {elapsed/3600:.2f}h. failed workers: {n_failed}/{len(procs)}")
    for i, code in sorted(exit_codes.items()):
        status = "OK" if code == 0 else f"FAILED (exit {code})"
        print(f"  worker {i}: {status} -- see {log_dir / f'worker_{i}.log'}")
    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
