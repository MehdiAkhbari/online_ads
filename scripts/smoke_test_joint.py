"""Capacity-planning smoke test for `adsim.estimate_joint.fit_joint`.

Runs the exact same joint-CausalForestDML fitting code used in
production, but on a random subsample of the Monopoly data, so we can
verify the pipeline runs end-to-end on this machine and measure actual
wall-clock time / peak RAM before committing to a full-scale run.

This intentionally bypasses `adsim.estimate.load_data_and_ranks` (which
assumes DATA_DIR/"Full Model/..." layout and overwrites the shared
results/main_scenario/ranks_list.pickle) so it can point straight at an
explicit --data-path and never touches real pipeline outputs.

Usage:
    python scripts/smoke_test_joint.py --frac 0.01 --n-jobs 6

Writes:
    <out-dir>/Joint CF.pkl              (same artifact format as prod)
    <out-dir>/smoke_test_summary.json   (timings, peak RAM, extrapolation)
    <out-dir>/smoke_test.log            (full log of the run)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

from adsim.estimate_joint import fit_joint
from adsim.simulation_steps import extract_ranks, prepare_data

log = logging.getLogger("smoke_test")


class MemoryMonitor:
    """Background sampler of RSS for this process + all its children.

    GridSearchCV / CausalForestDML fan out work to worker processes via
    joblib (loky), so the parent process's own RSS understates real
    usage; this walks psutil's children(recursive=True) each tick too.
    """

    def __init__(self, interval: float = 1.0):
        self.interval = interval
        self._peak_bytes = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._proc = psutil.Process(os.getpid())

    def _sample(self) -> int:
        total = 0
        procs = [self._proc] + self._proc.children(recursive=True)
        for p in procs:
            try:
                total += p.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return total

    def _run(self) -> None:
        while not self._stop.is_set():
            self._peak_bytes = max(self._peak_bytes, self._sample())
            self._stop.wait(self.interval)

    def __enter__(self) -> "MemoryMonitor":
        self._peak_bytes = self._sample()
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join(timeout=self.interval * 2)

    @property
    def peak_gb(self) -> float:
        return self._peak_bytes / 1e9


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--data-path", type=Path,
        default=Path(r"C:\Users\mehdiakhbari\Projects\Online Ads\Data\Estimation Data - Full Model - Monopoly.dta"),
    )
    p.add_argument("--frac", type=float, default=0.01, help="Row sampling fraction.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-count-per-rank", type=int, default=20,
                    help="Drop treatment ranks with fewer than this many rows "
                         "in the subsample (rare ranks vanish under subsampling; "
                         "this only affects which arms are fit in the SMOKE TEST, "
                         "not the real run).")
    p.add_argument("--n-jobs", type=int, default=min(8, os.cpu_count() or 4))
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--max-samples", type=float, default=0.01)
    p.add_argument(
        "--out-dir", type=Path,
        default=None,
        help="Defaults to results/Smoke Test/Monopoly - frac=<frac>/ under the repo.",
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    repo_root = Path(__file__).resolve().parent.parent
    out_dir = args.out_dir or (
        repo_root / "results" / "Smoke Test" / f"Monopoly - frac={args.frac}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(out_dir / "smoke_test.log", mode="w"),
        ],
    )

    log.info("=== smoke test start: frac=%s n_jobs=%d out_dir=%s ===",
              args.frac, args.n_jobs, out_dir)
    log.info("cpu_count=%d total_ram_gb=%.1f",
              os.cpu_count() or -1, psutil.virtual_memory().total / 1e9)

    summary: dict = {
        "frac": args.frac,
        "seed": args.seed,
        "n_jobs": args.n_jobs,
        "cpu_count": os.cpu_count(),
        "total_ram_gb": round(psutil.virtual_memory().total / 1e9, 2),
        "data_path": str(args.data_path),
    }

    # --- load full data (this cost is already full-scale: reading the
    # whole file is unavoidable regardless of --frac) ------------------
    with MemoryMonitor() as mon:
        t0 = time.perf_counter()
        data = pd.read_stata(args.data_path)
        t_load = time.perf_counter() - t0
    summary["n_rows_full"] = int(len(data))
    summary["elapsed_data_load_s"] = round(t_load, 1)
    summary["peak_rss_gb_data_load"] = round(mon.peak_gb, 2)
    summary["df_memory_gb"] = round(data.memory_usage(deep=True).sum() / 1e9, 2)
    log.info(
        "loaded full data: n_rows=%d elapsed=%.1fs peak_rss=%.2fGB df_mem=%.2fGB",
        summary["n_rows_full"], t_load, mon.peak_gb, summary["df_memory_gb"],
    )

    prepare_data(data, base_ad=50, max_ad=100)
    ranks_all = extract_ranks(data)
    # Mirror adsim.estimate.load_data_and_ranks: drop the base-ad category
    # (0) and the top fringe category from the arms to be fit.
    ranks_for_fit = list(ranks_all)
    if ranks_for_fit and ranks_for_fit[0] == 0:
        ranks_for_fit.pop(0)
    if ranks_for_fit:
        ranks_for_fit.pop(-1)
    summary["n_ranks_full_scale"] = len(ranks_for_fit)
    log.info("full data has %d candidate treatment ranks after collapsing fringe", len(ranks_for_fit))

    # --- subsample -------------------------------------------------------
    rng = np.random.RandomState(args.seed)
    mask = rng.uniform(size=len(data)) < args.frac
    sample = data.loc[mask].reset_index(drop=True)
    summary["n_rows_sample"] = int(len(sample))
    summary["frac_actual"] = round(len(sample) / len(data), 5)
    log.info("subsampled %d / %d rows (target frac=%.4f, actual=%.4f)",
              len(sample), len(data), args.frac, summary["frac_actual"])
    del data

    counts = sample["advertiser_rank"].value_counts()
    usable_ranks = [r for r in ranks_for_fit if counts.get(r, 0) >= args.min_count_per_rank]
    dropped = len(ranks_for_fit) - len(usable_ranks)
    summary["n_ranks_used_in_smoke_test"] = len(usable_ranks)
    summary["n_ranks_dropped_rare"] = dropped
    log.info(
        "using %d/%d ranks in the smoke-test fit (%d dropped for <%d rows "
        "after subsampling -- rare ranks only, does not affect full-scale run)",
        len(usable_ranks), len(ranks_for_fit), dropped, args.min_count_per_rank,
    )

    # --- fit (the actual production code path) ---------------------------
    with MemoryMonitor() as mon:
        t0 = time.perf_counter()
        fit_joint(
            ranks=usable_ranks,
            data=sample,
            out_dir=out_dir,
            n_jobs=args.n_jobs,
            random_state=args.random_state,
            max_samples=args.max_samples,
        )
        t_fit = time.perf_counter() - t0
    summary["elapsed_fit_s"] = round(t_fit, 1)
    summary["peak_rss_gb_fit"] = round(mon.peak_gb, 2)
    log.info("fit_joint done: elapsed=%.1fs peak_rss=%.2fGB", t_fit, mon.peak_gb)

    # --- naive linear extrapolation to full scale -------------------------
    # NOTE: fit_joint's own docstring warns max_samples is a FRACTION, so at
    # full scale each tree sees ~1/frac times MORE rows than in this test,
    # even though n_estimators is unchanged. Tree-building cost is
    # super-linear in rows-per-tree (roughly O(m log m)), so this linear
    # projection almost certainly UNDERSTATES the true full-scale fit time.
    # Treat it as a floor, not an estimate.
    scale = 1.0 / max(summary["frac_actual"], 1e-9)
    summary["naive_linear_scale_factor"] = round(scale, 1)
    summary["naive_extrapolated_fit_hours_LOWER_BOUND"] = round(t_fit * scale / 3600, 2)
    summary["naive_extrapolated_peak_rss_gb"] = round(mon.peak_gb * scale, 1)

    summary_path = out_dir / "smoke_test_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info("wrote summary -> %s", summary_path)
    log.info("=== smoke test done ===")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
