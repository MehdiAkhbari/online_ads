"""Legacy parallel monopoly simulation against the "Last 2 Days - Merged
Subjects Subsample" data.

Predates the Full Model pipeline. Saves a single combined .dta to
RESULTS_DIR/Simluation Results - Subsample.dta. Not part of the
canonical pipeline; use adsim.simulate.monopoly for the Full Model
scenario.

Replaces scripts/simulation_parallel.py.

Run with:
    python -m adsim.simulate.legacy.simulation_parallel
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing
import time
from pathlib import Path
from warnings import simplefilter

import pandas as pd

from adsim import config
from adsim.paths import DATA_DIR, RESULTS_DIR
from adsim.simulation_steps import (
    calc_base_ad_ctr,
    calc_ctrs,
    calc_tes,
    create_chosen_ad_columns,
    create_chosen_ad_vars,
    update_clicks,
    update_repeats,
)


log = logging.getLogger("adsim.simulate.legacy.simulation_parallel")

MAX_VISIT_NO = 100


def _init_worker() -> None:
    config.load_helpers()
    config.load_monopoly_forests()


def _simulate(data: pd.DataFrame, vals_data: pd.DataFrame, criteria: str) -> pd.DataFrame:
    create_chosen_ad_vars(data)
    overall_t0 = time.perf_counter()
    for i in range(1, MAX_VISIT_NO + 1):
        repeat_t0 = time.perf_counter()
        calc_tes(data, user_visit_no=i, ranks_list=config.ranks_list)
        calc_base_ad_ctr(data, user_visit_no=i)
        calc_ctrs(data, vals_data, user_visit_no=i)
        create_chosen_ad_columns(data, user_visit_no=i, criteria=criteria)
        update_repeats(data, user_visit_no=i)
        update_clicks(data, user_visit_no=i)
        log.info("repeat %d in %.1fs", i, time.perf_counter() - repeat_t0)
    log.info("all repeats finished in %.1fs", time.perf_counter() - overall_t0)
    return data


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="adsim.simulate.legacy.simulation_parallel", description=__doc__.split("\n\n")[0])
    p.add_argument("--processes", type=int, default=9)
    p.add_argument("--data", type=Path, default=None,
                   help="Default: DATA_DIR/Simulation Data - Last 2 Days - Merged Subjects Subsample.dta")
    p.add_argument("--vals-data", type=Path, default=None,
                   help="Default: DATA_DIR/Full Model/Advertiser Valuations.dta")
    p.add_argument("--criteria", choices=("CTR", "revenue"), default=None)
    p.add_argument("--chunk-users", type=int, default=820000)
    p.add_argument("-v", "--verbose", action="count", default=0)
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
    pd.options.mode.chained_assignment = None

    criteria = args.criteria or config.my_criteria

    data_path = args.data or (DATA_DIR / "Simulation Data - Last 2 Days - Merged Subjects Subsample.dta")
    vals_path = args.vals_data or (DATA_DIR / "Full Model" / "Advertiser Valuations.dta")
    log.info("loading data from %s", data_path)
    data = pd.read_stata(data_path)
    log.info("loading valuations from %s", vals_path)
    vals_data = pd.read_stata(vals_path)

    n_chunks = int(data["global_token_new"].max() / args.chunk_users) + 1
    data["chunk"] = (data["global_token_new"] / args.chunk_users).astype(int) + 1
    chunks = [data[data["chunk"] == c].copy() for c in range(1, n_chunks + 1)]
    log.info("created %d chunks for %d processes", n_chunks, args.processes)

    overall_t0 = time.perf_counter()
    multiprocessing.freeze_support()
    with multiprocessing.Pool(processes=args.processes, initializer=_init_worker) as pool:
        results = pool.starmap(_simulate, [(c, vals_data, criteria) for c in chunks])

    combined = pd.concat(results, ignore_index=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "Simluation Results - Subsample.dta"
    combined.to_stata(out_path)
    log.info("saved -> %s", out_path)
    log.info("done in %.1fs", time.perf_counter() - overall_t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
