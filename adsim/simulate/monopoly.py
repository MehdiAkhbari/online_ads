"""Monopoly forward simulation.

For each user-visit step (1..max_visit_no) and each chunk (parallelised
across processes), call the per-step simulation utilities to estimate
treatment effects, choose ads, and update repeat / click history. Saves
one .dta per chunk under `results/Full Model/Simulation Results/`.

Replaces scripts/monopoly_simulation.py.

Run with:
    python -m adsim.simulate.monopoly
    python -m adsim.simulate.monopoly --criteria CTR --processes 8
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing
import time
from pathlib import Path
from warnings import filterwarnings, simplefilter

import pandas as pd

from adsim import config
from adsim.paths import DATA_DIR, RESULTS_DIR
from adsim.simulation_steps import simulate_monopoly


log = logging.getLogger("adsim.simulate.monopoly")


# Multiprocessing workers re-import the module under the default `spawn`
# start method. Each worker calls _init_worker() once to load the forests
# into its own process memory.
def _init_worker() -> None:
    config.load_helpers()
    config.load_monopoly_forests()


def _save_chunk(chunk_data: pd.DataFrame, chunk_id: int, vals_data: pd.DataFrame, criteria: str) -> None:
    chunk_data = simulate_monopoly(chunk_data, vals_data, criteria)
    sim_results_dir = RESULTS_DIR / "Full Model" / "Simulation Results"
    sim_results_dir.mkdir(parents=True, exist_ok=True)
    if criteria == "CTR":
        filename = sim_results_dir / f"Simluation Results - Monopoly - chunk {chunk_id+1}.dta"
    elif criteria == "revenue":
        filename = sim_results_dir / f"Simluation Results - Monopoly Revenue Max - chunk {chunk_id+1}.dta"
    else:
        raise ValueError(f"unknown criteria {criteria!r}")
    chunk_data.to_stata(filename)
    log.info("chunk %d -> %s", chunk_id + 1, filename)


def _make_chunks(data: pd.DataFrame, n_processes: int) -> list[pd.DataFrame]:
    chunk_users_num = 1620000 / n_processes
    n_chunks = int(data["global_token_new"].max() / chunk_users_num) + 1
    data = data.copy()
    data["chunk"] = (data["global_token_new"] / chunk_users_num).astype(int) + 1
    return [data[data["chunk"] == c] for c in range(1, n_chunks + 1)]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="adsim.simulate.monopoly", description=__doc__.split("\n\n")[0])
    p.add_argument("--processes", type=int, default=3,
                   help="Pool size for the per-chunk simulation. Default 3.")
    p.add_argument("--criteria", choices=("CTR", "revenue"), default=None,
                   help="Optimisation criterion. Default: config.my_criteria.")
    p.add_argument("--data", type=Path, default=None,
                   help="Simulation .dta. Default: DATA_DIR/Full Model/Simulation Data - Full Model - Monopoly - Subsample.dta")
    p.add_argument("--vals-data", type=Path, default=None,
                   help="Advertiser valuations .dta. Default: DATA_DIR/Full Model/Advertiser Valuations.dta")
    p.add_argument("-v", "--verbose", action="count", default=0)
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
    pd.options.mode.chained_assignment = None
    filterwarnings(
        "ignore",
        message="Loky-backed parallel loops cannot be called in a multiprocessing, setting n_jobs=1",
    )

    criteria = args.criteria or config.my_criteria
    data_path = args.data or (DATA_DIR / "Full Model" / "Simulation Data - Full Model - Monopoly - Subsample.dta")
    vals_path = args.vals_data or (DATA_DIR / "Full Model" / "Advertiser Valuations.dta")

    log.info("loading data from %s", data_path)
    data = pd.read_stata(data_path)
    log.info("loading valuations from %s", vals_path)
    vals_data = pd.read_stata(vals_path)

    chunks = _make_chunks(data, args.processes)
    log.info("created %d chunks for %d processes (criteria=%s)", len(chunks), args.processes, criteria)

    overall_t0 = time.perf_counter()
    multiprocessing.freeze_support()
    with multiprocessing.Pool(processes=args.processes, initializer=_init_worker) as pool:
        pool.starmap(
            _save_chunk,
            [(chunk, i, vals_data, criteria) for i, chunk in enumerate(chunks)],
        )
    log.info("monopoly simulation finished in %.1fs", time.perf_counter() - overall_t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
