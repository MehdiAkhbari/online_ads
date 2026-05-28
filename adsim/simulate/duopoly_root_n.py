"""Duopoly forward simulation under the Root-N scenario.

Uses the Root-N variant of the per-split forests
(`results/Full Model/Split {N} - Root N/...`) and the Root-N
repeat/click update helpers. Saves one .dta per chunk.

Replaces scripts/duopoly_simulation_sqrt_n.py (Root-N scenario).

Run with:
    python -m adsim.simulate.duopoly_root_n
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
from adsim.simulation_steps import (
    calc_base_ad_split_ctr,
    calc_split_ctrs,
    calc_split_tes,
    create_chosen_ad_columns_split,
    create_chosen_split_ad_vars,
    update_clicks_on_main_and_split_root_n,
    update_repeats_on_main_and_split_root_n,
)


log = logging.getLogger("adsim.simulate.duopoly_root_n")

MAX_VISIT_NO = 100


def _init_worker(split_no_1: int, split_no_2: int) -> None:
    config.load_helpers()
    config.load_monopoly_forests()
    config.load_split_forests(split_no_1, root_n=True)
    config.load_split_forests(split_no_2, root_n=True)


def _simulate_duopoly_root_n(data: pd.DataFrame, split_no_1: int, split_no_2: int) -> pd.DataFrame:
    create_chosen_split_ad_vars(data)
    overall_t0 = time.perf_counter()
    for i in range(1, MAX_VISIT_NO + 1):
        repeat_t0 = time.perf_counter()
        log.info("repeat #%d", i)

        # Step 1: TEs and CTRs on each split.
        for split_no in (split_no_1, split_no_2):
            calc_split_tes(data, split_no=split_no, user_visit_no=i, ranks_list=config.ranks_list)
            calc_base_ad_split_ctr(data, split_no=split_no, user_visit_no=i)
            calc_split_ctrs(data, split_no=split_no, user_visit_no=i, ranks_list=config.ranks_list)

        # Step 2: choose ads on each split.
        for split_no in (split_no_1, split_no_2):
            create_chosen_ad_columns_split(data, split_no=split_no, user_visit_no=i)

        # Step 4: update repeats (Root-N updater).
        update_repeats_on_main_and_split_root_n(data, user_visit_no=i)

        # Step 5: update clicks (Root-N updater).
        update_clicks_on_main_and_split_root_n(data, user_visit_no=i)

        log.info("repeat #%d finished in %.1fs", i, time.perf_counter() - repeat_t0)

    log.info("all repeats finished in %.1fs", time.perf_counter() - overall_t0)
    return data


def _save_chunk(chunk_data: pd.DataFrame, chunk_id: int, split_no_1: int, split_no_2: int) -> None:
    chunk_data = _simulate_duopoly_root_n(chunk_data, split_no_1, split_no_2)
    sim_results_dir = RESULTS_DIR / "Full Model" / "Simulation Results"
    sim_results_dir.mkdir(parents=True, exist_ok=True)
    filename = sim_results_dir / f"Simluation Results - Split {split_no_1} {split_no_2} - Root N - chunk {chunk_id+1}.dta"
    chunk_data.to_stata(filename)
    log.info("chunk %d -> %s", chunk_id + 1, filename)


def _make_chunks(data: pd.DataFrame, n_processes: int) -> list[pd.DataFrame]:
    chunk_users_num = 1620000 / n_processes
    n_chunks = int(data["global_token_new"].max() / chunk_users_num) + 1
    data = data.copy()
    data["chunk"] = (data["global_token_new"] / chunk_users_num).astype(int) + 1
    return [data[data["chunk"] == c] for c in range(1, n_chunks + 1)]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="adsim.simulate.duopoly_root_n", description=__doc__.split("\n\n")[0])
    p.add_argument("--processes", type=int, default=4)
    p.add_argument("--split-1", type=int, default=None)
    p.add_argument("--split-2", type=int, default=None)
    p.add_argument("--data", type=Path, default=None)
    p.add_argument("-v", "--verbose", action="count", default=0)
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
    filterwarnings("ignore", category=UserWarning)
    pd.options.mode.chained_assignment = None

    split_no_1 = args.split_1 if args.split_1 is not None else config.split_no_1
    split_no_2 = args.split_2 if args.split_2 is not None else config.split_no_2

    data_path = args.data or (
        DATA_DIR / "Full Model"
        / f"Simulation Data - Full Model - Split {split_no_1} {split_no_2} - Root N - Subsample.dta"
    )

    log.info("loading data from %s", data_path)
    data = pd.read_stata(data_path)

    chunks = _make_chunks(data, args.processes)
    log.info("created %d chunks for %d processes (splits=%s/%s)",
             len(chunks), args.processes, split_no_1, split_no_2)

    overall_t0 = time.perf_counter()
    multiprocessing.freeze_support()
    with multiprocessing.Pool(
        processes=args.processes,
        initializer=_init_worker,
        initargs=(split_no_1, split_no_2),
    ) as pool:
        pool.starmap(
            _save_chunk,
            [(chunk, i, split_no_1, split_no_2) for i, chunk in enumerate(chunks)],
        )
    log.info("duopoly Root-N simulation finished in %.1fs", time.perf_counter() - overall_t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
