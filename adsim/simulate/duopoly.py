"""Duopoly / split forward simulation.

For each user-visit step, call the per-split simulation utilities for
both halves (split_no_1 and split_no_2). Saves one .dta per chunk under
`results/Full Model/Simulation Results/`.

Replaces scripts/duopoly_simulation.py.

Run with:
    python -m adsim.simulate.duopoly
    python -m adsim.simulate.duopoly --criteria CTR --processes 8
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
from adsim.utils import (
    calc_base_ad_split_ctr,
    calc_split_ctrs,
    calc_split_tes,
    create_chosen_ad_columns_split,
    create_chosen_split_ad_vars,
    update_clicks_on_main_and_split,
    update_repeats_on_main_and_split,
)


log = logging.getLogger("adsim.simulate.duopoly")

MAX_VISIT_NO = 100


def _simulate_duopoly(
    data: pd.DataFrame, vals_data: pd.DataFrame, criteria: str,
    split_no_1: int, split_no_2: int,
) -> pd.DataFrame:
    create_chosen_split_ad_vars(data)
    overall_t0 = time.perf_counter()
    for i in range(1, MAX_VISIT_NO + 1):
        repeat_t0 = time.perf_counter()
        log.info("repeat #%d", i)

        # Step 1: TEs and CTRs on each split.
        for split_no in (split_no_1, split_no_2):
            calc_split_tes(data, split_no=split_no, user_visit_no=i, ranks_list=config.ranks_list)
            calc_base_ad_split_ctr(data, split_no=split_no, user_visit_no=i)
            calc_split_ctrs(data, vals_data, split_no=split_no, user_visit_no=i, ranks_list=config.ranks_list)

        # Step 2: choose ads on each split.
        for split_no in (split_no_1, split_no_2):
            create_chosen_ad_columns_split(data, split_no=split_no, user_visit_no=i, criteria=criteria)

        # Step 3: update repeats on each split.
        for split_no in (split_no_1, split_no_2):
            update_repeats_on_main_and_split(data, split_no=split_no, user_visit_no=i)

        # Step 4: update clicks on each split.
        for split_no in (split_no_1, split_no_2):
            update_clicks_on_main_and_split(data, split_no=split_no, user_visit_no=i)

        log.info("repeat #%d finished in %.1fs", i, time.perf_counter() - repeat_t0)

    log.info("all repeats finished in %.1fs", time.perf_counter() - overall_t0)
    return data


# Each pool worker re-imports the module (spawn). Loading the forests in
# an initializer keeps the rest of this module import-safe.
def _init_worker(split_no_1: int, split_no_2: int) -> None:
    config.load_helpers()
    config.load_monopoly_forests()
    config.load_split_forests(split_no_1)
    config.load_split_forests(split_no_2)


def _save_chunk(
    chunk_data: pd.DataFrame,
    chunk_id: int,
    vals_data: pd.DataFrame,
    criteria: str,
    split_no_1: int,
    split_no_2: int,
) -> None:
    chunk_data = _simulate_duopoly(chunk_data, vals_data, criteria, split_no_1, split_no_2)
    sim_results_dir = RESULTS_DIR / "Full Model" / "Simulation Results"
    sim_results_dir.mkdir(parents=True, exist_ok=True)
    if criteria == "CTR":
        filename = sim_results_dir / f"Simluation Results - Split {split_no_1} {split_no_2} - chunk {chunk_id+1}.dta"
    elif criteria == "revenue":
        filename = sim_results_dir / f"Simluation Results - Split {split_no_1} {split_no_2} Revenue Max - chunk {chunk_id+1}.dta"
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
    p = argparse.ArgumentParser(prog="adsim.simulate.duopoly", description=__doc__.split("\n\n")[0])
    p.add_argument("--processes", type=int, default=4)
    p.add_argument("--criteria", choices=("CTR", "revenue"), default=None,
                   help="Optimisation criterion. Default: config.my_criteria.")
    p.add_argument("--split-1", type=int, default=None,
                   help="First split. Default: config.split_no_1.")
    p.add_argument("--split-2", type=int, default=None,
                   help="Second split. Default: config.split_no_2.")
    p.add_argument("--data", type=Path, default=None)
    p.add_argument("--vals-data", type=Path, default=None)
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

    criteria = args.criteria or config.my_criteria
    split_no_1 = args.split_1 if args.split_1 is not None else config.split_no_1
    split_no_2 = args.split_2 if args.split_2 is not None else config.split_no_2

    data_path = args.data or (
        DATA_DIR / "Full Model"
        / f"Simulation Data - Full Model - Split {split_no_1} {split_no_2} - Subsample.dta"
    )
    vals_path = args.vals_data or (DATA_DIR / "Full Model" / "Advertiser Valuations.dta")

    log.info("loading data from %s", data_path)
    data = pd.read_stata(data_path)
    log.info("loading valuations from %s", vals_path)
    vals_data = pd.read_stata(vals_path)

    chunks = _make_chunks(data, args.processes)
    log.info(
        "created %d chunks for %d processes (criteria=%s, splits=%s/%s)",
        len(chunks), args.processes, criteria, split_no_1, split_no_2,
    )

    overall_t0 = time.perf_counter()
    multiprocessing.freeze_support()
    with multiprocessing.Pool(
        processes=args.processes,
        initializer=_init_worker,
        initargs=(split_no_1, split_no_2),
    ) as pool:
        pool.starmap(
            _save_chunk,
            [(chunk, i, vals_data, criteria, split_no_1, split_no_2)
             for i, chunk in enumerate(chunks)],
        )
    log.info("duopoly simulation finished in %.1fs", time.perf_counter() - overall_t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
