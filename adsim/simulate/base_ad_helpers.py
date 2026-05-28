"""Fit the base-ad y0 helpers (m1.pkl, e1.pkl).

Loads the rank-1 causal forest, refits its inner Y model on Y and its
inner T model on T over the data subset where advertiser_rank in
{base_ad, 1}, and saves them as `results/Full Model/{m1,e1}.pkl`.

Replaces scripts/base_ad_ctr_estimation.py.

Run with:
    python -m adsim.simulate.base_ad_helpers
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import joblib
import pandas as pd

from adsim.paths import DATA_DIR, RESULTS_DIR
from adsim.simulation_steps import base_ad, define_xyt


log = logging.getLogger("adsim.simulate.base_ad_helpers")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="adsim.simulate.base_ad_helpers", description=__doc__.split("\n\n")[0])
    p.add_argument("--data", type=Path, default=None,
                   help="Estimation .dta. Default: DATA_DIR/Full Model/Estimation Data - Full Model - Monopoly.dta")
    p.add_argument("--rank-1-pkl", type=Path, default=None,
                   help="Rank-1 causal forest. Default: RESULTS_DIR/Full Model/Monopoly/CF - Rank 1.pkl")
    p.add_argument("-v", "--verbose", action="count", default=0)
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    data_path = args.data or (DATA_DIR / "Full Model" / "Estimation Data - Full Model - Monopoly.dta")
    cf_1_path = args.rank_1_pkl or (RESULTS_DIR / "Full Model" / "Monopoly" / "CF - Rank 1.pkl")

    t0 = time.perf_counter()
    log.info("loading data from %s", data_path)
    data = pd.read_stata(data_path)

    log.info("loading rank-1 forest from %s", cf_1_path)
    cf_1 = joblib.load(cf_1_path)

    # Restrict to base ad and rank-1 ad rows.
    data = data[(data["advertiser_rank"] == base_ad) | (data["advertiser_rank"] == 1)]

    X, Y, T = define_xyt(data)
    T = T.apply(lambda x: 0 if x == 0 else 1)

    t = time.perf_counter()
    m1 = cf_1.model_y.fit(X, Y)
    log.info("y model fitted in %.1fs", time.perf_counter() - t)

    t = time.perf_counter()
    e1 = cf_1.model_t.fit(X, T)
    log.info("t model fitted in %.1fs", time.perf_counter() - t)

    # Preserved from original behaviour: reassign e1 to the un-refitted
    # cf_1.model_t. Looks like a bug, but matches what was previously saved.
    e1 = cf_1.model_t

    out_dir = RESULTS_DIR / "Full Model"
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(m1, out_dir / "m1.pkl")
    joblib.dump(e1, out_dir / "e1.pkl")
    log.info("saved m1.pkl, e1.pkl to %s", out_dir)
    log.info("done in %.1fs", time.perf_counter() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
