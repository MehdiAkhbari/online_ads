"""Fit the base-ad y0 helpers (m1.pkl, e1.pkl).

The simulation recovers the base ad's click rate as
y0(x) = m1(x) - tau_1(x) * e1(x), where tau_1 is the rank-1 causal forest,
m1(x) = E[Y | X] and e1(x) = P(ad 1 shown | X) on the (base ad, rank 1)
pair. m1 and e1 must be fit on the same rows; this rebuilds the pair
exactly as `adsim.estimate` did for the rank-1 forest (same arm-balance cap
and seed, read from the forest's `adsim_plan_`) and fits fresh copies of
that forest's nuisance models on it.

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
from sklearn.base import clone

from adsim.estimate import build_pair
from adsim.hyperparams import ARM_RATIO_CAP
from adsim.paths import DATA_DIR, RESULTS_DIR
from adsim.simulation_steps import define_xyt, prepare_data


log = logging.getLogger("adsim.simulate.base_ad_helpers")


def fit_helpers(data: pd.DataFrame, cf_1) -> tuple[object, object]:
    """Fit (m1, e1) on the rank-1 pair. `data` must have gone through
    `prepare_data` (base ad coded 0)."""
    plan = getattr(cf_1, "adsim_plan_", {})
    pair = build_pair(
        data, 1,
        arm_ratio_cap=plan.get("arm_ratio_cap", ARM_RATIO_CAP),
        random_state=plan.get("random_state", 42),
    )
    X, Y, T = define_xyt(pair)
    T = (T != 0).astype(int)
    if T.nunique() != 2:
        raise ValueError(f"rank-1 pair has treatment values {sorted(T.unique())}; expected both 0 and 1")

    m1 = clone(cf_1.model_y).fit(X, Y)
    e1 = clone(cf_1.model_t).fit(X, T)
    return m1, e1


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
    prepare_data(data, base_ad=50, max_ad=100)

    log.info("loading rank-1 forest from %s", cf_1_path)
    cf_1 = joblib.load(cf_1_path)

    m1, e1 = fit_helpers(data, cf_1)

    out_dir = RESULTS_DIR / "Full Model"
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(m1, out_dir / "m1.pkl")
    joblib.dump(e1, out_dir / "e1.pkl")
    log.info("saved m1.pkl, e1.pkl to %s", out_dir)
    log.info("done in %.1fs", time.perf_counter() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
