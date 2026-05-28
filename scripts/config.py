"""Module-level config + eagerly loaded causal-forest artifacts.

Importing this module reads the per-rank `CF - Rank {r}.pkl` forests off
disk into module-level names (`cf_{rank}`, `cf_{rank}_s{split}`, ...).
The simulation scripts rely on this side-effect.

Paths come from `adsim.paths` (repo-root-relative by default; overridable
via `ADSIM_DATA_DIR` / `ADSIM_RESULTS_DIR` env vars).
"""

from __future__ import annotations

import pickle

import joblib
import numpy as np  # noqa: F401  (re-exported for legacy `from config import *` users)
import pandas as pd  # noqa: F401
from sklearn.base import BaseEstimator  # noqa: F401
from sklearn.ensemble import RandomForestClassifier  # noqa: F401
from sklearn.metrics import f1_score, log_loss, make_scorer  # noqa: F401

from adsim.paths import REPO_ROOT, RESULTS_DIR


simulation = True
root_n = False
sample_size_analysis = False
subsampling_ratio = 0.8

n_jobs = 30


my_criteria = "revenue"  # "CTR" or "revenue"


max_ads_per_page = 15

split_no_1 = 7
split_no_2 = 8


# --- ranks_list ---------------------------------------------------------
_RANKS_LIST_CANDIDATES = [
    RESULTS_DIR / "main_scenario" / "ranks_list.pickle",
    REPO_ROOT / "scripts" / "ranks_list.pickle",
]
for _path in _RANKS_LIST_CANDIDATES:
    if _path.is_file():
        with open(_path, "rb") as _f:
            ranks_list = pickle.load(_f)
        break
else:
    raise FileNotFoundError(
        "Could not find ranks_list.pickle. Looked in:\n  "
        + "\n  ".join(str(p) for p in _RANKS_LIST_CANDIDATES)
    )

# Drop the base ad (rank 0) and the >max-ad fringe (last entry).
ranks_list.pop(0)
ranks_list.pop(-1)


# --- forests ------------------------------------------------------------
if simulation:
    _full_model_dir = RESULTS_DIR / "Full Model"

    e1 = joblib.load(_full_model_dir / "e1.pkl")
    m1 = joblib.load(_full_model_dir / "m1.pkl")

    # Monopoly forests, exposed as cf_{rank}.
    for rank in ranks_list:
        cf = joblib.load(_full_model_dir / "Monopoly" / f"CF - Rank {rank}.pkl")
        globals()[f"cf_{rank}"] = cf
        if rank % 20 == 0:
            print(f"rank {rank} model loaded!")

    if root_n:
        # Root-N split forests, exposed as cf_{rank}_s{split}.
        for split_no in (split_no_1, split_no_2):
            split_dir = _full_model_dir / f"Split {split_no} - Root N"
            for rank in ranks_list:
                cf = joblib.load(split_dir / f"CF - Rank {rank}.pkl")
                globals()[f"cf_{rank}_s{split_no}"] = cf
                if rank % 20 == 0:
                    print(f"rank {rank} model loaded!")

    if sample_size_analysis:
        sub_dir = (
            _full_model_dir / "Root N - Random"
            / f"Subsampling Ratio = {subsampling_ratio}"
        )
        for rank in ranks_list:
            cf = joblib.load(sub_dir / f"CF - Rank {rank}.pkl")
            globals()[f"cf_{rank}_sub"] = cf
            if rank % 20 == 0:
                print(f"rank {rank} model loaded!")
