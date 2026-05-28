"""Module-level constants and the `ranks_list` shared by the simulations.

`ranks_list` is the list of advertiser ranks the study iterates over, with
the base ad (rank 0) and the >max-ad fringe (last entry) dropped.

The pickle is looked up first under `RESULTS_DIR/main_scenario/`, and falls
back to `scripts/ranks_list.pickle` (the version checked into the repo).
"""

from __future__ import annotations

import pickle

from adsim.paths import REPO_ROOT, RESULTS_DIR

max_ads_per_page = 15

split_no_1 = 7
split_no_2 = 8


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
