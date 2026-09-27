"""Module-level constants, the shared `ranks_list`, and explicit
loaders for the saved causal-forest artifacts.

Importing this module is **side-effect-free**: it does not load any
forests off disk. Call the explicit loaders below before running
simulations.

Typical usage:
    import adsim.config as config
    config.load_helpers()            # m1, e1
    config.load_monopoly_forests()   # populates config.forests
    # then run simulations that read config.forests / config.helpers

Forest storage
--------------
Each scenario directory holds either one `CF - Rank {r}.pkl` per rank
(written by `adsim.estimate`, the canonical pipeline) or a single
`Joint CF.pkl` (written by `adsim.estimate_joint`). The loaders prefer the
per-rank files and fall back to the joint file only when per-rank files
are missing; either way `config.forests[rank].const_marginal_effect(X)`
returns that rank's CATE.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Iterable

import joblib

from adsim.paths import REPO_ROOT, RESULTS_DIR

# --- Static config ------------------------------------------------------

max_ads_per_page: int = 15
n_jobs: int = 30

# The two halves of the duopoly / split scenario.
split_no_1: int = 7
split_no_2: int = 8

# Default optimisation criterion in the simulation. "CTR" or "revenue".
my_criteria: str = "revenue"

# Sample-size scenario knob.
subsampling_ratio: float = 0.8

# --- ranks_list ---------------------------------------------------------

_RANKS_LIST_CANDIDATES = [
    RESULTS_DIR / "main_scenario" / "ranks_list.pickle",
    REPO_ROOT / "scripts" / "ranks_list.pickle",
]


def _load_ranks_list() -> list[int]:
    for path in _RANKS_LIST_CANDIDATES:
        if path.is_file():
            with open(path, "rb") as f:
                ranks = pickle.load(f)
            # Drop the base ad (rank 0) and the >max-ad fringe (last entry).
            ranks.pop(0)
            ranks.pop(-1)
            return ranks
    raise FileNotFoundError(
        "Could not find ranks_list.pickle. Looked in:\n  "
        + "\n  ".join(str(p) for p in _RANKS_LIST_CANDIDATES)
    )


ranks_list: list[int] = _load_ranks_list()

# --- Forest registries (populated by the loaders below) -----------------

# Monopoly: cf for each advertiser rank.
forests: dict[int, object] = {}

# Split / duopoly: forests[split_no][rank] -> cf.
split_forests: dict[int, dict[int, object]] = {}

# Subsample / sample-size scenario: forests[rank] -> cf, fit on a subsample.
subsample_forests: dict[int, object] = {}

# {"m1": RandomForestRegressor, "e1": PropensityModel}.
helpers: dict[str, object] = {}


# --- Loaders ------------------------------------------------------------

def _full_model_dir() -> Path:
    return RESULTS_DIR / "Full Model"


def _load_scenario_forests(base: Path, ranks: list[int]) -> tuple[dict[int, object], str]:
    """Per-rank pickles if all are present, else `Joint CF.pkl` views."""
    per_rank = {rank: base / f"CF - Rank {rank}.pkl" for rank in ranks}
    missing = [rank for rank, path in per_rank.items() if not path.is_file()]
    if not missing:
        return {rank: joblib.load(path) for rank, path in per_rank.items()}, "per-rank pickles"

    joint_path = base / "Joint CF.pkl"
    if joint_path.is_file():
        # Imported here to avoid a circular import (estimate_joint -> estimate -> config).
        from adsim.estimate_joint import forests_dict_from_joint

        return forests_dict_from_joint(joint_path, ranks=ranks), str(joint_path)

    raise FileNotFoundError(
        f"No forests for {len(missing)} of {len(ranks)} ranks in {base} "
        f"(e.g. ranks {missing[:5]}) and no Joint CF.pkl. "
        "Fit them with `python -m adsim.estimate --scenario ...`."
    )


def load_helpers(verbose: bool = False) -> dict[str, object]:
    """Load m1 (Y model) and e1 (T model) from `results/Full Model/`."""
    base = _full_model_dir()
    helpers["m1"] = joblib.load(base / "m1.pkl")
    helpers["e1"] = joblib.load(base / "e1.pkl")
    if verbose:
        print(f"loaded helpers m1, e1 from {base}")
    return helpers


def load_monopoly_forests(
    ranks: Iterable[int] | None = None,
    verbose: bool = True,
) -> dict[int, object]:
    """Load the monopoly forests from `results/Full Model/Monopoly/`."""
    base = _full_model_dir() / "Monopoly"
    selected = list(ranks) if ranks is not None else ranks_list
    loaded, source = _load_scenario_forests(base, selected)
    forests.update(loaded)
    if verbose:
        print(f"loaded {len(selected)} monopoly forests from {base} ({source})")
    return forests


def load_split_forests(
    split_no: int,
    *,
    root_n: bool = False,
    ranks: Iterable[int] | None = None,
    verbose: bool = True,
) -> dict[int, object]:
    """Load one split's forests from `results/Full Model/Split {N}[ - Root N]/`."""
    suffix = " - Root N" if root_n else ""
    base = _full_model_dir() / f"Split {split_no}{suffix}"
    selected = list(ranks) if ranks is not None else ranks_list
    loaded, source = _load_scenario_forests(base, selected)
    bucket = split_forests.setdefault(split_no, {})
    bucket.update(loaded)
    if verbose:
        print(f"loaded {len(selected)} forests for split {split_no} from {base} ({source})")
    return bucket


def load_subsample_forests(
    subsampling_ratio: float = subsampling_ratio,
    ranks: Iterable[int] | None = None,
    verbose: bool = True,
) -> dict[int, object]:
    """Load the sample-size scenario's forests for one subsampling ratio."""
    base = (
        _full_model_dir() / "Root N - Random"
        / f"Subsampling Ratio = {subsampling_ratio}"
    )
    selected = list(ranks) if ranks is not None else ranks_list
    loaded, source = _load_scenario_forests(base, selected)
    subsample_forests.update(loaded)
    if verbose:
        print(f"loaded {len(selected)} subsample forests from {base} ({source})")
    return subsample_forests


__all__ = [
    "max_ads_per_page",
    "n_jobs",
    "split_no_1",
    "split_no_2",
    "my_criteria",
    "subsampling_ratio",
    "ranks_list",
    "forests",
    "split_forests",
    "subsample_forests",
    "helpers",
    "load_helpers",
    "load_monopoly_forests",
    "load_split_forests",
    "load_subsample_forests",
]
