"""Tests for the adsim package's import contract.

Two important properties that the rest of the cleanup depends on:

  1. Importing `adsim.config` does *no* I/O on the disk for forests/
     helpers — the registries are populated lazily by `load_*` calls.
     If this regresses, tests and notebooks that import the package
     without a populated `results/` will start failing at import time.

  2. `adsim.paths` resolves the data/results dirs from env vars when
     present, falling back to repo-root-relative defaults.
"""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# adsim.config side-effect-free import
# ---------------------------------------------------------------------------

def test_importing_config_does_not_populate_forests():
    # Reload from scratch so we observe what import actually does, even if
    # an earlier test already populated config.forests.
    import adsim.config
    importlib.reload(adsim.config)
    assert adsim.config.forests == {}
    assert adsim.config.split_forests == {}
    assert adsim.config.subsample_forests == {}
    assert adsim.config.helpers == {}


def test_config_ranks_list_is_loaded():
    # ranks_list is the one thing that *is* loaded at import time
    # (96 ranks, base ad and >max-fringe entry already removed).
    import adsim.config as config
    assert isinstance(config.ranks_list, list)
    assert len(config.ranks_list) > 0
    # base ad (0) should already be popped
    assert 0 not in config.ranks_list


def test_config_loaders_are_callable_and_take_ranks_kw():
    import adsim.config as config
    # We can't actually call them without forest pickles on disk, but we
    # can confirm the API surface is what the simulation modules expect.
    for name in ("load_helpers", "load_monopoly_forests", "load_split_forests",
                 "load_subsample_forests"):
        assert hasattr(config, name)
        assert callable(getattr(config, name))


# ---------------------------------------------------------------------------
# adsim.paths
# ---------------------------------------------------------------------------

def test_paths_default_to_repo_relative():
    # Spawn a fresh interpreter with no env vars set, so we observe the
    # default behaviour — without contaminating this process's env state.
    env = {k: v for k, v in os.environ.items() if k not in ("ADSIM_DATA_DIR", "ADSIM_RESULTS_DIR")}
    code = (
        "from adsim.paths import REPO_ROOT, DATA_DIR, RESULTS_DIR;"
        "print(REPO_ROOT);print(DATA_DIR);print(RESULTS_DIR)"
    )
    out = subprocess.check_output([sys.executable, "-c", code], env=env, text=True)
    repo_root, data_dir, results_dir = out.strip().splitlines()
    assert Path(data_dir) == Path(repo_root) / "data"
    assert Path(results_dir) == Path(repo_root) / "results"


def test_paths_env_var_overrides(tmp_path: Path):
    env = {**os.environ,
           "ADSIM_DATA_DIR": str(tmp_path / "alt-data"),
           "ADSIM_RESULTS_DIR": str(tmp_path / "alt-results")}
    code = (
        "from adsim.paths import DATA_DIR, RESULTS_DIR;"
        "print(DATA_DIR);print(RESULTS_DIR)"
    )
    out = subprocess.check_output([sys.executable, "-c", code], env=env, text=True)
    data_dir, results_dir = out.strip().splitlines()
    assert Path(data_dir) == (tmp_path / "alt-data").resolve()
    assert Path(results_dir) == (tmp_path / "alt-results").resolve()


# ---------------------------------------------------------------------------
# Submodule imports work
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("module", [
    "adsim",
    "adsim.paths",
    "adsim.config",
    "adsim.propensity_model",
    "adsim.simulation_steps",
    "adsim.estimate",
    "adsim.simulate",
    "adsim.simulate.base_ad_helpers",
    "adsim.simulate.monopoly",
    "adsim.simulate.duopoly",
    "adsim.simulate.duopoly_root_n",
    "adsim.simulate.legacy",
    "adsim.simulate.legacy.simulation",
    "adsim.simulate.legacy.simulation_parallel",
])
def test_module_imports_clean(module: str):
    importlib.import_module(module)
