"""Smoke-test that every `python -m adsim.X --help` works.

These are cheap end-to-end checks that catch:
  * an entry point losing its `if __name__ == "__main__"` block
  * a module-level import failing at module load
  * argparse breaking (e.g. dataclass/Path import errors)
"""

from __future__ import annotations

import subprocess
import sys

import pytest


ENTRY_POINTS = [
    "adsim.estimate",
    "adsim.simulate.base_ad_helpers",
    "adsim.simulate.monopoly",
    "adsim.simulate.duopoly",
    "adsim.simulate.duopoly_root_n",
    "adsim.simulate.legacy.simulation",
    "adsim.simulate.legacy.simulation_parallel",
]


@pytest.mark.parametrize("entry_point", ENTRY_POINTS)
def test_entry_point_help_exits_zero(entry_point: str):
    result = subprocess.run(
        [sys.executable, "-m", entry_point, "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"`python -m {entry_point} --help` failed:\n"
        f"stderr={result.stderr!r}\n"
        f"stdout={result.stdout!r}"
    )
    # Sanity: the help output mentions the entry point or a familiar token.
    assert "usage" in result.stdout.lower()


def test_check_old_pickles_help_exits_zero():
    # scripts/check_old_pickles.py is a standalone script, not importable as
    # adsim.something. Run it via python directly.
    result = subprocess.run(
        [sys.executable, "scripts/check_old_pickles.py", "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"check_old_pickles.py --help failed:\n"
        f"stderr={result.stderr!r}\n"
        f"stdout={result.stdout!r}"
    )
    assert "usage" in result.stdout.lower()


def test_estimate_dry_run_with_unknown_scenario_fails_cleanly():
    # If someone misspells a scenario, argparse should error (exit 2),
    # not crash with a stack trace.
    result = subprocess.run(
        [sys.executable, "-m", "adsim.estimate",
         "--scenario", "totally-made-up", "--dry-run"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 2
    assert "invalid choice" in result.stderr.lower()
