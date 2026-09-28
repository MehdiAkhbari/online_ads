"""End-to-end tests for `python -m adsim.diagnostics` on a synthetic data / results tree.

The tree mirrors the real layout: an estimation .dta, per-rank forests from
`adsim.estimate.fit_one_rank`, m1 / e1 from `base_ad_helpers`, a valuations
file, and a second monopoly fit (another seed) for the placebo-monopoly check.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import joblib
import pandas as pd
import pytest

from adsim.diagnostics.cli import CHECKS, select_checks
from adsim.diagnostics.output import to_latex
from adsim.estimate import fit_one_rank
from adsim.simulate.base_ad_helpers import fit_helpers
from adsim.simulation_steps import prepare_data


def test_select_checks_expands_groups_and_keeps_order():
    assert select_checks("all") == list(CHECKS)
    assert select_checks("identification")[0] == "propensity-info"
    assert "calibration" in select_checks("model")
    assert select_checks("balance,balance,overlap") == ["balance", "overlap"]
    with pytest.raises(ValueError, match="unknown check"):
        select_checks("balance,nope")


def test_to_latex_escapes_and_formats():
    tex = to_latex(pd.DataFrame({"feature_set": ["a_b", "50%"], "p_value": [0.000123456, float("nan")]}))
    assert r"feature\_set" in tex and r"50\%" in tex
    assert "0.0001235" in tex
    assert tex.startswith(r"\begin{tabular}{lr}") and r"\bottomrule" in tex


@pytest.fixture(scope="module")
def tree(tmp_path_factory, make_estimation_data):
    root = tmp_path_factory.mktemp("diagnostics")
    raw = make_estimation_data(n_users=1_200)
    (root / "data" / "Full Model").mkdir(parents=True)
    raw.to_stata(root / "data" / "Full Model" / "Estimation Data - Full Model - Monopoly.dta", write_index=False)
    pd.DataFrame({"advertiser_rank": [1, 2, 3], "advertiser_val_cents": [10.0, 40.0, 20.0]}).to_stata(
        root / "data" / "Full Model" / "Advertiser Valuations.dta", write_index=False)

    data = raw.copy()
    prepare_data(data)
    for seed, sub in ((0, "results/Full Model/Monopoly"), (1, "placebo/Monopoly")):
        for rank in (1, 2, 3):
            fit_one_rank(rank=rank, data=data, out_path=root / sub / f"CF - Rank {rank}.pkl",
                         n_jobs=2, random_state=seed, n_estimators=8)
    m1, e1 = fit_helpers(data, joblib.load(root / "results/Full Model/Monopoly/CF - Rank 1.pkl"))
    joblib.dump(m1, root / "results/Full Model/m1.pkl")
    joblib.dump(e1, root / "results/Full Model/e1.pkl")
    return root


def _run(tree, *args):
    env = {**os.environ, "ADSIM_DATA_DIR": str(tree / "data"), "ADSIM_RESULTS_DIR": str(tree / "results")}
    return subprocess.run(
        [sys.executable, "-m", "adsim.diagnostics", "--scenario", "monopoly", "--n-jobs", "2",
         "--refit-trees", "8", "--max-eval-rows", "3000", *args],
        capture_output=True, text=True, env=env, timeout=600,
    )


def test_cli_runs_every_check_and_writes_tables_and_figures(tree):
    result = _run(tree, "--top-k", "1", "--placebo-forest-dir", str(tree / "placebo" / "Monopoly"))
    assert result.returncode == 0, result.stderr[-3000:]

    out = tree / "results" / "Full Model" / "Diagnostics" / "Monopoly"
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["checks"] == list(CHECKS)
    assert manifest["failed"] == [] and manifest["skipped"] == []
    assert manifest["forest_found"] == {str(manifest["ranks"][0]): True}

    for check, table in [("propensity-info", "propensity_info"), ("balance", "balance_smd"),
                         ("placebo-outcome", "placebo_outcomes"), ("refuters", "refuters"),
                         ("sensitivity", "sensitivity"), ("overlap", "overlap_trimming"),
                         ("nuisance", "nuisance_scores"), ("cate-validation", "cate_validation"),
                         ("estimators", "estimators"), ("stability", "stability"),
                         ("calibration", "calibration"), ("out-of-range", "out_of_range"),
                         ("placebo-monopoly", "placebo_monopoly")]:
        assert (out / check / f"{table}.csv").is_file(), check
        assert (out / check / f"{table}.tex").is_file(), check
    for check, figure in [("propensity-info", "propensity_info"), ("balance", "balance_smd"),
                          ("overlap", "overlap_histogram"), ("refuters", "refuters"),
                          ("calibration", "reliability"), ("out-of-range", "out_of_range")]:
        assert (out / check / f"{figure}.pdf").is_file(), check

    per_rank = pd.read_csv(out / "propensity-info" / "propensity_info.csv")
    assert list(per_rank.columns[:2]) == ["rank", "support"]
    placebo = pd.read_csv(out / "placebo-monopoly" / "placebo_monopoly.csv")
    assert list(placebo["criterion"]) == ["CTR", "revenue"]


def test_cli_skips_pipeline_checks_without_helpers(tree, tmp_path):
    result = _run(tree, "--checks", "calibration,placebo-monopoly", "--helpers-dir", str(tmp_path),
                  "--out-dir", str(tmp_path / "out"))
    assert result.returncode == 0, result.stderr[-3000:]
    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text())
    assert {s["check"] for s in manifest["skipped"]} == {"calibration", "placebo-monopoly"}


def test_cli_rejects_unknown_checks(tree):
    result = _run(tree, "--checks", "balance,bogus")
    assert result.returncode == 2
    assert "unknown check" in result.stderr
