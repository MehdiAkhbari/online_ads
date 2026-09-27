"""Tests for adsim.config's forest loaders (per-rank pickles vs Joint CF.pkl)."""

from __future__ import annotations

import joblib
import numpy as np
import pytest

import adsim.config as config
from adsim.estimate_joint import JointForestArtifact, _JOINT_CACHE


class ConstantEffect:
    def __init__(self, value):
        self.value = value

    def const_marginal_effect(self, X):
        return np.full((len(X), 1), self.value)


class JointConstantEffect:
    def __init__(self, n_ranks):
        self.n_ranks = n_ranks

    def const_marginal_effect(self, X):
        return np.tile(np.arange(1, self.n_ranks + 1, dtype=float), (len(X), 1))


@pytest.fixture
def results_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(config, "forests", {})
    _JOINT_CACHE.clear()
    return tmp_path


def test_loads_per_rank_pickles(results_dir):
    base = results_dir / "Full Model" / "Monopoly"
    base.mkdir(parents=True)
    for rank in (1, 2):
        joblib.dump(ConstantEffect(rank / 10), base / f"CF - Rank {rank}.pkl")

    loaded = config.load_monopoly_forests(ranks=[1, 2], verbose=False)
    assert loaded[2].const_marginal_effect(np.zeros((3, 37))).ravel().tolist() == [0.2] * 3


def test_per_rank_pickles_take_precedence_over_joint(results_dir):
    base = results_dir / "Full Model" / "Split 7"
    base.mkdir(parents=True)
    joblib.dump(ConstantEffect(0.5), base / "CF - Rank 1.pkl")
    joblib.dump(JointForestArtifact(cf=JointConstantEffect(1), ranks=[1]), base / "Joint CF.pkl")

    bucket = config.load_split_forests(7, ranks=[1], verbose=False)
    assert isinstance(bucket[1], ConstantEffect)


def test_falls_back_to_joint_forest(results_dir):
    base = results_dir / "Full Model" / "Monopoly"
    base.mkdir(parents=True)
    joblib.dump(JointForestArtifact(cf=JointConstantEffect(2), ranks=[1, 2]), base / "Joint CF.pkl")

    loaded = config.load_monopoly_forests(ranks=[1, 2], verbose=False)
    assert loaded[2].const_marginal_effect(np.zeros((2, 37))).tolist() == [2.0, 2.0]


def test_missing_forests_raise_with_the_missing_ranks(results_dir):
    base = results_dir / "Full Model" / "Monopoly"
    base.mkdir(parents=True)
    joblib.dump(ConstantEffect(0.1), base / "CF - Rank 1.pkl")

    with pytest.raises(FileNotFoundError, match=r"1 of 2 ranks.*\[2\]"):
        config.load_monopoly_forests(ranks=[1, 2], verbose=False)
