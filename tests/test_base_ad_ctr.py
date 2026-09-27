"""Tests for the base-ad click rate y0 = m1 - tau_1 * e1 and its helpers."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor

import adsim.config as config
from adsim.propensity_model import PropensityModel
from adsim.simulate.base_ad_helpers import fit_helpers
from adsim.simulation_steps import calc_base_ad_ctr_vector, define_xyt, treated_probability


class FixedProba:
    def __init__(self, proba, classes=(0, 1)):
        self.proba = np.asarray(proba, dtype=float)
        self.classes_ = np.asarray(classes)

    def predict_proba(self, X):
        return np.tile(self.proba, (len(X), 1))


class FixedValue:
    def __init__(self, value):
        self.value = value

    def predict(self, X):
        return np.full(len(X), self.value)

    def const_marginal_effect(self, X):
        return np.full((len(X), 1), self.value)


def test_treated_probability_takes_the_class_1_column():
    out = treated_probability(FixedProba([0.8, 0.2]), np.zeros((3, 2)))
    assert out.shape == (3, 1)
    assert np.allclose(out, 0.2)


def test_treated_probability_respects_class_order():
    out = treated_probability(FixedProba([0.3, 0.7], classes=(1, 0)), np.zeros((2, 2)))
    assert np.allclose(out, 0.3)


def test_treated_probability_rejects_a_single_class_model():
    # What the old base_ad_helpers produced: T coded 1 for every row.
    with pytest.raises(ValueError, match="base_ad_helpers"):
        treated_probability(FixedProba([1.0], classes=(1,)), np.zeros((2, 2)))


def test_calc_base_ad_ctr_vector_formula(make_pair, monkeypatch):
    data = make_pair(3, 3)
    data["user_visit_no"] = [1, 1, 2, 1, 2, 1]
    for col in ("r_50", "c_50", "r_1", "c_1"):
        data[col] = 0
    monkeypatch.setattr(config, "forests", {1: FixedValue(0.002)})
    monkeypatch.setattr(config, "helpers", {"m1": FixedValue(0.01), "e1": FixedProba([0.25, 0.75])})

    y0 = calc_base_ad_ctr_vector(data, user_visit_no=1)
    assert y0.shape == (4, 1)
    assert np.allclose(y0, 0.01 - 0.002 * 0.75)


def test_fit_helpers_uses_base_ad_coded_zero(make_pair):
    data = make_pair(600, 1_800, rank=1)
    data.loc[:9, "advertiser_rank"] = 5
    cf_1 = type("CF", (), {})()
    cf_1.model_y = RandomForestRegressor(n_estimators=5, min_samples_leaf=20, random_state=0)
    cf_1.model_t = PropensityModel(n_estimators=5, min_samples_leaf=20, random_state=0)
    cf_1.adsim_plan_ = {"arm_ratio_cap": 3.0, "random_state": 0}

    m1, e1 = fit_helpers(data, cf_1)
    assert list(e1.classes_) == [0, 1]
    X = define_xyt(data.head(50))[0]
    assert treated_probability(e1, X).shape == (50, 1)
    assert m1.predict(X).shape == (50,)


def test_fit_helpers_rejects_data_without_a_base_ad(make_pair):
    data = make_pair(0, 100, rank=1)
    cf_1 = type("CF", (), {})()
    cf_1.model_y = RandomForestRegressor(n_estimators=2)
    cf_1.model_t = PropensityModel(n_estimators=2)
    with pytest.raises(ValueError):
        fit_helpers(data, cf_1)
