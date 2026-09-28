"""Data, feature blocks and per-rank containers shared by the diagnostic checks.

A `RankData` holds one (base ad, rank r) pair exactly as `adsim.estimate`
built it for that rank's saved forest (same arm cap and seed, read from
`cf.adsim_plan_`), plus cross-fitted nuisance predictions that several
checks share. Folds are stratified on the target and grouped by user,
with the same fallback to row folds as `adsim.estimate`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit, cross_val_predict

from adsim.estimate import build_forest, build_pair, pair_groups, plan_for_pair
from adsim.hyperparams import (
    ARM_RATIO_CAP,
    MAX_SAMPLES_CAP,
    MIN_CLICKS_PER_ARM,
    MIN_VAR_FRACTION_LEAF,
    _cv_splitter,
    e_model_leaf_grid,
    m_model_leaf_grid,
)
from adsim.propensity_model import X_treat_indices, PropensityModel
from adsim.simulation_steps import define_xyt, extract_ranks, prepare_data

log = logging.getLogger("adsim.diagnostics")

# The 37 forest features in three blocks (revision plan, section 4.6). The
# propensity model sees CONTEXT only, which is the identifying assumption.
AD_HISTORY_COLS = [
    "impression_repeat", "impression_repeat_base_ad",
    "previous_clicks", "previous_clicks_base_ad", "previous_clicks_all_ads",
]
BROWSING_COLS = ["total_visits", *[f"visit_s{i}" for i in range(1, 14)]]
CONTEXT_COLS = list(X_treat_indices)
FEATURE_COLS = AD_HISTORY_COLS + BROWSING_COLS + CONTEXT_COLS
HISTORY_COLS = AD_HISTORY_COLS + BROWSING_COLS

FEATURE_SETS: dict[str, list[str]] = {
    "context": CONTEXT_COLS,
    "context+browsing": CONTEXT_COLS + BROWSING_COLS,
    "context+ad_history": CONTEXT_COLS + AD_HISTORY_COLS,
    "all": FEATURE_COLS,
}

# Website (subject dummies x rank within subject) x day x hour.
DEFAULT_CELL_COLS = [*[f"sub_{i}" for i in range(1, 14)], "publisher_rank_sub", "day", "hour"]

PLACEBO_LAG_COL = "placebo_lag_click"
SEGMENT_COLS = ["segment_device", "segment_user", "segment_browsing"]


@dataclass(frozen=True)
class Settings:
    group_col: str | None = "global_token_new"
    cell_cols: tuple[str, ...] = tuple(DEFAULT_CELL_COLS)
    n_folds: int = 3
    n_jobs: int = 4
    random_state: int = 42
    refit_trees: int = 100
    holdout_frac: float = 0.3


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------

def load_estimation_data(path: Path, *, group_col: str | None, visit_col: str | None) -> pd.DataFrame:
    """Read an estimation .dta, recode the base ad to 0 and add the lagged-click placebo outcome."""
    log.info("loading %s", path)
    data = pd.read_stata(path)
    prepare_data(data, base_ad=50, max_ad=100)
    add_lag_click(data, group_col=group_col, visit_col=visit_col)
    add_segments(data)
    return data


def add_segments(data: pd.DataFrame) -> None:
    """User segments for the diagnostics tables (plan 4.3): device, first
    visit vs returning (assumes `total_visits` counts visits up to and
    including this one), and heavy vs light browsers (above / at or below
    the median `total_visits`)."""
    data["segment_device"] = np.where(data["mobile"] == 1, "mobile", "desktop")
    data["segment_user"] = np.where(data["total_visits"] <= 1, "first_visit", "returning")
    heavy = data["total_visits"] > data["total_visits"].median()
    data["segment_browsing"] = np.where(heavy, "heavy", "light")


def add_lag_click(data: pd.DataFrame, *, group_col: str | None, visit_col: str | None) -> None:
    """`placebo_lag_click` = the user's click on their previous impression (NaN on the first)."""
    if not group_col or not visit_col or group_col not in data or visit_col not in data:
        log.info("no %r / %r columns: lagged-click placebo unavailable", group_col, visit_col)
        return
    order = data.sort_values([group_col, visit_col]).index
    lagged = data.loc[order].groupby(group_col)["is_clicked"].shift(1)
    data[PLACEBO_LAG_COL] = lagged.reindex(data.index)


def diagnosable_ranks(data: pd.DataFrame) -> list[int]:
    """Ranks `adsim.estimate` fits: all but the base ad (0) and the fringe entry."""
    ranks = extract_ranks(data)
    return [r for r in ranks[:-1] if r != 0]


def top_ranks(data: pd.DataFrame, k: int) -> list[int]:
    counts = data["advertiser_rank"].value_counts()
    return [r for r in counts.index if r in set(diagnosable_ranks(data))][:k]


def load_rank_forest(forest_dir: Path, rank: int):
    path = Path(forest_dir) / f"CF - Rank {rank}.pkl"
    return joblib.load(path) if path.is_file() else None


def support_buckets(data: pd.DataFrame) -> pd.Series:
    """Impression-count terciles of the fitted ranks: rank -> low / medium / high."""
    counts = data["advertiser_rank"].value_counts()
    counts = counts[counts.index.isin(diagnosable_ranks(data))]
    if len(counts) < 3:
        return pd.Series("all", index=counts.index)
    labels = pd.qcut(counts.rank(method="first"), 3, labels=["low", "medium", "high"])
    return labels.astype(str)


# ----------------------------------------------------------------------
# Generic cross-fitting
# ----------------------------------------------------------------------

def cross_fit(model, X, y, groups, *, n_folds: int, random_state: int, proba: bool) -> np.ndarray:
    """Out-of-fold predictions, folds stratified on `y` and grouped by user."""
    cv = _cv_splitter(groups, random_state, n_splits=n_folds)
    if proba:
        return cross_val_predict(model, X, y, groups=groups, cv=cv, method="predict_proba")[:, 1]
    return cross_val_predict(model, X, y, groups=groups, cv=cv)


def user_split(groups, n: int, *, test_frac: float, random_state: int) -> np.ndarray:
    """Boolean mask of held-out rows; whole users are held out when ids exist."""
    if groups is None:
        splitter = ShuffleSplit(n_splits=1, test_size=test_frac, random_state=random_state)
        _, test = next(splitter.split(np.zeros(n)))
    else:
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=random_state)
        _, test = next(splitter.split(np.zeros(n), groups=groups))
    mask = np.zeros(n, dtype=bool)
    mask[test] = True
    return mask


# ----------------------------------------------------------------------
# One (base ad, rank) pair
# ----------------------------------------------------------------------

@dataclass
class RankData:
    rank: int
    pair: pd.DataFrame
    cf: object | None
    settings: Settings
    _cache: dict = field(default_factory=dict, repr=False)

    @classmethod
    def build(cls, data: pd.DataFrame, rank: int, forest_dir: Path | None, settings: Settings) -> "RankData":
        cf = load_rank_forest(forest_dir, rank) if forest_dir is not None else None
        plan = getattr(cf, "adsim_plan_", {})
        pair = build_pair(
            data, rank,
            arm_ratio_cap=plan.get("arm_ratio_cap", ARM_RATIO_CAP),
            random_state=plan.get("random_state", settings.random_state),
        )
        return cls(rank=rank, pair=pair, cf=cf, settings=settings)

    # -- arrays ---------------------------------------------------------

    @property
    def X(self) -> pd.DataFrame:
        return self.pair[FEATURE_COLS]

    @property
    def T(self) -> np.ndarray:
        return (self.pair["advertiser_rank"] != 0).astype(int).to_numpy()

    @property
    def Y(self) -> np.ndarray:
        return self.pair["is_clicked"].astype(int).to_numpy()

    @property
    def groups(self) -> np.ndarray | None:
        return groups_for(self.pair, self.settings.group_col)

    @property
    def cells(self) -> np.ndarray:
        cols = [c for c in self.settings.cell_cols if c in self.pair]
        return self.pair.groupby(cols, sort=False).ngroup().to_numpy()

    # -- learners: the saved forest's nuisance settings when available ----

    def propensity_params(self) -> dict:
        if self.cf is not None:
            params = dict(self.cf.model_t.get_params())
        else:
            rare = min(self.T.mean(), 1 - self.T.mean())
            params = {"n_estimators": 100, "max_features": "sqrt",
                      "min_samples_leaf": max(e_model_leaf_grid(len(self.pair), rare))}
        return {**params, "n_jobs": self.settings.n_jobs, "random_state": self.settings.random_state}

    def outcome_params(self) -> dict:
        if self.cf is not None:
            params = dict(self.cf.model_y.get_params())
        else:
            params = {"n_estimators": 100, "max_features": "sqrt",
                      "min_samples_leaf": max(m_model_leaf_grid(len(self.pair), float(self.Y.mean())))}
        return {**params, "n_jobs": self.settings.n_jobs, "random_state": self.settings.random_state}

    def propensity_learner(self) -> RandomForestClassifier:
        return RandomForestClassifier(**self.propensity_params())

    def outcome_learner(self) -> RandomForestRegressor:
        return RandomForestRegressor(**self.outcome_params())

    def propensity_model(self) -> PropensityModel:
        """The forest's own T-model (page context only)."""
        return PropensityModel(**self.propensity_params())

    # -- cached cross-fitted nuisances ------------------------------------

    def propensity(self, feature_set: str) -> np.ndarray:
        """Out-of-fold P(T = 1 | features in `feature_set`)."""
        key = ("e", feature_set)
        if key not in self._cache:
            self._cache[key] = cross_fit(
                self.propensity_learner(), self.pair[FEATURE_SETS[feature_set]], self.T, self.groups,
                n_folds=self.settings.n_folds, random_state=self.settings.random_state, proba=True,
            )
        return self._cache[key]

    def outcome(self, feature_set: str) -> np.ndarray:
        """Out-of-fold E[Y | features in `feature_set`] (the DML outcome model, without T)."""
        key = ("m", feature_set)
        if key not in self._cache:
            self._cache[key] = cross_fit(
                self.outcome_learner(), self.pair[FEATURE_SETS[feature_set]], self.Y, self.groups,
                n_folds=self.settings.n_folds, random_state=self.settings.random_state, proba=False,
            )
        return self._cache[key]

    # -- refitting the saved forest's configuration ----------------------

    def plan_for(self, pair: pd.DataFrame):
        """The saved forest's hyperparameter rule applied to `pair`'s row count and clicks."""
        plan_saved = getattr(self.cf, "adsim_plan_", {})
        return plan_for_pair(
            pair,
            max_samples_cap=plan_saved.get("max_samples", MAX_SAMPLES_CAP),
            min_clicks_per_arm=plan_saved.get("min_clicks_per_arm", MIN_CLICKS_PER_ARM),
            min_var_fraction_leaf=plan_saved.get("min_var_fraction_leaf", MIN_VAR_FRACTION_LEAF),
            n_estimators=self.settings.refit_trees,
        )

    def refit(self, pair: pd.DataFrame, *, W=None):
        """Fit a CausalForestDML on `pair` with the saved forest's rule and
        nuisance settings and `settings.refit_trees` trees."""
        X, Y, T = define_xyt(pair)
        T = (T != 0).astype(int)
        cf = build_forest(self.plan_for(pair), self.outcome_learner(), self.propensity_model(),
                          n_jobs=self.settings.n_jobs, random_state=self.settings.random_state)
        cf.fit(Y=Y, T=T, X=X, W=W, groups=groups_for(pair, self.settings.group_col), inference="blb")
        return cf

    def holdout_mask(self) -> np.ndarray:
        return user_split(self.groups, len(self.pair),
                          test_frac=self.settings.holdout_frac, random_state=self.settings.random_state)


def groups_for(pair: pd.DataFrame, group_col: str | None) -> np.ndarray | None:
    T = (pair["advertiser_rank"] != 0).astype(int)
    return pair_groups(pair, T, group_col)


def forest_ate(cf) -> tuple[float, float]:
    """The forest's doubly robust ATE and its standard error."""
    return float(np.ravel(cf.ate_)[0]), float(np.ravel(cf.ate_stderr_)[0])


def fitted_nuisance_predictions(cf, X) -> tuple[np.ndarray, np.ndarray]:
    """m(X) and e(X) for new rows, averaged over the forest's cross-fitted nuisance models."""
    X = np.asarray(X, dtype=float)
    m = np.mean([mdl.predict(X) for mdl in cf.models_y[0]], axis=0)
    e = np.mean([mdl.predict_proba(X)[:, 1] for mdl in cf.models_t[0]], axis=0)
    return np.ravel(m), np.ravel(e)
