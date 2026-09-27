"""Hyperparameter rule for `adsim.estimate`'s per-rank causal forests.

One (base ad, focal rank) pair at a time, three quantities decide
everything:

1. Arm balance. Neither arm may have more than `arm_ratio_cap` times the
   rows of the other; the larger arm is randomly subsampled down to that.
2. Rows per tree. `max_samples = min(max_samples_cap, N // 2)`, where N is
   the pair's row count after step 1. `N // 2` is the most econml accepts
   under `inference="blb"`, so this can never raise; the cap bounds runtime.
3. Leaf floor. `min_samples_leaf` is the leaf size at which a leaf is
   expected to hold `min_clicks_per_arm` clicks from EACH arm:
   `ceil(min_clicks_per_arm * N / c_min)`, where c_min is the expected click
   count of the arm with fewer clicks. econml enforces the floor on both
   honest halves of every tree, so a pair whose half-sample cannot hold two
   such leaves gets unsplit trees, i.e. an average effect only.

No thresholds or branches: max_samples is piecewise linear in N, and the
leaf floor depends only on the arms' shares and click rates, so the number
of leaves a tree can grow (`max_samples / 2 / min_samples_leaf`) rises
smoothly with N until the per-tree cap binds. See
docs/hyperparameter_choices.txt for the reasoning behind each default.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

# sklearn / adsim.propensity_model are imported lazily inside the
# *_best_estimator helpers so the pure-arithmetic rule stays importable
# (and unit-testable) without them.

ARM_RATIO_CAP: float = 3.0
MAX_SAMPLES_CAP: int = 200_000
MIN_CLICKS_PER_ARM: float = 5.0
MIN_VAR_FRACTION_LEAF: float = 0.1
N_ESTIMATORS: int = 500


def arm_sizes_after_cap(n_base: int, n_focal: int, arm_ratio_cap: float = ARM_RATIO_CAP) -> tuple[int, int]:
    """Rows kept in each arm so that neither exceeds `arm_ratio_cap` x the other."""
    if arm_ratio_cap < 1:
        raise ValueError(f"arm_ratio_cap must be >= 1, got {arm_ratio_cap}")
    cap = math.floor(arm_ratio_cap * min(n_base, n_focal))
    return min(n_base, cap), min(n_focal, cap)


@dataclass(frozen=True)
class CFPlan:
    """Causal-forest settings for one pair, plus the inputs that produced them."""

    n_base: int
    n_focal: int
    n_total: int
    expected_clicks_base: float
    expected_clicks_focal: float
    max_samples: int
    min_samples_leaf: int
    min_samples_split: int
    min_var_fraction_leaf: float
    n_estimators: int
    max_leaves_per_tree: float

    def as_dict(self) -> dict:
        return asdict(self)


def cf_hyperparams(
    n_base: int,
    n_focal: int,
    click_rate_base: float,
    click_rate_focal: float,
    *,
    max_samples_cap: int = MAX_SAMPLES_CAP,
    min_clicks_per_arm: float = MIN_CLICKS_PER_ARM,
    min_var_fraction_leaf: float = MIN_VAR_FRACTION_LEAF,
    n_estimators: int = N_ESTIMATORS,
) -> CFPlan:
    """Derive one pair's causal-forest settings.

    `n_base` / `n_focal` are the arm sizes AFTER `arm_sizes_after_cap`;
    the click rates are each arm's observed click rate.
    """
    n_total = n_base + n_focal
    if n_total < 2:
        raise ValueError(f"need at least 2 rows to fit a forest, got {n_total}")

    max_samples = min(max_samples_cap, n_total // 2)

    clicks_base = n_base * click_rate_base
    clicks_focal = n_focal * click_rate_focal
    c_min = min(clicks_base, clicks_focal)
    if c_min > 0:
        min_samples_leaf = min(n_total, math.ceil(min_clicks_per_arm * n_total / c_min))
    else:
        min_samples_leaf = n_total

    return CFPlan(
        n_base=n_base,
        n_focal=n_focal,
        n_total=n_total,
        expected_clicks_base=clicks_base,
        expected_clicks_focal=clicks_focal,
        max_samples=max_samples,
        min_samples_leaf=min_samples_leaf,
        min_samples_split=2 * min_samples_leaf,
        min_var_fraction_leaf=min_var_fraction_leaf,
        n_estimators=n_estimators,
        max_leaves_per_tree=(max_samples / 2) / min_samples_leaf,
    )


def _leaf_grid(n_total: int, rate: float, targets: tuple[float, float], caps: tuple[float, float]) -> list[int]:
    rate = max(rate, 1.0 / n_total)
    candidates = [
        min(target / rate, n_total / cap)
        for target, cap in zip(targets, caps)
    ]
    return sorted({max(1, int(round(c))) for c in candidates})


def m_model_leaf_grid(n_total: int, click_rate: float) -> list[int]:
    """Leaf-size candidates for the outcome model m(X) = E[Y|X]: target
    ~10 and ~100 expected clicks per leaf, capped at N/40 and N/10.
    """
    return _leaf_grid(n_total, click_rate, targets=(10.0, 100.0), caps=(40.0, 10.0))


def e_model_leaf_grid(n_total: int, rare_arm_rate: float) -> list[int]:
    """Leaf-size candidates for the propensity model e(X) = P(T=1|X):
    target ~10 and ~50 expected rare-arm rows per leaf, capped at N/100
    and N/20.
    """
    return _leaf_grid(n_total, rare_arm_rate, targets=(10.0, 50.0), caps=(100.0, 20.0))


def _cv_splitter(groups, random_state: int, n_splits: int = 3):
    """Stratified folds (by the target being predicted), grouped by user
    when user ids are available so one user's impressions never straddle
    a train/validation boundary."""
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

    if groups is None:
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def m_model_best_estimator(X, Y, leaf_grid: list[int], n_jobs: int, groups=None, random_state: int = 42):
    """Grid-search the outcome model m(X) over `leaf_grid` (scored by MSE,
    i.e. the Brier score for a binary Y). Returns the best parameters and
    an UNFITTED estimator configured with them."""
    from sklearn.base import clone
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import GridSearchCV

    model = RandomForestRegressor(n_estimators=100, max_features="sqrt", n_jobs=n_jobs,
                                  random_state=random_state, verbose=0)
    grid = GridSearchCV(
        estimator=model,
        param_grid={"min_samples_leaf": leaf_grid},
        scoring="neg_mean_squared_error",
        cv=_cv_splitter(groups, random_state),
        refit=False,
    )
    # Stratify on Y so every fold sees clicks.
    grid.fit(X, Y, groups=groups)
    return grid.best_params_, clone(model).set_params(**grid.best_params_)


def e_model_best_estimator(X, T, leaf_grid: list[int], n_jobs: int, groups=None, random_state: int = 42):
    """Grid-search the propensity model e(X) over `leaf_grid`, scored by
    log-loss: DML needs calibrated probabilities, and a thresholded metric
    such as F1 scores every candidate ~0 when one arm is rare. Returns the
    best parameters and an UNFITTED estimator configured with them."""
    from sklearn.base import clone
    from sklearn.model_selection import GridSearchCV

    from adsim.propensity_model import PropensityModel

    model = PropensityModel(n_estimators=100, max_features="sqrt", n_jobs=n_jobs, random_state=random_state)
    grid = GridSearchCV(
        estimator=model,
        param_grid={"min_samples_leaf": leaf_grid},
        scoring="neg_log_loss",
        cv=_cv_splitter(groups, random_state),
        refit=False,
    )
    grid.fit(X, T, groups=groups)
    return grid.best_params_, clone(model).set_params(**grid.best_params_)
