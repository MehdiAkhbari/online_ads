"""N-adaptive hyperparameter rule for `adsim.estimate`'s per-rank causal
forests.

Supersedes the fixed absolute values (min_samples_split=20000,
min_samples_leaf=10000, max_samples=50000 for *every* rank) previously
described in docs/hyperparameter_choices.txt. See that file for the full
derivation; this module is the implementation.

Deliberately isolated from `adsim.simulation_steps` / `adsim.estimate_joint`:
those still use the old static `param_grid` / `cf_param_grid` / grid-search
helpers unchanged. Only `adsim.estimate` (the per-rank/per-pair pipeline)
uses this module.
"""

from __future__ import annotations

from dataclasses import dataclass

# sklearn / adsim.propensity_model are imported lazily inside
# m_model_best_estimator / e_model_best_estimator below, not at module
# level: cf_hyperparams / m_model_leaf_grid / e_model_leaf_grid are pure
# arithmetic and should be importable (and unit-testable) without those
# heavier dependencies installed.


@dataclass(frozen=True)
class CFHyperparams:
    """Derived causal-forest hyperparameters for one (base ad, focal rank)
    pair. `min_samples_leaf_grid` and `min_var_fraction_leaf_grid` are
    meant to be cross-validated via `cf.tune()`; every other field is a
    fixed value or formula, not searched.
    """

    n_focal_used: int
    n_total: int
    max_samples: int
    min_samples_split: int
    min_samples_leaf_grid: list[int]
    min_var_fraction_leaf_grid: list[float]
    max_depth: int
    n_estimators: int
    min_balancedness_tol: float


def cf_hyperparams(
    n_base: int,
    n_focal: int,
    *,
    focal_cap_ratio: float = 3.0,
    max_samples_cap: int = 100_000,
    n_estimators: int = 500,
    max_depth: int = 3,
) -> CFHyperparams:
    """Derive one pair's causal-forest hyperparameters from its sample
    composition.

    - Caps the focal (usually majority) arm at `focal_cap_ratio` x the
      base arm: past ~3:1 the majority arm's contribution to
      Var(tau_hat) is already negligible, but it still consumes most of
      each tree's per-tree sample budget. Capping trades away that
      near-zero marginal benefit for far more base-arm representation
      per tree, and bounds every pair's N at a manageable size.
    - `max_samples = min(max_samples_cap, n_total // 2)`. The `// 2` is
      not optional: econml raises `ValueError` if an integer
      `max_samples` exceeds `n_samples // 2` under `inference="blb"`.
    - `min_samples_leaf` gets two candidates, `H // 4` and `H // 2` where
      `H = max_samples // 2`: honest=True splits each tree's subsample
      into a splitting half and an estimation half, and the leaf floor
      binds on both halves and both children, so `H // k` means "allow
      up to k leaves per honest half" -- a real, interpretable
      complexity difference for `cf.tune()`'s R-score to compare.
    - `min_samples_split = 2 * min(leaf grid)`: the loosest floor that
      still keeps min_samples_split non-binding (a node can't split
      unless both children can meet the leaf floor).
    """
    n_focal_used = max(1, min(n_focal, round(focal_cap_ratio * n_base)))
    n_total = n_base + n_focal_used
    max_samples = max(2, min(max_samples_cap, n_total // 2))
    h = max(1, max_samples // 2)
    leaf_grid = sorted({max(1, h // 4), max(1, h // 2)})
    min_samples_split = 2 * leaf_grid[0]
    return CFHyperparams(
        n_focal_used=n_focal_used,
        n_total=n_total,
        max_samples=max_samples,
        min_samples_split=min_samples_split,
        min_samples_leaf_grid=leaf_grid,
        min_var_fraction_leaf_grid=[0.01, 0.1],
        max_depth=max_depth,
        n_estimators=n_estimators,
        min_balancedness_tol=0.45,
    )


def _leaf_grid(n_total: int, rate: float, targets: tuple[float, float], caps: tuple[float, float]) -> list[int]:
    # Guards a same-pair zero-rate edge case (e.g. a pair with no clicks
    # at all) against ZeroDivisionError. This is a minimal safety net,
    # not the click-based degenerate/ATE-only handling described in the
    # design discussion -- that ladder is deferred; see
    # docs/hyperparameter_choices.txt.
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
    target ~10 and ~50 expected rare-arm units per leaf, capped at N/100
    and N/20. Uses the rare arm's rate (not the click rate) -- what e(X)
    needs bounded away from 0/1 is the rare arm's representation, since
    the binding count here is treated units, not clicks.
    """
    return _leaf_grid(n_total, rare_arm_rate, targets=(10.0, 50.0), caps=(100.0, 20.0))


def m_model_best_estimator(X, Y, leaf_grid: list[int], n_jobs: int):
    """Grid-search the outcome model m(X) over `leaf_grid`.

    max_features="sqrt" is fixed (not searched): sklearn's
    RandomForestRegressor defaults to max_features=1.0, i.e. searching
    all covariates at every split, which is the single biggest avoidable
    cost in nuisance fitting. n_estimators=100 is fixed for the same
    reason tune()'s own internal forests use 100 -- this is a nuisance
    model, not the final causal forest.
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import GridSearchCV

    model = RandomForestRegressor(n_estimators=100, max_features="sqrt", n_jobs=n_jobs, verbose=0)
    grid = GridSearchCV(
        estimator=model,
        param_grid={"min_samples_leaf": leaf_grid},
        scoring="neg_mean_squared_error",
        cv=3,
    )
    grid.fit(X, Y)
    return grid.best_params_, grid.best_estimator_


def e_model_best_estimator(X, T, leaf_grid: list[int], n_jobs: int):
    """Grid-search the propensity model e(X) over `leaf_grid`.

    Scored by neg_log_loss, not F1: DML needs a calibrated probability,
    not a thresholded decision, and with arm shares as lopsided as
    150-vs-100,000 rows, F1 will happily select a model that predicts
    the majority class everywhere.
    """
    from sklearn.model_selection import GridSearchCV

    from adsim.propensity_model import PropensityModel

    model = PropensityModel(n_estimators=100, max_features="sqrt", n_jobs=n_jobs)
    grid = GridSearchCV(
        estimator=model,
        param_grid={"min_samples_leaf": leaf_grid},
        scoring="neg_log_loss",
        cv=3,
    )
    grid.fit(X, T)
    return grid.best_params_, grid.best_estimator_
