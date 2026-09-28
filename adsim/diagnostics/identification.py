"""Identification tests (revision plan, section 4.2; referees R1.3a, R2, R3).

Assumption 1 says the ad shown is as good as random given the page and
time context C that the propensity model conditions on. Each check below
tests one implication of it on a (base ad, rank r) pair and returns a
dict of named tables. Check 7 of section 4.2 (conditioning on bids and
pacing) needs allocation inputs the data does not contain.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from adsim.diagnostics.data import (
    AD_HISTORY_COLS,
    BROWSING_COLS,
    CONTEXT_COLS,
    FEATURE_SETS,
    PLACEBO_LAG_COL,
    RankData,
    cross_fit,
    forest_ate,
)
from adsim.diagnostics.stats import (
    EPS,
    clustered_mean,
    cluster_robust_wald,
    crump_alpha,
    demean_within,
    mixed_cell_share,
    null_draw_test,
    ovb_bounds,
    ovb_params,
    partial_r2,
    partialling_out_ate,
    permute_within,
    robustness_value,
    standardized_mean_difference,
    within_cell_smd,
)

HISTORY_BLOCKS = {"browsing": BROWSING_COLS, "ad_history": AD_HISTORY_COLS}


def _log_loss_rows(t: np.ndarray, e: np.ndarray) -> np.ndarray:
    e = np.clip(e, EPS, 1 - EPS)
    return -(t * np.log(e) + (1 - t) * np.log(1 - e))


def propensity_information(rd: RankData, *, n_strata: int = 20) -> dict[str, pd.DataFrame]:
    """4.2.1: does user history help predict which ad is shown, beyond C?

    The same learner (the forest's propensity settings) is cross-fitted on
    C alone and on C plus each history block; `delta_log_loss` is the
    out-of-sample log-loss gain over e(C). Two forests on different feature
    sets overfit differently, so even an irrelevant block shifts the loss,
    and at N ~ 10^5 that shift looks significant. The test therefore
    compares the gain with `settings.n_permutations` refits in which the
    added block is shuffled within `n_strata` quantile strata of e(C)
    (under Assumption 1, T is independent of history given e(C), so this is
    the null). `excess_gain` = gain minus the null mean; `z_perm` and
    `p_perm` test it against the null draws (`stats.null_draw_test`);
    `p_perm_empirical` is the rank-based permutation p-value.

    Caveat: if `impression_repeat` / `previous_clicks` count the ad actually
    shown rather than rank r, the ad-history block predicts T mechanically.
    """
    t, groups = rd.T, rd.groups
    e_context = rd.propensity("context")
    base_ll = _log_loss_rows(t, e_context)
    base_brier = np.mean((t - e_context) ** 2)
    strata = pd.qcut(e_context, n_strata, labels=False, duplicates="drop")
    rng = np.random.default_rng(rd.settings.random_state)

    rows = []
    for name, cols in FEATURE_SETS.items():
        e = rd.propensity(name)
        ll = _log_loss_rows(t, e)
        gain, se = clustered_mean(base_ll - ll, groups)
        row = {"feature_set": name, "n": len(t), "log_loss": ll.mean(), "auc": roc_auc_score(t, e),
               "brier": np.mean((t - e) ** 2), "partial_r2_T": 1 - np.mean((t - e) ** 2) / base_brier,
               "gain": gain, "gain_se": se}
        if name != "context" and rd.settings.n_permutations > 0:
            added = [c for c in cols if c not in CONTEXT_COLS]
            null = np.array([
                np.mean(base_ll - _log_loss_rows(t, _permuted_block_propensity(rd, cols, added, strata, rng)))
                for _ in range(rd.settings.n_permutations)
            ])
            z, p = null_draw_test(gain, null)
            row.update({"null_gain_mean": null.mean(), "null_gain_sd": null.std(ddof=1),
                        "excess_gain": gain - null.mean(), "z_perm": z, "p_perm": p,
                        "p_perm_empirical": (1 + np.sum(null >= gain)) / (1 + len(null))})
        rows.append(row)
    return {"propensity_info": pd.DataFrame(rows)}


def _permuted_block_propensity(rd: RankData, cols: list[str], block: list[str], strata, rng) -> np.ndarray:
    """Cross-fitted P(T = 1 | cols) with the `block` columns shuffled within strata."""
    X = rd.pair[cols].copy()
    X[block] = X[block].to_numpy()[permute_within(np.arange(len(X)), strata, rng)]
    return cross_fit(rd.propensity_learner(), X, rd.T, rd.groups,
                     n_folds=rd.settings.n_folds, random_state=rd.settings.random_state, proba=True)


def covariate_balance(rd: RankData) -> dict[str, pd.DataFrame]:
    """4.2.2: standardized mean differences of user history, focal vs base ad.

    Three versions: raw, within website x day x hour cells, and inverse-
    propensity weighted with e(C). The omnibus test regresses T on each
    history block with cell fixed effects (user-clustered Wald F). Only rows
    in cells showing both ads inform the within-cell versions; their share
    is reported, and coarser `--cell-cols` raise it.
    """
    t, cells, groups = rd.T, rd.cells, rd.groups
    e = np.clip(rd.propensity("context"), EPS, 1 - EPS)
    ipw = np.where(t == 1, 1 / e, 1 / (1 - e))
    smd_rows = []
    for block, cols in HISTORY_BLOCKS.items():
        for col in cols:
            x = rd.pair[col].to_numpy(dtype=float)
            smd_rows.append({
                "covariate": col,
                "block": block,
                "smd_raw": standardized_mean_difference(x, t),
                "smd_within_cell": within_cell_smd(x, t, cells),
                "smd_ipw": standardized_mean_difference(x, t, weights=ipw),
            })

    t_dm = demean_within(t, cells).ravel()
    mixed = mixed_cell_share(t, cells)
    omnibus_rows = []
    for block, cols in {**HISTORY_BLOCKS, "all_history": BROWSING_COLS + AD_HISTORY_COLS}.items():
        test = cluster_robust_wald(t_dm, demean_within(rd.pair[cols], cells), groups)
        omnibus_rows.append({"block": block, "n": len(t), "n_cells": len(np.unique(cells)),
                             "share_rows_in_mixed_cells": mixed, **test})
    return {"balance_smd": pd.DataFrame(smd_rows), "balance_omnibus": pd.DataFrame(omnibus_rows)}


def placebo_outcomes(rd: RankData) -> dict[str, pd.DataFrame]:
    """4.2.3: the ad shown now should not "affect" outcomes realised before it.

    Placebo outcomes: the user's click on their previous impression (when the
    data has a visit order) and whether they had clicked any ad before. The
    effect is a DML partialling-out estimate controlling for C only, the
    conditioning set of Assumption 1.
    """
    placebos = {"any_prior_click": (rd.pair["previous_clicks_all_ads"] > 0).astype(float)}
    if PLACEBO_LAG_COL in rd.pair:
        placebos["lag_click"] = rd.pair[PLACEBO_LAG_COL]

    e_all = rd.propensity("context")
    rows = []
    for name, outcome in placebos.items():
        valid = outcome.notna().to_numpy()
        y = outcome.to_numpy(dtype=float)[valid]
        if valid.sum() < 50 or np.unique(y).size < 2:
            continue
        groups = None if rd.groups is None else rd.groups[valid]
        m = cross_fit(rd.outcome_learner(), rd.pair.loc[valid, CONTEXT_COLS], y.astype(int), groups,
                      n_folds=rd.settings.n_folds, random_state=rd.settings.random_state, proba=False)
        est = partialling_out_ate(y, rd.T[valid], m, e_all[valid], groups)
        rows.append({"placebo_outcome": name, "outcome_mean": y.mean(), **est,
                     "effect_relative_to_mean": est["theta"] / y.mean()})
    return {"placebo_outcomes": pd.DataFrame(rows)}


def refutation_tests(rd: RankData, *, subset_frac: float = 0.8) -> dict[str, pd.DataFrame]:
    """4.2.4: refit the saved forest's configuration on altered data.

    - placebo_treatment: T permuted within website x day x hour cells
      (keeps each cell's ad mix); the ATE and CATE spread should vanish.
      Labels can only move in cells showing both ads, so `share_T_changed`
      is reported: if it is small, the placebo is uninformative and
      `--cell-cols` should be coarser.
    - random_common_cause: an extra N(0, 1) control; the ATE should not move.
    - data_subset: a random `subset_frac` of users; the ATE should not move.
    These are the DoWhy refuters the plan names, done natively because
    econml's `dowhy` wrapper needs the optional dowhy package.
    """
    rng = np.random.default_rng(rd.settings.random_state)
    pair = rd.pair
    variants = {"observed": (pair, None)}

    permuted = pair.copy()
    permuted["advertiser_rank"] = permute_within(pair["advertiser_rank"].to_numpy(), rd.cells, rng)
    variants["placebo_treatment"] = (permuted, None)
    variants["random_common_cause"] = (pair, rng.normal(size=(len(pair), 1)))

    if rd.groups is not None:
        users = np.unique(rd.groups)
        keep_users = rng.choice(users, size=max(1, int(subset_frac * len(users))), replace=False)
        subset = pair[np.isin(rd.groups, keep_users)].reset_index(drop=True)
    else:
        subset = pair.sample(frac=subset_frac, random_state=rd.settings.random_state).reset_index(drop=True)
    variants["data_subset"] = (subset, None)

    X_eval = rd.X.sample(n=min(5_000, len(pair)), random_state=rd.settings.random_state)
    share_changed = float(np.mean((permuted["advertiser_rank"] != 0).to_numpy() != (rd.T == 1)))
    rows = []
    for name, (data, W) in variants.items():
        cf = rd.refit(data, W=W)
        ate, se = forest_ate(cf)
        tau = np.ravel(cf.const_marginal_effect(X_eval))
        lo, hi = cf.effect_interval(X_eval, alpha=0.05)
        rows.append({
            "test": name,
            "n": len(data),
            "share_T_changed": share_changed if name == "placebo_treatment" else np.nan,
            "ate": ate,
            "ate_se": se,
            "ate_ci_lo": ate - 1.96 * se,
            "ate_ci_hi": ate + 1.96 * se,
            "cate_sd": tau.std(),
            "share_cate_ci_excludes_0": float(np.mean((np.ravel(lo) > 0) | (np.ravel(hi) < 0))),
        })
    out = pd.DataFrame(rows)
    observed = out.loc[out["test"] == "observed", "ate"].iloc[0]
    out["expected"] = out["test"].map({"observed": "", "placebo_treatment": "ate ~ 0",
                                       "random_common_cause": "ate ~ observed", "data_subset": "ate ~ observed"})
    out["ate_minus_observed"] = out["ate"] - observed
    return {"refuters": out}


def ovb_sensitivity(rd: RankData, *, alpha: float = 0.05,
                    grid: tuple[float, ...] = (0.01, 0.02, 0.05, 0.1)) -> dict[str, pd.DataFrame]:
    """4.2.5: omitted-variable-bias bounds (Chernozhukov et al., REStat 2026).

    Uses the forest's DML residuals (outcome model on all 37 features,
    propensity on C), cross-fitted here. The benchmark row sets (c_y, c_t)
    to the explanatory power of the observed browsing history: its partial
    R-squared for the outcome beyond the other features, and for the ad shown
    beyond C. Robustness values give the c = c_y = c_t at which the point
    bound (rv_theta) or the confidence bound (rv_ci) first reaches 0.
    """
    t, y, groups = rd.T, rd.Y, rd.groups
    y_res = y - rd.outcome("all")
    t_res = t - rd.propensity("context")
    params = ovb_params(y_res, t_res, groups)

    c_y_bench = partial_r2(y - rd.outcome("context+ad_history"), y_res)
    c_t_bench = partial_r2(t_res, t - rd.propensity("context+browsing"))
    scenarios = [("none", 0.0, 0.0), ("benchmark: browsing history", c_y_bench, c_t_bench)]
    scenarios += [(f"c_y = c_t = {c:g}", c, c) for c in grid]

    se = float(np.sqrt(params["cov"][0, 0]))
    rows = [{"scenario": name, "c_y": c_y, "c_t": c_t, "theta": params["theta"], "theta_se": se,
             **ovb_bounds(params, c_y, c_t, alpha=alpha)}
            for name, c_y, c_t in scenarios]
    out = pd.DataFrame(rows)
    out["rv_theta"] = robustness_value(params, alpha=alpha, interval="theta")
    out["rv_ci"] = robustness_value(params, alpha=alpha, interval="ci")
    out["base_ctr"] = y[t == 0].mean()
    return {"sensitivity": out}


def overlap(rd: RankData, *, n_bins: int = 40) -> dict[str, pd.DataFrame]:
    """4.2.6: distribution of e(C) by arm and the effect of Crump et al. trimming.

    Reports the optimal Crump-Hotz-Imbens-Mitnik threshold and the DML
    effect (outcome model on all features, propensity on C) on the full
    pair, after Crump trimming and after the 0.1 / 0.9 rule of thumb. The
    change in simulated gains needs forests re-fit on the trimmed sample.
    """
    t, y, groups = rd.T, rd.Y, rd.groups
    e = rd.propensity("context")
    m = rd.outcome("all")
    alpha = crump_alpha(e)

    summary = {"n": len(e), "crump_alpha": alpha}
    for arm, label in ((0, "base"), (1, "focal")):
        q = np.quantile(e[t == arm], [0, 0.01, 0.05, 0.5, 0.95, 0.99, 1])
        summary.update({f"e_{label}_{k}": v for k, v in zip(["min", "p1", "p5", "p50", "p95", "p99", "max"], q)})
    summary["share_outside_crump"] = float(np.mean((e < alpha) | (e > 1 - alpha)))
    summary["share_outside_0.1_0.9"] = float(np.mean((e < 0.1) | (e > 0.9)))

    trim_rows = []
    for rule, lo in (("none", 0.0), ("crump", alpha), ("0.1-0.9", 0.1)):
        keep = (e >= lo) & (e <= 1 - lo)
        g = None if groups is None else groups[keep]
        est = partialling_out_ate(y[keep], t[keep], m[keep], e[keep], g) if keep.sum() > 10 else {}
        trim_rows.append({"trimming": rule, "threshold": lo, "share_kept": keep.mean(), **est})

    edges = np.linspace(0, 1, n_bins + 1)
    hist = pd.DataFrame({
        "bin_lo": edges[:-1],
        "bin_hi": edges[1:],
        "base": np.histogram(e[t == 0], bins=edges)[0],
        "focal": np.histogram(e[t == 1], bins=edges)[0],
    })
    return {"overlap_summary": pd.DataFrame([summary]), "overlap_trimming": pd.DataFrame(trim_rows),
            "overlap_histogram": hist}


__all__ = [
    "propensity_information",
    "covariate_balance",
    "placebo_outcomes",
    "refutation_tests",
    "ovb_sensitivity",
    "overlap",
]
