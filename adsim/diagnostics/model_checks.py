"""Model diagnostics and robustness (revision plan, sections 2.2 and 4.3;
referees R1.4, R3, R4 CV).

Per-rank checks take a `RankData`. The pipeline checks (`level_calibration`,
`out_of_range`, `placebo_monopoly`) take the click rates the simulation
itself computes, y_r(x) = y0(x) + tau_r(x) with y0 = m1 - tau_1 * e1, from
`pipeline_ctrs`. Every check returns a dict of named tables.
"""

from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd
from econml.dml import CausalForestDML
from econml.dr import ForestDRLearner, LinearDRLearner
from econml.score import RScorer
from econml.validate import DRTester
from scipy import stats as sps
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from adsim.diagnostics.data import (
    FEATURE_COLS,
    SEGMENT_COLS,
    RankData,
    fitted_nuisance_predictions,
    forest_ate,
)
from adsim.diagnostics.stats import (
    EPS,
    brier_score,
    clustered_mean,
    expected_calibration_error,
    reliability_table,
)
from adsim.hyperparams import _cv_splitter
from adsim.simulation_steps import treated_probability

log = logging.getLogger("adsim.diagnostics")


def _is_constant(a: np.ndarray) -> bool:
    """True for a forest with no splits (its CATEs differ only by float noise)."""
    return bool(np.ptp(a) <= 1e-12 * (1 + np.abs(a).max()))


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if _is_constant(a) or _is_constant(b):
        return np.nan
    return float(sps.spearmanr(a, b).statistic)


def _prob_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    """Out-of-sample fit of a probability forecast `p` for a binary `y`."""
    pc = np.clip(p, EPS, 1 - EPS)
    return {
        "n": len(y),
        "mean_observed": y.mean(),
        "mean_predicted": p.mean(),
        "log_loss": float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc))),
        "brier": brier_score(p, y),
        "auc": roc_auc_score(y, p) if np.unique(y).size == 2 else np.nan,
        "r2": 1 - brier_score(p, y) / y.var() if y.var() > 0 else np.nan,
        "share_below_0": float(np.mean(p < 0)),
        "share_above_1": float(np.mean(p > 1)),
    }


# ----------------------------------------------------------------------
# Per-rank checks
# ----------------------------------------------------------------------

def nuisance_scores(rd: RankData) -> dict[str, pd.DataFrame]:
    """4.3 nuisance models: cross-fitted log loss, Brier, AUC and R-squared of
    the outcome model m(X) and the propensity model e(C), overall and by
    user segment. `econml_fold_score` is the fit-time score econml stored
    (R-squared for m, minus log loss for e)."""
    targets = {
        "outcome m(X)": (rd.Y, rd.outcome("all"), getattr(rd.cf, "nuisance_scores_y", None)),
        "propensity e(C)": (rd.T, rd.propensity("context"), getattr(rd.cf, "nuisance_scores_t", None)),
    }
    overall, by_segment = [], []
    for model, (y, p, stored) in targets.items():
        stored_mean = float(np.mean(stored)) if stored is not None else np.nan
        overall.append({"model": model, **_prob_metrics(y, p), "econml_fold_score": stored_mean})
        for col in SEGMENT_COLS:
            if col not in rd.pair:
                continue
            for level, idx in rd.pair.groupby(col).indices.items():
                by_segment.append({"model": model, "segment": col.removeprefix("segment_"), "level": level,
                                   **_prob_metrics(y[idx], p[idx])})
    return {"nuisance_scores": pd.DataFrame(overall), "nuisance_scores_by_segment": pd.DataFrame(by_segment)}


def cate_validation(rd: RankData, *, n_groups: int = 4, n_bootstrap: int = 200) -> dict[str, pd.DataFrame]:
    """4.3 CATE: refit the forest's configuration on training users, evaluate on held-out users.

    - R-score (econml RScorer): > 0 means the CATE beats a constant effect.
    - DRTester: BLP slope (1 = well calibrated heterogeneity, 0 = none),
      calibration R-squared (Dwivedi et al.), QINI and AUTOC with p-values.
    - Held-out level calibration of the shown arm's click probability,
      m(x) + (t - e(x)) tau(x), and its share outside [0, 1].
    DRTester needs CATE variation; a forest with no splits gets NaNs there.
    """
    val = rd.holdout_mask()
    cf = rd.refit(rd.pair[~val].reset_index(drop=True))
    X_tr, X_val = rd.X[~val].to_numpy(float), rd.X[val].to_numpy(float)
    T_tr, T_val, Y_tr, Y_val = rd.T[~val], rd.T[val], rd.Y[~val], rd.Y[val]
    g_tr = None if rd.groups is None else rd.groups[~val]
    g_val = None if rd.groups is None else rd.groups[val]
    tau_val = np.ravel(cf.const_marginal_effect(X_val))

    scorer = RScorer(model_y=rd.outcome_learner(), model_t=rd.propensity_model(), discrete_treatment=True,
                     cv=rd.settings.n_folds, random_state=rd.settings.random_state)
    scorer.fit(Y_val, T_val, X=X_val, groups=g_val)

    row = {"n_train": int((~val).sum()), "n_val": int(val.sum()), "rscore": scorer.score(cf),
           "forest_ate_train": forest_ate(cf)[0], "cate_sd_val": tau_val.std()}
    groups_table = pd.DataFrame()
    if not _is_constant(tau_val):
        splits = list(_cv_splitter(g_tr, rd.settings.random_state, rd.settings.n_folds).split(X_tr, T_tr, g_tr))
        tester = DRTester(model_regression=rd.outcome_learner(), model_propensity=rd.propensity_model(),
                          cate=cf, cv=splits)
        tester.fit_nuisance(X_val, T_val, Y_val, X_tr, T_tr, Y_tr)
        res = tester.evaluate_all(X_val, X_tr, n_groups=n_groups, n_bootstrap=n_bootstrap)
        row.update({
            "dr_ate_val": float(np.ravel(tester.ate_val)[0]),
            "blp_slope": res.blp.params[0], "blp_se": res.blp.errs[0], "blp_p": res.blp.pvals[0],
            "cal_r2": float(res.cal.cal_r_squared[0]),
            "qini": res.qini.params[0], "qini_se": res.qini.errs[0], "qini_p": res.qini.pvals[0],
            "autoc": res.toc.params[0], "autoc_se": res.toc.errs[0], "autoc_p": res.toc.pvals[0],
        })
        groups_table = next(iter(res.cal.plot_data_dict.values())).rename(columns={"ind": "group"})
    else:
        log.info("rank=%d: constant CATE on held-out users; DRTester skipped", rd.rank)

    m, e = fitted_nuisance_predictions(cf, X_val)
    p = m + (T_val - e) * tau_val
    row.update({f"shown_{k}": v for k, v in _prob_metrics(Y_val, p).items() if k != "n"})
    row["shown_ece"] = expected_calibration_error(p, Y_val)
    return {"cate_validation": pd.DataFrame([row]), "cate_calibration_groups": groups_table,
            "pair_reliability": reliability_table(p, Y_val)}


class ProbaTLearner:
    """CATE = P(click | x, focal ad) - P(click | x, base ad), one classifier per arm.

    Click-rate levels stay in [0, 1] by construction. With a logistic
    regression per arm this is the "logit with ad x covariate interactions"
    benchmark (referee R3).
    """

    def __init__(self, model):
        self.model = model

    def fit(self, Y, T, *, X):
        self.arms_ = {}
        for arm in (0, 1):
            y = np.asarray(Y)[np.asarray(T) == arm]
            if np.unique(y).size < 2:
                self.arms_[arm] = float(y.mean()) if len(y) else 0.0
            else:
                self.arms_[arm] = clone(self.model).fit(np.asarray(X)[np.asarray(T) == arm], y)
        return self

    def predict_arm(self, X, arm: int) -> np.ndarray:
        model = self.arms_[arm]
        if isinstance(model, float):
            return np.full(len(X), model)
        return model.predict_proba(np.asarray(X))[:, 1]

    def const_marginal_effect(self, X):
        return (self.predict_arm(X, 1) - self.predict_arm(X, 0)).reshape(-1, 1)


def _alternative_estimators(rd: RankData, train: pd.DataFrame) -> dict[str, object]:
    """Plan 4.3 estimators 2-6; forests share the baseline's leaf floor and rows per tree."""
    plan = rd.plan_for(train)
    params_y = rd.outcome_params()
    classifier_y = RandomForestClassifier(
        n_estimators=params_y.get("n_estimators", 100), max_features="sqrt",
        min_samples_leaf=params_y.get("min_samples_leaf", 1), n_jobs=rd.settings.n_jobs,
        random_state=rd.settings.random_state)
    forest_kw = {"n_estimators": plan.n_estimators, "max_samples": plan.max_samples,
                 "min_samples_leaf": plan.min_samples_leaf, "min_samples_split": plan.min_samples_split,
                 "n_jobs": rd.settings.n_jobs, "random_state": rd.settings.random_state}
    return {
        "causal_forest_discrete_outcome": CausalForestDML(
            model_y=classifier_y, model_t=rd.propensity_model(), discrete_treatment=True, discrete_outcome=True,
            criterion="het", min_var_fraction_leaf=plan.min_var_fraction_leaf, **forest_kw),
        "forest_dr_learner": ForestDRLearner(
            model_regression=rd.outcome_learner(), model_propensity=rd.propensity_model(), **forest_kw),
        "linear_dr_learner": LinearDRLearner(
            model_regression=rd.outcome_learner(), model_propensity=rd.propensity_model(),
            random_state=rd.settings.random_state),
        "t_learner_gbm": ProbaTLearner(HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.05, min_samples_leaf=max(20, len(train) // 200),
            random_state=rd.settings.random_state)),
        "logit_interactions": ProbaTLearner(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))),
    }


def estimator_comparison(rd: RankData) -> dict[str, pd.DataFrame]:
    """4.3 alternative estimators, fit on training users and compared on held-out users.

    Columns: held-out R-score (one RScorer shared by all), mean CATE, CATE
    spread, Spearman correlation with the baseline forest's CATE, and the
    share of negative implied focal-ad click rates mu0(x) + tau(x), where
    mu0 is a common base-arm click model. The joint multi-arm forest and
    the simulated gain g per estimator need full re-estimation and
    simulation runs and are not computed here.
    """
    val = rd.holdout_mask()
    train = rd.pair[~val].reset_index(drop=True)
    X_tr, X_val = rd.X[~val].to_numpy(float), rd.X[val].to_numpy(float)
    T_tr, T_val, Y_tr, Y_val = rd.T[~val], rd.T[val], rd.Y[~val], rd.Y[val]
    g_tr = None if rd.groups is None else rd.groups[~val]
    g_val = None if rd.groups is None else rd.groups[val]

    scorer = RScorer(model_y=rd.outcome_learner(), model_t=rd.propensity_model(), discrete_treatment=True,
                     cv=rd.settings.n_folds, random_state=rd.settings.random_state)
    scorer.fit(Y_val, T_val, X=X_val, groups=g_val)
    mu0 = ProbaTLearner(RandomForestClassifier(**rd.propensity_params())).fit(Y_tr, T_tr, X=X_tr)
    base_rate_val = mu0.predict_arm(X_val, 0)

    t0 = time.perf_counter()
    baseline = rd.refit(train)
    fitted = {"causal_forest": (baseline, time.perf_counter() - t0)}
    for name, est in _alternative_estimators(rd, train).items():
        t0 = time.perf_counter()
        try:
            if isinstance(est, ProbaTLearner):
                est.fit(Y_tr, T_tr, X=X_tr)
            else:
                est.fit(Y_tr, T_tr, X=X_tr, groups=g_tr)
            fitted[name] = (est, time.perf_counter() - t0)
        except Exception as exc:  # one failing estimator should not sink the table
            log.warning("rank=%d estimator %s failed: %s", rd.rank, name, exc)

    tau_base = np.ravel(baseline.const_marginal_effect(X_val))
    rows = []
    for name, (est, seconds) in fitted.items():
        tau = np.ravel(est.const_marginal_effect(X_val))
        rows.append({
            "estimator": name,
            "rscore": scorer.score(est),
            "mean_cate": tau.mean(),
            "cate_sd": tau.std(),
            "spearman_vs_baseline": _spearman(tau, tau_base),
            "share_implied_ctr_below_0": float(np.mean(base_rate_val + tau < 0)),
            "fit_seconds": seconds,
        })
    return {"estimators": pd.DataFrame(rows)}


def stability(rd: RankData, *, split_col: str = "day") -> dict[str, pd.DataFrame]:
    """R3 data: refit on the first vs second half of the estimation days and
    compare the two forests' CATEs on the same rows."""
    days = rd.pair[split_col]
    if days.nunique() < 2:
        return {"stability": pd.DataFrame([{"skipped": f"only one value of {split_col}"}])}
    cut = np.median(np.unique(days))
    first = (days <= cut).to_numpy()
    halves = {"first": rd.pair[first].reset_index(drop=True), "second": rd.pair[~first].reset_index(drop=True)}
    X_eval = rd.X.sample(n=min(20_000, len(rd.pair)), random_state=rd.settings.random_state)
    row = {"split_col": split_col, "cut": cut}
    taus = {}
    for name, half in halves.items():
        cf = rd.refit(half)
        ate, se = forest_ate(cf)
        taus[name] = np.ravel(cf.const_marginal_effect(X_eval))
        row.update({f"n_{name}": len(half), f"ate_{name}": ate, f"ate_se_{name}": se})
    row["ate_diff_z"] = (row["ate_first"] - row["ate_second"]) / np.hypot(row["ate_se_first"], row["ate_se_second"])
    row["cate_spearman"] = _spearman(taus["first"], taus["second"])
    varies = not (_is_constant(taus["first"]) or _is_constant(taus["second"]))
    row["cate_pearson"] = np.corrcoef(taus["first"], taus["second"])[0, 1] if varies else np.nan
    return {"stability": pd.DataFrame([row])}


# ----------------------------------------------------------------------
# Pipeline click rates (what the simulation uses)
# ----------------------------------------------------------------------

def base_ad_ctr(X: pd.DataFrame, cf_1, helpers: dict) -> np.ndarray:
    """y0(x) = m1(x) - tau_1(x) * e1(x), before clipping (as in `calc_base_ad_ctr_vector`)."""
    tau_1 = np.ravel(cf_1.const_marginal_effect(X))
    return np.ravel(helpers["m1"].predict(X)) - tau_1 * np.ravel(treated_probability(helpers["e1"], X))


def pipeline_ctrs(X: pd.DataFrame, forests: dict, y0: np.ndarray) -> pd.DataFrame:
    """Unclipped y_r(x) = y0(x) + tau_r(x) for every rank in `forests` (columns)."""
    return pd.DataFrame({rank: y0 + np.ravel(cf.const_marginal_effect(X)) for rank, cf in forests.items()},
                        index=X.index)


def shown_ad_predictions(sample: pd.DataFrame, forests: dict, y0: np.ndarray,
                         support: pd.Series) -> pd.DataFrame:
    """Predicted click rate of the ad actually shown on each impression (base ad: y0)."""
    X = sample[FEATURE_COLS]
    p_raw = pd.Series(np.nan, index=sample.index)
    shown = sample["advertiser_rank"]
    p_raw[shown == 0] = y0[(shown == 0).to_numpy()]
    for rank, cf in forests.items():
        rows = (shown == rank).to_numpy()
        if rows.any():
            p_raw[rows] = y0[rows] + np.ravel(cf.const_marginal_effect(X[rows]))
    out = sample[["advertiser_rank", "is_clicked", *[c for c in SEGMENT_COLS if c in sample]]].copy()
    out["p_raw"] = p_raw
    out["p"] = p_raw.clip(lower=0)
    out["support"] = shown.map(support).where(shown != 0, "base ad")
    return out.dropna(subset=["p_raw"])


def level_calibration(preds: pd.DataFrame, *, n_bins: int = 10) -> dict[str, pd.DataFrame]:
    """4.3 level calibration: predicted CTR of the shown ad vs realised clicks.

    Predictions are the simulation's clipped y_r(x). Brier score, expected
    calibration error (also relative to the mean CTR, since CTRs are ~0.1%),
    and the share of raw predictions outside [0, 1], overall, by ad-support
    bucket and by user segment; plus reliability tables.
    """
    def summarise(frame: pd.DataFrame) -> dict:
        y, p = frame["is_clicked"].to_numpy(float), frame["p"].to_numpy(float)
        ece = expected_calibration_error(p, y, n_bins)
        return {"n": len(y), "mean_predicted": p.mean(), "observed_ctr": y.mean(),
                "predicted_over_observed": p.mean() / y.mean() if y.mean() > 0 else np.nan,
                "brier": brier_score(p, y), "ece": ece, "ece_relative": ece / y.mean() if y.mean() > 0 else np.nan,
                "share_raw_below_0": float(np.mean(frame["p_raw"] < 0)),
                "share_raw_above_1": float(np.mean(frame["p_raw"] > 1))}

    rows = [{"group": "all", "level": "all", **summarise(preds)}]
    for col in ["support", *[c for c in SEGMENT_COLS if c in preds]]:
        for level, frame in preds.groupby(col):
            rows.append({"group": col.removeprefix("segment_"), "level": level, **summarise(frame)})

    reliability = [reliability_table(preds["p"], preds["is_clicked"], n_bins).assign(support="all")]
    for level, frame in preds.groupby("support"):
        reliability.append(reliability_table(frame["p"], frame["is_clicked"], n_bins).assign(support=level))
    return {"calibration": pd.DataFrame(rows), "reliability": pd.concat(reliability, ignore_index=True)}


def out_of_range(ctrs: pd.DataFrame, y0: np.ndarray) -> dict[str, pd.DataFrame]:
    """4.3 / item 2.9: share of y_r(x) the simulation clips at 0 (and above 1), by rank.

    Evaluated for every rank on every sampled impression, i.e. over the
    simulation's whole choice set, at the impressions' observed covariates.
    """
    def row(label, values):
        values = np.asarray(values, dtype=float).ravel()
        return {"rank": label, "n": values.size, "share_below_0": float(np.mean(values < 0)),
                "share_above_1": float(np.mean(values > 1)), "mean": values.mean(),
                "p01": np.quantile(values, 0.01), "p99": np.quantile(values, 0.99)}

    rows = [row("base ad (y0)", y0)]
    rows += [row(rank, ctrs[rank]) for rank in ctrs.columns]
    rows.append(row("all ranks", ctrs.to_numpy()))
    return {"out_of_range": pd.DataFrame(rows)}


def _chosen_mask(scores: np.ndarray, n_slots: np.ndarray) -> np.ndarray:
    """True for the top `n_slots[i]` columns of each row of `scores`."""
    order = np.argsort(-scores, axis=1, kind="stable")
    position = np.empty_like(order)
    np.put_along_axis(position, order, np.arange(scores.shape[1])[None, :].repeat(len(scores), 0), axis=1)
    return position < n_slots[:, None]


def placebo_monopoly(sample: pd.DataFrame, ctrs: pd.DataFrame, y0: np.ndarray,
                     placebo_ctrs: pd.DataFrame, placebo_y0: np.ndarray, *,
                     valuations: pd.Series | None, groups: np.ndarray | None,
                     max_ads: int = 15) -> dict[str, pd.DataFrame]:
    """Item 2.2 placebo check: the optimizer's-curse part of the merger gain.

    A second monopoly forest set (same data, different seed or resample)
    chooses the ads on each impression's first visit and is scored by the
    main monopoly forests, exactly as a duopolist is. The main set choosing
    for itself is what the simulation records for the monopoly. Their gap,
    `mechanical_gain`, is pure estimation noise. CTRs are recorded for the
    ads actually chosen (bug 2.1 does not apply). The base ad competes
    under the CTR criterion only, as in the simulation. Under the revenue
    criterion read `value_gain` (CTR x value, the objective); its CTR gap
    can take either sign.
    """
    ranks = [r for r in ctrs.columns if r in placebo_ctrs.columns]
    main = ctrs[ranks].clip(lower=0).to_numpy()
    plc = placebo_ctrs[ranks].clip(lower=0).to_numpy()
    n_slots = np.minimum(sample["ads_on_page"].fillna(0).to_numpy(), max_ads).astype(int)

    criteria = {"CTR": None}
    if valuations is not None:
        vals = valuations.reindex(ranks).to_numpy(dtype=float)
        if np.isnan(vals).any():
            log.warning("no valuation for ranks %s; excluded from the revenue criterion",
                        [r for r, v in zip(ranks, vals) if np.isnan(v)])
        criteria["revenue"] = np.nan_to_num(vals, nan=0.0)

    slots = n_slots.sum()
    rows = []
    for criterion, vals in criteria.items():
        if vals is None:
            truth = np.c_[main, np.clip(y0, 0, None)]
            truth_plc = np.c_[plc, np.clip(placebo_y0, 0, None)]
            chosen_main, chosen_plc = _chosen_mask(truth, n_slots), _chosen_mask(truth_plc, n_slots)
            value_gain = np.nan
        else:
            truth, truth_plc = main, plc
            chosen_main, chosen_plc = _chosen_mask(main * vals, n_slots), _chosen_mask(plc * vals, n_slots)
            value_gain = (main * vals * chosen_main).sum() / (main * vals * chosen_plc).sum() - 1
        self_sum = (truth * chosen_main).sum(1)
        plc_sum = (truth * chosen_plc).sum(1)
        ctr_self, ctr_plc = self_sum.sum() / slots, plc_sum.sum() / slots
        diff, diff_se = clustered_mean((self_sum - plc_sum) * len(n_slots) / slots, groups)
        rows.append({
            "criterion": criterion,
            "n_impressions": int((n_slots > 0).sum()),
            "n_slots": int(slots),
            "ctr_monopoly_self_scored": ctr_self,
            "ctr_placebo_scored_by_monopoly": ctr_plc,
            "ctr_placebo_self_scored": (truth_plc * chosen_plc).sum() / slots,
            "mechanical_gain": ctr_self / ctr_plc - 1,
            "diff": diff,
            "diff_se": diff_se,
            "value_gain": value_gain,
            "share_slots_same_ad": float((chosen_main & chosen_plc).sum() / slots),
        })
    return {"placebo_monopoly": pd.DataFrame(rows)}
