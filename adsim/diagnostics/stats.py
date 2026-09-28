"""Statistical helpers for adsim.diagnostics (numpy / scipy only).

Every standard error here is clustered by user when `groups` is given,
because users contribute many impressions (revision plan, item 2.7).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps

EPS = 1e-6


# ----------------------------------------------------------------------
# Means and clustered standard errors
# ----------------------------------------------------------------------

def _cluster_totals(values: np.ndarray, groups: np.ndarray | None) -> np.ndarray:
    """Row sums of `values` within each cluster (rows unchanged if no groups)."""
    values = np.asarray(values, dtype=float)
    if groups is None:
        return values
    codes = pd.factorize(np.asarray(groups))[0]
    if values.ndim == 1:
        return np.bincount(codes, weights=values)
    return np.stack([np.bincount(codes, weights=values[:, j]) for j in range(values.shape[1])], axis=1)


def clustered_mean(values, groups=None) -> tuple[float, float]:
    """Mean of `values` and its standard error, clustered by `groups`."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    mean = float(values.mean())
    totals = _cluster_totals(values - mean, groups)
    g = len(totals)
    se = float(np.sqrt(g / max(g - 1, 1) * np.sum(totals ** 2)) / n)
    return mean, se


def z_test(estimate: float, se: float, *, one_sided: bool = False) -> tuple[float, float]:
    """(z statistic, p-value) for H0: estimate = 0. One-sided tests H1: estimate > 0."""
    if not np.isfinite(se) or se <= 0:
        return float("nan"), float("nan")
    z = estimate / se
    p = sps.norm.sf(z) if one_sided else 2 * sps.norm.sf(abs(z))
    return float(z), float(p)


def null_draw_test(value: float, null) -> tuple[float, float]:
    """Is `value` larger than a new draw from the distribution sampled by `null`?

    Studentised with prediction-interval scaling, (value - mean) / (sd *
    sqrt(1 + 1/B)), and a one-sided p-value from t with B - 1 df. Used when
    `value` and each null draw carry the same model-fitting noise.
    """
    null = np.asarray(null, dtype=float)
    b = len(null)
    scale = null.std(ddof=1) * np.sqrt(1 + 1 / b) if b > 1 else float("nan")
    if not np.isfinite(scale) or scale <= 0:
        return float("nan"), float("nan")
    z = (value - null.mean()) / scale
    return float(z), float(sps.t.sf(z, df=b - 1))


# ----------------------------------------------------------------------
# Balance
# ----------------------------------------------------------------------

def standardized_mean_difference(x, t, weights=None) -> float:
    """(mean_treated - mean_control) / sqrt((var_treated + var_control) / 2).

    With `weights` the arm means are weighted (e.g. inverse-propensity
    weights) but the denominator keeps the unweighted variances, so raw and
    adjusted SMDs share one scale.
    """
    x = np.asarray(x, dtype=float)
    t = np.asarray(t).astype(bool)
    w = np.ones_like(x) if weights is None else np.asarray(weights, dtype=float)
    diff = np.average(x[t], weights=w[t]) - np.average(x[~t], weights=w[~t])
    scale = np.sqrt((x[t].var(ddof=1) + x[~t].var(ddof=1)) / 2)
    return float(diff / scale) if scale > 0 else 0.0


def demean_within(values, cells) -> np.ndarray:
    """Subtract each cell's mean (the within transformation for cell fixed effects)."""
    frame = pd.DataFrame(np.asarray(values, dtype=float))
    return (frame - frame.groupby(np.asarray(cells)).transform("mean")).to_numpy()


def within_cell_smd(x, t, cells) -> float:
    """Treated-minus-control difference in `x` within cells, scaled by the raw SMD denominator."""
    x = np.asarray(x, dtype=float)
    t = np.asarray(t, dtype=float)
    x_dm, t_dm = demean_within(np.c_[x, t], cells).T
    denom = np.sum(t_dm ** 2)
    if denom <= 0:
        return float("nan")
    diff = np.sum(x_dm * t_dm) / denom
    tb = t.astype(bool)
    scale = np.sqrt((x[tb].var(ddof=1) + x[~tb].var(ddof=1)) / 2)
    return float(diff / scale) if scale > 0 else 0.0


def mixed_cell_share(t, cells) -> float:
    """Share of rows in cells that contain both arms. Only these rows carry
    within-cell information (and can change label in a within-cell permutation)."""
    arm_mean = pd.Series(np.asarray(t, dtype=float)).groupby(np.asarray(cells)).transform("mean")
    return float(((arm_mean > 0) & (arm_mean < 1)).mean())


def permute_within(values, cells, rng: np.random.Generator) -> np.ndarray:
    """Shuffle `values` independently inside each cell."""
    values = np.asarray(values)
    cells = np.asarray(cells)
    by_cell = np.lexsort((np.arange(len(values)), cells))
    by_cell_shuffled = np.lexsort((rng.random(len(values)), cells))
    out = np.empty_like(values)
    out[by_cell] = values[by_cell_shuffled]
    return out


def cluster_robust_wald(y, X, groups=None) -> dict:
    """Joint test that all OLS slopes of y on X are zero, with a
    cluster-robust (CR1) covariance. Pass already-demeaned y and X for a
    fixed-effects regression. Returns F, its degrees of freedom and p-value.
    """
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    keep = X.std(axis=0) > 0
    X = X[:, keep]
    n, k = X.shape
    if k == 0:
        return {"F": float("nan"), "df_num": 0, "df_den": 0, "p_value": float("nan")}
    xtx_inv = np.linalg.pinv(X.T @ X)
    beta = xtx_inv @ X.T @ y
    scores = _cluster_totals(X * (y - X @ beta)[:, None], groups)
    g = len(scores)
    correction = g / max(g - 1, 1) * (n - 1) / max(n - k, 1)
    cov = correction * xtx_inv @ (scores.T @ scores) @ xtx_inv
    wald = float(beta @ np.linalg.pinv(cov) @ beta)
    df_den = max(g - 1, 1)
    F = wald / k
    return {"F": F, "df_num": k, "df_den": df_den, "p_value": float(sps.f.sf(F, k, df_den))}


# ----------------------------------------------------------------------
# Double machine learning (partialling out) and omitted-variable bounds
# ----------------------------------------------------------------------

def partialling_out_ate(y, t, m_hat, e_hat, groups=None) -> dict:
    """Robinson / DML estimate of a constant effect from cross-fitted
    nuisances m_hat = E[y|X] and e_hat = P(t=1|X), with a clustered SE."""
    y_res = np.asarray(y, dtype=float) - np.asarray(m_hat, dtype=float)
    t_res = np.asarray(t, dtype=float) - np.asarray(e_hat, dtype=float)
    denom = np.mean(t_res ** 2)
    theta = float(np.mean(y_res * t_res) / denom)
    psi = (y_res - theta * t_res) * t_res / denom
    _, se = clustered_mean(psi, groups)
    z, p = z_test(theta, se)
    return {"theta": theta, "se": se, "z": z, "p_value": p, "n": len(y_res)}


def ovb_params(y_res, t_res, groups=None) -> dict:
    """Ingredients of the Chernozhukov, Cinelli, Newey, Sharma & Syrgkanis
    omitted-variable-bias bounds for the partially linear model.

    theta = E[y~ t~] / E[t~^2]; sigma2 = E[(y~ - theta t~)^2] (outcome
    residual variance); nu2 = 1 / E[t~^2] (second moment of the Riesz
    representer). `cov` is the (clustered) covariance of the three
    estimates, used for confidence bounds. Mirrors econml 0.16's
    `dml_sensitivity_values`, which econml 0.15 does not have.
    """
    y_res = np.asarray(y_res, dtype=float)
    t_res = np.asarray(t_res, dtype=float)
    n = len(y_res)
    t2 = np.mean(t_res ** 2)
    theta = np.mean(y_res * t_res) / t2
    sigma2 = np.mean((y_res - theta * t_res) ** 2)
    nu2 = 1 / t2
    moments = np.c_[
        y_res * t_res - theta * t_res ** 2,
        (y_res - theta * t_res) ** 2 - sigma2,
        t_res ** 2 * nu2 - 1,
    ]
    jac_inv = np.diag(1 / np.array([t2, 1.0, t2]))
    totals = _cluster_totals(moments, groups)
    omega = totals.T @ totals / n
    cov = jac_inv @ omega @ jac_inv / n
    return {"theta": float(theta), "sigma2": float(sigma2), "nu2": float(nu2), "cov": cov}


def ovb_bounds(params: dict, c_y: float, c_t: float, *, rho: float = 1.0, alpha: float = 0.05) -> dict:
    """Bounds on theta when an omitted confounder explains a share `c_y` of
    the outcome's residual variance and `c_t` of the treatment's (Riesz
    representer's), with correlation `rho` between the two biases.

    `theta_lo/hi` bound the point estimate; `ci_lo/hi` add sampling error.
    """
    theta, sigma2, nu2, cov = params["theta"], params["sigma2"], params["nu2"], params["cov"]
    c_t = min(c_t, 1 - EPS)
    half = abs(rho) * np.sqrt(c_y) * np.sqrt(c_t / (1 - c_t)) / 2
    # sqrt(sigma2 * nu2) linearised in (sigma2, nu2) so the delta method gives its variance.
    grad = np.array([half * np.sqrt(nu2 / sigma2), half * np.sqrt(sigma2 / nu2)])
    coef_hi = np.r_[1.0, grad]
    coef_lo = np.r_[1.0, -grad]
    est = np.array([theta, sigma2, nu2])
    lo, hi = float(est @ coef_lo), float(est @ coef_hi)
    z = sps.norm.ppf(1 - alpha / 2)
    return {
        "theta_lo": lo,
        "theta_hi": hi,
        "ci_lo": lo - z * float(np.sqrt(coef_lo @ cov @ coef_lo)),
        "ci_hi": hi + z * float(np.sqrt(coef_hi @ cov @ coef_hi)),
    }


def robustness_value(params: dict, *, alpha: float = 0.05, interval: str = "ci", null: float = 0.0) -> float:
    """Smallest c (= c_y = c_t) at which the bound first includes `null`.

    `interval="theta"` uses the point bounds; `"ci"` the confidence bounds.
    Returns 0 if the unadjusted interval already includes `null`.
    """
    lo_key, hi_key = ("theta_lo", "theta_hi") if interval == "theta" else ("ci_lo", "ci_hi")

    def covers(c):
        b = ovb_bounds(params, c, c, alpha=alpha)
        return b[lo_key] <= null <= b[hi_key]

    if covers(0.0):
        return 0.0
    lo, hi = 0.0, 1 - EPS
    if not covers(hi):
        return 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if covers(mid) else (mid, hi)
    return float(hi)


def partial_r2(resid_short, resid_long) -> float:
    """Share of the short model's residual variance removed by the long model."""
    short = np.mean(np.asarray(resid_short, dtype=float) ** 2)
    long = np.mean(np.asarray(resid_long, dtype=float) ** 2)
    return float(max(0.0, 1 - long / short)) if short > 0 else 0.0


# ----------------------------------------------------------------------
# Overlap
# ----------------------------------------------------------------------

def crump_alpha(e) -> float:
    """Optimal trimming threshold of Crump, Hotz, Imbens & Mitnik (2009):
    keep units with alpha <= e <= 1 - alpha. Returns 0 when no trimming
    is needed."""
    e = np.clip(np.asarray(e, dtype=float), EPS, 1 - EPS)
    h = 1 / (e * (1 - e))
    if h.max() <= 2 * h.mean():
        return 0.0
    h_sorted = np.sort(h)
    running_mean = np.cumsum(h_sorted) / np.arange(1, len(h_sorted) + 1)
    k = int(np.argmax(h_sorted >= 2 * running_mean))
    gamma = h_sorted[k]
    return float(0.5 - np.sqrt(max(0.25 - 1 / gamma, 0.0)))


# ----------------------------------------------------------------------
# Probability calibration
# ----------------------------------------------------------------------

def brier_score(p, y) -> float:
    return float(np.mean((np.asarray(p, dtype=float) - np.asarray(y, dtype=float)) ** 2))


def reliability_table(p, y, n_bins: int = 10) -> pd.DataFrame:
    """Predicted vs observed click rate in quantile bins of the prediction."""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    bins = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, max(len(edges) - 2, 0))
    frame = pd.DataFrame({"bin": bins, "p": p, "y": y})
    out = frame.groupby("bin").agg(n=("y", "size"), mean_predicted=("p", "mean"), observed_rate=("y", "mean"))
    out["observed_se"] = np.sqrt(out["observed_rate"] * (1 - out["observed_rate"]) / out["n"])
    return out.reset_index()


def expected_calibration_error(p, y, n_bins: int = 10) -> float:
    """Share-weighted mean |predicted - observed| over quantile bins."""
    table = reliability_table(p, y, n_bins)
    weights = table["n"] / table["n"].sum()
    return float(np.sum(weights * (table["mean_predicted"] - table["observed_rate"]).abs()))
