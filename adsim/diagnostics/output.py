"""Writing diagnostic tables (CSV + LaTeX) and figures.

Figures are drawn only from the saved tables, so each can be redrawn from
its CSV. LaTeX is written by hand (booktabs) because pandas' `to_latex`
needs jinja2, which is not a dependency.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_LATEX_ESCAPES = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
                  "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\^{}"}


def _escape(text) -> str:
    return "".join(_LATEX_ESCAPES.get(ch, ch) for ch in str(text))


def _format_cell(value) -> str:
    if isinstance(value, (float, np.floating)):
        return "" if math.isnan(value) else f"{value:.4g}"
    return _escape(value)


def to_latex(df: pd.DataFrame) -> str:
    """A booktabs tabular of `df` (4 significant digits)."""
    header = " & ".join(_escape(c) for c in df.columns) + r" \\"
    body = [" & ".join(_format_cell(v) for v in row) + r" \\" for row in df.itertuples(index=False)]
    align = "".join("r" if pd.api.types.is_numeric_dtype(df[c]) else "l" for c in df.columns)
    return "\n".join([rf"\begin{{tabular}}{{{align}}}", r"\toprule", header, r"\midrule", *body,
                      r"\bottomrule", r"\end{tabular}", ""])


def write_table(df: pd.DataFrame, out_dir: Path, name: str) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv, tex = out_dir / f"{name}.csv", out_dir / f"{name}.tex"
    df.to_csv(csv, index=False)
    tex.write_text(to_latex(df))
    return [csv, tex]


def save_figure(fig, out_dir: Path, name: str) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / f"{name}.pdf", out_dir / f"{name}.png"]
    for path in paths:
        fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return paths


def _grid(n: int, width: float = 3.2, height: float = 2.4, **kw):
    cols = min(n, 4)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(width * cols, height * rows + 0.5), squeeze=False,
                             layout="constrained", **kw)
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    return fig, axes.ravel()


# ----------------------------------------------------------------------
# One function per figure, keyed by the table it draws from
# ----------------------------------------------------------------------

def plot_propensity_info(df: pd.DataFrame):
    """Log-loss gain from adding user history to e(C) (bars), against the
    gain from the same block permuted (black: null mean +/- 1.96 SD), by rank."""
    df = df[df["feature_set"] != "context"]
    sets = list(dict.fromkeys(df["feature_set"]))
    ranks = list(dict.fromkeys(df["rank"]))
    x = np.arange(len(ranks))
    width = 0.8 / len(sets)
    fig, ax = plt.subplots(figsize=(max(5, 0.6 * len(ranks) + 2), 3.2))
    for i, fs in enumerate(sets):
        sub = df[df["feature_set"] == fs].set_index("rank").reindex(ranks)
        ax.bar(x + i * width, sub["gain"], width, label=fs)
        if "null_gain_mean" in sub:
            ax.errorbar(x + i * width, sub["null_gain_mean"], yerr=1.96 * sub["null_gain_sd"],
                        fmt="_", color="black", capsize=3)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x + width * (len(sets) - 1) / 2, [str(r) for r in ranks])
    ax.set_xlabel("advertiser rank")
    ax.set_ylabel("log-loss gain vs e(C)")
    ax.legend(fontsize=8)
    return fig


def plot_love(df: pd.DataFrame):
    """Love plot: largest |SMD| across the diagnosed ranks, per covariate."""
    cols = {"smd_raw": "raw", "smd_within_cell": "within cell", "smd_ipw": "IPW e(C)"}
    worst = df.groupby("covariate", sort=False)[list(cols)].agg(lambda s: s.abs().max())
    y = np.arange(len(worst))
    fig, ax = plt.subplots(figsize=(5, 0.28 * len(worst) + 1))
    for col, label in cols.items():
        ax.scatter(worst[col], y, label=label, s=18)
    ax.axvline(0.1, color="grey", ls="--", lw=0.8)
    ax.set_yticks(y, worst.index)
    ax.invert_yaxis()
    ax.set_xlabel("max |standardized mean difference| over ranks")
    ax.legend(fontsize=8)
    return fig


def plot_overlap(df: pd.DataFrame):
    """Histograms of e(C) for base-ad and focal-ad impressions, one panel per rank."""
    ranks = list(dict.fromkeys(df["rank"]))
    fig, axes = _grid(len(ranks))
    for ax, rank in zip(axes, ranks):
        sub = df[df["rank"] == rank]
        mid = (sub["bin_lo"] + sub["bin_hi"]) / 2
        width = (sub["bin_hi"] - sub["bin_lo"]).iloc[0]
        for arm in ("base", "focal"):
            ax.bar(mid, sub[arm] / sub[arm].sum(), width=width, alpha=0.5, label=arm)
        ax.set_title(f"rank {rank}", fontsize=9)
        ax.set_xlim(0, 1)
    axes[0].legend(fontsize=7)
    fig.supxlabel("e(C) = P(focal ad | page context)")
    return fig


def plot_refuters(df: pd.DataFrame):
    """ATE with 95% CI for the observed data and each refutation, per rank."""
    ranks = list(dict.fromkeys(df["rank"]))
    tests = list(dict.fromkeys(df["test"]))
    fig, axes = _grid(len(ranks), sharey=True)
    for ax, rank in zip(axes, ranks):
        sub = df[df["rank"] == rank].set_index("test").reindex(tests)
        ax.errorbar(sub["ate"], np.arange(len(tests)), xerr=1.96 * sub["ate_se"], fmt="o", capsize=2)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_title(f"rank {rank}", fontsize=9)
    axes[0].set_yticks(np.arange(len(tests)), tests, fontsize=7)
    axes[0].invert_yaxis()
    fig.supxlabel("average treatment effect (click probability)")
    return fig


def plot_cate_calibration(df: pd.DataFrame):
    """Held-out group ATE (doubly robust) vs mean predicted CATE by CATE quantile group."""
    ranks = list(dict.fromkeys(df["rank"]))
    fig, axes = _grid(len(ranks))
    for ax, rank in zip(axes, ranks):
        sub = df[df["rank"] == rank]
        ax.errorbar(sub["g_cate"], sub["gate"], yerr=1.96 * sub["se_gate"], fmt="o", capsize=2)
        lims = [min(sub["g_cate"].min(), sub["gate"].min()), max(sub["g_cate"].max(), sub["gate"].max())]
        ax.plot(lims, lims, color="grey", ls="--", lw=0.8)
        ax.set_title(f"rank {rank}", fontsize=9)
    fig.supxlabel("mean predicted CATE in group")
    fig.supylabel("held-out DR group ATE")
    return fig


def plot_reliability(df: pd.DataFrame):
    """Reliability diagram of the simulation's predicted CTR for the shown ad (percent)."""
    fig, ax = plt.subplots(figsize=(4.5, 4))
    for support, sub in df.groupby("support", sort=False):
        style = {"color": "black", "lw": 1.5, "zorder": 3} if support == "all" else {"lw": 0.8, "alpha": 0.7}
        ax.errorbar(100 * sub["mean_predicted"], 100 * sub["observed_rate"], yerr=196 * sub["observed_se"],
                    fmt="o-", ms=3, capsize=2, label=support, **style)
    hi = 100 * max(df["mean_predicted"].max(), df["observed_rate"].max())
    ax.plot([0, hi], [0, hi], color="grey", ls="--", lw=0.8)
    ax.set_xlabel("predicted CTR (%)")
    ax.set_ylabel("observed CTR (%)")
    ax.legend(fontsize=8, title="ad support")
    return fig


def plot_out_of_range(df: pd.DataFrame):
    """Share of each rank's y_r(x) below zero (clipped by the simulation)."""
    sub = df[~df["rank"].isin(["all ranks"])]
    fig, ax = plt.subplots(figsize=(max(5, 0.12 * len(sub) + 2), 3))
    ax.bar(np.arange(len(sub)), 100 * sub["share_below_0"])
    ax.set_xticks(np.arange(len(sub)), [str(r) for r in sub["rank"]], rotation=90, fontsize=6)
    ax.set_ylabel("% of impressions with y < 0")
    return fig


FIGURES = {
    "propensity_info": plot_propensity_info,
    "balance_smd": plot_love,
    "overlap_histogram": plot_overlap,
    "refuters": plot_refuters,
    "cate_calibration_groups": plot_cate_calibration,
    "reliability": plot_reliability,
    "out_of_range": plot_out_of_range,
}
