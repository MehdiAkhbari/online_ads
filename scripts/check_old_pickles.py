"""Compatibility check for old pickled artifacts in the new env.

Run this on a machine that has the old `results/Full Model/...` directory
(the pickles produced by the original `estimation.py` runs) AND the new
`adsim` env (Python 3.11, scikit-learn 1.5, econml 0.15) installed.

What it does:
    1. Walks the results directory and finds every `CF - Rank *.pkl`,
       plus `m1.pkl` and `e1.pkl` if present.
    2. For each artifact:
         - tries `joblib.load(...)` and reports load success / failure;
         - if loaded, runs a tiny inference call on synthetic input with
           the same column shape the simulation uses;
         - reports whether the call returned a finite array of the
           expected shape.
    3. Prints a summary with a single verdict at the end:
         OK         -> artifacts work in this env, no re-fit needed
         DEGRADED   -> loads succeed but inference is broken/wrong
         INCOMPATIBLE -> loads themselves fail, re-fit required

Run with:
    python scripts/check_old_pickles.py
    # or, if your results dir is elsewhere:
    ADSIM_RESULTS_DIR=/path/to/results python scripts/check_old_pickles.py
    # or:
    python scripts/check_old_pickles.py --results-dir /path/to/results
"""

from __future__ import annotations

import argparse
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import joblib
except ImportError as e:
    print("FATAL: joblib not installed. Activate the new env first.", file=sys.stderr)
    raise SystemExit(2) from e


# Column list mirrored from scripts/utils.py:define_xyt / construct_X.
# These are the X columns the saved CausalForestDML forests were fit on.
X_COLUMNS = [
    "impression_repeat", "impression_repeat_base_ad",
    "previous_clicks", "previous_clicks_base_ad", "previous_clicks_all_ads",
    "total_visits",
    *[f"visit_s{i}" for i in range(1, 14)],
    *[f"sub_{i}" for i in range(1, 14)],
    "publisher_rank_sub", "day", "hour", "mobile", "ads_on_page",
]


def make_synthetic_X(n_rows: int = 16, seed: int = 0) -> pd.DataFrame:
    """Tiny realistic-ish X for smoke-testing inference."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        col: rng.integers(0, 5, size=n_rows).astype(float)
        for col in X_COLUMNS
    })


@dataclass
class Result:
    artifact: str
    path: Path
    load_ok: bool = False
    load_error: str = ""
    obj_type: str = ""
    infer_ok: bool = False
    infer_error: str = ""
    notes: list[str] = field(default_factory=list)


def _short_traceback(exc: BaseException) -> str:
    """One-line summary of the exception, plus the most relevant frames."""
    head = f"{type(exc).__name__}: {exc}"
    tb = traceback.format_exception_only(type(exc), exc)[-1].strip()
    return tb if tb else head


def check_causal_forest(path: Path, X: pd.DataFrame) -> Result:
    r = Result(artifact="CausalForestDML", path=path)
    try:
        cf = joblib.load(path)
        r.load_ok = True
        r.obj_type = type(cf).__name__
    except Exception as e:
        r.load_error = _short_traceback(e)
        return r

    # Inference smoke test.
    try:
        out = cf.const_marginal_effect(X)
        out = np.asarray(out).reshape(-1)
        if out.shape[0] != len(X):
            r.notes.append(f"unexpected output length {out.shape[0]} != {len(X)}")
        elif not np.isfinite(out).all():
            r.notes.append(f"output contains {np.isnan(out).sum()} NaN / "
                           f"{np.isinf(out).sum()} inf values")
        else:
            r.notes.append(
                f"output range [{out.min():+.4f}, {out.max():+.4f}], "
                f"mean {out.mean():+.4f}"
            )
        r.infer_ok = (out.shape[0] == len(X)) and np.isfinite(out).all()
    except Exception as e:
        r.infer_error = _short_traceback(e)

    return r


def check_y_helper(path: Path, X: pd.DataFrame) -> Result:
    """m1.pkl is a fitted RandomForestRegressor (Y model)."""
    r = Result(artifact="m1 (Y model)", path=path)
    try:
        model = joblib.load(path)
        r.load_ok = True
        r.obj_type = type(model).__name__
    except Exception as e:
        r.load_error = _short_traceback(e)
        return r

    try:
        out = np.asarray(model.predict(X)).reshape(-1)
        if out.shape[0] != len(X):
            r.notes.append(f"unexpected output length {out.shape[0]} != {len(X)}")
        elif not np.isfinite(out).all():
            r.notes.append(f"output contains non-finite values")
        else:
            r.notes.append(
                f"predict output range [{out.min():+.4f}, {out.max():+.4f}]"
            )
        r.infer_ok = (out.shape[0] == len(X)) and np.isfinite(out).all()
    except Exception as e:
        r.infer_error = _short_traceback(e)
    return r


def check_t_helper(path: Path, X: pd.DataFrame) -> Result:
    """e1.pkl is a fitted PropensityModel (T model)."""
    r = Result(artifact="e1 (T model)", path=path)
    try:
        model = joblib.load(path)
        r.load_ok = True
        r.obj_type = type(model).__name__
    except Exception as e:
        r.load_error = _short_traceback(e)
        return r

    try:
        proba = np.asarray(model.predict_proba(X))
        if proba.ndim != 2 or proba.shape[0] != len(X):
            r.notes.append(f"predict_proba returned shape {proba.shape} (expected ({len(X)}, n_classes))")
        elif not np.isfinite(proba).all():
            r.notes.append("predict_proba contains non-finite values")
        else:
            r.notes.append(
                f"predict_proba shape {proba.shape}, "
                f"row sums {proba.sum(axis=1).min():.4f}..{proba.sum(axis=1).max():.4f}"
            )
        r.infer_ok = (proba.ndim == 2 and proba.shape[0] == len(X)
                      and np.isfinite(proba).all())
    except Exception as e:
        r.infer_error = _short_traceback(e)
    return r


def discover_artifacts(results_dir: Path) -> dict[str, list[Path]]:
    """Find pkl artifacts produced by the original estimation runs."""
    full_model = results_dir / "Full Model"
    monopoly_dir = full_model / "Monopoly"

    return {
        "monopoly_forests": sorted(monopoly_dir.glob("CF - Rank *.pkl")) if monopoly_dir.is_dir() else [],
        "y_helper": [full_model / "m1.pkl"] if (full_model / "m1.pkl").is_file() else [],
        "t_helper": [full_model / "e1.pkl"] if (full_model / "e1.pkl").is_file() else [],
        # Other variants worth checking if present.
        "split_forests": sorted(full_model.glob("Split */CF - Rank *.pkl")) if full_model.is_dir() else [],
    }


def render_row(r: Result) -> str:
    if not r.load_ok:
        status = "LOAD-FAIL "
        detail = r.load_error
    elif not r.infer_ok:
        status = "INFER-FAIL"
        detail = r.infer_error or "; ".join(r.notes)
    else:
        status = "OK        "
        detail = "; ".join(r.notes)
    return f"  [{status}] {r.path.name:32s} ({r.obj_type or '-'})  {detail}"


def run(results_dir: Path, sample_size: int, full_scan: bool) -> int:
    print(f"results_dir = {results_dir}\n")
    artifacts = discover_artifacts(results_dir)

    if not any(artifacts.values()):
        print("No pickled artifacts found under that path.")
        print("Looked for:")
        print(f"  {results_dir / 'Full Model' / 'Monopoly' / 'CF - Rank *.pkl'}")
        print(f"  {results_dir / 'Full Model' / 'm1.pkl'}")
        print(f"  {results_dir / 'Full Model' / 'e1.pkl'}")
        print(f"  {results_dir / 'Full Model' / 'Split */CF - Rank *.pkl'}")
        return 2

    X = make_synthetic_X(n_rows=sample_size)
    results: list[Result] = []

    monopoly = artifacts["monopoly_forests"]
    if monopoly:
        sample = monopoly if full_scan else monopoly[: min(3, len(monopoly))]
        print(f"== Causal forests (Monopoly): {len(sample)}/{len(monopoly)} sampled ==")
        for p in sample:
            r = check_causal_forest(p, X)
            results.append(r)
            print(render_row(r))
        print()

    if artifacts["y_helper"]:
        print("== Y-model helper (m1.pkl) ==")
        r = check_y_helper(artifacts["y_helper"][0], X)
        results.append(r)
        print(render_row(r))
        print()

    if artifacts["t_helper"]:
        print("== T-model helper (e1.pkl) ==")
        r = check_t_helper(artifacts["t_helper"][0], X)
        results.append(r)
        print(render_row(r))
        print()

    if full_scan and artifacts["split_forests"]:
        print(f"== Causal forests (Split): {len(artifacts['split_forests'])} ==")
        for p in artifacts["split_forests"]:
            r = check_causal_forest(p, X)
            results.append(r)
            print(render_row(r))
        print()

    # ---- Verdict ----
    n = len(results)
    n_load_fail = sum(1 for r in results if not r.load_ok)
    n_infer_fail = sum(1 for r in results if r.load_ok and not r.infer_ok)
    n_ok = sum(1 for r in results if r.load_ok and r.infer_ok)

    print("=" * 60)
    print(f"Artifacts checked:  {n}")
    print(f"  OK              : {n_ok}")
    print(f"  Load-fail       : {n_load_fail}")
    print(f"  Inference-fail  : {n_infer_fail}")

    if n_load_fail > 0:
        verdict = "INCOMPATIBLE — at least one artifact failed to unpickle. Re-fit required."
        exit_code = 1
    elif n_infer_fail > 0:
        verdict = "DEGRADED — artifacts load but inference is broken. Re-fit recommended."
        exit_code = 1
    else:
        verdict = "OK — old artifacts work in this env."
        exit_code = 0

    print()
    print(f"Verdict: {verdict}")
    return exit_code


def main(argv: list[str] | None = None) -> int:
    # Default to the repo's RESULTS_DIR if adsim is installed; otherwise ./results.
    try:
        from adsim.paths import RESULTS_DIR as _DEFAULT_RESULTS_DIR
        default_results = str(_DEFAULT_RESULTS_DIR)
    except ImportError:
        default_results = "./results"

    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--results-dir", type=Path, default=Path(default_results),
        help=f"Results directory containing 'Full Model/...'. Default: {default_results}",
    )
    p.add_argument(
        "--sample-size", type=int, default=16,
        help="Number of synthetic rows to feed the inference smoke test.",
    )
    p.add_argument(
        "--full-scan", action="store_true",
        help="Check every CF rank, not just a small sample.",
    )
    args = p.parse_args(argv)

    if not args.results_dir.exists():
        print(f"results-dir does not exist: {args.results_dir}", file=sys.stderr)
        return 2

    return run(args.results_dir.resolve(), args.sample_size, args.full_scan)


if __name__ == "__main__":
    raise SystemExit(main())
