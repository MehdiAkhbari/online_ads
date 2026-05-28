# Cleanup changelog — what moved, what got renamed, what got deleted

A high-level summary of the refactor that turned the original
`scripts/`-based codebase into the `adsim/` package. Use this if you're
returning to the project and need to map old filenames in your memory
to the new ones.

> For *why* each change was made, read the commit messages
> (`git log --oneline`). For the new layout in detail, read `README.md`.

---

## TL;DR

The whole pipeline used to live as flat `.py` files under `scripts/`,
imported each other via `from utils import *`, and hardcoded Windows
paths like `"..\\data\\Full Model\\..."`. Today every runnable thing is
a `python -m adsim.<X>` entry point, every path goes through
`adsim.paths.DATA_DIR` / `RESULTS_DIR`, and `scripts/` only holds the
standalone pickle-compat checker plus the `ranks_list.pickle`
artifact.

13 commits across the day, ending with **58 passing tests**. Old
notebooks worked but were broken against the new layout — every
`.ipynb` was ported to import from `adsim.*` and use the new path
constants.

---

## Code modules — old name → new name

### Estimation entry points (6 → 1)

| Old | New |
|---|---|
| `scripts/estimation.py` (split 7) | `python -m adsim.estimate --scenario split --split 7` |
| `scripts/estimation2.py` (split 7, ranks > 10) | `python -m adsim.estimate --scenario split --split 7 --ranks-filter "rank > 10"` |
| `scripts/estimation3.py` (split 6, Root-N) | `python -m adsim.estimate --scenario split-root-n --split 6` |
| `scripts/estimation_split_5.py` | **deleted** (legacy LogisticRegression PropensityModel; pre-dates `propensity_model.py`) |
| `scripts/estimation_split_6.py` | **deleted** (same as above) |
| `scripts/estimation_sqrt_n.py` | `python -m adsim.estimate --scenario root-n --subsample-ratio 0.8` |
| *(no monopoly entry point existed; was inline-commented)* | `python -m adsim.estimate --scenario monopoly` |

The new `adsim/estimate.py` is one CLI parameterised by `--scenario`,
with auto-resume (skips ranks whose `.pkl` already exists), filtering
(`--ranks`, `--ranks-filter`, `--limit`), and structured per-rank logs.
Designed for HPC: one scenario per job, fault-isolated reruns.

### Simulation entry points (5 → 6 in `adsim.simulate/`)

| Old | New |
|---|---|
| `scripts/base_ad_ctr_estimation.py` | `python -m adsim.simulate.base_ad_helpers` |
| `scripts/monopoly_simulation.py` | `python -m adsim.simulate.monopoly` |
| `scripts/duopoly_simulation.py` | `python -m adsim.simulate.duopoly` |
| `scripts/duopoly_simulation_sqrt_n.py` | `python -m adsim.simulate.duopoly_root_n` |
| `scripts/simulation.py` (legacy "Last 2 Days") | `python -m adsim.simulate.legacy.simulation` |
| `scripts/simulation_parallel.py` (legacy parallel) | `python -m adsim.simulate.legacy.simulation_parallel` |

Each module now has a real `argparse` CLI (`--processes`, `--criteria`,
`--data`, `--vals-data`, `--split-1`, `--split-2`, ...). Multiprocessing
workers load forests via `Pool(initializer=...)` so module imports stay
side-effect-free.

### Shared modules — moved into `adsim/`

| Old | New |
|---|---|
| `scripts/utils.py` (956 lines) + the older `adsim/utils.py` (852 lines) | `adsim/simulation_steps.py` |
| `scripts/config.py` | `adsim/config.py` (rewritten — see "Behavioural changes" below) |
| `scripts/propensity_model.py` | `adsim/propensity_model.py` |
| `adsim/constants.py` (`PATH_ROOT` shim) | **deleted** |
| *(new)* | `adsim/paths.py` — `REPO_ROOT`, `DATA_DIR`, `RESULTS_DIR` |

`adsim/utils.py` was renamed to `adsim/simulation_steps.py` because
"utils" was misleading for a 950-line module that's mostly per-step
simulation primitives (`calc_tes`, `calc_split_tes`,
`create_chosen_ad_columns`, `update_repeats`, `update_clicks`,
`simulate_monopoly`, ...). Every import site was updated.

### Tooling

| Old | New |
|---|---|
| *(none)* | `scripts/check_old_pickles.py` — verifies whether old `CF - Rank *.pkl` artifacts still load + do inference in the new env. Reports `OK` / `DEGRADED` / `INCOMPATIBLE`. |

---

## Notebooks — old name → new name

### Analysis (paper-ready) — kept, renamed

| Old | New |
|---|---|
| `scripts/Results Analysis.ipynb` | `notebooks/analysis/results_analysis.ipynb` |
| `scripts/Subject Correlation Matrix.ipynb` | `notebooks/analysis/subject_correlation_matrix.ipynb` |
| `scripts/advertiser_welfare_analysis.ipynb` | `notebooks/analysis/advertiser_welfare_analysis.ipynb` |
| `scripts/ctr_vs_repeat.ipynb` | `notebooks/analysis/ctr_vs_repeat.ipynb` |
| `scripts/merge_simulation_results.ipynb` | `notebooks/analysis/merge_simulation_results.ipynb` |

### Sample-size — 4 → 1

| Old | New |
|---|---|
| `scripts/Sample Size Analysis copy.ipynb` | `notebooks/sample_size/sample_size_analysis.ipynb` |
| `scripts/Sample Size Analysis copy 2/3/4.ipynb` | **deleted** (re-execution snapshots; cached outputs already captured in `.dta` artifacts) |

### Exploration — 10 → 2

| Old | New |
|---|---|
| `scripts/Model Estimation for Ads 1, 2.ipynb` | `notebooks/exploration/pairwise_estimation_ad_1_vs_2.ipynb` |
| `scripts/Ad 1,3 Estimation - Full Week.ipynb` | `notebooks/exploration/pairwise_estimation_ad_1_vs_3_full_week.ipynb` |
| `scripts/Two Ads Estimation.ipynb` | **deleted** (earlier draft of rank {1,3} pair) |
| `scripts/Two Ads Estimation-3.ipynb` | **deleted** (duplicate of rank {1,3} pair) |
| `scripts/test2.ipynb` (`debug_duopoly_revenue` after rename) | **deleted** (subsumed by `adsim.simulate.duopoly`) |
| `scripts/test3.ipynb` (`prototype_subsampling` after rename) | **deleted** (subsumed by `adsim.estimate --scenario root-n`) |
| `scripts/test4.ipynb` (`prototype_split_simulation`) | **deleted** (abandoned design) |
| `scripts/test4-copy.ipynb` (`..._with_outputs`) | **deleted** (same prototype + cached outputs) |
| `scripts/test5.ipynb` (`debug_update_clicks`) | **deleted** (subsumed by `adsim.simulation_steps.update_clicks`) |
| `scripts/test_6.ipynb` (`prototype_revenue_ctrs`) | **deleted** (subsumed by `adsim.simulation_steps`) |

The intermediate `test*.ipynb` → `debug_*` / `prototype_*` rename was a
documentation step (a separate commit) before deletion, so the file
names in `git log` reflect what each scratch notebook was actually
prototyping.

### Notebook content fixes (every notebook)

Every `.ipynb` had:

- `from utils import *` → `from adsim.simulation_steps import *`
- `import config` → `from adsim import config`
- `from propensity_model import …` → `from adsim.propensity_model import …`
- Windows backslash paths (`"..\\data\\Full Model\\..."`) replaced with
  `str(DATA_DIR / "Full Model/...")` or
  `f"{DATA_DIR}/Full Model/...{var}..."` for f-strings
- `from adsim.paths import DATA_DIR, RESULTS_DIR` injected into the
  first cell when needed

`ctr_vs_repeat.ipynb` additionally had `config.cf_<rank>` references
replaced with `config.forests[<rank>]` lookups, and a
`config.load_monopoly_forests()` call added to the top.

Two pre-existing syntax bugs unrelated to the refactor were also
fixed: a stray `tune the model:` line in
`pairwise_estimation_ad_1_vs_2.ipynb`, and a malformed
`for i in range(1, range(1, max_visit_no + 1):` in one of the
prototype notebooks before it was deleted.

---

## Top-level: deleted

| Path | Why |
|---|---|
| `Two Ads Estimation-3.ipynb` (root) | Older duplicate of `scripts/Two Ads Estimation-3.ipynb` |
| `notebooks/test.ipynb` | 146 KB scratch in an otherwise empty `notebooks/` dir |
| `scripts/Untitled-1.py` | Scratch (notebook export) |
| `scripts/file.pkl` | 5-byte file containing `pickle.dump(123, ...)` — leftover REPL test |
| `scripts/propensity_score.pkl` | 0-byte empty file — failed/aborted `joblib.dump` |
| `environment.yml` | Windows conda export; replaced by `pyproject.toml` + `requirements.txt` |
| `setup.py` | Replaced by `pyproject.toml` |

---

## Behavioural changes worth knowing

These are the cases where running the new code does something
*different* from the old, not just where the same code lives in a
different file.

1. **Importing `adsim.config` does no I/O.** The old `config.py`
   read every `CF - Rank *.pkl` off disk at import time, populating
   module-level names like `cf_1`, `cf_2`, ..., `cf_<N>_s7`. The new
   `adsim.config` exposes empty registries (`config.forests`,
   `config.split_forests`, `config.helpers`) and explicit loaders
   (`load_helpers()`, `load_monopoly_forests()`,
   `load_split_forests(N)`, `load_subsample_forests(ratio)`). The
   simulation entry points call these explicitly before their main
   loop. Multiprocessing workers call them in their `Pool` initializer.

2. **Forest lookup is `config.forests[<rank>]`, not
   `config.cf_<rank>`.** Every `exec(f"config.cf_{rank}.const_marginal_effect(...)")`
   was replaced with a plain dict lookup. Same for the duopoly variant
   (`config.split_forests[<split>][<rank>]`) and the helpers
   (`config.helpers["m1"]` / `["e1"]`).

3. **Paths come from env vars by default.** `ADSIM_DATA_DIR` /
   `ADSIM_RESULTS_DIR` override `<repo_root>/data` /
   `<repo_root>/results`. Useful on HPC where the inputs live on a
   shared filesystem.

4. **`sqrt-n` and `root-n` are now one name.** The old code mixed both
   spellings (`--scenario sqrt-n` alongside `update_*_sqrt_n`
   functions, while data dirs and result subdirs called it `Root N`).
   Today everything is `root-n` (function names, scenario flag);
   on-disk filenames are still `Root N` to match the original `.dta`
   files.

5. **Re-running estimation auto-resumes by default.** If
   `CF - Rank 7.pkl` already exists, `python -m adsim.estimate --scenario
   monopoly` will skip rank 7 and continue. Pass `--force` to override.
   This matters most for HPC: a job that dies mid-run can be
   re-submitted as-is.

---

## Tests

Before: `tests/test_utils.py` had one stub function with `pass`.

After: 4 files, **58 tests, all passing**.

```
tests/test_simulation_steps.py    10 tests   pure-function helpers
tests/test_estimate_cli.py        20 tests   CLI parsing + scenario dispatch
tests/test_package_imports.py     19 tests   import-time invariants + env vars
tests/test_cli_smoke.py            9 tests   `python -m adsim.X --help`
```

The point isn't 100% coverage — it's anchoring the package's *public
surface* (CLI args, scenario keys, import contract, pure helpers)
against accidental regressions.

---

## What you can throw away from your memory

- `from utils import *` (old)
- `import config` (old, pulled forests at import)
- `..\\data\\Full Model\\…` paths
- `config.cf_<rank>` attribute access
- `config.cf_<rank>_s<split>` for splits
- `config.m1` / `config.e1` (now `config.helpers["m1"]` / `["e1"]`)
- `config.cf_<rank>_sub` for the subsample scenario
- The 6 `estimation*.py` scripts
- The 5 `*_simulation*.py` scripts
- The 4 "Sample Size Analysis copy*.ipynb" duplicates
- The 6 `test*.ipynb` / `test4-copy.ipynb` scratch notebooks
- `environment.yml`, `setup.py`, `adsim/constants.py`

If you find yourself reaching for any of those, check `git log --all`
or `git log --diff-filter=D --summary` — they're in history.
