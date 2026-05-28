# Online Ads

Causal-inference-driven simulation of online ad serving. The study fits one
`CausalForestDML` (econml) per advertiser rank on observational impression
data, then replays user visits forward with those forests in three
scenarios — **monopoly**, **duopoly / split**, and **sample-size /
Root-N** — and analyses the simulated outcomes (CTR, advertiser welfare)
in a set of Jupyter notebooks.

## Status

- ✅ Python 3.11 / scikit-learn 1.5 / econml 0.15 environment, pinned
- ✅ All importable code lives under the `adsim/` package
- ✅ All file I/O goes through `adsim.paths.DATA_DIR` / `RESULTS_DIR`
- ✅ Importing `adsim.config` does no I/O — forests load on demand
- ✅ One CLI per pipeline stage (`python -m adsim.<thing>`)
- ✅ HPC-friendly estimation (auto-resume, per-rank logs, fault-isolated)
- ✅ Test suite: 58 tests passing
- ⚠️  Input `.dta` files are not in git — must be supplied locally
- ⚠️  Old `CF - Rank *.pkl` artifacts (sklearn 1.2 / econml 0.14) likely
  won't load in this env — re-fit with `python -m adsim.estimate`, or
  verify with `scripts/check_old_pickles.py`

---

## TL;DR

```bash
# 1. Set up the env (Python 3.11 required; with pyenv: pyenv install 3.11.4)
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel setuptools
pip install -r requirements.txt
pip install -e ".[notebooks,dev]"

# 2. Drop the input .dta files under ./data/ (see "Required input data").

# 3. Fit the per-rank causal forests (slow — hours per scenario; HPC-friendly).
python -m adsim.estimate --scenario monopoly
python -m adsim.estimate --scenario split --split 7
python -m adsim.estimate --scenario split --split 8

# 4. Fit the base-ad helpers (m1, e1).
python -m adsim.simulate.base_ad_helpers

# 5. Run the simulation.
python -m adsim.simulate.monopoly        # or .duopoly / .duopoly_root_n

# 6. Analyse — open notebooks under notebooks/analysis/.
```

If you have pre-existing pickled forests from a prior run, check whether
they still load before re-fitting:

```bash
python scripts/check_old_pickles.py --results-dir /path/to/your/results
```

---

## Pipeline

```
data/Full Model/Estimation Data - Full Model - Monopoly.dta
data/Full Model/Estimation Data - Full Model - Split {5,6,7,8}[ - Root N].dta
        │
        ▼   python -m adsim.estimate --scenario {monopoly,split,split-root-n,root-n}
results/Full Model/Monopoly/CF - Rank {r}.pkl                (~95 forests)
results/Full Model/Split {N}[ - Root N]/CF - Rank {r}.pkl
        │
        ▼   python -m adsim.simulate.base_ad_helpers
results/Full Model/{m1,e1}.pkl              (base-ad y0 helpers)
        │
        ▼   python -m adsim.simulate.{monopoly,duopoly,duopoly_root_n}
            (loads forests via adsim.config.load_*_forests())
results/Full Model/Simulation Results/Simluation Results - * - chunk N.dta
        │
        ▼   notebooks
notebooks/analysis/merge_simulation_results.ipynb     → combines chunks
notebooks/analysis/results_analysis.ipynb             → main figures/tables
notebooks/analysis/advertiser_welfare_analysis.ipynb
notebooks/analysis/ctr_vs_repeat.ipynb
notebooks/analysis/subject_correlation_matrix.ipynb
notebooks/sample_size/sample_size_analysis.ipynb
```

> The misspelling **"Simluation"** in the output filenames is preserved on
> purpose — it matches the original naming, and the analysis notebooks read
> files by that exact name.

---

## Layout

| Path | Purpose |
|---|---|
| `adsim/` | Installable Python package (`pip install -e .`). All importable + runnable code lives here. |
| `adsim/paths.py` | `REPO_ROOT`, `DATA_DIR`, `RESULTS_DIR` (overridable via env vars). |
| `adsim/config.py` | Static knobs (`split_no_1/2`, `my_criteria`, ...), `ranks_list`, and explicit forest loaders. **Importing it does no I/O** — call `load_helpers()` / `load_monopoly_forests()` / `load_split_forests(n)` explicitly. |
| `adsim/simulation_steps.py` | Estimation helpers (`define_xyt`, `prepare_data`, `m_model_best_estimator`, ...) and per-step simulation primitives (`calc_tes`, `update_clicks`, `simulate_monopoly`, ...). |
| `adsim/propensity_model.py` | `PropensityModel`, the T-model used by `CausalForestDML`. |
| `adsim/estimate.py` | `python -m adsim.estimate` — fit per-rank causal forests, one CLI for all four scenarios. |
| `adsim/simulate/` | `python -m adsim.simulate.{base_ad_helpers,monopoly,duopoly,duopoly_root_n}` — forward-simulation entry points. |
| `adsim/simulate/legacy/` | `python -m adsim.simulate.legacy.{simulation,simulation_parallel}` — older "Last 2 Days" simulations, kept for reference. |
| `scripts/check_old_pickles.py` | Standalone tool: verifies whether old `CF - Rank *.pkl` artifacts still load in this env. |
| `scripts/ranks_list.pickle` | Canonical input artifact (96 advertiser ranks; checked into git). |
| `notebooks/analysis/` | Paper-ready analysis notebooks (`results_analysis`, `advertiser_welfare_analysis`, `ctr_vs_repeat`, `subject_correlation_matrix`, `merge_simulation_results`). |
| `notebooks/sample_size/` | Sample-size sensitivity aggregation (`sample_size_analysis.ipynb`). |
| `notebooks/exploration/` | Pairwise-estimation studies (`pairwise_estimation_ad_1_vs_2`, `pairwise_estimation_ad_1_vs_3_full_week`). See [`notebooks/README.md`](notebooks/README.md) for details. |
| `tests/` | Pytest suite (58 tests). |
| `data/`, `results/` | **Not in git.** Populate locally before running anything. |

---

## Setup

Requires Python **3.11**. With pyenv:

```bash
pyenv install 3.11.4
~/.pyenv/versions/3.11.4/bin/python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel setuptools
pip install -r requirements.txt
pip install -e .                       # installs the adsim package, editable
pip install -e ".[notebooks,dev]"      # optional: jupyter, pytest, ruff
```

Each new shell needs `source .venv/bin/activate`.

### Verify the install

```bash
python -c "from econml.dml import CausalForestDML; import adsim; print(adsim.__version__)"
python -m pytest tests/                # 58 passing
python -m adsim.estimate --help        # CLI sanity
```

### Configure data + results paths

By default `adsim.paths` resolves:

- `DATA_DIR    = <repo_root>/data`
- `RESULTS_DIR = <repo_root>/results`

To point either at an external location (shared filesystem, external
drive, etc.), export env vars before running:

```bash
export ADSIM_DATA_DIR=/Volumes/research/online_ads/data
export ADSIM_RESULTS_DIR=/Volumes/research/online_ads/results
```

### Notes on env migration

The original env was a Windows conda export (Python 3.8.18, scikit-learn
1.2.2, econml 0.14.1). The new env is Python 3.11 + scikit-learn 1.5 +
econml 0.15. Pickled `CF - Rank *.pkl` artifacts produced under the old
env are **likely not loadable** in this one — `scripts/check_old_pickles.py`
reports OK / DEGRADED / INCOMPATIBLE for any saved artifacts.

---

## Required input data

None of these are checked in. Place them under `data/` (or under
`$ADSIM_DATA_DIR`).

### Estimation inputs
- `data/Full Model/Estimation Data - Full Model - Monopoly.dta`
- `data/Full Model/Estimation Data - Full Model - Split {5,6,7,8}.dta`
- `data/Full Model/Estimation Data - Full Model - Split {N} - Root N.dta`

### Simulation inputs
- `data/Full Model/Simulation Data - Full Model - Monopoly - Subsample.dta`
- `data/Full Model/Simulation Data - Full Model - Split 7 8 - Subsample.dta`
- `data/Full Model/Simulation Data - Full Model - Split 7 8 - Root N - Subsample.dta`
- `data/Full Model/Advertiser Valuations.dta`

### Notebook-only inputs
- `data/Subjects Visited by Each User.dta` (used by `notebooks/analysis/subject_correlation_matrix.ipynb`)
- `data/Full Model/Estimation Data - Full Model - Monopoly - Whole Week.dta` (used by the two `notebooks/exploration/pairwise_estimation_*` notebooks)

### Legacy inputs (only needed for `adsim.simulate.legacy.*`)
- `data/Simulation Data - Last 2 Days.dta`
- `data/Simulation Data - Last 2 Days - Merged Subjects Subsample.dta`

The list of advertiser ranks the study iterates over lives in
`scripts/ranks_list.pickle` (96 ranks: 0..101 with gaps).
`adsim.config.ranks_list` exposes it with rank 0 (base ad) and the
>max-ad fringe entry already removed.

---

## Reproducing the study

From the repo root, with `.venv` activated.

### 1. Fit the per-rank causal forests (slow — hours per scenario)

A single CLI entry point covers all four scenarios:

```bash
# Monopoly
python -m adsim.estimate --scenario monopoly

# Duopoly / split (one job per split)
python -m adsim.estimate --scenario split --split 7
python -m adsim.estimate --scenario split --split 8

# Root-N split
python -m adsim.estimate --scenario split-root-n --split 6

# Sample-size (subsampled monopoly)
python -m adsim.estimate --scenario root-n --subsample-ratio 0.8
```

Outputs land under `results/Full Model/<scenario_dir>/CF - Rank {r}.pkl`.

The CLI is **HPC-friendly**:

- **Auto-resumes** by default — skips ranks whose pickle already exists.
  Pass `--force` to override. Re-running a job after a transient failure
  is just re-submitting it.
- **Filtering**: `--ranks 1,2,3` (explicit), `--ranks-filter "rank > 10"`
  (Python expression on `rank`), or `--limit N` (cap to first N ranks).
- **Structured logs**: each rank prints scenario, walltime, and output
  path.
- **`--dry-run`** prints what would run without fitting anything.

Run `python -m adsim.estimate --help` for the full list. Example HPC
submission (Slurm):

```bash
sbatch --time=24:00:00 --cpus-per-task=30 --wrap="python -m adsim.estimate --scenario monopoly"
sbatch --time=24:00:00 --cpus-per-task=30 --wrap="python -m adsim.estimate --scenario split --split 7"
sbatch --time=24:00:00 --cpus-per-task=30 --wrap="python -m adsim.estimate --scenario split --split 8"
```

### 2. Fit the base-ad y0 helpers

```bash
python -m adsim.simulate.base_ad_helpers
```

Outputs: `results/Full Model/m1.pkl`, `results/Full Model/e1.pkl`.

### 3. Run the simulation

```bash
python -m adsim.simulate.monopoly                # monopoly
python -m adsim.simulate.duopoly                 # duopoly / split 7+8
python -m adsim.simulate.duopoly_root_n          # duopoly under Root-N
```

Each takes `--processes N`, `--data PATH`, `--vals-data PATH`, and (where
applicable) `--criteria {CTR,revenue}`, `--split-1 N`, `--split-2 N`. Run
with `--help` for the full list.

Outputs: `results/Full Model/Simulation Results/Simluation Results - * - chunk N.dta`
(one `.dta` per worker process).

### 4. Analyse

Open notebooks under `notebooks/analysis/` (with `jupyter lab`):

1. `merge_simulation_results.ipynb` — combine the per-chunk `.dta` outputs
   into one DataFrame per scenario.
2. `results_analysis.ipynb` — main figures / tables.
3. `advertiser_welfare_analysis.ipynb`, `ctr_vs_repeat.ipynb`,
   `subject_correlation_matrix.ipynb` — secondary analyses.
4. Sample-size scenario: `notebooks/sample_size/sample_size_analysis.ipynb`.
5. Pairwise studies (paper artifacts): `notebooks/exploration/pairwise_estimation_*.ipynb`.

See [`notebooks/README.md`](notebooks/README.md) for what each one does
and which inputs it needs.

---

## Working with the package programmatically

```python
from adsim import config
from adsim.simulation_steps import calc_tes, calc_base_ad_ctr, simulate_monopoly

# By default importing config does no I/O. Load what you need:
config.load_helpers()                      # config.helpers = {"m1": ..., "e1": ...}
config.load_monopoly_forests()             # config.forests = {1: cf, 2: cf, ...}

# For the duopoly scenario:
config.load_split_forests(config.split_no_1)
config.load_split_forests(config.split_no_2)
# config.split_forests[7], config.split_forests[8] now populated.

# For the sample-size / root-n scenario:
config.load_subsample_forests(subsampling_ratio=0.8)
# config.subsample_forests = {1: cf, 2: cf, ...}
```

This is the contract every simulation script relies on.

---

## Tests

```bash
python -m pytest tests/                 # 58 passing
python -m pytest tests/ -v              # verbose
python -m pytest tests/test_estimate_cli.py    # one file
```

Coverage:

- `tests/test_simulation_steps.py` — pure-function helpers (`prepare_data`,
  `extract_ranks`, `find_optimal_ads`).
- `tests/test_estimate_cli.py` — CLI argument parsing + scenario dispatch
  for `adsim.estimate`.
- `tests/test_package_imports.py` — package-level invariants (importing
  `adsim.config` is side-effect-free; env-var path overrides work; every
  submodule imports cleanly).
- `tests/test_cli_smoke.py` — every `python -m adsim.X --help` exits 0,
  and `--scenario unknown` fails cleanly with exit code 2.

What the tests **don't** cover (out of scope without real data + fitted
forests): the actual per-step simulation primitives (`calc_tes`,
`calc_split_tes`, ...) and the full `python -m adsim.simulate.*` runs.

---

## I'm coming back to this project — what changed?

If you stepped away and forgot the layout, the short version:

- **Everything runnable is `python -m adsim.<thing>`.** No more
  `python scripts/foo.py`. The only thing left in `scripts/` is the
  pickle-compat checker and the canonical `ranks_list.pickle`.
- **Estimation is one CLI** (`python -m adsim.estimate --scenario X`).
  The 6 old `estimation*.py` scripts are gone.
- **Simulations are explicit modules** (`python -m adsim.simulate.monopoly`,
  `.duopoly`, `.duopoly_root_n`, `.base_ad_helpers`).
- **`adsim.utils` was renamed to `adsim.simulation_steps`.** Old name
  is gone — every notebook + module reference was updated.
- **Forests are loaded on demand.** `import adsim.config` does no I/O;
  the simulation entry points call `config.load_*_forests()` themselves.
  Inside the package code, look up forests as `config.forests[rank]`,
  not `config.cf_<rank>`.
- **Notebooks were reorganised**: 19 → 8. The originals are recoverable
  from git history; the cleanup is documented in `notebooks/README.md`.
- **Old `.pkl` forests probably don't load in the new env** — env moved
  from sklearn 1.2 / econml 0.14 to 1.5 / 0.15.

Quick recovery checklist:

```bash
source .venv/bin/activate
python -m pytest tests/                 # tests should pass
python -m adsim.estimate --help         # CLI should help
python scripts/check_old_pickles.py --results-dir /path/to/old/results
```

If the tests pass and the CLIs help-print, the package is healthy.

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'adsim'`**
You forgot `pip install -e .`. Or you forgot `source .venv/bin/activate`.

**`FileNotFoundError: ranks_list.pickle`**
`adsim.config` looks first under `RESULTS_DIR/main_scenario/ranks_list.pickle`,
then falls back to `<repo_root>/scripts/ranks_list.pickle`. The fallback
*is* checked into git; if it's missing, run `git checkout scripts/ranks_list.pickle`.

**`KeyError: <rank>` from `config.forests[<rank>]`**
You forgot to call `config.load_monopoly_forests()` (or the relevant
`load_*` function) before invoking simulation logic. Each `python -m
adsim.simulate.*` entry point calls these for you; doing it from a
notebook or a custom script needs explicit calls.

**`scripts/check_old_pickles.py` reports INCOMPATIBLE**
Old artifacts won't load in the new env. Re-fit with
`python -m adsim.estimate --scenario <whichever>` — it auto-resumes, so
you can run it incrementally on HPC without re-doing finished ranks.

**Tests fail after a refactor**
That's the design — the test suite anchors against accidental
regressions in the package's public surface (CLI args, scenario keys,
import-time side effects, pure-function helpers). Read the failure and
either fix the code or update the test if the behaviour change is
intentional.

---

## Reproducibility notes for reviewers

- **Pinned env.** `pyproject.toml` and `requirements.txt` pin Python 3.11
  + scikit-learn 1.5 + econml 0.15 + pandas 2.2 + numpy 1.26 (compatible
  ranges). The original conda env (`environment.yml`) is gone.
- **No hardcoded paths.** All file I/O goes through `adsim.paths.DATA_DIR`
  / `RESULTS_DIR`. Override with `ADSIM_DATA_DIR` / `ADSIM_RESULTS_DIR`
  env vars; default to repo-relative.
- **Side-effect-free imports.** `import adsim.config` does no I/O. Forests
  are loaded only when an explicit `config.load_*_forests()` call runs.
  This means you can `import adsim.simulation_steps` from a notebook or
  test without needing a populated `results/` dir.
- **Dropped string-execed code.** Per-rank lookups previously written with
  `exec(f"config.cf_{rank}.const_marginal_effect(...)")` are now plain
  dict lookups against `config.forests[rank]` /
  `config.split_forests[split][rank]`.
- **Explicit verdict for old artifacts.** `scripts/check_old_pickles.py`
  reports OK / DEGRADED / INCOMPATIBLE for any pre-existing
  `CF - Rank *.pkl` so you don't have to guess whether a pickle from
  sklearn 1.2 / econml 0.14 will work here.
- **Tests pass.** `python -m pytest tests/` → 58 passing tests covering
  pure-function helpers, CLI argument parsing, package import
  invariants, and CLI smoke tests.

---

## Known cleanup tasks

- [ ] The `predict_proba(...).reshape(-1, 1)` in
  `adsim.simulation_steps.calc_base_ad_ctr_vector` is suspicious
  (`predict_proba` returns shape `(n, 2)`, not `(n,)`); audit before
  re-running the duopoly simulation.
- [ ] Hardcoded chunk sizes (`1620000 / n_processes`, `820000`,
  `300000`) in the simulation modules. Make them either CLI args or
  auto-derived from `data.global_token_new.nunique()`.
- [ ] `notebooks/analysis/merge_simulation_results.ipynb` references chunk
  outputs for splits 1/2, 3/4, 5/6 that aren't produced by the canonical
  pipeline (only split 7/8 is). Either restore those simulator runs or
  trim the notebook to the canonical scenarios.
- [ ] `notebooks/analysis/advertiser_welfare_analysis.ipynb` cell 17 has
  a pre-existing `pd.merge(...)` call missing arguments — broken before
  the refactor, still broken now.
- [ ] `find_optimal_ads` in `adsim.simulation_steps` slices a list with
  `row['ads_on_page']`, which is a `numpy.float64` if the row's Series
  has uniform numeric dtype — it works in production because the input
  DataFrame has at least one non-numeric column (forcing dtype `object`).
  Worth coercing to `int` for safety.
