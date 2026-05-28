# online_ads

Causal-inference-driven simulation of online ad serving. The study fits one
`CausalForestDML` (econml) per advertiser rank on observational impression
data, then replays user visits forward with those forests in three scenarios
— **monopoly**, **duopoly / split**, and **sample-size / Root-N** — and
analyses the simulated outcomes (CTR, advertiser welfare, etc.) in a set of
Jupyter notebooks.

## TL;DR

```bash
# 1. set up the env (Python 3.11 required; with pyenv: pyenv install 3.11.4)
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel setuptools
pip install -r requirements.txt
pip install -e ".[notebooks]"

# 2. drop the input .dta files under ./data/ (see "Required input data")

# 3. fit the causal forests (slow — hours per scenario)
python -m adsim.estimate --scenario monopoly
python -m adsim.estimate --scenario split --split 7
python -m adsim.estimate --scenario split --split 8
python -m adsim.simulate.base_ad_helpers

# 4. run the simulation
python -m adsim.simulate.monopoly        # or adsim.simulate.duopoly
                                         # or adsim.simulate.duopoly_root_n

# 5. analyse the outputs in scripts/*.ipynb
```

If you have pre-existing pickled forests from a prior run, check whether
they still load in this env before re-fitting:

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
notebooks/sample_size/sample_size_analysis.ipynb
```

---

## Layout

| Path | What lives there |
|---|---|
| `adsim/` | Installable Python package (`pip install -e .`). Canonical home for all importable + runnable code. |
| `adsim/paths.py` | `REPO_ROOT`, `DATA_DIR`, `RESULTS_DIR` (overridable via env vars). |
| `adsim/config.py` | Static knobs (`split_no_1/2`, `my_criteria`, ...), `ranks_list`, and explicit forest loaders. **Importing it does no I/O** — call `load_helpers()` / `load_monopoly_forests()` / `load_split_forests(n)` explicitly when you need the artifacts. |
| `adsim/utils.py` | Estimation helpers (`define_xyt`, `prepare_data`, `m_model_best_estimator`, ...) and per-step simulation primitives (`calc_tes`, `update_clicks`, `simulate_monopoly`, ...). |
| `adsim/propensity_model.py` | `PropensityModel`, the T-model used by `CausalForestDML`. |
| `adsim/estimate.py` | `python -m adsim.estimate` — fit per-rank causal forests (one CLI for all four scenarios). |
| `adsim/simulate/` | `python -m adsim.simulate.{base_ad_helpers,monopoly,duopoly,duopoly_root_n}` — forward simulation entry points. |
| `adsim/simulate/legacy/` | `python -m adsim.simulate.legacy.{simulation,simulation_parallel}` — older "Last 2 Days" simulations, kept for reference. |
| `scripts/check_old_pickles.py` | Standalone tool: verifies whether old `CF - Rank *.pkl` artifacts still load in this env. |
| `scripts/ranks_list.pickle` | Canonical input artifact (96 advertiser ranks). |
| `notebooks/analysis/` | Canonical / paper-ready analysis notebooks: `results_analysis`, `advertiser_welfare_analysis`, `ctr_vs_repeat`, `subject_correlation_matrix`, `merge_simulation_results`. |
| `notebooks/sample_size/` | Sample-size sensitivity aggregation: `sample_size_analysis.ipynb`. |
| `notebooks/exploration/` | Pairwise-estimation studies that produce paper artifacts (Shapley plots, LaTeX summaries) outside the canonical per-rank pipeline: `pairwise_estimation_ad_1_vs_2.ipynb`, `pairwise_estimation_ad_1_vs_3_full_week.ipynb`. See `notebooks/README.md` for details. |
| `tests/` | Pytest tests. (Currently a stub — see [cleanup tasks](#known-cleanup-tasks).) |
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

Verify the install:

```bash
python -c "from econml.dml import CausalForestDML; import adsim; print(adsim.__version__)"
```

### Configuring data + results paths

By default `adsim.paths` resolves:
- `DATA_DIR    = <repo_root>/data`
- `RESULTS_DIR = <repo_root>/results`

To point either at an external location (shared filesystem, external drive, ...), set env vars before running:

```bash
export ADSIM_DATA_DIR=/Volumes/research/online_ads/data
export ADSIM_RESULTS_DIR=/Volumes/research/online_ads/results
```

### Notes on env migration

The original env was a Windows conda export (Python 3.8.18, scikit-learn 1.2.2, econml 0.14.1). The new env is Python 3.11 + scikit-learn 1.5 + econml 0.15. Pickled `CF - Rank *.pkl` artifacts produced under the old env are **likely not loadable** in this one — `scripts/check_old_pickles.py` will tell you for sure.

---

## Required input data

None of these are checked in. Place them under `data/` (or under `$ADSIM_DATA_DIR`).

### Estimation inputs
- `data/Full Model/Estimation Data - Full Model - Monopoly.dta`
- `data/Full Model/Estimation Data - Full Model - Split {5,6,7,8}.dta`
- `data/Full Model/Estimation Data - Full Model - Split {N} - Root N.dta`

### Simulation inputs
- `data/Full Model/Simulation Data - Full Model - Monopoly - Subsample.dta`
- `data/Full Model/Simulation Data - Full Model - Split 7 8 - Subsample.dta`
- `data/Full Model/Simulation Data - Full Model - Split 7 8 - Root N - Subsample.dta`
- `data/Full Model/Advertiser Valuations.dta`

### Legacy / older runs (only needed for `scripts/simulation*.py`)
- `data/Simulation Data - Last 2 Days.dta`
- `data/Simulation Data - Last 2 Days - Merged Subjects Subsample.dta`

The list of advertiser ranks the study iterates over lives in `scripts/ranks_list.pickle` (96 ranks: 0..101 with gaps). `adsim.config.ranks_list` exposes it with rank 0 (base ad) and the >max-ad fringe entry already removed.

---

## Reproducing the study

From the repo root, with `.venv` activated.

### 1. Fit the per-rank causal forests (slow — hours per scenario)

A single CLI entrypoint covers all four scenarios:

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
- **Auto-resumes**: skips ranks whose pickle already exists. Pass `--force` to override. Re-running a job after a transient failure is just re-submitting it.
- **Filtering**: `--ranks 1,2,3` (explicit), `--ranks-filter "rank > 10"` (Python expression on `rank`, recovers the old `estimation2.py` continuation behaviour), or `--limit N` (cap to first N ranks).
- **Structured logs**: each rank prints scenario, walltime, and output path.
- **`--dry-run`** prints what would run without fitting anything.

Run `python -m adsim.estimate --help` for the full list.

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

Each takes `--processes N`, `--data PATH`, `--vals-data PATH`, and (where applicable) `--criteria {CTR,revenue}`, `--split-1 N`, `--split-2 N`. Run with `--help` for the full list.

Outputs: `results/Full Model/Simulation Results/Simluation Results - * - chunk N.dta` (one file per worker).

### 4. Analyse

Open notebooks under `notebooks/analysis/` (with `jupyter lab`):

1. `notebooks/analysis/merge_simulation_results.ipynb` — combine the per-chunk `.dta` outputs into one DataFrame.
2. `notebooks/analysis/results_analysis.ipynb` — main figures / tables.
3. `notebooks/analysis/advertiser_welfare_analysis.ipynb`, `ctr_vs_repeat.ipynb`, `subject_correlation_matrix.ipynb`.
4. Sample-size scenario: `notebooks/sample_size/`.
5. Earlier exploratory work: `notebooks/exploration/` (not part of the canonical pipeline).

---

## Working with the package programmatically

```python
from adsim import config
from adsim.utils import calc_tes, calc_base_ad_ctr, simulate_monopoly

# By default importing config does no I/O. Load what you need:
config.load_helpers()
config.load_monopoly_forests()
# config.forests is now a dict[int, CausalForestDML]
# config.helpers is now {"m1": ..., "e1": ...}

# For the duopoly scenario:
config.load_split_forests(config.split_no_1)
config.load_split_forests(config.split_no_2)
# config.split_forests[7], config.split_forests[8]
```

This is the contract the simulation scripts rely on.

---

## Known cleanup tasks

- [ ] Add real tests under `tests/` — `tests/test_utils.py` is currently a stub.
- [ ] The `predict_proba(...).reshape(-1, 1)` in `calc_base_ad_ctr_vector` is suspicious (predict_proba returns `(n, 2)`); audit before re-running.
- [ ] Hardcoded chunk sizes (`1620000 / n_processes`, `820000`, `300000`) in the simulation scripts. Make them either CLI args or auto-derived from `data.global_token_new.nunique()`.

---

## Reproducibility notes for reviewers

- **Pinned env.** `pyproject.toml` and `requirements.txt` pin Python 3.11 + scikit-learn 1.5 + econml 0.15 + pandas 2.2 + numpy 1.26 (compatible ranges). The original conda env (`environment.yml`) is gone.
- **No hardcoded paths.** All file I/O goes through `adsim.paths.DATA_DIR` / `RESULTS_DIR`. Override with `ADSIM_DATA_DIR` / `ADSIM_RESULTS_DIR` env vars; default to repo-relative.
- **Side-effect-free imports.** `import adsim.config` does no I/O. Forests are loaded only when an explicit `config.load_*_forests()` call runs. This means you can `import adsim.utils` from a notebook or test without needing a populated `results/` dir.
- **Dropped string-execed code.** Per-rank lookups previously written with `exec(f"config.cf_{rank}.const_marginal_effect(...)")` are now plain dict lookups against `config.forests[rank]` / `config.split_forests[split][rank]`.
- **Explicit verdict for old artifacts.** `scripts/check_old_pickles.py` reports OK / DEGRADED / INCOMPATIBLE for any pre-existing `CF - Rank *.pkl` so you don't have to guess whether a pickle from sklearn 1.2 / econml 0.14 will work here.
