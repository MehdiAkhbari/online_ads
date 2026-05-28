# online_ads

Causal-inference-driven simulation of online ad serving. The study fits one
`CausalForestDML` (econml) per advertiser rank on observational impression
data, then replays user visits forward with those forests in three scenarios
— **monopoly**, **duopoly / split**, and **sample-size / Root-N** — and
analyses the simulated outcomes (CTR, advertiser welfare, etc.) in a set of
notebooks.

> Status: research code mid-cleanup. The repo still contains several
> near-duplicate scripts from individual experimental runs; see
> [Known cleanup tasks](#known-cleanup-tasks) at the bottom.

---

## Pipeline

```
data/Full Model/Estimation Data - Full Model - Monopoly.dta
data/Full Model/Estimation Data - Full Model - Split {5,6,7,8}[ - Root N].dta
        │
        ▼   scripts/estimation.py            (and split / sqrt-n variants)
results/Full Model/Monopoly/CF - Rank {r}.pkl                (~95 forests)
results/Full Model/Split {N}[ - Root N]/CF - Rank {r}.pkl
        │
        ▼   scripts/base_ad_ctr_estimation.py
results/Full Model/{m1,e1}.pkl              (base-ad y0 helpers)
        │
        ▼   scripts/{monopoly,duopoly}_simulation[_sqrt_n].py
            (loads forests via scripts/config.py)
results/Full Model/Simulation Results/Simluation Results - * - chunk N.dta
        │
        ▼   notebooks
scripts/merge_simulation_results.ipynb       → combines chunks
scripts/Results Analysis.ipynb               → main figures/tables
scripts/advertiser_welfare_analysis.ipynb
scripts/ctr_vs_repeat.ipynb
scripts/Sample Size Analysis copy*.ipynb
```

The forests are loaded eagerly at import time inside `scripts/config.py`
(via `joblib.load`) and made available as module-level names `cf_{rank}`,
`cf_{rank}_s{split}`, etc. The simulation scripts then reach into those
through `import config`.

---

## Layout

| Path | What lives there |
|---|---|
| `adsim/` | Installable Python package (`pip install -e .`). The intended long-term home for shared utilities. Currently contains an older fork of `utils.py` that the scripts do not import yet. |
| `scripts/` | The runnable research code (estimation, simulations, analysis notebooks). `scripts/utils.py` is the canonical utilities module today. |
| `notebooks/` | A single scratch notebook (`test.ipynb`). |
| `tests/` | Placeholder pytest skeleton. |
| `data/`, `results/` | **Not in git.** Populate locally before running anything (see [Required input data](#required-input-data)). |

---

## Setup

Requires Python **3.11**.

```bash
# from the repo root
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel setuptools
pip install -r requirements.txt
pip install -e .              # installs the `adsim` package in editable mode
# optional: notebook + dev extras
pip install -e ".[notebooks,dev]"
```

Verify the install:

```bash
python -c "from econml.dml import CausalForestDML; print('ok')"
```

### Notes on the env change

The original env was a Windows conda export (`environment.yml`, Python
3.8.18, scikit-learn 1.2.2, econml 0.14.1). It is **gone**. The new env
targets Python 3.11 + scikit-learn 1.5 + econml 0.15. Pickled
`CF - Rank *.pkl` artifacts produced under the old env are likely **not**
loadable in this one — plan on re-fitting the forests once the input
`.dta` files are in place.

---

## Required input data

None of these are checked in. They must be placed under `data/` before any
script will run.

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

The list of advertiser ranks the study iterates over lives in
`scripts/ranks_list.pickle` (96 ranks: 0..101 with gaps).

---

## Running the study end-to-end

> All paths in the existing scripts use Windows backslashes (e.g.
> `..\\data\\Full Model\\...`). On macOS / Linux you currently need to
> either run from `scripts/` with a symlinked or POSIX-equivalent layout,
> or fix the paths. This is one of the [known cleanup tasks](#known-cleanup-tasks).

From the repo root, with `.venv` activated:

```bash
cd scripts

# 1. Fit per-rank causal forests (monopoly + one split). Slow.
python estimation.py

# 2. Fit base-ad y0 helpers (m1.pkl, e1.pkl).
python base_ad_ctr_estimation.py

# 3. Replay user visits forward.
python monopoly_simulation.py
# or:
python duopoly_simulation.py
```

Then open the analysis notebooks in `scripts/`:

1. `merge_simulation_results.ipynb` — combine the per-chunk `.dta` outputs.
2. `Results Analysis.ipynb` — main figures/tables.
3. `advertiser_welfare_analysis.ipynb`, `ctr_vs_repeat.ipynb`, etc.

---

## Canonical scripts vs. variants

The `scripts/` directory still contains several near-duplicates from
individual experimental runs. The canonical ones to drive the pipeline:

| Stage | Canonical | Variants kept for now |
|---|---|---|
| Estimation | `estimation.py` (split 7) | `estimation2.py` (rank > 10 only), `estimation3.py` (split 6 + Root-N), `estimation_split_5.py`, `estimation_split_6.py` (older logistic-regression PropensityModel), `estimation_sqrt_n.py` |
| Base-ad CTR | `base_ad_ctr_estimation.py` | — |
| Monopoly sim | `monopoly_simulation.py` | `simulation.py`, `simulation_parallel.py` (older "Last 2 Days" data) |
| Duopoly sim | `duopoly_simulation.py` | `duopoly_simulation_sqrt_n.py` |
| Shared utils | `scripts/utils.py` | `adsim/utils.py` (older fork, not imported by the scripts today) |

---

## Known cleanup tasks

- [ ] Fix Windows-only paths (`..\\data\\...`) so the code is portable.
- [ ] Replace `adsim/constants.py` (hardcoded OneDrive path) with paths
      derived from a `DATA_DIR` / `RESULTS_DIR` env var or config.
- [ ] Reconcile `scripts/utils.py` (956 lines, current) with
      `adsim/utils.py` (852 lines, older); pick one home.
- [ ] Collapse the `estimation*.py` family into a single parameterised
      script (`--scenario`, `--split`, `--data`, `--out`).
- [ ] Delete leftover/stub files:
      `scripts/Untitled-1.py`, `scripts/file.pkl` (5 bytes),
      `scripts/propensity_score.pkl` (empty), `notebooks/test.ipynb`,
      and the duplicate `Two Ads Estimation-3.ipynb` at the repo root.
- [ ] Replace `exec(...)` / `globals()[...]` patterns (used to manage the
      ~95 per-rank forests) with explicit dicts.
- [ ] Add real tests under `tests/` — `tests/test_utils.py` is currently
      a stub.
