# notebooks/

Jupyter notebooks for the online_ads study. All notebooks have been ported to
the modern `adsim` package layout:

- imports go through `adsim.simulation_steps` / `adsim.config` / `adsim.paths`
- file paths use `DATA_DIR` / `RESULTS_DIR` (from `adsim.paths`); override via
  `ADSIM_DATA_DIR` / `ADSIM_RESULTS_DIR` env vars
- forests are loaded explicitly via `config.load_helpers()` /
  `config.load_monopoly_forests()` / `config.load_split_forests(N)`

Every cell parses as valid Python 3.11. Notebooks still need their input data
(`.dta` files under `data/`) and forest pickles (under `results/`) to actually
execute end-to-end.

---

## `analysis/` — paper-ready analysis

Run in this order after the simulation has produced its per-chunk `.dta`
files:

| Notebook | Purpose |
|---|---|
| `merge_simulation_results.ipynb` | Combine per-chunk simulation outputs into one `.dta` per scenario. References chunk files for splits 1/2, 3/4, 5/6 that aren't produced by the canonical pipeline (`adsim.simulate.duopoly` does split 7/8 only) — those cells will `FileNotFoundError` unless you have legacy outputs. |
| `results_analysis.ipynb` | Main figures + tables. Self-contained: hand-rolls its own forest loader inside the notebook, doesn't depend on `config.forests`. Writes figures to `<RESULTS_DIR>/Full Model/Figures/`. |
| `advertiser_welfare_analysis.ipynb` | Welfare-side analysis. Reads merged simulation outputs + `Advertiser Valuations.dta`. Has one pre-existing bug at cell 17 (incomplete `pd.merge(...)` call) that's unrelated to the refactor. |
| `ctr_vs_repeat.ipynb` | CTR vs ad-impression-repeats heatmaps. Loads monopoly forests via `config.load_monopoly_forests()` and computes treatment effects across visit counts. |
| `subject_correlation_matrix.ipynb` | Correlation matrix across the 13 subject categories. Reads `<DATA_DIR>/Subjects Visited by Each User.dta` (not in the canonical input list — drop the file under `data/` to run). |

## `exploration/` — pairwise estimation studies

Two pairwise estimation studies that produce paper artifacts (Shapley values,
treatment-effect plots, LaTeX summary tables, feature-importance bar charts).
Each fits one `CausalForestDML` restricted to two advertiser ranks and
saves a "Full Data" forest pickle that lives outside the per-rank pipeline.

| Notebook | Pair | Data | Forest output |
|---|---|---|---|
| `pairwise_estimation_ad_1_vs_2.ipynb` | ranks {1, 2} | `Estimation Data - Full Model - Monopoly - Whole Week.dta` | `CF - Rank 1, 2 - Full Data.pkl`, plus `M -` / `E -` companions |
| `pairwise_estimation_ad_1_vs_3_full_week.ipynb` | ranks {1, 3} | `Estimation Data - Full Model - Monopoly - Whole Week.dta` | `CF - Rank 1, 3 - Full Data - New.pkl`, plus `M -` / `E -` companions |

Both reference data + forests that are **not** part of the canonical
`adsim.estimate` pipeline:
- `Estimation Data - Full Model - Monopoly - Whole Week.dta` (different from
  the regular monopoly estimation file)
- `CF - Rank 1, 2 - Full Data.pkl`, `CF - Rank 1, 3 - Full Data - New.pkl`,
  and their `M -` / `E -` companions

If you need to re-run these notebooks, place the `Whole Week` `.dta` under
`data/Full Model/` and the pairwise pickles under
`results/Full Model/Monopoly/`.

## `sample_size/` — sample-size sensitivity aggregation

| Notebook | Purpose |
|---|---|
| `sample_size_analysis.ipynb` | Aggregates simulation results across subsampling ratios (20%, 40%, 60%, 80%, 100%) and produces summary `.dta` tables + plots. |

The simulation step that generates the per-ratio outputs is now
`python -m adsim.estimate --scenario root-n --subsample-ratio R` followed by
the monopoly simulation; the notebook reads the resulting
`Simluation Results - SQRT N Sub {ratio}.dta` files.

The notebook still contains its own `calc_tes_sub` driver (the pre-package
prototype) that uses `exec()` and references `config.cf_<rank>_sub` /
`config.cf_<rank>` attributes that no longer exist. **Those driver cells
will fail at runtime** — but the aggregation/plotting cells (which only
read pre-computed `.dta` files) are unaffected. To re-run: skip the driver
cells and start from the aggregation step.

---

## Removed in cleanup

The following notebooks were deleted because their content was either
identical to a sibling, or fully subsumed by `adsim.simulation_steps` /
`adsim.simulate.*`. All recoverable from git history.

From `sample_size/`:
- `Sample Size Analysis copy 2/3/4.ipynb` — same notebook re-executed at
  different `subsampling_ratio` values; cached outputs are already
  captured in the on-disk `.dta` artifacts.

From `exploration/`:
- `Two Ads Estimation.ipynb`, `Two Ads Estimation-3.ipynb` — earlier drafts
  of the rank {1, 3} pairwise estimation; superseded by
  `pairwise_estimation_ad_1_vs_3_full_week.ipynb`
- `debug_duopoly_revenue.ipynb` — debugging the revenue-criterion path,
  superseded by `adsim.simulate.duopoly`
- `debug_update_clicks.ipynb` — sanity check of `update_clicks`,
  superseded by `adsim.simulation_steps.update_clicks`
- `prototype_revenue_ctrs.ipynb` — prototype that became
  `adsim.simulation_steps`
- `prototype_split_simulation.ipynb`,
  `prototype_split_simulation_with_outputs.ipynb` — abandoned
  "decide-on-split, evaluate-on-main" design; replaced by the current
  `adsim.simulate.duopoly` design
- `prototype_subsampling.ipynb` — prototype of the sample-size scenario,
  superseded by `python -m adsim.estimate --scenario root-n` and
  `python -m adsim.simulate.base_ad_helpers`
