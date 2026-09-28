# Referee diagnostics (`python -m adsim.diagnostics`)

The diagnostic checks from the revision plan (sections 4.2 and 4.3, plus
the item 2.2 placebo-monopoly check), run on the saved per-rank forests.
Each check is one function in `adsim/diagnostics/identification.py` or
`adsim/diagnostics/model_checks.py` and returns named tables.

Per-rank checks rebuild the (base ad, rank) pair each forest was fit on
(same arm cap and seed, from `cf.adsim_plan_`) and reuse that forest's
nuisance-model settings. Cross-fitting folds are grouped by user, and all
standard errors are clustered by user.

## Running it

```bash
export ADSIM_DATA_DIR=/path/to/data ADSIM_RESULTS_DIR=/path/to/results

# All checks, the 10 most-shown ads, monopoly forests
python -m adsim.diagnostics --scenario monopoly --n-jobs 30

# Cheap identification tests only, for chosen ranks
python -m adsim.diagnostics --scenario monopoly --checks identification --ranks 1,2,3

# One split platform
python -m adsim.diagnostics --scenario split --split 7 --checks model

# Winner's-curse check: fit a second monopoly with another seed, then compare
ADSIM_RESULTS_DIR=/path/to/results-seed7 python -m adsim.estimate --scenario monopoly --random-state 7
python -m adsim.diagnostics --scenario monopoly --checks placebo-monopoly \
    --placebo-forest-dir "/path/to/results-seed7/Full Model/Monopoly"
```

The per-rank checks need only the estimation data. Without a saved forest
for a rank they use default nuisance settings and log a warning
(`forest_found` in the manifest), so the identification tests can run
before re-estimation. `calibration`, `out-of-range` and `placebo-monopoly`
need every rank's forest plus `m1.pkl` / `e1.pkl`, and are skipped
otherwise.

Output goes to `RESULTS_DIR/Full Model/Diagnostics/<scenario dir>/<check>/`
as `<table>.csv`, `<table>.tex` (booktabs) and `<figure>.pdf/.png`, plus
`manifest.json` with the settings, ranks, and any skipped or failed checks.
Per-rank tables start with `rank` and `support` (low / medium / high
impression tercile). The run exits with code 1 if any check failed.

Useful flags: `--checks` (names, `identification`, `model`, or `all`),
`--top-k` / `--ranks`, `--cell-cols` (page-time cells, default website x
day x hour), `--refit-trees` (trees per refit forest, default 100),
`--max-eval-rows` (impressions for the pipeline checks), `--eval-data`
(held-out impressions for calibration), `--visit-col` (visit order for the
lagged-click placebo). `--help` lists everything.

## The checks

| Check | Plan / referee | What it computes | Supports Assumption 1 / a good model when |
|---|---|---|---|
| `propensity-info` | 4.2.1; R1.3a, R2 | Out-of-sample log loss, AUC and Brier of e(C) vs e(C + browsing), e(C + ad history), e(all), same learner. The log-loss gain is tested against refits with the added block permuted within e(C) strata | `p_perm` not small; `excess_gain` ~0 |
| `balance` | 4.2.2 | SMDs of the 19 history covariates, focal vs base ad: raw, within cells, IPW with e(C); cell fixed-effects Wald F per block; love plot | \|SMD\| < 0.1 and omnibus p not small |
| `placebo-outcome` | 4.2.3 | DML effect of the ad shown now on the previous impression's click and on "clicked any ad before", controlling for C | effects ~0 |
| `refuters` | 4.2.4 | Refits the forest with T permuted within cells, with a random extra control, and on 80% of users | placebo ATE ~0; other two ~ observed |
| `sensitivity` | 4.2.5 | Chernozhukov et al. OVB bounds for the DML effect, benchmarked on the browsing block's partial R², plus robustness values | the benchmark bound excludes 0; robustness value > benchmark |
| `overlap` | 4.2.6 | e(C) histograms by arm, Crump et al. threshold, DML effect with and without trimming | little mass near 0 or 1; trimming barely moves the effect |
| `nuisance` | 4.3; R1.4, R4 CV | Cross-fitted log loss, Brier, AUC, R² of m(X) and e(C), overall and by segment | — (report) |
| `cate-validation` | 4.3; R4 CV | Refit on 70% of users; on the rest: R-score, DRTester BLP slope, calibration R², QINI, AUTOC, and calibration of the shown arm's click probability | R-score > 0, BLP slope near 1 with small p |
| `estimators` | 4.3; R1.4, R3 | Held-out comparison of the forest with a discrete-outcome forest, ForestDRLearner, LinearDRLearner, a gradient-boosting T-learner and a logit with ad x covariate interactions | the forest's R-score is competitive; CATE ranks agree |
| `stability` | R3 data | Forests fit on the first vs second half of the days; ATE difference and CATE rank correlation | correlation high, ATE difference small |
| `calibration` | 4.3; R1.4 | The simulation's predicted CTR for the shown ad (y0 + tau, clipped) vs realised clicks: Brier, ECE, reliability diagram, by support bucket and segment | predicted / observed ~1 |
| `out-of-range` | 4.3, 2.9; R1.4 | Share of every rank's y_r(x) below 0 (clipped) or above 1, over sampled impressions | small shares |
| `placebo-monopoly` | 2.2 | A second monopoly fit chooses first-visit ads and is scored by the main monopoly, as duopolists are | `mechanical_gain` (CTR) / `value_gain` (revenue) is the optimizer's-curse share of the merger gain |

Segments are device, first visit vs returning (`total_visits <= 1`), and
heavy vs light browsers (`total_visits` above the median).

## Caveats when reading the results

- **Why `propensity-info` uses a permutation null.** Two random forests on
  different feature sets overfit differently, so adding an irrelevant block
  still moves the out-of-sample log loss. At N ~ 10^5 that shift looks
  significant under a plain z-test (on a 400K-row synthetic pair with
  random allocation, z was 3.6-5.7; the permutation test gave p = 0.13-0.44).
  The null refits reproduce this shift, so read `excess_gain` and `p_perm`,
  not `gain` alone.
- **Ad-history block.** If `impression_repeat` / `previous_clicks` count
  the ad actually shown (not rank r on every row), they predict T
  mechanically, so a positive `context+ad_history` gain in
  `propensity-info` is then not evidence of selection. The browsing block
  has no such problem.
- **Within-cell tests need mixed cells.** Only rows in cells that show both
  ads inform the within-cell SMDs and the permutation placebo.
  `share_rows_in_mixed_cells` (balance) and `share_T_changed` (refuters)
  report this; if they are small, pass coarser `--cell-cols`.
- **Calibration is in-sample by default.** The forests were fit on the
  estimation data. Pass held-out impressions (for example another week or
  days 6-7 with clicks) as `--eval-data`. `cate-validation` is always
  out-of-sample (refit on training users).
- **Pipeline checks use observed covariates.** `calibration`,
  `out-of-range` and `placebo-monopoly` evaluate every rank at each sampled
  impression's observed `impression_repeat` / `previous_clicks`, whereas
  the simulation substitutes each ad's own running counts.
- **Sensitivity is for the pair's average effect** (partially linear DML,
  outcome model on all features, propensity on C), not for each CATE.
  econml 0.15 has no `sensitivity_summary`, so the bounds are computed in
  `adsim.diagnostics.stats` following econml 0.16's formulas, with
  user-clustered variances.

## Compute

Measured on a synthetic pair the size of the largest capped monopoly pair
(400,000 rows, 0.1% CTR, 8 vCPU, `--refit-trees 100`):

| Check | Time per rank |
|---|---|
| `propensity-info` (10 permutations per block) | ~5.5 min (34 cross-fits; scales with `--n-permutations`) |
| `balance`, `overlap`, `nuisance` | < 1 s each (share cached cross-fits) |
| `placebo-outcome`, `sensitivity` | ~10 s each |
| `refuters` (4 refits) | ~60 s |
| `cate-validation` | ~25 s |
| `stability` (2 refits) | ~15 s |
| `estimators` (6 estimators) | ~40 s |

Smaller pairs are proportionally faster. The pipeline checks
(`calibration`, `out-of-range`, `placebo-monopoly`) evaluate every rank's
500-tree forest on `--max-eval-rows` impressions (default 100,000); that
is ~95 forest predictions, plus ~95 more for the placebo set.

## Not implemented here

- **4.2.7, conditioning on bids, pacing and new-ad boosts.** The data has
  no allocation inputs. If they exist, add them to the context block.
- **DoWhy refuters.** `est.dowhy` needs the optional dowhy package; the
  same three refutations are done natively in `refuters`.
- **Change in simulated gains** after trimming, per alternative estimator,
  or for the joint multi-arm forest (4.3 item 7). Each needs a full
  re-estimation and simulation run.
- **Simulation-side analyses** (4.1 uncertainty and off-policy evaluation,
  4.4 sensitivity grid and participation, 4.5 break-even prices, 4.6
  grouped importance, 4.7 remedies). These are results, not diagnostics,
  and depend on the simulation fixes (items 2.1, 2.2).
