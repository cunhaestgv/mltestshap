# Stage 3: can margins, SHAP values and trees choose the fix for a fault?

Written on 2026-10-30, before any Stage 3 run. Scenarios, candidates, rules, thresholds, measures and
success criteria below are fixed. A smoke run on one case only checks that the code runs and how long
it takes; nothing is changed because of what a scenario does. If a rule or scenario turns out badly,
that is reported as a result. Any later change is reported as post hoc, next to the original result.

## Question

Given a model with faults, does a rule that recommends one fix choose the fix that lowers the test
error most, and does each information source add to that choice?

- **L1, errors only:** error tree, seeds, densities and a gradient-boosting probe. No SHAP.
- **L2:** L1 + SHAP values (ordinary and on the loss).
- **L3:** L2 + the margin diagnosis (`diagnose.py`, unchanged thresholds), including its SHAP trees.

Baselines: a uniformly random candidate, and the best single action for all cases, picked after seeing
the results (which favours the baseline).

## Synthetic cases

x1..x7 ~ U(0,1), n = 5000: 3000 train, 500 validation (early stopping), 750 discovery, 750 test. 5
replicates per scenario and task. As in Stage 2, f = 0.3 x3 − 0.3 x4 + 0.2 x5 + g(x), ε ~ N(0, 0.05²).
Regression: y = f + ε. Classification: y = [f + ε > c], c = median of f, and the clean label is [f > c].
Defects affect the training split unless stated; validation is also affected by `spurious`, and all
splits by `missing`. Discovery and test are clean otherwise. σ(u) = +1 if u > 0.5 else −1.

| scenario | g(x) and defect | task | injected remedy |
|---|---|---|---|
| noeffect | g = 0 (control) | both | none needed |
| sharp | 0.5·[x1 > 0.5] | both | capacity |
| inter | 0.8 (x1 − 0.5) σ(x2) | both | capacity |
| complex | 0.8 (x1 − 0.5) σ(x2) σ(x6) σ(x7) | both | capacity |
| nonmono | 0.5·[0.3 < x1 < 0.7] | both | capacity |
| missing | 0.8 (x1 − 0.5) σ(z); z is hidden, x2 replaced by z + N(0, 0.15²) | both | audit (gives z) |
| noise | as inter; 40% of train with 0.25 < x1 < 0.75 have permuted targets | both | audit (clean labels) |
| sparse | as sharp; train with 0.25 < x1 < 0.75 kept with probability 0.2 | both | more data |
| unstable | as sharp; only 300 train samples and 20 extra noise inputs | both | ensemble or regularize |
| spurious | as sharp; in train and validation x7 is a noisy copy of the target, elsewhere uniform | both | drop the feature |
| outliers | as sharp; 5% of train targets shifted by ±3 | regression | robust loss |
| imbalance | as sharp; c = 90th percentile of f | classification | threshold or class weights |

"Injected remedy" is only the reason for the scenario. Scoring uses the measured gains, not this column.

## Real cases

Adult (classification) and Bike (regression), 3 replicates (other seeds; Adult also reshuffles the
discovery/test split). The original test split is halved into discovery and test; Bike keeps time order
(first half is discovery). Diet is excluded: 18 test days cannot be split. `more_data` and `audit`
cannot be tried on real data; such recommendations are counted as not evaluable and left out of regret.

## Candidates, all actually tried

Trained with 3 seeds, test error averaged over them, except where noted.

| candidate | what it does |
|---|---|
| none | base model, mean over 5 seeds |
| ensemble | prediction averaged over the 5 base seeds |
| capacity | hidden layers 128-64-32 (LSTM: 4× units) |
| regularize | L2 1e-3 on kernels, dropout 0.2 |
| more_data | 1000 extra samples inside the worst slice (synthetic), collected with the same defects, except sparsity |
| drop_top | the feature with the highest loss Shapley value is replaced by its training mean |
| audit | synthetic only: `missing` gets z, `noise` and `outliers` get clean targets; otherwise equal to none |
| robust | regression: Huber loss, δ = 0.1 |
| threshold | classification: threshold maximizing balanced accuracy on discovery, base models |
| class_weight | classification: inverse-frequency class weights |

## Signals, all from discovery data or earlier

Error of a sample: regression |y − ŷ|; classification |y − p|, mean over the 5 seeds.
- **Slice:** regression tree on errors, depth 3, min leaf max(20, 5% of discovery). Worst leaf = S.
- **ratio:** mean error in S / mean error overall.
- **density:** share of train in S / share of discovery in S.
- **disagreement D:** mean |ŷ_seed − ŷ_mean| in S / mean error in S.
- **gbm:** error in S of sklearn HistGradientBoosting (default settings) fitted on train / model error in S.
- **outlier share:** share of train residuals with |r| > 5 × median |r|.
- **prevalence:** share of the minority class in train.
- **Loss Shapley:** Shapley values of the per-sample loss (squared error, log-loss), estimated with
  30 sampled permutations per sample on up to 300 discovery samples, reference seed.
  top_share = highest positive mean value / sum of positive means; top_rel = that value / mean loss.
- **shap_total:** sum over features of mean |SHAP| in S, DeepSHAP, reference seed, up to 1000 discovery
  samples. For classification SHAP explains the probability.
- **sign agreement:** mean per-sample sign agreement across seeds of the feature with top |SHAP| in S.
- **row:** `diagnose.diagnose` on discovery data (so errors are held out) for the root feature of the
  error tree.

## Rules (first match wins)

**L1**
1. prevalence < 0.2 (classification) → threshold
2. outlier share ≥ 0.01 (regression) → robust
3. density < 0.5 → more_data
4. D ≥ 0.5 → ensemble
5. gbm ≤ 0.8 → capacity
6. ratio ≥ 1.5 → audit
7. otherwise → none

**L2:** L1, with these changes: before rule 3, top_share ≥ 0.5 and top_rel ≥ 0.1 → drop_top; rule 4 also
fires when sign agreement < 0.8 (`THRESHOLDS["agree"]`); rule 6 also fires when shap_total < mean error in S.

**L3:** L2, with this step before rule 3 and after the drop_top check: row 5 → audit; row 6 → more_data;
row 7 → ensemble; rows 1, 3, 4, 9 with ratio ≥ 1.5 → capacity. Other rows continue with L2.

## Measures

Test error, against clean targets: regression MAE; classification balanced error rate. On real data,
against the observed targets. Gain of a candidate = (err_none − err_candidate) / err_none.

1. **Regret** (primary): best gain among evaluable candidates and none, minus the gain of the recommendation.
2. **Hit rate:** the recommendation's gain is at least 90% of the best gain, or the best gain is under 1% and
   the recommendation does not lose more than 1%.
3. **Culprit feature** (`spurious` only): does x7 rank first by loss Shapley, by error-tree root, by |SHAP|?
4. **Slicing:** worst leaf of the error tree against the worst of 8 k-means clusters of SHAP vectors, both
   chosen on discovery: error lift on test (slice error / overall error), and F1 against the true region
   0.25 < x1 < 0.75 for `sparse` and `noise`.

## What counts as support

A level adds value if its mean regret over all evaluable synthetic cases is lower than the level before it
and the 95% paired bootstrap interval (2000 resamples of cases) of the difference excludes zero. L1 is
useful if the same holds against the best single action. Real cases are reported separately, as a check;
3 replicates of 2 datasets are too few for a verdict.

## Outcome (2026-10-31)

All 116 cases ran (110 synthetic, 6 real). Tables: `results/stage3/summary.md`; checks: `src/stage3_check.py`.
Repeating all synthetic signals gave identical values and recommendations.

**Verdict: none of the support criteria was met.**

- Mean regret, synthetic: L1 0.227, L2 0.219, L3 0.283; always audit (best single action) 0.244; random 0.356.
  Hit rates 25%, 26%, 19%. Rules beat random, but not the best single action.
- L1 vs always audit: −0.017 [−0.057, 0.022]. L2 vs L1: −0.008 [−0.028, 0.005]. Neither supported.
- L3 vs L2: +0.064 [0.024, 0.107]. Adding the margin diagnosis made the choice significantly worse. 93 of 110
  root features were rows 1 or 4, and with the worst leaf's ratio nearly always ≥ 1.5 (median 2.27) both map to capacity,
  one of the worst single actions. Row 4 is also a remainder class, not a detected cause (see `diagnose.py`).
- Rule 2 fired in 33 of 55 regression cases, including ones without outliers: a model that fits well has
  residuals with heavy relative tails. Post hoc: requiring a 5% share, or removing the rule, did not help
  (L1 0.281 and 0.276); L2 vs L1 stayed non-significant and L3 stayed worse.
- Culprit feature (`spurious`): error-tree root 10/10, |SHAP| 10/10, loss Shapley 5/10 (5/5 regression, 0/5
  classification). In regression dropping it gained 91–93%, but the L2 rule fired only once, as top_rel ≥ 0.1 failed in 4 of 5.
- Slicing: error lift on test 2.18 (error tree) vs 2.47 (SHAP clusters), difference 0.288 [0.063, 0.540], clusters
  higher in 70/116. F1 on the true region (sparse, noise): 0.37 vs 0.25. Clusters isolate errors at least as
  well, but do not recover the injected region. No criterion was set for this measure.
- Real data, as a check: Bike rules all said robust (gain 3%), capacity gained 17%. Adult L1 and L2 said audit (not
  evaluable), L3 said capacity (1%); threshold gained 22% on balanced error, a measure that favours it.

Corrections to the run, none based on results: real-data replicates used the same seeds (fixed before the full run);
the cluster lift compared misaligned samples (fixed, recomputed for all cases; no other measure uses it); thread limits
added to the numeric libraries after the machine was overloaded.
