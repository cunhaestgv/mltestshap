# Stage 7: diagnosing the type of fault against a retraining reference

Written and approved by the author on 2026-11-11, before any Stage 7 run. No Stage 7 feature, model output or classifier
had been computed. Everything below is fixed. Bad outcomes are results; any later change is reported as post hoc.

**Why.** Stage 6 showed that accuracy OR F-pairs-H flags 142 of 144 new mutants. The next question is diagnosis: given
a flagged model, what type of fault is it? Stage 2 diagnosed why one model's margin is wide, with no reference. That
cannot tell a faulty model from a healthy one, since healthy models have wide margins too.

Stage 7 diagnoses faults by comparing a model with the healthy retrains. It is also the last fair test of entropy
margins and decision trees for diagnosis: do they add anything to simpler measures?

## Models and labels
- **Datasets:** Adult, Bike and Diet, ungrouped, θ = 0.1, the same generation set X and background B as Stages 4–6.
- **Reference:** the 10 healthy retrains (seeds 1–10). The seed-0 model is **not** used as a reference. Weight
  mutants are perturbations of seed 0, so any comparison with seed 0 would reveal them trivially.
- **Training set:** the 144 exploration mutants (Stage 4).
- **Test set:** the 144 confirmation mutants (Stage 6).
- **Classes, 4:**

  | class | operators | per set |
  |---|---|---|
  | weights | GF, WS, NEB, NAI | 108 |
  | label noise (LN) | LN 0.1, 0.3 | 12 |
  | feature shuffled (FS) | FS on the top 3 variables | 18 |
  | data removed (DS) | DS on the top variable | 6 |

  DS has only 6 test models, so its recall is reported with that caveat.

## Fingerprint
Every measure is computed per model, relative to the 10 healthy models of its dataset, so that the three datasets can
be pooled. Notation:
- S is the model's SHAP matrix on X; μ_H and σ_H are the per-pair mean and sd over H.
- a_v is the mean |SHAP| summed over the features of variable v; ā_v is its mean over H.
- x_v is the value of variable v at the last time step (sequences) or the value itself (Adult).
- ŷ(x) is the model's output on X; for Adult it is the logit of the probability, clipped to [1e-6, 1 − 1e-6].
- ŷ_H(x) is the mean output of H.

**A. Baseline: accuracy and SHAP magnitude**
- A1 `acc_z`: (error − mean_H) / sd_H. Error is RMSE, or 1 − AUC for Adult.
- A2 `drop_max`: max over v of (ā_v − a_v) / ā_v. This is the Stage 5 one-sided drop.
- A3 `scale`: log(Σ_v a_v / Σ_v ā_v).

**B. Concentration of the SHAP differences**
- B1 `F-pairs-H`: the Stage 6 statistic.
- B2 `Z-pairs`: the Stage 6 statistic.
- B3 `var_conc`: d_v = mean over X of Σ_{f ∈ v} |S − μ_H|. B3 is max_v d_v / Σ_v d_v.
- B4 `ratio_sd`: sd over v of log(a_v / ā_v). Uniform shrinkage gives a small value.
- B5 `region_conc`: let r(x) = Σ_f |S(x, f) − μ_H(x, f)|. For each variable v and each tail (x_v in its bottom
  quartile, or in its top quartile on X), compute mean r in the tail / mean r elsewhere. B5 is the maximum over v and
  tails.

**C. Entropy margins and decision trees (Algorithm 1 and `method.generate`, θ = 0.1, wide_frac 0.1, depth 4)**
- C1 `W-margin`: the Stage 6 statistic, mean_f |w_f − w̄_f|.
- C2 `margin_shift`: mean_f (w_f − w̄_f). A signed value: positive when margins widen.
- C3 `margin_conc`: max_f |w_f − w̄_f| / Σ_f |w_f − w̄_f|. Set to 0 when the sum is 0.
- C4 `tree_res`: the share of wide-feature samples that fall in pure tree leaves (entropy < θ), minus its mean over H.
- C5 `tree_leaves`: log((n + 1) / (n̄_H + 1)), where n is the number of pure leaves.

**D. Outputs (black box, no SHAP)**
- D1 `slope` and D2 `r2`: slope and R² of the least-squares fit ŷ = a + b · ŷ_H over X.
- D3 `resid_var`: max over v of |Spearman corr(residual, x_v)|.
- D4 `resid_region`: as B5, with r(x) = |ŷ(x) − ŷ_H(x)|.

**E. With the model's own training data (descriptive only)**
For healthy and weight models, the training data is the clean one. For data mutants it is the corrupted one, rebuilt
with `mutants.corrupt_data` and rng 500 + seed, as in `run.get_weights`.
- E1 `own_fit_z`: the model's loss on its own training data, z-scored against the healthy losses on the clean
  training data.
- E2 `label_dis`: the share of the model's training samples whose label disagrees with H, minus the same share on the
  clean data. Disagreement means:
  - regression: |y − ŷ_H| > 3 × the healthy training RMSE;
  - classification: the healthy mean probability of the given label < 0.1.
- E3 `coverage`: max over v of the KS distance between the model's training data and the clean training data.
- E4 `relation`: max over v of the drop in |Spearman corr(x_v, y)| from the clean to the model's training data.

E needs the model's training data, and with it the data faults are expected to be nearly trivial to find. It is
reported to show this, not tested.

White-box weight measures, such as dead or inverted neurons, are **excluded**. The weight mutants are seed 0 with
edited weights, so such measures would detect the mutation operator rather than diagnose a fault.

## Classifiers
**Learned.**
- Multinomial logistic regression (scikit-learn): standardized features, C = 1.0, `class_weight="balanced"`,
  max_iter 5000.
- Fitted on the exploration set and evaluated once on the confirmation set.

| Set | Features |
|---|---|
| S0 | A |
| S1 | A + B |
| S2 | A + B + C |
| S3 | A + B + D |
| S4 | A + B + C + D |
| S5 | S4 + E |

**Rule-based (no training).** Thresholds fixed here, before any feature is computed:
1. `drop_max` ≥ 0.5 and `var_conc` ≥ 0.4: **FS**
2. else, `region_conc` ≥ 2.0: **DS**
3. else, `scale` ≤ log(0.8) and `ratio_sd` ≤ 0.25: **LN**
4. else: **weights**

## Claims
- **Metric:** balanced accuracy, the mean recall over the 4 classes, on the 144 confirmation mutants. Chance is 0.25.
- **Tests:** a stratified bootstrap (10,000 resamples within each class, seed 0) gives a one-sided p-value: the share
  of resamples in which the difference is ≤ 0. Holm correction over the 4 tests, α = 0.05.

1. **Concentration helps:** S1 > S0.
2. **Margins and trees help:** S2 > S1. If not supported, the paper says margins and trees did not help diagnose the
   fault type.
3. **Outputs help:** S3 > S1.
4. **The a priori rules work:** rule-based > 0.25.

**Also reported, not tested:**
- balanced accuracy and the confusion matrix for every set, S0–S5, and for the rules;
- per-class recall;
- per dataset;
- the same on the exploration set by 5-fold stratified cross-validation (exploratory);
- leave-one-dataset-out: train on the exploration mutants of two datasets, test on the confirmation mutants of the
  third;
- restricted to the mutants flagged by accuracy OR F-pairs-H (Stage 6, post hoc rule);
- a random forest (500 trees, min_samples_leaf 2, balanced, seed 0) on S4;
- localization:
  - FS: the top-drop variable equals the shuffled variable;
  - DS: the B5 and D4 argmax variable equals the removed variable, and the tail is the top one;
- among weight mutants only: a 4-way GF / WS / NEB / NAI classification on S4.

## Checks
- On the exploration set, B1, B2 and C1 must reproduce the Stage 6 statistics exactly.
- Healthy models, cross-fitted against the other 9, must give values near 0 for A2, A3, C2 and C4, and slope ≈ 1. Their
  fingerprints are reported next to the mutants'.

## What was known before this design
- Stages 5–6 already showed that the one-sided drop localizes FS variables (17–18/18). A2 is therefore expected to
  separate FS well, and S0 is a strong baseline.
- No other measure above has been computed on any model.

## Implementation details (fixed with the code, before any run)
- C4: a model with no wide feature gets `tree_res` = 1, since nothing is blurred.
- B5 and D4: a tail is skipped when it holds no samples or all of them, which can happen for binary variables.
- Spearman correlations of a constant column count as 0.
- Healthy rows are cross-fitted: their reference is the other 9 healthy models. ε of Z-pairs is the Stage 6 one.

Code: `src/stage7.py`. The steps are `features --dataset <ds>`, which computes the fingerprints, and `summarize`, which
fits the classifiers, runs the tests and writes the tables. Results: `results/stage7/`.

## Outcome (2026-11-11)

Tables: `results/stage7/summary.md`. Fingerprints: `results/stage7/features_<dataset>.csv`. Check passed: B1, B2 and C1
reproduce Stage 6 exactly (318 models, difference 0). Healthy models: slope 1.01, scale −0.01, margin shift 0.00 (medians).

**Pre-registered tests (confirmation set, 144 mutants, balanced accuracy, chance 0.25, Holm over 4)**

| claim | result | difference [95% CI] | Holm p |
|---|---|---|---|
| 1. S1 (A + concentration) > S0 (A) | **not supported**: 0.586 vs 0.590 | −0.005 [−0.130, +0.120] | 0.56 |
| 2. S2 (+ margins and trees) > S1 | **not supported**: 0.623 vs 0.586 | +0.037 [−0.086, +0.162] | 0.56 |
| 3. S3 (+ outputs) > S1 | **supported**: 0.718 vs 0.586 | +0.132 [+0.046, +0.241] | < 0.001 |
| 4. a priori rules > 0.25 | **supported, weakly**: 0.324 | +0.074 [+0.021, +0.132] | 0.006 |

Margins and trees did not significantly improve the diagnosis of the fault type, so the paper says they did not help.

The rules call almost everything "weights" (104/108 correct). They recognise 6/18 FS and no LN or DS: the thresholds
were too strict.

**Also reported**
- Balanced accuracy: S4 (all model-only measures) 0.734; S5 (with the model's own training data) 0.944, including DS 6/6.
  As expected, the training data makes data faults easy to find.
- Leave-one-dataset-out collapses (S4 0.17–0.65; S0 0.32–0.48). The classifiers learn dataset-specific values and do
  not transfer to a new dataset without labelled mutants from it.
- Flagged mutants only (142): S3 0.780, S4 0.713.
- Weight operators, 4-way: 0.611 (chance 0.25).
- Localization:
  - FS: the top-drop variable is the shuffled variable in 17/18.
  - DS: the SHAP region pointer is on the removed variable, top tail, in 5/6.
  - Control (post hoc): the removed variable is the dataset's most important one, so the pointer often lands on it for
    other models too (Diet weights 99%, Bike FS 67%). It is informative on Bike only (DS 4/4 vs healthy 0% and weights
    10%, top tail). Over both sets, DS 7/12.
- Median fingerprints: weight and LN mutants fit even their own training data badly (`own_fit_z` 26 and 48, vs
  FS 6 and DS −0.7). LN shrinks outputs (slope 0.72) and |SHAP| (scale −0.12).
