# Stage 6: confirming detectors based on a retraining reference

Written on 2026-11-03, before any Stage 6 run. Everything below is fixed. Bad outcomes are results; any later
change is reported as post hoc.

**Why.** The author chose to reposition the method. Partitions, entropy margins and trees describe the model and
generate tests. Detection uses agreement among healthy retrains and |SHAP| magnitude, as a complement to accuracy.

The best Stage 4 detector (F-pairs) was found post hoc, on the same mutants it was scored on. Stage 6 therefore
scores a fixed set of detectors on **new mutants**. Those mutants use the same operators and levels but new seeds;
the healthy models are unchanged.

## Models
- **Datasets:** Adult, Bike and Diet, ungrouped, θ = 0.1.
- **Healthy:** the 10 healthy retrains of `run.py`.
- **Exploration set:** the 144 general mutants of `run.py` and Stage 4.
- **Confirmation set (new):** 144 mutants, 48 per dataset, from the operators of `run.model_plan`:
  - weight mutants GF, WS, NEB, NAI × levels 0.02, 0.05, 0.1 × seeds 3, 4, 5 (36 per dataset);
  - data mutants LN 0.1 and 0.3, FS on the top 3 variables, DS on the top variable, each × seeds 2, 3 (12 per
    dataset).

  Seeds and training follow `run.get_weights`.
- **SHAP:** DeepSHAP on the same generation set and background as `run.py`.

## Detectors
All statistics are computed per model. Each healthy model is scored against the other 9 healthy models (cross-fitted).
"Pairs" are all (input, feature) pairs of the generation set. H means the healthy models used as reference.

| Name | Statistic |
|---|---|
| F-pairs | As in Stage 4: share of pairs whose sign differs from the seed-0 sign, over pairs where H disagrees with seed 0 in ≤ 10% of models |
| F-pairs-H | Share of pairs whose sign differs from the majority sign of H, over pairs where ≤ 10% of H differ from that majority (majority as in `stage5_localize.majority`) |
| Z-pairs | Mean over pairs of min(\|S − mean_H\| / (sd_H + ε), 10); ε = 0.01 × mean \|S\| of H over all pairs |
| M-var | Max over variables of \|a_v − ā_v\| / ā_v, with a_v the mean \|SHAP\| summed over the variable's features and ā_v its mean over H |
| W-margin | Mean over features of \|w_f − w̄_f\|, with w_f the `margin_frac` of Algorithm 1 (`method.analyse_feature`, θ = 0.1, wide_frac 0.1) on the model's own SHAP signs: 0 for a single partition, 1 if wide with no boundary; w̄_f its mean over H |
| ours, bva, random | As in `run.py`: failure rate with the SHAP oracle, suites generated from seed 0; random is 5 repetitions, each scored separately |
| accuracy | RMSE (Bike, Diet) or −AUC (Adult) |
| acc+X | max(z_acc, z_X), with z = (statistic − mean_H) / sd_H |

**Detection.** A model is detected when its statistic is above that of every healthy model (cross-fitted statistics),
as in `run.py`.

**Choosing the best new detector (claim 2).** On the exploration set, choose the one of {F-pairs-H, Z-pairs, M-var}
with the highest pooled detection rate (ties: highest pooled AUROC). The choice is recorded below before any
confirmation-set SHAP exists.

## Claims
Pooled over the 144 confirmation mutants. Each claim is a one-sided reading of an exact two-sided McNemar test; the
difference must be in the stated direction. Holm correction over the 5 tests, α = 0.05.
1. F-pairs detects more than ours (test 1) and more than bva (test 2). Supported if both are significant.
2. The chosen detector detects more than F-pairs (test 3).
3. acc+F-pairs detects more than accuracy (test 4). The number of healthy models it flags (cross-fitted, out of 30)
   is reported.
4. W-margin detects more than ours (test 5). If not, the paper presents margins as a description of the model only.

**Also reported, not tested:**
- detection by dataset, and by weight and data mutants;
- AUROC, mutants vs healthy;
- detection of the mutants that accuracy misses;
- M-var localization on the FS mutants (top variable = shuffled variable);
- all detectors on the exploration set, labelled exploratory.

## Background sweep (descriptive, for the case study)
For every narrow feature of the seed-0 model of each dataset, recompute SHAP with two other backgrounds. These are
100 training samples at or below the feature's median, and 100 above it, as in `stage4_background.py`.

Label each boundary:
- **reference crossing:** narrow under both backgrounds, and |boundary_high − boundary_low| /
  |bg_high − bg_low| ≥ 0.5;
- **learned step:** narrow under both, ratio < 0.5;
- **not stable:** not narrow under at least one background;
- **no contrast:** the two backgrounds have the same mean of the feature, as happens for some binary features.

## Checks
- On the exploration set, the Stage 6 code must reproduce Stage 4's F-pairs, ours and bva statistics.
- The accuracy of the confirmation mutants is reported next to that of the exploration mutants.

Code: `src/stage6.py`. Results: `results/stage6/`.

## Choice for claim 2 (recorded 2026-11-03, before any confirmation-set model existed)
Exploration set, pooled detection (AUROC): F-pairs-H 90% (0.978), Z-pairs 89% (0.985), M-var 84% (0.928).
**Chosen: F-pairs-H.** Check passed: the Stage 6 code reproduces Stage 4's F-pairs, ours, bva, random and accuracy statistics exactly
(174 models, max difference 0).

## Outcome (2026-11-03)

Tables: `results/stage6/summary.md`. Statistics: `results/stage6/detect.csv`. The confirmation mutants have accuracy in the same range
as the exploration mutants (median RMSE Bike 98 vs 110, Diet 0.58 vs 0.57; median AUC Adult 0.888 vs 0.890).

**Pre-registered tests (confirmation set, 144 general mutants, Holm over 5)**

| claim | result | discordant | Holm p |
|---|---|---|---|
| 1. F-pairs > ours, > bva | **supported**: 82% vs 33% and 40% | 71/0, 60/0 | < 0.0001 |
| 2. F-pairs-H > F-pairs | **supported**: 90% vs 82% | 12/0 | 0.001 |
| 3. acc+F-pairs > accuracy | **not supported**: 80% vs 89%, accuracy better | 6/19 | 0.015 (wrong direction) |
| 4. W-margin > ours | **supported**: 66% vs 33% | 53/5 | < 0.0001 |

The confirmation set reproduces the exploration set closely:
- F-pairs 82% (81%), F-pairs-H 90% (90%), Z-pairs 87% (89%), M-var 76% (84%), W-margin 66% (67%);
- ours 33% (34%), bva 40% (40%), accuracy 89% (91%).

Of the 16 confirmation mutants that accuracy misses, F-pairs-H and Z-pairs each detect 14 (88%), F-pairs 8, W-margin 5,
ours 0.

Every single detector flags 3 of the 30 healthy models when cross-fitted (bva 2). With the rule "above every other
healthy model", this is about one per dataset by construction. The false-alarm rate per dataset is about 1/10 whatever
the statistic.

M-var as pre-registered (two-sided relative change) localizes the shuffled variable in only 2/18 FS mutants (4/18 on the
exploration set). Its two-sided relative change is dominated by variables with a small mean |SHAP|.

**Post hoc (`src/stage6_posthoc.py`)**
1. **One-sided relative drop** of mean |SHAP| (the Stage 5 definition) localizes the shuffled variable in 17/18
   confirmation FS mutants (18/18 exploration). For localization, the right statistic is the drop, not the absolute change.
2. **Union with accuracy** (each detector with its own rule):
   - accuracy OR F-pairs-H: 142/144 (98.6%) vs accuracy alone 128/144;
   - accuracy OR Z-pairs: 142/144 (98.6%);
   - healthy models flagged: 6/30 and 4/30, vs 3/30 for accuracy alone.

   The pre-registered z-max combination failed because the healthy bar of a maximum of two z-scores is high. Combining
   by union works, at the cost of up to twice the false alarms.
3. **Background sweep labels** (`results/stage6/background.csv`):
   - Every continuous feature that stays narrow under both backgrounds is a reference crossing: 8 features, ratio
     0.95–1.01. No learned step was found.
   - The 14 Diet and 1 Bike "not stable" binary features are an artefact of the sweep. When the low background is all 0,
     inputs at 0 get a SHAP value of exactly 0, so one partition disappears.
   - "No contrast": the feature's median is its maximum, so there is no high background.
   - Adult's 3 continuous features and Bike's hum_t-1 and cnt_t-1 lose their narrow margin under the low background.
   - Diet with all 95 training samples as background: the boundary of calories (3,100–3,200 kcal) contains the background
     mean (3,161 kcal). The boundary of Weight_Kg (75.98–76.20 kg) contains its background mean (76.19 kg).
