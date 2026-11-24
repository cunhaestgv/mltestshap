# Stage 4: group-level fault detectors and a culprit screen

Written on 2026-11-01, before any Stage 4 run. Everything below is fixed. A smoke run only checks that the code
runs and how long it takes. Bad outcomes are results; any later change is reported as post hoc.

Ideas tested, in priority order: (1) boundary drift, (2) region-level checks, (6) culprit screen.
Motivation (`stage3` and the stability analysis): per-sample signs near zero are noisy, while healthy boundaries
of continuous features moved by only 0–2% of the range. The question is whether comparing whole partitions
separates faults from retraining noise better than the per-sample tests.

## Part A: detection

**Models.** Adult, Bike and Diet: reference model, the 10 healthy retrains and 48 mutants of `run.py`, θ = 0.1.
Weight mutants are rebuilt from their seeds; data mutants are the saved ones. Bike is also analysed with grouped
lags (`bike_grouped`) as a separate view, reusing the same SHAP values.

**New boundary-shift mutants (BS).** None of the 48 targets a boundary. For each dataset, take the narrow feature of
the reference model with more than 10 distinct values and the highest mean |SHAP|, and its variable v. Add δ to v
(scaled units, so δ is a share of its range; clipped to [0,1]) in the training and validation inputs and retrain;
all time steps for sequences. δ ∈ {0.05, 0.1, 0.2}, 2 seeds (100, 101), 6 mutants per dataset. This models a
calibration or unit fault in the training pipeline, and moves the boundary of v by about δ on clean inputs.

**SHAP.** DeepSHAP on the generation set, for every model, as in `run.py`. Detectors use the SHAP oracle only.

**D1, boundary drift.** On the features that are narrow in the reference model, recompute Algorithm 1 on the
model under test. d_f = |c − c_ref| / range, with c the midpoint of lo and hi; d_f = 1 if the feature is no longer
narrow or its direction reversed. Statistic: mean of d_f. Not applicable with no narrow features (`bike_grouped`).

**D2, region-level checks.** Regions come from the reference model: partitions of single and narrow features, and
the pure leaves (entropy < θ) of the trees of wide features, with the settings of `method.tree_partitions`. Violation
of a region: share of its samples whose sign differs from the region's sign. A region is kept if its mean violation over
healthy retrains is ≤ 0.1. Statistic: mean violation over kept regions. **D2-all** is the same without the filter.
Cross-fitting: for a healthy model, kept regions are chosen from the other 9 healthy models, so the filter cannot
lower its own statistic. Mutants use all 10.

**Detection** is as in `run.py`: statistic above that of every healthy model. Comparators, same rule: our suite, BVA
and random (mean of 5 repetitions) with the SHAP oracle, and accuracy (RMSE or AUC worse than every healthy model).
Recomputed from the SHAP values; the results must reproduce `results/summary.md`.

**Measures.** Detection rate on the 48 general mutants, and by weight and data mutants, because weight mutants are small
edits of the reference while healthy models are full retrains, which biases any comparison with the reference.
Detection on BS mutants. Detection of mutants accuracy misses. AUROC of each statistic, mutants vs healthy.
Localization for BS: does the feature whose d_f most exceeds its healthy maximum belong to v?

**Support.** Primary view pools Adult, Bike and Diet (144 general mutants, 18 BS). D1 and D2 are each compared with our
suite and BVA, on general and BS mutants: 8 exact McNemar tests, α = 0.05 / 8. A detector is supported on a set if
it detects more than both comparators, significantly. `bike_grouped` and accuracy are reported, not tested.

## Part B: culprit screen

**Cases.** The 110 synthetic Stage 3 cases and 6 real cases, plus harder spurious cases: noise of the copy of the target
× 2 and × 4, both tasks, 5 replicates each (reps 5–9 and 10–14), 20 cases. So 30 spurious cases, 100 others, 6 real.

**Flags.** From discovery data: root feature of the error tree (R) and feature with top mean |SHAP| (H).
- **Screen:** flag if R = H, confirm by retraining.
- **R-only** and **H-only:** flag that feature, confirm by retraining. They show whether SHAP adds to the error tree.
- **No confirm:** the screen's flag, dropped without confirming.
- **Stage 3 L2 rule:** drop the top loss Shapley feature when top_share ≥ 0.5 and top_rel ≥ 0.1, no confirmation.

Confirm: replace the feature by its training mean, retrain 3 seeds, and accept if the mean error on discovery falls by at
least 5% against the base models. Test error as in Stage 3.

**Measures.** Recall: spurious cases where x7 is dropped. Harm: other cases where a drop is accepted and test error rises by more
than 1%. Mean test gain. Per task.

**Support.** Screen recall ≥ 80% (24/30) and harm ≤ 5% (5/100), and recall at least that of the Stage 3 rule. SHAP
adds value if the screen has lower harm than R-only without lower recall; both are reported. Real cases are a check.

## Outcome (2026-11-01)

Tables: `results/stage4/summary.md`. Recomputed ours, BVA, random and accuracy reproduce `results/summary.md`.

**As pre-registered**
- D1, boundary drift: not supported. General 40% vs ours 34% and BVA 40%; BS 22% vs 22% and 17%. Very uneven: Adult 83%, Bike 10%, Diet 27%.
- D2, region checks: supported on general mutants (68% vs 34% and 40%, 50/1 and 46/5, p < 0.0001); not on BS (28% vs 22% and 17%).
- Accuracy stays ahead: 91% general. On the 13 general mutants accuracy misses, D2 finds 15%.
- Part B: screen not supported. Recall 11/30 (needed 24), harm 0/100. By copy noise: ×1 10/10, ×2 1/10, ×4 0/10.
  SHAP added nothing: root-only recall 13/30, also 0 harm. Confirmation is what prevents damage: without it, 56/100 harmed.
  Stage 3 rule 1/30. Real data: every flagged feature was correctly rejected (test error would rise 20–60%).

**Post hoc, not pre-registered**
1. *Scoring error found and fixed:* random was first scored on the statistic averaged over repetitions, not per repetition as in
   `summarize.py`, which inflated it (Adult 52% instead of 32%). Fixed before any report.
2. *The filter, not the regions, gives D2's gain.* The healthy filter without regions, general mutants: F-ours 73%, F-bva 76%,
   F-random 73%, all (input, feature) pairs 81% (vs D2 2/21, p = 0.0001). Filtered tests on the 13 mutants accuracy misses: 23–46%.
3. *BS mutants failed their manipulation check:* the target boundary moved by ≤ 0.014 for δ up to 0.2, so the BS results above do
   not test boundary drift. Localization 2/18 (ties with no excess counted as misses).
4. *Boundaries of continuous features follow the SHAP background* (`src/stage4_background.py`). Same model, same samples, only
   the background changed; 13 narrow continuous features. Boundary vs background mean: r = 0.99, mean gap 0.017. When both
   backgrounds kept a boundary (8 features), it moved 0.310 for a background move of 0.317 (median ratio 0.98). The other 5
   were no longer narrow under at least one background.
5. *A real step is partly different* (`src/stage4_threshold.py`, 5 replicates, step at 0.7): boundary 0.69 and 0.71 under the
   full and high backgrounds, but 0.33 under the low one. A smooth line follows the background exactly (0.20, 0.50, 0.80).

Consequence: for smooth effects, a sign boundary is where the input crosses the SHAP reference, not a threshold the model learned. This
also explains why boundaries looked stable across seeds. Binary features and real steps are less affected.
