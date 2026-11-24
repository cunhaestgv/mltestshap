# Stage 5: can SHAP disagreement and a tree locate a known fault?

Written on 2026-11-03, before any Stage 5 run. This is an exploratory feasibility check of idea 1 from the Stage 4
discussion. It is not part of the Stage 3 or Stage 4 pre-registrations. It uses only the SHAP values already
saved by Stage 4 (`results/stage4/<ds>/shap/*.npz`, generation set), with no retraining.

Two mutant types have a known answer:
- **FS:** the shuffled variable v. This is a column fault.
- **DS:** the training rows with v above its 75th percentile were removed. This is a region fault.

## Design
**Reference sign.** For each (input, feature) pair, the reference is the majority sign over the 10 healthy models
(+1 vs −1 counts; zero counts as neither; ties go to +1). h is the share of the healthy models that differ from the
majority. For a model under test, excess e = 1[its sign differs from the majority] − h.

**Null control.** Each healthy model is scored against the majority and h of the other 9 (cross-fitted).

**Views.** Adult, Bike and Diet, ungrouped. A flattened feature f belongs to variable f % n_vars. The time-t column of
v is the last column of its group, as in `run.to_view`.

**FS, which variable?** 18 mutants.
- **Method:** rank variables by mean excess over their features and all rows. Hit if the top variable is v. Also
  report the rank of v.
- **Baseline A:** the variable with the largest relative drop in mean |SHAP|, (healthy mean − mutant) / healthy mean,
  over its features.
- **Baseline B:** chance, 1 / number of variables.
- **Null:** for healthy models, the rate at which the top variable is one of the dataset's FS variables, per
  variable.

**DS, which region?** 6 mutants.
- **Method:**
  1. Row target = mean excess over the row's features.
  2. Fit `DecisionTreeRegressor(max_depth=2, min_samples_leaf=max(5, 2% of rows), random_state=0)` on the flattened
     inputs.
- **Root hit:** the root splits on variable v, any lag (time t reported separately).
- **True cut:** the threshold DS used. That is q75 of v at time t over the full training data. If no row is above
  q75, DS removed only rows at the maximum, and the cut is then the midpoint between the maximum and the next value.
- **Threshold hit:** |root threshold − true cut| ≤ 0.05. Inputs are scaled to [0,1].
- **Lift:** mean row target above the true cut minus the mean below it, on the generation set.

**Control.** Run the same tree on the healthy models (cross-fitted), the 4 LN mutants and the 36 weight mutants of
the same dataset. Report how often each group's root is on the dataset's DS variable.

**Exploratory.** Root feature, threshold and top FS-style variable for every model, in `localize.csv`.

**Judged promising if:**
- FS top-1 hits ≥ 12/18 and at least baseline A;
- DS root hits ≥ 4/6, with healthy root-on-v rate ≤ 30%;
- DS threshold hits ≥ 3/6.

No significance test, since the numbers are small. Results are reported as they come out.

Code: `src/stage5_localize.py`. Results: `results/stage5/`.

## Outcome (2026-11-03)

Tables: `results/stage5/summary.md`. Sanity: mean excess of healthy models (cross-fitted) 0.024, mutants 0.138.

**As pre-registered: no criterion met.**
- FS: sign excess top-1 11/18 (Adult 4/6, Bike 4/6, Diet 3/6; median rank 1; chance 0.08–0.12; healthy null 0.03–0.07).
  Baseline A, the |SHAP| drop, 18/18. A shuffled variable is found perfectly by magnitude; signs add nothing.
- DS: root on v 0/6, threshold hit 0/6. Healthy root-on-v rate 0.13.

**Post hoc check** (`scratchpad s5check.py`, not saved in the repo): mean row excess above vs below the true cut.
- Adult (v = married, binary): DS removed all married = 1 rows (47% of the data). This is a column fault, not a region
  fault: the sign of `married` differs from the healthy majority on every row (excess 1.0 above and below). Rows
  excess 0.62 vs 0.55.
- Bike (v = cnt): 0.206 vs 0.204. The DS mutants disagree everywhere about equally.
- Diet (v = calories): 0.017 vs 0.011. DS barely changes any sign.
- |SHAP| deviation from the healthy mean is 1.2–1.4× higher above the cut in DS mutants. Healthy and LN models give
  1.0–1.4 (Adult healthy 1.4), so the signal is weak and not clearly specific.

Consequence: SHAP signs do not show where a missing data region is. This fits the background finding: for smooth effects
the sign mostly says whether an input lies above or below the background mean, which removing the top quartile does not
change. Column faults are found, but by |SHAP| magnitude, not by signs or trees.
