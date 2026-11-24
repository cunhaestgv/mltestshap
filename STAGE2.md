# Stage 2: controlled validation of the margin diagnosis

Written on 2026-10-28, before any Stage 2 run. The scenarios, expected rows,
fixes and measures below are fixed; if a scenario turns out not to produce
the intended effect, that is reported as a result, not fixed by changing the
scenario. Thresholds are `diagnose.THRESHOLDS`, unchanged.

## Setup

- Inputs x1..x7 ~ U(0, 1), independent; n = 4000 per replicate (70/15/15 random
  train/validation/test split). The generation set for SHAP and the diagnosis
  is the training set (up to 3000 samples), background = 100 training samples.
- Target: y = 0.3 x3 − 0.3 x4 + 0.2 x5 + g(x) + ε, ε ~ N(0, 0.05²).
  x3, x4, x5 have monotonic main effects; x6, x7 have none unless g uses them.
- Model: MLP 32-16-1 (ReLU, linear output, MSE, Adam), early stopping on
  validation loss (patience 10, max 200 epochs, batch 64).
- Diagnosis: reference model (seed 0) + 4 retrains (seeds 1–4) on the same data.
- 5 replicates (data seeds 0–4) per scenario. Target feature: x1.

## Scenarios

| id | cause injected | g(x) and data | expected row | fix |
|---|---|---|---|---|
| sharp | (control) sharp boundary | 0.5·[x1 > 0.5] | 1 sharp boundary | none |
| noeffect | no real effect | 0 | 2 no real effect | drop x1 |
| inter | simple interaction | 0.8 (x1 − 0.5) σ(x2) | 3 simple interaction | none (tree partitions) |
| complex | complex interaction | 0.8 (x1 − 0.5) σ(x2) σ(x6) σ(x7) | 4 complex interaction | none (tree partitions) |
| nonmono | non-monotonic effect | 0.5·[0.3 < x1 < 0.7] | 9 non-monotonic effect | none |
| copy | correlated copy | 0.5·[x1 > 0.5]; extra input x8 = x1 + N(0, 0.02²) | 8 explainer artifact | group x1 and x8 (sum SHAP) |
| missing | missing feature | 0.8 (x1 − 0.5) σ(z), z ~ U(0,1) not an input; input x2 replaced by w = z + N(0, 0.15²) | 5 missing feature or label noise | give the model z instead of w |
| noise | label noise | as `inter`; for 40% of the training samples with 0.25 < x1 < 0.75 the target is permuted among them | 5 missing feature or label noise | train on the clean targets |
| sparse | too little data | 0.5·[x1 > 0.5]; training samples with 0.25 < x1 < 0.75 kept with probability 0.2 | 6 too little data | train on all samples |

σ(u) = +1 if u > 0.5 else −1. In `noise` and `sparse` only the training split
is altered; validation and test are clean and complete.

## Measures

Per scenario and replicate:

1. **Diagnosis** of x1 (row), and whether it equals the expected row.
   Reported as accuracy per scenario (out of 5) and as a confusion table.
2. **Margin share** of x1 (share of generation samples in its margin) before
   and after the fix. For `copy` the fix is grouping, so the "after" margin is
   that of the grouped feature x1+x8, from the same models.
3. **Local error**: mean |y − ŷ| of the reference model on test samples whose
   x1 lies in the reference model's margin, before and after the fix (same
   region), relative to the mean |error| on the whole test set before the fix.
   For `copy`, the model does not change, so only the margin is reported.

The noise-free targets of the test set are used for the local error, so that
label noise and ε do not mask the effect of a fix.

## What counts as support for the diagnosis table

The table is supported for a cause when the expected row is given in at least
4 of 5 replicates, and the fix narrows the margin or lowers the local error in
at least 4 of 5. Other outcomes are reported as they are.

## Outcome (2026-10-29)

- As pre-specified: 14/45 correct (`results/stage2/prereg/`). After one post hoc correction to
  row 9 (sign changes counted over the whole feature range, not only inside the margin, because for
  a bump the margin is the pure middle): 19/45 (`results/stage2/summary.md`). Both reported in the paper.
- Supported (>= 4/5): sharp, complex, nonmono (after the correction). Not supported: noeffect 3/5,
  inter 1/5 (XOR-shaped sign; greedy depth-2 tree fails), copy and sparse 0/5 (no margin produced),
  missing and noise 0/5 (margin = whole range, so err_ratio = 1 and row 5 is unreachable;
  resid_corr is 0.85–0.97 in every scenario). The fixes lowered the local error in 5/5 for missing and
  noise (= overall error, as the margin spans the whole range) and never narrowed a margin.
