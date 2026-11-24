# Stage 6 results

Design: `../../STAGE6.md`. Statistics: `detect.csv`.

## Exploration set (144 general mutants), exploratory

| method | pooled | adult | bike | diet | weight | data | missed by accuracy | AUROC |
|---|---|---|---|---|---|---|---|---|
| F-pairs | 81% | 83% | 92% | 69% | 79% | 89% | 46% (13) | 0.94 |
| F-pairs-H | 90% | 94% | 96% | 81% | 90% | 92% | 62% (13) | 0.98 |
| Z-pairs | 89% | 100% | 81% | 85% | 90% | 86% | 92% (13) | 0.98 |
| M-var | 84% | 92% | 79% | 81% | 84% | 83% | 77% (13) | 0.93 |
| W-margin | 67% | 67% | 79% | 56% | 67% | 69% | 31% (13) | 0.91 |
| ours | 34% | 44% | 29% | 29% | 31% | 42% | 0% (13) | 0.56 |
| bva | 40% | 50% | 29% | 40% | 36% | 50% | 23% (13) | 0.60 |
| random | 34% | 32% | 28% | 42% | 32% | 41% | 28% (13) | 0.57 |
| accuracy | 91% | 92% | 98% | 83% | 94% | 83% | 0% (13) | 0.97 |
| acc+F-pairs | 81% | 62% | 98% | 83% | 79% | 89% | 46% (13) | 0.97 |

M-var localization on FS mutants: 4/18.

## Confirmation set (144 general mutants)

| method | pooled | adult | bike | diet | weight | data | missed by accuracy | AUROC |
|---|---|---|---|---|---|---|---|---|
| F-pairs | 82% | 88% | 88% | 71% | 78% | 94% | 50% (16) | 0.92 |
| F-pairs-H | 90% | 96% | 100% | 75% | 89% | 94% | 88% (16) | 0.97 |
| Z-pairs | 87% | 100% | 75% | 85% | 86% | 89% | 88% (16) | 0.99 |
| M-var | 76% | 85% | 73% | 71% | 71% | 92% | 75% (16) | 0.91 |
| W-margin | 66% | 77% | 73% | 48% | 62% | 78% | 31% (16) | 0.90 |
| ours | 33% | 42% | 29% | 27% | 30% | 42% | 0% (16) | 0.58 |
| bva | 40% | 48% | 27% | 46% | 38% | 47% | 19% (16) | 0.59 |
| random | 34% | 35% | 24% | 43% | 31% | 43% | 21% (16) | 0.59 |
| accuracy | 89% | 90% | 96% | 81% | 91% | 83% | 0% (16) | 0.96 |
| acc+F-pairs | 80% | 65% | 92% | 83% | 77% | 89% | 38% (16) | 0.96 |

M-var localization on FS mutants: 2/18.

Healthy models flagged (cross-fitted, 30): F-pairs 3, F-pairs-H 3, Z-pairs 3, M-var 3, W-margin 3, ours 3, bva 2, accuracy 3, acc+F-pairs 3

## Pre-registered tests (confirmation set, Holm over 5)

| test | A only | B only | p | Holm p | supported |
|---|---|---|---|---|---|
| F-pairs > ours | 71 | 0 | 0.0000 | 0.0000 | yes |
| F-pairs > bva | 60 | 0 | 0.0000 | 0.0000 | yes |
| F-pairs-H > F-pairs | 12 | 0 | 0.0005 | 0.0010 | yes |
| acc+F-pairs > accuracy | 6 | 19 | 0.0146 | 0.0146 | no |
| W-margin > ours | 53 | 5 | 0.0000 | 0.0000 | yes |

Claim 1 (F-pairs > ours and bva): supported.
Claim 2 (F-pairs-H > F-pairs): supported.
Claim 3 (acc+F-pairs > accuracy): not supported; healthy flagged 3/30.
Claim 4 (W-margin > ours): supported.

## Accuracy of mutants (RMSE, or -AUC for Adult)

```
                       min   median      max
view  set                                   
adult confirmation  -0.901   -0.888   -0.160
      exploration   -0.901   -0.890   -0.292
bike  confirmation  54.287   98.091  691.772
      exploration   54.101  109.535  773.306
diet  confirmation   0.442    0.580    1.855
      exploration    0.437    0.573    1.837
```

## Background sweep

```
label    no contrast  not stable  reference crossing
dataset                                             
adult              0           4                   0
bike               2           3                   2
diet               2          14                   6
```

