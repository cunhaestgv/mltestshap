# Stage 7 results

Design: `../../STAGE7.md`. Fingerprints: `features_<dataset>.csv`; predictions: `predictions.csv`.

Check (Stage 6 statistics reproduced, 318 models): max |difference| F-pairs-H 0, Z-pairs 0, W-margin 0.

## Pre-registered tests (confirmation set, 144 mutants; balanced accuracy; Holm over 4)

| test | difference | 95% CI | p | Holm p | supported |
|---|---|---|---|---|---|
| S1 > S0 | -0.005 | [-0.130, +0.120] | 0.5341 | 0.5624 | no |
| S2 > S1 | +0.037 | [-0.086, +0.162] | 0.2812 | 0.5624 | no |
| S3 > S1 | +0.132 | [+0.046, +0.241] | 0.0000 | 0.0000 | yes |
| rules > 0.25 | +0.074 | [+0.021, +0.132] | 0.0020 | 0.0060 | yes |

## Balanced accuracy and recall per class (confirmation set)

| classifier | balanced acc. | weights | LN | FS | DS | adult | bike | diet |
|---|---|---|---|---|---|---|---|---|
| S0 | 0.590 | 0.56 | 0.58 | 0.89 | 0.33 | 0.89 | 0.62 | 0.26 |
| S1 | 0.586 | 0.62 | 0.50 | 0.89 | 0.33 | 0.76 | 0.69 | 0.31 |
| S2 | 0.623 | 0.63 | 0.58 | 0.94 | 0.33 | 0.71 | 0.73 | 0.43 |
| S3 | 0.718 | 0.73 | 0.75 | 0.89 | 0.50 | 0.75 | 0.99 | 0.41 |
| S4 | 0.734 | 0.82 | 0.83 | 0.94 | 0.33 | 0.72 | 0.85 | 0.63 |
| S5 | 0.944 | 0.97 | 0.92 | 0.89 | 1.00 | 1.00 | 0.99 | 0.84 |
| rules | 0.324 | 0.96 | 0.00 | 0.33 | 0.00 | 0.48 | 0.25 | 0.24 |
| RF-S4 | 0.618 | 1.00 | 0.42 | 0.89 | 0.17 | 0.69 | 0.75 | 0.42 |

## Confusion matrices (rows = true, columns = predicted; order weights, LN, FS, DS)

**S0**
```
[[60 29  2 17]
 [ 4  7  1  0]
 [ 1  1 16  0]
 [ 3  0  1  2]]
```

**S1**
```
[[67 29  5  7]
 [ 4  6  0  2]
 [ 1  0 16  1]
 [ 1  3  0  2]]
```

**S2**
```
[[68 28  1 11]
 [ 5  7  0  0]
 [ 1  0 17  0]
 [ 2  1  1  2]]
```

**S3**
```
[[79 16  5  8]
 [ 3  9  0  0]
 [ 1  0 16  1]
 [ 2  1  0  3]]
```

**S4**
```
[[89 11  1  7]
 [ 2 10  0  0]
 [ 0  1 17  0]
 [ 1  2  1  2]]
```

**S5**
```
[[105   3   0   0]
 [  1  11   0   0]
 [  2   0  16   0]
 [  0   0   0   6]]
```

**rules**
```
[[104   2   1   1]
 [ 12   0   0   0]
 [ 12   0   6   0]
 [  5   0   1   0]]
```

## Exploratory

| analysis | S0 | S1 | S2 | S3 | S4 | S5 |
|---|---|---|---|---|---|---|
| exploration, 5-fold CV | 0.593 | 0.549 | 0.650 | 0.669 | 0.657 | 0.822 |
| leave out adult | 0.389 | 0.688 | 0.111 | 0.208 | 0.174 | 0.688 |
| leave out bike | 0.479 | 0.451 | 0.569 | 0.410 | 0.653 | 0.944 |
| leave out diet | 0.319 | 0.201 | 0.167 | 0.257 | 0.222 | 0.562 |
| flagged by accuracy OR F-pairs-H (142) | 0.632 | 0.627 | 0.602 | 0.780 | 0.713 | 0.944 |

**Localization (confirmation):** FS top-drop variable = shuffled variable 17/18. DS: SHAP region variable = removed variable 5/6 (top tail 5); output region variable 5/6 (top tail 5).

**Weight operators (GF/WS/NEB/NAI), S4, confirmation:** balanced accuracy 0.611 (chance 0.25).

## Median fingerprint per class (all sets; healthy cross-fitted)

| cls | acc_z | drop_max | scale | F-pairs-H | Z-pairs | var_conc | ratio_sd | region_conc | W-margin | margin_shift | margin_conc | tree_res | tree_leaves | slope | r2 | resid_var | resid_region | own_fit_z | label_dis | coverage | relation |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| healthy | -0.11 | 0.29 | -0.01 | 0.01 | 0.85 | 0.21 | 0.21 | 1.28 | 0.16 | -0.00 | 0.20 | 0.01 | -0.01 | 1.01 | 0.99 | 0.35 | 1.71 | -0.35 | 0.00 | 0.00 | 0.00 |
| weights | 12.89 | 0.36 | -0.01 | 0.13 | 2.12 | 0.25 | 0.40 | 1.31 | 0.24 | -0.08 | 0.17 | 0.07 | -0.09 | 0.82 | 0.90 | 0.48 | 1.90 | 25.93 | 0.00 | 0.00 | 0.00 |
| LN | 5.64 | 0.45 | -0.12 | 0.10 | 1.64 | 0.22 | 0.42 | 1.32 | 0.23 | -0.07 | 0.16 | -0.02 | -0.14 | 0.72 | 0.92 | 0.26 | 1.79 | 48.44 | 0.05 | 0.00 | 0.11 |
| FS | 8.35 | 0.93 | -0.01 | 0.16 | 2.44 | 0.35 | 0.87 | 1.39 | 0.23 | 0.02 | 0.14 | -0.00 | -0.05 | 0.89 | 0.90 | 0.32 | 1.74 | 5.96 | 0.06 | 0.00 | 0.39 |
| DS | 14.72 | 0.63 | -0.01 | 0.21 | 1.97 | 0.27 | 0.58 | 1.33 | 0.30 | -0.15 | 0.19 | 0.01 | -0.51 | 0.86 | 0.87 | 0.46 | 2.63 | -0.68 | 0.00 | 0.25 | 0.23 |
