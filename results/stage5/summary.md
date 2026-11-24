# Stage 5: fault localization from SHAP disagreement

Design: `STAGE5.md`. Per-model rows: `localize.csv`.

Sanity: mean excess, healthy (cross-fitted) 0.0237; mutants 0.1379.

## FS: which variable

| view | n | sign-excess top-1 | |SHAP|-drop top-1 | chance | median rank (excess) | healthy null top=v |
|---|---|---|---|---|---|---|
| adult | 6 | 4 | 6 | 0.09 | 1 | 0.03 |
| bike | 6 | 4 | 6 | 0.08 | 1 | 0.03 |
| diet | 6 | 3 | 6 | 0.12 | 2 | 0.07 |
| all | 18 | 11 | 18 | | 1 | |

## DS: which region

| view | model | root feature | threshold | true cut | root on v | threshold hit | lift |
|---|---|---|---|---|---|---|---|
| adult | DS_6_0 | age | 0.116 | 0.500 (skewed) | False | False | +0.068 |
| adult | DS_6_1 | age | 0.226 | 0.500 (skewed) | False | False | +0.037 |
| bike | DS_12_0 | hr_t-1 | 0.717 | 0.244 | False | False | +0.002 |
| bike | DS_12_1 | hr_t-2 | 0.761 | 0.244 | False | False | +0.010 |
| diet | DS_1_0 | walk_t-2 | 0.500 | 0.407 | False | False | +0.006 |
| diet | DS_1_1 | weight_t | 0.500 | 0.407 | False | False | +0.002 |

Root on v: 0/6; threshold hit: 0/6.

Control, share of models whose root splits on the DS variable, and mean lift:

| view | healthy | LN | weight | FS | DS |
|---|---|---|---|---|---|
| adult | 0.00 (-0.006) | 0.25 (-0.020) | 0.25 (+0.032) | 0.00 (-0.038) | 0.00 (+0.052) |
| bike | 0.10 (-0.013) | 0.00 (+0.009) | 0.22 (+0.002) | 0.17 (+0.013) | 0.00 (+0.006) |
| diet | 0.30 (+0.002) | 0.00 (+0.000) | 0.06 (-0.001) | 0.50 (+0.018) | 0.00 (+0.004) |

## Criteria

- FS top-1 >= 12/18 and >= |SHAP|-drop: not met
- DS root on v >= 4/6 and healthy rate <= 0.30: not met
- DS threshold hit >= 3/6: not met
- healthy root-on-v rate, pooled: 0.13
