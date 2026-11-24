# Stage 3 results

Design: ../../STAGE3.md. 116 cases.

## Synthetic cases

| rule | mean regret | hit rate | evaluable |
|---|---|---|---|
| L1 | 0.227 | 25% | 110/110 |
| L2 | 0.219 | 26% | 110/110 |
| L3 | 0.283 | 19% | 110/110 |
| always audit | 0.244 | | 110/110 |
| always ensemble | 0.261 | | 110/110 |
| always more_data | 0.279 | | 110/110 |
| always drop_top | 0.342 | | 110/110 |
| always none | 0.385 | | 110/110 |
| always capacity | 0.390 | | 110/110 |
| always regularize | 0.607 | | 110/110 |
| random candidate | 0.356 | | 110/110 |

Paired differences in regret (95% bootstrap interval; negative = first is better):

- L1 vs always audit: -0.017 [-0.057, 0.022]
- L2 vs L1: -0.008 [-0.028, 0.005]
- L3 vs L2: 0.064 [0.024, 0.107]

## By scenario

| scenario | task | best fix (cases) | best gain | L1 recs | L2 recs | L3 recs | regret L1/L2/L3 |
|---|---|---|---|---|---|---|---|
| complex | cla | capacity×4, ensemble×1 | 24% | none×3, audit×2 | none×3, audit×2 | capacity×2, none×2, ensemble×1 | 0.24 / 0.24 / 0.13 |
| complex | reg | capacity×5 | 19% | audit×4, robust×1 | audit×4, robust×1 | capacity×4, robust×1 | 0.18 / 0.18 / 0.02 |
| imbalance | cla | threshold×2, class_weight×2, regularize×1 | 38% | threshold×5 | threshold×5 | threshold×5 | 0.12 / 0.12 / 0.12 |
| inter | cla | ensemble×2, more_data×2, capacity×1 | 24% | capacity×3, audit×1, none×1 | capacity×2, drop_top×1, audit×1, none×1 | capacity×3, drop_top×1, none×1 | 0.23 / 0.25 / 0.25 |
| inter | reg | ensemble×4, more_data×1 | 18% | robust×5 | robust×5 | robust×5 | 0.14 / 0.14 / 0.14 |
| missing | cla | audit×5 | 65% | audit×5 | audit×5 | capacity×5 | 0.00 / 0.00 / 0.66 |
| missing | reg | audit×5 | 74% | robust×5 | robust×5 | robust×5 | 0.53 / 0.53 / 0.53 |
| noeffect | cla | ensemble×4, drop_top×1 | 34% | audit×4, none×1 | audit×4, none×1 | capacity×4, none×1 | 0.34 / 0.34 / 0.65 |
| noeffect | reg | ensemble×5 | 37% | none×5 | none×5 | none×4, ensemble×1 | 0.37 / 0.37 / 0.29 |
| noise | cla | audit×5 | 41% | audit×4, capacity×1 | audit×4, capacity×1 | capacity×5 | 0.11 / 0.11 / 0.35 |
| noise | reg | audit×5 | 49% | robust×5 | robust×5 | robust×5 | 0.17 / 0.17 / 0.17 |
| nonmono | cla | regularize×2, more_data×2, ensemble×1 | 33% | capacity×3, audit×2 | capacity×3, audit×2 | capacity×5 | 0.32 / 0.32 / 0.36 |
| nonmono | reg | ensemble×2, robust×2, more_data×1 | 25% | robust×5 | robust×5 | robust×5 | 0.05 / 0.05 / 0.05 |
| outliers | reg | robust×4, audit×1 | 81% | robust×5 | robust×5 | robust×5 | 0.01 / 0.01 / 0.01 |
| sharp | cla | more_data×2, threshold×1, regularize×1, ensemble×1 | 27% | capacity×3, audit×2 | capacity×3, drop_top×1, audit×1 | capacity×4, drop_top×1 | 0.34 / 0.29 / 0.33 |
| sharp | reg | ensemble×5 | 23% | robust×5 | robust×5 | robust×5 | 0.16 / 0.16 / 0.16 |
| sparse | cla | more_data×4, capacity×1 | 52% | more_data×5 | more_data×4, drop_top×1 | capacity×3, audit×1, drop_top×1 | 0.06 / 0.10 / 0.50 |
| sparse | reg | ensemble×3, drop_top×1, more_data×1 | 23% | more_data×3, robust×2 | more_data×3, robust×2 | capacity×3, robust×2 | 0.23 / 0.23 / 0.29 |
| spurious | cla | capacity×3, more_data×2 | 23% | more_data×5 | more_data×5 | capacity×5 | 0.05 / 0.05 / 0.07 |
| spurious | reg | drop_top×5 | 92% | more_data×5 | more_data×4, drop_top×1 | capacity×4, drop_top×1 | 0.93 / 0.74 / 0.75 |
| unstable | cla | more_data×5 | 36% | capacity×5 | capacity×5 | capacity×5 | 0.35 / 0.35 / 0.35 |
| unstable | reg | ensemble×5 | 8% | capacity×5 | capacity×5 | capacity×5 | 0.05 / 0.05 / 0.05 |

## Culprit feature (spurious, x7 ranked first)

Loss Shapley 5/10, error-tree root 10/10, |SHAP| 10/10.

## Real data: adult (classification)

| rule | mean regret | hit rate | evaluable |
|---|---|---|---|
| L1 | nan | nan% | 0/3 |
| L2 | nan | nan% | 0/3 |
| L3 | 0.202 | 0% | 3/3 |
| always threshold | 0.000 | | 3/3 |
| always class_weight | 0.030 | | 3/3 |
| always capacity | 0.202 | | 3/3 |
| always ensemble | 0.211 | | 3/3 |
| always none | 0.216 | | 3/3 |
| always drop_top | 0.216 | | 3/3 |
| always regularize | 0.235 | | 3/3 |
| random candidate | 0.159 | | 3/3 |

Recommendations: L1 ['audit'], L2 ['audit'], L3 ['capacity']

Mean gain per candidate: none 0.0%, ensemble 0.5%, capacity 1.4%, regularize -1.9%, drop_top -0.0%, threshold 21.6%, class_weight 18.6%

## Real data: bike (regression)

| rule | mean regret | hit rate | evaluable |
|---|---|---|---|
| L1 | 0.145 | 0% | 3/3 |
| L2 | 0.145 | 0% | 3/3 |
| L3 | 0.145 | 0% | 3/3 |
| always capacity | 0.004 | | 3/3 |
| always ensemble | 0.066 | | 3/3 |
| always robust | 0.145 | | 3/3 |
| always none | 0.178 | | 3/3 |
| always drop_top | 0.241 | | 3/3 |
| always regularize | 0.523 | | 3/3 |
| random candidate | 0.193 | | 3/3 |

Recommendations: L1 ['robust'], L2 ['robust'], L3 ['robust']

Mean gain per candidate: none 0.0%, ensemble 11.2%, capacity 17.4%, regularize -34.5%, drop_top -6.3%, robust 3.3%

## Slicing

Mean error lift on test: error tree 2.18, SHAP clusters 2.47.
F1 against the true region (sparse, noise): error tree 0.37, SHAP clusters 0.25.
