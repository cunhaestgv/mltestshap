# Stage 2 results

Design: ../../STAGE2.md. Target feature x1; 5 models per replicate.

| scenario | expected | correct | diagnoses | margin before | margin after | local err before | local err after | err all |
|---|---|---|---|---|---|---|---|---|
| sharp | 1 sharp boundary | 5/5 | sharp boundary ×5 | 0.00 | – | 0.20 | – | 0.02 |
| noeffect | 2 no real effect | 3/5 | no real effect ×3, unstable training ×2 | 1.00 | – | 0.01 | 0.01 | 0.01 |
| inter | 3 simple interaction | 1/5 | complex interaction ×4, simple interaction ×1 | 1.00 | – | 0.02 | – | 0.02 |
| complex | 4 complex interaction | 5/5 | complex interaction ×5 | 1.00 | – | 0.13 | – | 0.13 |
| nonmono | 9 non-monotonic effect | 0/5 | simple interaction ×5 | 0.40 | – | 0.04 | – | 0.04 |
| copy | 8 explainer artifact | 0/5 | sharp boundary ×5 | 0.00 | 0.00 | 0.16 | – | 0.02 |
| missing | 5 missing feature or label noise | 0/5 | complex interaction ×5 | 1.00 | 1.00 | 0.08 | 0.02 | 0.08 |
| noise | 5 missing feature or label noise | 0/5 | complex interaction ×4, simple interaction ×1 | 1.00 | 1.00 | 0.05 | 0.02 | 0.05 |
| sparse | 6 too little data | 0/5 | sharp boundary ×5 | 0.00 | 0.00 | 0.21 | 0.23 | 0.04 |

**copy, after grouping x1 and x8:** sharp boundary ×5

Margin = share of generation samples in the margin of x1 (0 when it is not wide). Local error = mean |f(x) − ŷ| of the reference model on test samples in the margin region (noise-free targets); after = model trained on the fixed data, same region. Means over replicates.

Per-replicate scores (agreement, self, shallow, deep, err_ratio, resid_corr, density): results.csv and *.json.
