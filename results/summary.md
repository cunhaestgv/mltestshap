# Fault-detection results

## adult

**Model quality (test set).** Original: accuracy 0.8426, auc 0.902. Healthy retrains (n=10): accuracy 0.845 ± 0.001, auc 0.9009 ± 0.00063.

**Suite size per theta:** θ=0.05: 69 tests, θ=0.1: 72 tests, θ=0.2: 87 tests.

**False-alarm level** (mean failure rate on healthy retrains, θ=0.1):

| suite/oracle | mean | max |
|---|---|---|
| bva/output | 0.181 | 0.222 |
| bva/shap | 0.128 | 0.153 |
| ours/output | 0.158 | 0.222 |
| ours/shap | 0.083 | 0.167 |
| random/output | 0.214 | 0.319 |
| random/shap | 0.119 | 0.222 |

**Mutants detected (θ=0.1).** Random is averaged over its repetitions.

| op | bva/output | bva/shap | ours/output | ours/shap | random/output | random/shap | accuracy/- | n/mutants |
|---|---|---|---|---|---|---|---|---|
| GF | 33% | 44% | 33% | 22% | 24% | 22% | 100% | 9 |
| WS | 89% | 100% | 89% | 89% | 67% | 91% | 100% | 9 |
| NEB | 0% | 0% | 0% | 0% | 0% | 0% | 56% | 9 |
| NAI | 11% | 44% | 22% | 44% | 4% | 18% | 100% | 9 |
| LN | 100% | 100% | 75% | 75% | 50% | 25% | 100% | 4 |
| FS | 50% | 17% | 67% | 33% | 3% | 13% | 100% | 6 |
| DS | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 2 |
| All | 44% | 50% | 46% | 44% | 27% | 32% | 92% | 48 |

**Faults that accuracy misses** (4 mutants whose test metric stays within the healthy range), detected by:

| suite | output | shap |
|---|---|---|
| bva | 0% | 0% |
| ours | 0% | 0% |
| random | 0% | 0% |

**Sensitivity to θ** (all mutants detected):

| theta | bva/output | bva/shap | ours/output | ours/shap | random/output | random/shap |
|---|---|---|---|---|---|---|
| 0.05 | 38% | 50% | 46% | 46% | 30% | 35% |
| 0.1 | 44% | 50% | 46% | 44% | 27% | 32% |
| 0.2 | 31% | 42% | 33% | 44% | 38% | 40% |

## bike

**Model quality (test set).** Original: rmse 54.1, mae 37.35. Healthy retrains (n=10): rmse 53.38 ± 3.5, mae 37.45 ± 3.

**Suite size per theta:** θ=0.05: 378 tests, θ=0.1: 390 tests, θ=0.2: 417 tests.

**False-alarm level** (mean failure rate on healthy retrains, θ=0.1):

| suite/oracle | mean | max |
|---|---|---|
| bva/output | 0.357 | 0.459 |
| bva/shap | 0.296 | 0.392 |
| ours/output | 0.449 | 0.487 |
| ours/shap | 0.366 | 0.431 |
| random/output | 0.38 | 0.474 |
| random/shap | 0.301 | 0.421 |

**Mutants detected (θ=0.1).** Random is averaged over its repetitions.

| op | bva/output | bva/shap | ours/output | ours/shap | random/output | random/shap | accuracy/- | n/mutants |
|---|---|---|---|---|---|---|---|---|
| GF | 0% | 11% | 11% | 11% | 7% | 9% | 100% | 9 |
| WS | 78% | 89% | 78% | 78% | 84% | 89% | 100% | 9 |
| NEB | 0% | 0% | 0% | 0% | 0% | 0% | 89% | 9 |
| NAI | 0% | 22% | 11% | 22% | 18% | 22% | 100% | 9 |
| LN | 50% | 25% | 25% | 25% | 10% | 10% | 100% | 4 |
| FS | 0% | 17% | 17% | 17% | 3% | 10% | 100% | 6 |
| DS | 0% | 50% | 0% | 100% | 20% | 90% | 100% | 2 |
| All | 19% | 29% | 23% | 29% | 22% | 28% | 98% | 48 |

**Faults that accuracy misses** (1 mutants whose test metric stays within the healthy range), detected by:

| suite | output | shap |
|---|---|---|
| bva | 0% | 0% |
| ours | 0% | 0% |
| random | 0% | 0% |

**Sensitivity to θ** (all mutants detected):

| theta | bva/output | bva/shap | ours/output | ours/shap | random/output | random/shap |
|---|---|---|---|---|---|---|
| 0.05 | 15% | 29% | 12% | 27% | 21% | 32% |
| 0.1 | 19% | 29% | 23% | 29% | 22% | 28% |
| 0.2 | 17% | 29% | 29% | 42% | 21% | 29% |

## bike_grouped

**Model quality (test set).** Original: rmse 54.1, mae 37.35. Healthy retrains (n=10): rmse 53.38 ± 3.5, mae 37.45 ± 3.

**Suite size per theta:** θ=0.05: 120 tests, θ=0.1: 135 tests, θ=0.2: 162 tests.

**False-alarm level** (mean failure rate on healthy retrains, θ=0.1):

| suite/oracle | mean | max |
|---|---|---|
| bva/output | 0.296 | 0.341 |
| bva/shap | 0.151 | 0.178 |
| ours/output | 0.258 | 0.356 |
| ours/shap | 0.113 | 0.141 |
| random/output | 0.304 | 0.393 |
| random/shap | 0.16 | 0.244 |

**Mutants detected (θ=0.1).** Random is averaged over its repetitions.

| op | bva/output | bva/shap | ours/output | ours/shap | random/output | random/shap | accuracy/- | n/mutants |
|---|---|---|---|---|---|---|---|---|
| GF | 44% | 67% | 22% | 56% | 47% | 56% | 100% | 9 |
| WS | 78% | 100% | 89% | 100% | 91% | 100% | 100% | 9 |
| NEB | 56% | 78% | 33% | 78% | 31% | 56% | 89% | 9 |
| NAI | 67% | 89% | 44% | 67% | 64% | 73% | 100% | 9 |
| LN | 25% | 50% | 0% | 50% | 25% | 55% | 100% | 4 |
| FS | 33% | 67% | 67% | 83% | 40% | 60% | 100% | 6 |
| DS | 0% | 100% | 0% | 100% | 20% | 90% | 100% | 2 |
| All | 52% | 79% | 44% | 75% | 52% | 69% | 98% | 48 |

**Faults that accuracy misses** (1 mutants whose test metric stays within the healthy range), detected by:

| suite | output | shap |
|---|---|---|
| bva | 0% | 0% |
| ours | 0% | 0% |
| random | 0% | 0% |

**Sensitivity to θ** (all mutants detected):

| theta | bva/output | bva/shap | ours/output | ours/shap | random/output | random/shap |
|---|---|---|---|---|---|---|
| 0.05 | 50% | 71% | 46% | 81% | 60% | 76% |
| 0.1 | 52% | 79% | 44% | 75% | 52% | 69% |
| 0.2 | 58% | 81% | 29% | 73% | 53% | 67% |

## bike_rerun

**Model quality (test set).** Original: rmse 54.1, mae 37.35. Healthy retrains (n=10): rmse 53.38 ± 3.5, mae 37.45 ± 3.

**Suite size per theta:** θ=0.05: 378 tests, θ=0.1: 390 tests, θ=0.2: 417 tests.

**False-alarm level** (mean failure rate on healthy retrains, θ=0.1):

| suite/oracle | mean | max |
|---|---|---|
| bva/output | 0.361 | 0.462 |
| bva/shap | 0.296 | 0.392 |
| ours/output | 0.447 | 0.49 |
| ours/shap | 0.366 | 0.431 |
| random/output | 0.379 | 0.474 |
| random/shap | 0.301 | 0.421 |

**Mutants detected (θ=0.1).** Random is averaged over its repetitions.

| op | bva/output | bva/shap | ours/output | ours/shap | random/output | random/shap | accuracy/- | n/mutants |
|---|---|---|---|---|---|---|---|---|
| GF | 0% | 11% | 11% | 11% | 7% | 9% | 100% | 9 |
| WS | 78% | 89% | 67% | 78% | 84% | 89% | 100% | 9 |
| NEB | 0% | 0% | 0% | 0% | 0% | 0% | 89% | 9 |
| NAI | 0% | 22% | 11% | 22% | 13% | 22% | 100% | 9 |
| LN | 25% | 25% | 25% | 25% | 5% | 10% | 100% | 4 |
| FS | 0% | 17% | 17% | 17% | 7% | 10% | 100% | 6 |
| DS | 0% | 50% | 0% | 100% | 10% | 90% | 100% | 2 |
| All | 17% | 29% | 21% | 29% | 21% | 28% | 98% | 48 |

**Faults that accuracy misses** (1 mutants whose test metric stays within the healthy range), detected by:

| suite | output | shap |
|---|---|---|
| bva | 0% | 0% |
| ours | 0% | 0% |
| random | 0% | 0% |

**Sensitivity to θ** (all mutants detected):

| theta | bva/output | bva/shap | ours/output | ours/shap | random/output | random/shap |
|---|---|---|---|---|---|---|
| 0.05 | 15% | 29% | 17% | 27% | 21% | 32% |
| 0.1 | 17% | 29% | 21% | 29% | 21% | 28% |
| 0.2 | 21% | 29% | 27% | 42% | 23% | 29% |

## diet

**Model quality (test set).** Original: rmse 0.5518, mae 0.4643. Healthy retrains (n=10): rmse 0.4728 ± 0.029, mae 0.4004 ± 0.027.

**Suite size per theta:** θ=0.05: 150 tests, θ=0.1: 150 tests, θ=0.2: 150 tests.

**False-alarm level** (mean failure rate on healthy retrains, θ=0.1):

| suite/oracle | mean | max |
|---|---|---|
| bva/output | 0.077 | 0.107 |
| bva/shap | 0.065 | 0.113 |
| ours/output | 0.151 | 0.207 |
| ours/shap | 0.119 | 0.207 |
| random/output | 0.099 | 0.2 |
| random/shap | 0.07 | 0.173 |

**Mutants detected (θ=0.1).** Random is averaged over its repetitions.

| op | bva/output | bva/shap | ours/output | ours/shap | random/output | random/shap | accuracy/- | n/mutants |
|---|---|---|---|---|---|---|---|---|
| GF | 0% | 0% | 0% | 0% | 0% | 0% | 89% | 9 |
| WS | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 9 |
| NEB | 0% | 0% | 0% | 0% | 0% | 7% | 89% | 9 |
| NAI | 22% | 22% | 22% | 11% | 24% | 29% | 100% | 9 |
| LN | 100% | 100% | 75% | 25% | 60% | 95% | 50% | 4 |
| FS | 83% | 67% | 50% | 50% | 67% | 67% | 67% | 6 |
| DS | 0% | 0% | 0% | 0% | 0% | 10% | 0% | 2 |
| All | 42% | 40% | 35% | 29% | 37% | 42% | 83% | 48 |

**Faults that accuracy misses** (8 mutants whose test metric stays within the healthy range), detected by:

| suite | output | shap |
|---|---|---|
| bva | 50% | 38% |
| ours | 25% | 0% |
| random | 32% | 45% |

**Sensitivity to θ** (all mutants detected):

| theta | bva/output | bva/shap | ours/output | ours/shap | random/output | random/shap |
|---|---|---|---|---|---|---|
| 0.05 | 42% | 40% | 35% | 29% | 37% | 42% |
| 0.1 | 42% | 40% | 35% | 29% | 37% | 42% |
| 0.2 | 42% | 40% | 35% | 29% | 37% | 42% |

## diet_date

**Model quality (test set).** Original: rmse 1.211, mae 1.102. Healthy retrains (n=10): rmse 0.6484 ± 0.18, mae 0.5239 ± 0.2.

**Suite size per theta:** θ=0.05: 168 tests, θ=0.1: 168 tests, θ=0.2: 168 tests.

**False-alarm level** (mean failure rate on healthy retrains, θ=0.1):

| suite/oracle | mean | max |
|---|---|---|
| bva/output | 0.296 | 0.327 |
| bva/shap | 0.302 | 0.345 |
| ours/output | 0.373 | 0.405 |
| ours/shap | 0.34 | 0.429 |
| random/output | 0.278 | 0.345 |
| random/shap | 0.287 | 0.357 |

**Mutants detected (θ=0.1).** Random is averaged over its repetitions.

| op | bva/output | bva/shap | ours/output | ours/shap | random/output | random/shap | accuracy/- | n/mutants |
|---|---|---|---|---|---|---|---|---|
| GF | 0% | 0% | 0% | 0% | 0% | 0% | 44% | 9 |
| WS | 67% | 33% | 22% | 22% | 53% | 40% | 44% | 9 |
| NEB | 0% | 0% | 0% | 0% | 0% | 0% | 44% | 9 |
| NAI | 0% | 0% | 0% | 0% | 0% | 0% | 22% | 9 |
| LN | 50% | 50% | 0% | 0% | 50% | 50% | 0% | 4 |
| FS | 50% | 33% | 33% | 0% | 47% | 30% | 33% | 6 |
| DS | 0% | 0% | 0% | 0% | 0% | 0% | 100% | 2 |
| All | 23% | 15% | 8% | 4% | 20% | 15% | 38% | 48 |

**Faults that accuracy misses** (30 mutants whose test metric stays within the healthy range), detected by:

| suite | output | shap |
|---|---|---|
| bva | 30% | 23% |
| ours | 13% | 7% |
| random | 26% | 22% |

**Sensitivity to θ** (all mutants detected):

| theta | bva/output | bva/shap | ours/output | ours/shap | random/output | random/shap |
|---|---|---|---|---|---|---|
| 0.05 | 23% | 15% | 8% | 4% | 20% | 15% |
| 0.1 | 23% | 15% | 8% | 4% | 20% | 15% |
| 0.2 | 23% | 15% | 8% | 4% | 20% | 15% |
