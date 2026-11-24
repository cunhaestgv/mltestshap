# Stage 4 results

Design: ../../STAGE4.md.

## Part A: detection

| view | set | n | D1 | D2 | D2-all | ours | bva | random | accuracy |
|---|---|---|---|---|---|---|---|---|---|
| adult | general | 48 | 83% | 73% | 48% | 44% | 50% | 32% | 92% |
| adult | weight | 36 | 83% | 67% | 42% | 39% | 47% | 33% | 89% |
| adult | data | 12 | 83% | 92% | 67% | 58% | 58% | 32% | 100% |
| adult | BS | 6 | 33% | 17% | 33% | 33% | 17% | 23% | 67% |
| bike | general | 48 | 10% | 85% | 40% | 29% | 29% | 28% | 98% |
| bike | weight | 36 | 11% | 89% | 33% | 28% | 31% | 30% | 97% |
| bike | data | 12 | 8% | 75% | 58% | 33% | 25% | 23% | 100% |
| bike | BS | 6 | 33% | 67% | 83% | 33% | 17% | 30% | 100% |
| diet | general | 48 | 27% | 46% | 40% | 29% | 40% | 42% | 83% |
| diet | weight | 36 | 22% | 44% | 31% | 28% | 31% | 34% | 94% |
| diet | data | 12 | 42% | 50% | 67% | 33% | 67% | 67% | 50% |
| diet | BS | 6 | 0% | 0% | 17% | 0% | 17% | 23% | 0% |
| bike_grouped | general | 48 | – | 94% | 83% | 75% | 79% | 69% | 98% |
| bike_grouped | weight | 36 | – | 94% | 81% | 75% | 83% | 71% | 97% |
| bike_grouped | data | 12 | – | 92% | 92% | 75% | 67% | 63% | 100% |
| bike_grouped | BS | 6 | – | 100% | 100% | 83% | 67% | 77% | 100% |

Pooled over Adult, Bike and Diet; α = 0.05/8. a/b = mutants only the detector / only the comparator found.

| set | test | a/b | p | detector | comparator |
|---|---|---|---|---|---|
| general | D1 vs ours | 23/14 | 0.1877 | 40% | 34% |
| general | D1 vs bva | 20/19 | 1.0000 | 40% | 40% |
| general | D2 vs ours | 50/1 | 0.0000 | 68% | 34% |
| general | D2 vs bva | 46/5 | 0.0000 | 68% | 40% |
| BS | D1 vs ours | 1/1 | 1.0000 | 22% | 22% |
| BS | D1 vs bva | 3/2 | 1.0000 | 22% | 17% |
| BS | D2 vs ours | 2/1 | 1.0000 | 28% | 22% |
| BS | D2 vs bva | 3/1 | 0.6250 | 28% | 17% |

Mutants accuracy misses (pooled, n=21): D1 24%, D2 10%, D2-all 24%, ours 0%, bva 19%, random 25%

AUROC, mutants vs healthy (general / data + BS):

| view | D1 | D2 | D2-all | ours | bva | random | accuracy |
|---|---|---|---|---|---|---|---|
| adult | 0.92 / 0.83 | 0.81 / 0.92 | 0.67 / 0.79 | 0.74 / 0.81 | 0.65 / 0.73 | 0.63 / 0.84 | 0.98 / 0.95 |
| bike | 0.56 / 0.69 | 0.91 / 0.87 | 0.49 / 0.83 | 0.46 / 0.81 | 0.53 / 0.74 | 0.49 / 0.80 | 0.99 / 1.00 |
| diet | 0.39 / 0.48 | 0.73 / 0.62 | 0.60 / 0.76 | 0.50 / 0.69 | 0.63 / 0.78 | 0.59 / 0.74 | 0.93 / 0.66 |
| bike_grouped | – | 0.98 / 0.99 | 0.84 / 0.94 | 0.83 / 0.93 | 0.84 / 0.72 | 0.85 / 0.93 | 0.99 / 1.00 |

Regions, kept regions, narrow features: adult 24, 18, 4; bike 130, 34, 7; diet 50, 38, 22; bike_grouped 45, 34, 0

Localization of BS mutants (hit = the top drifting feature belongs to the shifted variable): adult 0/6, bike 2/6, diet 0/6
By δ: 0.05 1/6, 0.1 1/6, 0.2 0/6

Did BS mutants move the boundary? Midpoint shift of the target feature, scaled units:

| view | target | δ | moved (seeds 0, 1) | healthy max d_f |
|---|---|---|---|---|
| adult | education_num | 0.05 | +0.000, +0.000 | 0.000 |
| adult | education_num | 0.1 | +0.000, +0.000 | 0.000 |
| adult | education_num | 0.2 | +0.000, +0.000 | 0.000 |
| bike | cnt_t | 0.05 | +0.004, -0.014 | 0.006 |
| bike | cnt_t | 0.1 | +0.001, -0.010 | 0.006 |
| bike | cnt_t | 0.2 | +0.003, +0.000 | 0.006 |
| diet | calories_t | 0.05 | +0.000, +0.000 | 0.000 |
| diet | calories_t | 0.1 | +0.000, +0.000 | 0.000 |
| diet | calories_t | 0.2 | +0.000, +0.000 | 0.000 |

## Post hoc controls (not pre-registered)

The healthy filter of D2 applied without regions. Cross-fitted like D2. Pooled over Adult, Bike and Diet.

| set | D2 | D2-all | ours | F-ours | bva | F-bva | random | F-random | F-pairs |
|---|---|---|---|---|---|---|---|---|---|
| general | 68% | 42% | 34% | 73% | 40% | 76% | 34% | 73% | 81% |
| BS | 28% | 44% | 22% | 44% | 17% | 39% | 26% | 49% | 67% |

D2 vs F-pairs, general: 2/21, p = 0.0001.

## Part B: culprit screen

136 cases.

| policy | recall (spurious) | regression | classification | harm (others) | gain spurious | gain others |
|---|---|---|---|---|---|---|
| screen | 11/30 | 6/15 | 5/15 | 0/100 | 33.9% | 0.0% |
| root-only | 13/30 | 7/15 | 6/15 | 0/100 | 37.9% | 0.0% |
| shap-only | 11/30 | 6/15 | 5/15 | 0/100 | 33.9% | 0.0% |
| no-confirm | 11/30 | 6/15 | 5/15 | 56/100 | -464.5% | -370.3% |
| stage3-rule | 1/30 | 1/15 | 0/15 | 1/100 | 0.3% | -0.0% |

Screen recall by copy noise: ×1 10/10, ×2 1/10, ×4 0/10

Real: adult #0 root 6, shap 6, drop 6: discovery gain -17.6%, test gain -19.5%
Real: adult #1 root 6, shap 6, drop 6: discovery gain -15.7%, test gain -21.1%
Real: adult #2 root 6, shap 6, drop 6: discovery gain -16.8%, test gain -19.9%
Real: bike #0 root 38, shap 38, drop 38: discovery gain -32.7%, test gain -41.2%
Real: bike #1 root 38, shap 38, drop 38: discovery gain -41.3%, test gain -39.0%
Real: bike #2 root 38, shap 38, drop 38: discovery gain -38.3%, test gain -59.6%
