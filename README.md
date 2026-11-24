# Replication package: SHAP-sign partitions as test cases for ML models

Code, generated test suites and results of the paper *Advancing Machine Learning Software
Testing with Interpretable Black-Box Techniques* (Cunha and Pombo), submitted to Software Testing, Verification and Reliability (Wiley).

The method derives per-feature equivalence partitions from the sign of SHAP values, delimits
the uncertainty margin around each boundary with entropy (Algorithm 1 of the paper), uses
decision trees for wide margins, and writes boundary test cases (input, feature, expected sign).

## Where each result of the paper comes from

| Paper | Command | Output |
|---|---|---|
| Case study (Table 1, figures) | `src/run.py --dataset diet`, `src/figures.py diet` | `results/diet/partitions_theta0.1.csv`, `trees_theta0.1.txt`, `tests_theta0.1.csv` |
| Evaluation, fault detection (Table "mutants detected") | `src/run.py --dataset {adult,bike,diet}`, `src/summarize.py` | `results/summary.md`, `results/*_detection.csv` |
| Same, with the day counter as input | `src/run.py --dataset diet_date` | `results/diet_date/` |
| Diagnosis on Bike, lagged copies grouped | `src/diagnose.py bike [--group-lags]`, `src/run.py --dataset bike --group-lags` | `results/bike/diagnosis.csv`, `results/bike_grouped/` |
| Controlled diagnosis (injected causes) | `src/stage2.py` (design: `STAGE2.md`) | `results/stage2/summary.md`; as pre-specified: `results/stage2/prereg/` |
| Reproducibility check | `src/run.py --dataset bike --out results/bike_rerun` | `results/bike_rerun/` |
| RQ5: region checks and filtered statistics on the exploration set | `src/stage4.py shap`, `detect`, `summarize` (design: `STAGE4.md`) | `results/stage4/summary.md` |
| RQ5: detectors against retrained models on the confirmation set (Table "confirmation mutants detected") | `src/stage6.py detect`, `choose`, `models`, `detect`, `summarize`; post hoc `src/stage6_posthoc.py` (design: `STAGE6.md`) | `results/stage6/summary.md`, `detect.csv`, `choice.json` |
| RQ5: localization by sign disagreement and trees | `src/stage5_localize.py` (design: `STAGE5.md`) | `results/stage5/summary.md` |
| RQ5: boundaries vs SHAP background | `src/stage6.py background`, `src/stage4_background.py`, `src/stage4_threshold.py` | `results/stage6/background.csv`, `results/stage4/background.csv`, `threshold.csv` |
| RQ6: type of fault of a flagged model (Table "diagnosis of the type of fault") | `src/stage7.py features --dataset {adult,bike,diet}`, `summarize` (design: `STAGE7.md`) | `results/stage7/summary.md`, `features_*.csv`, `predictions.csv` |
| RQ6: choosing a fix | `src/stage3.py` (design: `STAGE3.md`), `src/stage3_check.py` | `results/stage3/summary.md` |

## Setup

Python 3.8–3.11 (TensorFlow 2.13 has no wheels for 3.12):

```sh
python3.11 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt            # + requirements-figures.txt for src/figures.py
```

The results were produced on CPU (Python 3.8, macOS) and re-run on Linux with Python 3.11
and an NVIDIA GPU (CUDA 11.8 / cuDNN 8.6, from the `nvidia-*-cu11` pip packages); the re-run
reproduces them (`results/bike_rerun`). The small MLPs of `stage2.py` run much faster on the CPU
(`CUDA_VISIBLE_DEVICES=""`). TensorFlow runs in graph mode because `shap.DeepExplainer` does not
support LSTMs in eager mode.

## Data

The datasets are not redistributed here; see `data/README.md` for where to put them.

## Running

```sh
python3 -I src/run.py --dataset adult        # also bike, diet, diet_date
python3 -I src/summarize.py                  # all datasets with results
python3 -I src/diagnose.py bike --group-lags
python3 -I src/stage2.py
python3 -I src/stage3.py                     # see STAGE3.md for --shard and --summarize
python3 -I src/stage4.py shap; python3 -I src/stage4.py detect; python3 -I src/stage4.py summarize
python3 -I src/stage5_localize.py
python3 -I src/stage6.py detect; python3 -I src/stage6.py choose     # exploration set only, before the new mutants exist
python3 -I src/stage6.py models; python3 -I src/stage6.py detect; python3 -I src/stage6.py background; python3 -I src/stage6.py summarize
python3 -I src/stage7.py features --dataset adult   # also bike, diet
python3 -I src/stage7.py summarize
python3 -I src/figures.py diet --out figures --margins calories_t run_t-1 weight_t-1 --trees weight_t-1
```

Every trained model and every evaluation is saved as soon as it is done (`results/<dataset>/weights`,
`evals`), so an interrupted run resumes where it stopped, and the shipped results are reused
instead of retraining. Delete a dataset's folder to recompute it from scratch.

## What `run.py` does

1. Trains the original model (seed 0) with early stopping on the validation loss. Time series are
   split chronologically (70/15/15); Adult uses its official test file. Scaling uses training
   statistics only.
2. Computes DeepSHAP values on up to 3,000 training samples, with 100 training samples as background.
3. Generates test suites for each entropy threshold θ ∈ {0.05, 0.1, 0.2}: **ours** (Algorithm 1,
   boundary values for narrow margins, decision-tree leaves of depth 4 for wide margins, k = 3 real
   samples per boundary or leaf), **random** (same size, 5 repetitions) and **bva** (samples at and
   next to each feature's minimum and maximum). A test is (input, feature, expected sign).
4. Builds the models under test: 10 healthy retrains (other seeds), 36 model-level mutants
   (DeepMutation GF, WS, NEB, NAI at 2/5/10%, 3 seeds) and 12 data-level mutants (label noise 10/30%,
   shuffling of each of the 3 most important variables, removal of the top quartile of the most
   important variable; 2 seeds).
5. Runs every suite on every model with two oracles: the SHAP sign of the model under test, and a
   black-box output effect M(x) − mean_b M(x with the feature replaced by b) over 30 background samples.

A mutant is **detected** by a suite when the suite's failure rate on it is higher than on every
healthy model. Accuracy-based detection flags a mutant whose test RMSE (or AUC) is worse than that
of every healthy model.

`--group-lags` analyses the time steps of each variable (var_t-2, var_t-1, var_t) as one feature:
its SHAP value is the sum over the steps, its value is the one at time t, and the black-box oracle
replaces all its steps at once.

## Diagnosis of uncertainty margins

`src/diagnose.py` classifies each feature's margin into one of nine rows (sharp boundary, no real
effect, simple/complex interaction, missing feature or label noise, too little data, unstable
training, explainer artifact, non-monotonic effect), from the reference model and its healthy
retrains. The thresholds (`THRESHOLDS`) were fixed before any diagnosis was computed. Row 9 and its
two corrections were defined after seeing results; the docstring documents both, and `STAGE2.md`
reports the controlled results before and after.

## Licence and citation

The code is under the MIT licence. The results and the design documents (`STAGE*.md`) are under
CC BY 4.0. See `LICENSE`. To cite this package, use `CITATION.cff` or the Zenodo record linked
from the repository.
