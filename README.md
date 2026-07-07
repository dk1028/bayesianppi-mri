# Bayesian Chain-Rule Prediction-Powered Inference for Binary Prevalence Estimation

Reproducibility repository for the TMLR manuscript **“Bayesian Chain-Rule Prediction-Powered Inference for Binary Prevalence Estimation.”**

This repository contains the conjugate Bayesian chain-rule estimator (CRE), comparison baselines, simulation-based calibration, K-bin sensitivity analyses, and the ADNI MRI case-study pipeline.

## What this revision changes

The manuscript revision uses one authoritative analysis pipeline:

1. Train the MRI autorater using **five-fold subject-grouped out-of-fold (OOF) splits**.
2. Match age at the **scan level** using subject identifier and acquisition date.
3. Derive the age 65--70 subset from the full OOF table, rather than training a separate age-specific CNN.
4. Run the full-cohort and age 65--70 repeated-labeling experiments from the same analysis code.
5. Compare CRE with labeled-only Bayes, the binary difference estimator, continuous-score PPI, and the power-tuned PPI++-style baseline used in the manuscript.
6. Run the revised case-study audits, K-bin sensitivity analysis, and SBC diagnostics.

Legacy scripts from the earlier workflow are moved to `code/legacy/` by the supplied update script. They are retained for history but are **not used** for manuscript results.

## Estimands and baselines

The main binary chain-rule estimand is

\[
g = P(H=1)
  = P(A=1)P(H=1\mid A=1)
  + P(A=0)P(H=1\mid A=0),
\]

where:

- `H = 1` denotes Alzheimer’s disease (AD),
- `H = 0` denotes cognitively normal (CN),
- `A = 1{p >= 0.5}` is the thresholded autorater decision,
- `p` is the continuous OOF autorater probability.

CRE and the binary difference estimator use the thresholded decision `A`. Continuous-score PPI and the power-tuned PPI baseline use `p`. They are complementary comparisons and need not have identical operational interpretations.

## Repository layout

```text
bayesianppi-mri/
├── README.md
├── README_DATA.md
├── REPRODUCIBILITY.md
├── RESULTS.md
├── CITATION.cff
├── requirements.txt
├── requirements-ci.txt
├── environment.yml
├── Makefile
├── configs/
│   └── paper.yaml
├── code/
│   ├── autorater/
│   │   ├── _shared.py
│   │   ├── fold_utils.py
│   │   ├── train_subject_oof.py
│   │   ├── change_dicom.py
│   │   └── process_all.py
│   ├── data/
│   │   └── attach_scan_age.py
│   ├── ppi/
│   │   ├── _shared.py
│   │   ├── coverage_experiment.py
│   │   ├── case_study_analysis.py
│   │   ├── k_chain_rule.py
│   │   ├── sbc.py
│   │   ├── histogram_posterior.py
│   │   └── prior_predictive_checks.py
│   └── legacy/
├── data/
│   ├── README_DATA.md
│   └── example/
│       └── synthetic_oof_example.csv
├── results/
│   └── reference/
├── tests/
└── paper/
```

## macOS installation

Tested commands for Terminal on macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

Apple Silicon users can use PyTorch’s MPS backend automatically when available. The autorater code selects devices in this order: CUDA, MPS, CPU.

Run the lightweight test suite:

```bash
pytest -q
```

## Reproduction tiers

### Tier 1: synthetic analyses

No ADNI data are required.

```bash
python code/ppi/sbc.py \
  --output-dir results/sbc \
  --M 500 \
  --S 1000 \
  --NA 2116 \
  --NH 100 \
  --priors jeffreys uniform \
  --bins 20 \
  --seed 2025
```

The existing posterior and prior-predictive scripts can also be run directly:

```bash
python code/ppi/histogram_posterior.py
python code/ppi/prior_predictive_checks.py
```

### Tier 2: ADNI case study

ADNI data cannot be redistributed. After obtaining authorized ADNI access, place local input data according to `README_DATA.md`.

#### Step 1: create subject-grouped OOF predictions

```bash
python code/autorater/train_subject_oof.py \
  --input-csv data/csv/matched_cn_ad_labels_all.csv \
  --output-csv results/autorater/autorater_predictions_all_subject_oof.csv \
  --folds 5 \
  --epochs 5 \
  --batch-size 2 \
  --learning-rate 1e-4 \
  --seed 42
```

The script aborts if any subject appears in more than one OOF fold.

#### Step 2: attach scan-level age and derive age 65--70

```bash
python code/data/attach_scan_age.py \
  --oof-csv results/autorater/autorater_predictions_all_subject_oof.csv \
  --metadata-csv data/csv/adni_mrpage_t1_search.csv \
  --output-dir results/autorater \
  --expect-paper-counts
```

Expected manuscript QC:

| Dataset | Scans | Subjects | Prevalence |
|---|---:|---:|---:|
| Full cohort | 2116 | 503 | 0.3081285 |
| Age 65--70 | 291 | 93 | 0.2371134 |

#### Step 3: repeated-labeling coverage and width

```bash
python code/ppi/coverage_experiment.py \
  --full-csv results/autorater/autorater_predictions_all_subject_oof_with_scan_age.csv \
  --age6570-csv results/autorater/autorater_predictions_6570_subject_oof_scan_age.csv \
  --output-dir results/coverage \
  --nsim 500 \
  --posterior-draws 5000 \
  --bootstrap 1000 \
  --seed 2025
```

#### Step 4: case-study ROC, threshold, calibration, and prevalence audits

```bash
python code/ppi/case_study_analysis.py \
  --input-csv results/autorater/autorater_predictions_all_subject_oof_with_scan_age.csv \
  --output-dir results/case_study \
  --bootstrap 2000 \
  --permutations 5000 \
  --threshold-bootstrap 2000 \
  --seed 2025
```

This script replaces the historically named `delong_permutation.ipynb`. The reported age-bin comparison is a permutation test, not a DeLong test.

#### Step 5: K-bin sensitivity

```bash
python code/ppi/k_chain_rule.py \
  --full-csv results/autorater/autorater_predictions_all_subject_oof_with_scan_age.csv \
  --age6570-csv results/autorater/autorater_predictions_6570_subject_oof_scan_age.csv \
  --output-dir results/kbin \
  --k 2 4 5 \
  --draws 10000 \
  --seed 2025
```

The independent Dirichlet--Beta K-bin model remains conjugate. It is presented as a score-discretization sensitivity analysis, not as evidence for a non-conjugate NUTS extension.

## PPI++ implementation note

The manuscript results are reproduced by the self-contained power-tuned PPI formula in `code/ppi/_shared.py`, using the observed cohort as the monitoring pool and a randomly labeled subset as the rectifier sample. The result columns are named `ppipp_*` for consistency with the manuscript.

For a canonical disjoint labeled/unlabeled analysis using the maintained `ppi-python` package, use the optional `official_ppipp_mean_ci()` wrapper and provide the complement of the labeled indices as the unlabeled prediction sample. This optional mode is not used to regenerate the fixed manuscript numbers.

## Reference results

See `RESULTS.md` and `results/reference/` for summary-level manuscript values. Participant-level rows and identifiers are not included.

Key QC values:

| Scope | AUC | ACC at 0.5 | TPR | TNR |
|---|---:|---:|---:|---:|
| Full cohort | 0.6880144 | 0.7136106 | 0.2653374 | 0.9132514 |
| Age 65--70 | 0.7717391 | 0.8350515 | 0.4057971 | 0.9684685 |

## Data privacy and ADNI terms

Do not commit any of the following:

- ADNI participant-level metadata,
- subject identifiers,
- local NIfTI paths,
- DICOM or NIfTI files,
- row-level OOF prediction tables.

The repository tracks code, schemas, synthetic fixtures, summary-level result tables, and figures only.

## Citation

See `CITATION.cff`.
