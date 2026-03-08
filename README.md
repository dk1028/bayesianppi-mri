# Conjugate and MCMC Bayesian Chain-Rule Prediction-Powered Inference for Binary Prevalence Estimation

> Code artifact for the manuscript and preprint.

The goal of this repository is to make it easy to:

1. Reproduce the **conjugate Bayesian chain-rule PPI** experiments and diagnostics.
2. Reproduce the **ADNI MRI case study** (3D CNN autorater + CRE / Naïve / Difference / prior-free analytic PPI baselines).
3. Regenerate the **coverage, interval-width, ROC/AUC, threshold-audit, and example figures** used in the manuscript.

The repository is organized so that readers can move from raw ADNI data and metadata, to matched tables, to out-of-fold (OOF) autorater predictions, and finally to prevalence inference and figures.

---

## 1. Repository layout

We assume the repository is cloned as `bayesianppi-mri/`.

```text
bayesianppi-mri/
├─ paper/
│   └─ tmlr_bayesianppi_final.pdf
├─ data/
│   ├─ raw_dicom/              # ADNI DICOM (user-supplied, not tracked)
│   ├─ nifti/                  # NIfTI volumes (dcm2niix output; not tracked)
│   ├─ csv/                    # Metadata and matched label tables
│   │   ├─ adni_mrpage_t1_search.csv        # ADNI Advanced Search export (user-supplied)
│   │   ├─ matched_cn_ad_labels_all.csv     # Full-cohort matched labels (generated)
│   │   └─ matched_cn_ad_labels_6570.csv    # 65–70 subset matched labels (generated)
│   └─ README_DATA.md
├─ results/
│   ├─ autorater/              # Autorater predictions, metrics, logs
│   ├─ coverage/               # Coverage & width summaries
│   └─ age_analysis/           # Age-stratified ROC/AUC results
├─ figs/                       # Final and intermediate figures used in the paper
└─ code/
    ├─ autorater/
    │   ├─ _shared.py          # Shared preprocessing / model / OOF helpers
    │   ├─ _viz_examples.py    # Shared example-figure helpers
    │   ├─ change_dicom.py     # DICOM → NIfTI (dcm2niix wrapper)
    │   ├─ process_all.py      # Build matched CN/AD table for full cohort
    │   ├─ process_6570.py     # Build matched CN/AD table for 65–70 subset
    │   ├─ cnn_all.py          # 3D CNN autorater (full cohort; saves OOF predictions)
    │   ├─ cnn_6570.py         # 3D CNN autorater (65–70 subset)
    │   ├─ process.py          # CNN architecture / processing figure helper
    │   ├─ autorater.py        # Example AD vs CN cases and bar plots
    │   ├─ model_ADCN.py       # Additional model / architecture figure script
    │   └─ fig1_pipeline.py    # NIfTI → crop/pad → resize → preprocessing pipeline figure
    └─ ppi/
        ├─ _shared.py                        # Shared CRE / baseline / CI helpers
        ├─ histogram_posterior.py            # Conjugate Bernoulli chain-rule posterior experiment
        ├─ prior_predictive_checks.py        # Prior predictive checks for Bayesian PPI
        ├─ implementaion_coverage_all.py     # Coverage & width (full cohort, Colab-oriented)
        ├─ implementaion_coverage_6570.py    # Coverage & width (65–70 subset, Colab-oriented)
        ├─ coverage_intervalall.py           # Plot coverage vs width (full cohort)
        ├─ coverage_interval6570.py          # Plot coverage vs width (65–70 subset)
        ├─ k_chain_rule.py                   # K-bin chain-rule estimator experiments
        ├─ sbc.py                            # Simulation-based calibration (SBC)
        └─ age_analysis.py                   # Age-stratified ROC / threshold / calibration analysis
```

> **Important:** raw MRI data and ADNI metadata are *not* included, due to licensing.
> The scripts are written so that, given access to ADNI and the CSV export from ADNI Advanced Search, a reader can reproduce the case study.

---

## 2. Core ideas and components

The repository contains three main components that correspond to the manuscript:

1. **Conjugate Bayesian chain-rule PPI experiments and diagnostics**

   * `code/ppi/histogram_posterior.py` implements a simple Bernoulli chain-rule posterior experiment using **direct conjugate Beta sampling**.
   * `code/ppi/prior_predictive_checks.py` runs prior predictive checks for the Bayesian chain-rule model.
   * `code/ppi/k_chain_rule.py` and `code/ppi/sbc.py` explore K-bin chain-rule estimators and SBC.

2. **MRI autorater (ADNI case study)**

   * `code/autorater/change_dicom.py` converts ADNI DICOMs to NIfTI using `dcm2niix`.
   * `code/autorater/process_all.py` and `process_6570.py` join the ADNI metadata CSV with NIfTI paths.
   * `code/autorater/cnn_all.py` and `cnn_6570.py` train a lightweight 3D CNN autorater for CN vs AD classification.
   * The full-cohort autorater workflow is designed to save **out-of-fold (OOF)** predicted probabilities for downstream threshold and prevalence analyses.

3. **Coverage/interval-width and age-stratified analyses**

   * `code/ppi/implementaion_coverage_all.py` and `implementaion_coverage_6570.py` run **repeated-labeling resampling** experiments using autorater predictions and clinician labels.
   * `code/ppi/coverage_intervalall.py` and `coverage_interval6570.py` recreate the coverage/width plots by **reading generated CSV summaries** (rather than plotting hard-coded numbers).
   * `code/ppi/age_analysis.py` examines ROC/AUC, threshold selection, calibration, and overlap diagnostics by age band (e.g. 50–73, 74–79, 80–100).

Each script’s purpose is documented in its docstring and in the sections below.

---

## 3. Requirements and installation

We recommend a fresh Python virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The main dependencies (exact versions may be pinned in `requirements.txt`) are:

* `numpy`, `scipy`, `pandas`, `matplotlib`
* `torch`, `torchvision`
* `pymc>=5` (used for non-conjugate extensions; the main base CRE experiments do **not** require MCMC)
* `scikit-learn` (ROC, AUC, calibration curves, overlap diagnostics)
* `tqdm` for progress bars
* `nibabel`, `scikit-image` for MRI loading and preprocessing

> **Note:** the base chain-rule experiments and the main coverage scripts use **direct conjugate Beta sampling** for the core Bayesian estimator. PyMC is retained for non-conjugate extensions and optional exploratory scripts.

---

## 4. Data (ADNI) and directory setup

### 4.1. ADNI access and licensing

The real-data experiments use the Alzheimer’s Disease Neuroimaging Initiative (ADNI) dataset.
Due to licensing restrictions, we cannot redistribute ADNI data in this repository.

To reproduce our case study, you will need:

1. An ADNI account with access to the MRI data.
2. The ability to run **Advanced Search** on the ADNI website.
3. Permission to download both **images (DICOM)** and the **associated metadata CSV**.

Please see `data/README_DATA.md` for a concise description of the columns used in the matching step (subject ID, acquisition date, diagnosis label, etc.).

### 4.2. ADNI Advanced Search settings for this project

We constructed our main metadata CSV using the ADNI **Advanced Search** interface as follows:

1. Go to ADNI **Advanced Search**.
2. Under **Project**, check: `ADNI`.
3. Under **Phase**, check (only):

   * `ADNI 1`
   * `ADNI GO`
   * `ADNI 2`
   * `ADNI 3`
4. Under **STUDY/VISIT**, ensure the logical operator is `AND`.
5. In **Image Description**, type `MPRAGE` (or `MRPAGE` if that is how your ADNI interface auto-completes; the intention is to select T1-weighted MPRAGE sequences).
6. Under **Modality**, select `MRI`.
7. Under **Acquisition Type**, select `3D`.
8. Under **Weighting**, select `T1`.
9. Click **Search**.
10. Click **Select All** (or equivalent) and then **Export CSV** to download the metadata.

We recommend saving or renaming this CSV as:

```text
bayesianppi-mri/data/csv/adni_mrpage_t1_search.csv
```

The scripts `process_all.py` and `process_6570.py` assume this CSV is present in `data/csv/`.
If you use a different filename, you can either rename the file or update the `META_CSV` path in those scripts.

### 4.3. Directory structure for ADNI data

Once you have the metadata CSV and MRI images, place them as follows:

* **Metadata CSV:**

  * `data/csv/adni_mrpage_t1_search.csv` (Advanced Search export)
  * Additional metadata tables (if any) can also live in `data/csv/`.

* **DICOM images:**

  * Organize raw DICOM images under `data/raw_dicom/` in one folder per subject or per ADNI download batch.

* **NIfTI images:**

  * After running `change_dicom.py` (see below), NIfTI files will be written to `data/nifti/`.

This layout is reflected in the code through repo-relative paths, so no hard-coded absolute paths need to be edited.

---

## 5. Paths and repo-relative imports

All Python scripts in `code/autorater/` and `code/ppi/` are designed to work with **repo-relative** paths. At the top of each script, we follow a pattern similar to:

```python
from pathlib import Path

REPO_ROOT   = Path(__file__).resolve().parents[2]
DATA_ROOT   = REPO_ROOT / "data"
CSV_ROOT    = DATA_ROOT / "csv"
NIFTI_ROOT  = DATA_ROOT / "nifti"
RESULTS_ROOT= REPO_ROOT / "results"
FIGS_ROOT   = REPO_ROOT / "figs"
```

This means that once you:

```bash
cd bayesianppi-mri
python code/autorater/cnn_all.py
```

all file input/output is resolved relative to the repository root, and you do **not** need to change absolute paths manually.

---

## 6. How to run the experiments

### 6.1. Conjugate Bayesian chain-rule experiments (no ADNI required)

These are the simplest to reproduce and do not require any medical data.

1. **Posterior histogram experiment** (`histogram_posterior.py`)

   ```bash
   cd bayesianppi-mri
   python code/ppi/histogram_posterior.py
   ```

   This script:

   * Simulates multiple datasets under a Bernoulli chain-rule model with known true parameters.
   * Uses **conjugate Beta posterior updates** (rather than MCMC) for the base estimator.
   * Computes interval summaries and plots histograms of posterior means / medians of the target quantity.

2. **Prior predictive checks** (`prior_predictive_checks.py`)

   ```bash
   python code/ppi/prior_predictive_checks.py
   ```

   This script:

   * Draws parameters from the prior for the chain-rule model.
   * Simulates observables (A, H) under those draws.
   * Produces prior predictive plots for summary statistics used in the paper.

3. **K-bin chain-rule experiments** (`k_chain_rule.py`)

   ```bash
   python code/ppi/k_chain_rule.py --outdir results/coverage/k_chain_rule
   ```

   This script explores how the number of bins (K) affects the chain-rule estimator’s behaviour, using conjugate Dirichlet–Beta updates.

4. **Simulation-based calibration** (`sbc.py`)

   ```bash
   python code/ppi/sbc.py --outdir results/coverage/sbc
   ```

   This script runs SBC experiments to check whether the conjugate Bayesian chain-rule model is well calibrated, using rank histograms and diagnostic summaries.

### 6.2. DICOM → NIfTI conversion (optional, ADNI)

If you start from ADNI DICOM images, use `change_dicom.py` to convert them to NIfTI using `dcm2niix`:

```bash
cd bayesianppi-mri
python code/autorater/change_dicom.py
```

* Input: `data/raw_dicom/` (organized as you downloaded from ADNI).
* Output: `data/nifti/` (mirroring the directory structure of the input).

You may need to set the `DCM2NIIX_PATH` environment variable if `dcm2niix` is not on your `PATH`.

### 6.3. Build matched CN/AD tables (full cohort and 65–70)

With the ADNI metadata CSV in place (e.g. `data/csv/adni_mrpage_t1_search.csv`):

```bash
cd bayesianppi-mri

# Full cohort
python code/autorater/process_all.py

# 65–70 subset
python code/autorater/process_6570.py
```

These scripts:

* Read the Advanced Search CSV.
* Filter to the relevant subset of ADNI phases and diagnosis labels (CN vs AD).
* Match each MRI to a NIfTI path in `data/nifti/`.
* Create:

  * `data/csv/matched_cn_ad_labels_all.csv`
  * `data/csv/matched_cn_ad_labels_6570.csv`

These matched tables are the main inputs to the CNN and prevalence scripts.

### 6.4. Train the CNN autorater

To train the 3D CNN autorater on the **full cohort**:

```bash
cd bayesianppi-mri
python code/autorater/cnn_all.py
```

For the **65–70 subset**:

```bash
python code/autorater/cnn_6570.py
```

Each script:

* Loads the corresponding matched CSV from `data/csv/`.
* Loads 3D NIfTI volumes from `data/nifti/`.
* Trains a lightweight 3D CNN (using PyTorch) to predict AD vs CN.
* Saves autorater predictions and (optionally) model checkpoints to `results/autorater/`.

The full-cohort pipeline is designed so that the exported prediction CSV used downstream corresponds to **out-of-fold predicted probabilities**, which are then used for threshold selection, calibration, and prevalence estimation.

### 6.5. Coverage and interval-width experiments (ADNI-based)

The scripts `implementaion_coverage_all.py` and `implementaion_coverage_6570.py` are written in a Colab-friendly style, but can also be run locally once dependencies and paths are configured.

```bash
cd bayesianppi-mri

# Full cohort coverage / width
python code/ppi/implementaion_coverage_all.py

# 65–70 subset coverage / width
python code/ppi/implementaion_coverage_6570.py
```

These scripts:

* Read autorater predictions and clinician labels.
* Implement the conjugate Bayesian chain-rule estimator (CRE), labeled-only Bayesian baseline, Difference estimator, and prior-free analytic PPI baseline.
* For label budgets such as `n = 10, 20, 40, 80`, run **repeated-labeling resampling** with the cohort prevalence treated as the finite-population target.
* Use `M=500` replications in the main coverage experiments.
* Save summary tables to `results/coverage/` (or to the configured output directory in Colab).

Once the coverage tables are available, regenerate the figures with:

```bash
python code/ppi/coverage_intervalall.py
python code/ppi/coverage_interval6570.py
```

The plotting scripts now read the generated CSV summaries directly and recreate the coverage / width figures without relying on hard-coded values.

### 6.6. Age-stratified analysis

The script `code/ppi/age_analysis.py` implements the age-stratified analysis:

```bash
cd bayesianppi-mri
python code/ppi/age_analysis.py
```

It:

* Joins autorater predictions with ADNI metadata.
* Splits the cohort into pre-defined age bands (e.g. 50–73, 74–79, 80–100).
* Computes ROC curves, AUC, and confidence intervals in each age band.
* Performs threshold analysis (fixed `t=0.5`, Youden cut-point, OOF vs leaky comparison).
* Produces calibration summaries and overlap diagnostics.
* Saves intermediate results and plots under `results/age_analysis/` and `figs/`.

---

## 7. Figures

The following scripts are used to recreate figures in the manuscript:

* `code/autorater/fig1_pipeline.py` – preprocessing pipeline figure: NIfTI → crop/pad → resize → normalization.
* `code/autorater/autorater.py` – example AD vs CN volumes and autorater outputs.
* `code/autorater/model_ADCN.py` – CNN architecture / model diagram.
* `code/autorater/process.py` – updated processing / architecture figure helper.
* `code/ppi/coverage_intervalall.py` – coverage vs width (full cohort).
* `code/ppi/coverage_interval6570.py` – coverage vs width (65–70 subset).
* `code/ppi/age_analysis.py` – ROC / age-band / threshold-audit figures.

Each script writes `.png` and/or `.pdf` files to the `figs/` directory (or to the configured results folder in Colab).

---

## 8. Reproducibility checklist (informal)

In the spirit of a reproducibility checklist, we highlight the following:

* **Environment:**

  * We developed and tested the code primarily with Python 3.11.
  * `requirements.txt` lists the Python dependencies.

* **Randomness & seeds:**

  * Conjugate chain-rule scripts (`histogram_posterior.py`, `prior_predictive_checks.py`, `k_chain_rule.py`, `sbc.py`) set random seeds where appropriate for repeatable simulations.
  * CNN training scripts may include a fixed seed for PyTorch, but exact weight realizations can still vary across hardware and CuDNN versions.

* **Data availability:**

  * Base diagnostics and conjugate experiments are self-contained.
  * ADNI-based experiments require access to ADNI MRI and metadata, which we do not redistribute.
  * We document how the ADNI metadata CSV is constructed via Advanced Search (Project / Phase / Modality / Image Description filters).

* **Code completeness:**

  * All main experiments described in the manuscript are covered by scripts in `code/autorater` and `code/ppi`.
  * For ADNI, we provide the full path from the Advanced Search CSV and DICOM images to matched label tables, OOF autorater predictions, repeated-labeling prevalence estimation, and final figures.

* **Compute:**

  * Base PPI diagnostics and coverage scripts run on CPU-only machines.
  * CNN training is much more practical on a GPU, but can be scaled down (fewer epochs, smaller batch sizes) if needed.
