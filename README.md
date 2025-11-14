# Bayesian Prediction-Powered Inference: Conjugate Base Model and MCMC Extensions

> **“Bayesian Prediction-Powered Inference with MCMC: Methods and a Medical Imaging Case Study”**
> by Dowoo Kim and Russell Steele

The goal of this repository is to make it easy to:

1. Reproduce the **toy Bayesian prediction-powered inference (PPI)** experiments.
2. Reproduce the **ADNI MRI case study** (3D CNN autorater + PPI / CRE / Naïve / Difference estimators).
3. Regenerate the **coverage, interval-width, and example figures** used in the manuscript.

The structure and README are written to follow common practices of other TMLR artifacts: clear layout, instructions for data access, and a minimal but complete path from raw data to final figures.

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
    │   ├─ change_dicom.py     # DICOM → NIfTI (dcm2niix wrapper)
    │   ├─ process_all.py      # Build matched CN/AD table for full cohort
    │   ├─ process_6570.py     # Build matched CN/AD table for 65–70 subset
    │   ├─ cnn_all.py          # 3D CNN autorater (full cohort)
    │   ├─ cnn_6570.py         # 3D CNN autorater (65–70 subset)
    │   ├─ autorater.py        # Example AD vs CN cases and bar plots
    │   ├─ model_ADCN.py       # Additional model / architecture figure script
    │   └─ fig1_pipeline.py    # NIfTI → slice → preprocessing pipeline figure
    ├─ ppi/
    │   ├─ histogram_posterior.py           # Toy Bernoulli chain-rule coverage experiment
    │   ├─ prior_predictive_checks.py       # Prior predictive checks for Bayesian PPI
    │   ├─ implementaion_coverage_all.py    # Coverage & width (full cohort, Colab-oriented)
    │   ├─ implementaion_coverage_6570.py   # Coverage & width (65–70 subset, Colab-oriented)
    │   ├─ coverage_intervalall.py          # Plot coverage vs width (full cohort)
    │   ├─ coverage_interval6570.py         # Plot coverage vs width (65–70 subset)
    │   ├─ k_chain_rule.py                  # K-bin chain-rule estimator experiments
    │   ├─ sbc.py                           # Simulation-based calibration (SBC) for PPI models
    │   └─ age_analysis.py                  # Age-stratified performance analysis (Colab/desktop)
    └─ utils/
        └─ paths_example.py                 # Example of repo-relative paths (optional helper)
```

> **Important:** raw MRI data and ADNI metadata are *not* included, due to licensing.
> The scripts are written so that, given access to ADNI and the CSV export from ADNI Advanced Search, a reader can reproduce the case study.

---

## 2. Core ideas and components

The repository contains three main components that correspond to the manuscript:

1. **Toy PPI experiments** (no ADNI data needed)

   * `code/ppi/histogram_posterior.py` implements a simple Bernoulli PPI experiment.
   * `code/ppi/prior_predictive_checks.py` runs prior predictive checks for the Bayesian chain-rule model.
   * `code/ppi/k_chain_rule.py` and `code/ppi/sbc.py` explore K-bin chain-rule estimators and SBC.

2. **MRI autorater (ADNI case study)**

   * `code/autorater/change_dicom.py` converts ADNI DICOMs to NIfTI using `dcm2niix`.
   * `code/autorater/process_all.py` and `process_6570.py` join the ADNI metadata CSV with NIfTI paths.
   * `code/autorater/cnn_all.py` and `cnn_6570.py` train a 3D CNN autorater for CN vs AD classification.

3. **Coverage/interval-width and age-stratified analyses**

   * `code/ppi/implementaion_coverage_all.py` and `implementaion_coverage_6570.py` run coverage simulations using autorater predictions and clinician labels.
   * `code/ppi/coverage_intervalall.py` and `coverage_interval6570.py` recreate the coverage/width plots.
   * `code/ppi/age_analysis.py` examines AUC and ROC curves by age band (e.g., 50–73, 74–79, 80–100).

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

* `numpy`, `scipy`, `pandas`, `matplotlib`, `seaborn`
* `torch`, `torchvision`, possibly `monai` (depending on the final CNN implementation)
* `pymc>=5` (for MCMC-based Bayesian inference; chain-rule estimator implementation)
* `scikit-learn` (ROC, AUC, calibration curves)
* `tqdm` for progress bars
* `nibabel`, `scikit-image` for MRI loading and preprocessing

> **Note:** Toy PPI experiments (`histogram_posterior.py`, `prior_predictive_checks.py`, `k_chain_rule.py`, `sbc.py`) only require PyMC, NumPy, SciPy, Matplotlib and do not depend on PyTorch or MRI data.

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

all file input/output is resolved relative to the repository root, and you do **not** need to change `C:\...` style absolute paths.

---

## 6. How to run the experiments

### 6.1. Toy PPI experiments (no ADNI required)

These are the simplest to reproduce and do not require any medical data.

1. **Posterior histogram experiment** (`histogram_posterior.py`)

   ```bash
   cd bayesianppi-mri
   python code/ppi/histogram_posterior.py
   ```

   This script:

   * Simulates multiple datasets under a Bernoulli chain-rule model with known true parameters.
   * Fits the Bayesian chain-rule model using PyMC.
   * Computes empirical coverage of 95% credible intervals.
   * Plots histograms of posterior means and medians of the target quantity.

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

   This script explores how the number of bins (K) affects the chain-rule estimator’s behaviour, using conjugate Dirichlet–Beta updates and simulated predictions.

4. **Simulation-based calibration** (`sbc.py`)

   ```bash
   python code/ppi/sbc.py --outdir results/coverage/sbc
   ```

   This script runs SBC experiments to check whether the Bayesian PPI models are well-calibrated, using rank histograms and diagnostic summaries.

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

These matched tables are the main inputs to the CNN and PPI coverage scripts.

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
* Trains a 3D CNN (using PyTorch) to predict AD vs CN.
* Saves autorater predictions and (optionally) model checkpoints to `results/autorater/`.

The PPI coverage experiments use these autorater predictions as the “machine learning predictor” (A) in the chain-rule estimator.

### 6.5. Coverage and interval-width experiments (ADNI-based)

The scripts `implementaion_coverage_all.py` and `implementaion_coverage_6570.py` were originally written in a Colab style (with some `!pip` commands). In this repository, we keep them as standalone scripts, but recommend running them either:

* In a local Python environment after ensuring all dependencies are installed, **or**
* In Google Colab (by uploading the script and mounting the repository or copying paths).

```bash
cd bayesianppi-mri

# Full cohort coverage
python code/ppi/implementaion_coverage_all.py

# 65–70 coverage
python code/ppi/implementaion_coverage_6570.py
```

These scripts:

* Read autorater predictions and clinician labels.
* Implement the Bayesian PPI chain-rule estimator, Naïve estimator, and Difference estimator.
* For several label budgets (e.g. n = 10, 20, 40, 80), estimate coverage and average width of 95% intervals.
* Save summary tables to `results/coverage/`.

Once the coverage tables are available, you can regenerate the figures with:

```bash
python code/ppi/coverage_intervalall.py
python code/ppi/coverage_interval6570.py
```

The plotting scripts use the reported summary numbers (either hard-coded or loaded from CSV) to reproduce the coverage vs width figures in the paper.

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
* Optionally performs permutation tests to compare performance across bands.
* Saves intermediate results and plots under `results/age_analysis/` and `figs/`.

---

## 7. Figures

The following scripts are used to recreate figures in the manuscript:

* `code/autorater/fig1_pipeline.py` – data-processing pipeline figure: NIfTI → slice → preprocessing.
* `code/autorater/autorater.py` – example AD vs CN volumes and autorater outputs.
* `code/autorater/model_ADCN.py` – CNN architecture / model diagram.
* `code/ppi/coverage_intervalall.py` – coverage vs width (full cohort).
* `code/ppi/coverage_interval6570.py` – coverage vs width (65–70 subset).

Each script writes `.png` (or `.pdf`) files to the `figs/` directory.

---

## 8. Reproducibility checklist (informal)

In the spirit of the TMLR reproducibility checklist, we highlight the following:

* **Environment:**

  * We developed and tested the code with Python 3.11 on a Linux environment, and also on Windows 10 using Git Bash.
  * `requirements.txt` lists all Python dependencies.

* **Randomness & seeds:**

  * Toy PPI scripts (`histogram_posterior.py`, `prior_predictive_checks.py`, `k_chain_rule.py`, `sbc.py`) set random seeds where appropriate for repeatable simulations.
  * CNN training scripts may include a fixed seed for PyTorch, but exact weight realizations can still vary across hardware and CuDNN versions.

* **Data availability:**

  * Toy experiments are fully self-contained.
  * ADNI-based experiments require access to ADNI MRI and metadata, which we do not redistribute.
  * We document exactly how the ADNI metadata CSV is constructed via Advanced Search (Project/Phase/Modality/Image Description filters).

* **Code completeness:**

  * All main experiments described in the paper are covered by scripts in `code/autorater` and `code/ppi`.
  * For ADNI, we provide the full path from the Advanced Search CSV and DICOM images to matched label tables, autorater predictions, coverage estimates, and final figures.

* **Compute:**

  * Toy PPI experiments run in minutes on a CPU-only machine.
  * CNN training requires a GPU for practical runtime (e.g., a single commodity GPU, such as NVIDIA RTX-series), but can be scaled down (fewer epochs, smaller batch sizes) if needed.

---

## 9. Citing this work

If you find this code useful in your own research, please consider citing the TMLR paper (citation details to be updated once available):

```text
@article{kim202Xbayesianppi,
  title   = {Bayesian Prediction-Powered Inference: Conjugate Base Model and MCMC Extensions},
  author  = {Kim, Dowoo and Steele, Russell},
  journal = {Transactions on Machine Learning Research},
  year    = {202X}
}
```

---

## 10. Contact

For questions, bug reports, or suggestions, please open a GitHub issue on the repository or contact the corresponding author.

This repository is intended to make the Bayesian PPI methodology and the ADNI MRI case study as transparent and reproducible as permitted under the ADNI data-use agreement.
