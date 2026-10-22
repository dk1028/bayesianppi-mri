# Reproducibility guide

## Authoritative analysis version

The authoritative TMLR revision workflow uses:

- subject-grouped 5-fold OOF predictions,
- scan-level age matched by subject and acquisition date,
- full cohort: 2116 scans, 503 subjects,
- age 65--70: 291 scans, 93 subjects,
- fixed deployment threshold `0.5`,
- repeated-labeling budgets `10, 20, 40, 80`,
- 500 repetitions,
- Uniform and Jeffreys priors.

## Random seeds

| Component | Seed |
|---|---:|
| Autorater OOF | 42 |
| Coverage and width | 2025 |
| Case-study audits | 2025 |
| K-bin sensitivity | 2025 |
| SBC | 2025 |

Every command writes a JSON configuration or embeds run settings in its result CSV.

## macOS hardware notes

- CNN training is the expensive step.
- Apple Silicon uses the PyTorch MPS backend when available.
- All downstream PPI/CRE analyses run comfortably on CPU.
- Synthetic SBC and K-bin sensitivity typically finish within seconds or minutes.

## Manuscript-to-code map

| Manuscript component | Script |
|---|---|
| Subject-level OOF predictions | `code/autorater/train_subject_oof.py` |
| Scan-level age merge | `code/data/attach_scan_age.py` |
| Coverage/width tables and figures | `code/ppi/coverage_experiment.py` |
| ROC, age, threshold, calibration, prevalence | `code/ppi/case_study_analysis.py` |
| K-bin sensitivity | `code/ppi/k_chain_rule.py` |
| SBC | `code/ppi/sbc.py` |

## Important statistical scope notes

1. Subject grouping is enforced during CNN training and threshold-selection cross-validation.
2. The manuscript repeated-labeling experiment samples labeled scan rows, matching the reported label-budget interpretation.
3. The primary AUC confidence intervals reproduce the manuscript’s scan-level bootstrap. Researchers interested in within-subject dependence should additionally run a subject-cluster bootstrap sensitivity analysis.
4. The pseudo-labeled versus unlabeled propensity analysis is a random-subsampling sanity check. It does not prove exchangeability under arbitrary real-world missingness.
5. K-bin Dirichlet--Beta updates remain conjugate under the independent-bin specification.
6. SBC validates computation under the assumed synthetic model, not the adequacy of the ADNI data-generating model.

## Verification

Run:

```bash
pytest -q
python code/ppi/sbc.py --output-dir /tmp/sbc-smoke --M 20 --S 100 --NA 100 --NH 20 --priors uniform --seed 1
```

Then compare generated outputs to `results/reference/` with:

```bash
python scripts/check_reference_results.py --results-root results
```


## Camera-ready reconciliation (2026-10-21)

The supplied original MRI notebook is the authority for preprocessing: resize the NIfTI directly to 64 cubed by trilinear interpolation, then use `(image - image.mean()) / (image.std() + 1e-5)`. Do not use the earlier patch's crop/pad and min-max pipeline to claim reproduction of the supplied OOF table. Training uses BCEWithLogitsLoss and prediction applies sigmoid. cuDNN deterministic/benchmark flags match the original notebook. No MRI training was rerun while preparing this package.

`coverage_experiment.py` exports four summary CSVs, one per cohort and prior. The complete repeated-labeling notebook was rerun on the provided fixed OOF predictions with its original settings. The previous Uniform summary CSVs agree exactly at full precision, and all 328 visible numeric notebook output cells match their saved output precision. The original notebook did not contain a complete dependency lockfile. Versions recorded in its installation output are provided in `requirements-paper-oof.txt`; use the fixed OOF prediction table for exact downstream reproduction.

For small label budgets, PPI and scalar power-tuned PPI use t_(n-1), not Satterthwaite degrees of freedom. The full prediction pool includes labeled rows; the working variance and scan-level labeling design are evaluated empirically. Neither this nor the original subject-grouped split provides cluster-robust uncertainty for repeated visits.

The final SBC diagnostic uses a labeled subset of the observed A pool. Its ranks and summaries are included with the manuscript. The K-bin sensitivity uses all available labels and quantile bins; its K=2 partition differs from the primary threshold of 0.5.

The Jupyter notebooks under `notebooks/` preserve the supplied estimator code, have private row outputs removed, and use repository-relative paths. Their source hashes are recorded in `notebooks/PROVENANCE.json`. The SBC notebook archives the original separate-pool diagnostic; use `code/ppi/sbc.py` or `make sbc` for the corrected labeled-subset design and final reference diagnostics. The command-line Python scripts remain the main reproducibility entry points.
