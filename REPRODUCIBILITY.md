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
| Case-study audits | 2025; AUC scope offsets below |
| K-bin sensitivity | 2025 |
| SBC | 2025 |

The case-study scan-level AUC bootstrap matches the original notebook: overall uses 2025; ages 50--73, 74--79, and 80--100 use 2026, 2027, and 2028; the exact age 65--70 subset uses 3024 (`2025 + 999`). The optional cluster bootstrap uses separate seeds and is not the source of manuscript AUC intervals.

Each main analysis command writes a JSON configuration or embeds run settings in its result CSV. `configs/paper.yaml` is a readable settings record, not an automatically loaded configuration file.

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
3. The primary AUC confidence intervals reproduce the manuscript’s scan-level bootstrap. Add `--cluster-bootstrap 2000` to `case_study_analysis.py` for a separate subject-cluster sensitivity analysis. It is skipped by default; the extra `AUC_cluster_*` columns are then empty.
4. The pseudo-labeled versus unlabeled propensity analysis is a random-subsampling sanity check. It does not prove exchangeability under arbitrary real-world missingness.
5. K-bin Dirichlet--Beta updates remain conjugate under the independent-bin specification.
6. SBC validates computation under the assumed synthetic model, not the adequacy of the ADNI data-generating model.

## Verification

Run:

```bash
pytest -q
python code/ppi/sbc.py --output-dir /tmp/sbc-smoke --M 20 --S 100 --NA 100 --NH 20 --priors uniform --seed 1
```

After running the full repeated-labeling coverage command on the authorized OOF inputs, compare its four generated prior/cohort summary tables with:

```bash
python scripts/check_reference_results.py --results-root results
```

That command checks coverage summary exports; it does not compare every case-study, K-bin, or SBC file. See [results/reference/README.md](results/reference/README.md) for the reference-file map. Without ADNI inputs, `pytest -q` checks the public aggregate-to-summary exports and synthetic computations.

## Fixed OOF inputs

Exact downstream reproduction uses the supplied input rows in their original order. The following SHA-256 hashes identify those local-only inputs; the inputs themselves are not redistributed:

| File | SHA-256 |
| --- | --- |
| `autorater_predictions_all_subject_oof_with_scan_age.csv` | `26f964357e7b8adb417a9e0e31b58290fc9c57874b847e4103853eabeda823cf` |
| `autorater_predictions_6570_subject_oof_scan_age.csv` | `7595e736b0b37fe7b00795967db513f207f02ae60fd6e313cfceb7d0b5ef07b1` |

Most installation requirements are version ranges, not a complete historical lockfile. scikit-learn is pinned to `1.5.1`: its subject-grouped fold assignments reproduce the original CNN OOF table and threshold audit, while `1.8.0` gives different folds for the same seed. `requirements-paper-oof.txt` lists versions explicitly recorded by the original MRI notebook. Re-training on other hardware, including MPS, is supported as a new run but is not guaranteed to reproduce the fixed OOF probabilities byte for byte.


## Camera-ready reconciliation (2026-10-21)

The supplied original MRI notebook is the authority for preprocessing: resize the NIfTI directly to 64 cubed by trilinear interpolation, then use `(image - image.mean()) / (image.std() + 1e-5)`. Do not use the earlier patch's crop/pad and min-max pipeline to claim reproduction of the supplied OOF table. Training uses BCEWithLogitsLoss and prediction applies sigmoid. cuDNN deterministic/benchmark flags match the original notebook. No MRI training was rerun while preparing this package.

`coverage_experiment.py` exports four summary CSVs, one per cohort and prior. The complete repeated-labeling notebook was rerun on the provided fixed OOF predictions with its original settings. The previous Uniform summary CSVs agree exactly at full precision, and all 328 visible numeric notebook output cells match their saved output precision. The original notebook did not contain a complete dependency lockfile. Versions recorded in its installation output are provided in `requirements-paper-oof.txt`; use the fixed OOF prediction table for exact downstream reproduction.

For small label budgets, PPI and scalar power-tuned PPI use t_(n-1), not Satterthwaite degrees of freedom. The full prediction pool includes labeled rows; the working variance and scan-level labeling design are evaluated empirically. Neither this nor the original subject-grouped split provides cluster-robust uncertainty for repeated visits.

The final SBC diagnostic uses a labeled subset of the observed A pool. Its ranks and summaries are included with the manuscript. The K-bin sensitivity uses all available labels and quantile bins; its K=2 partition differs from the primary threshold of 0.5.

The Jupyter notebooks under `notebooks/` preserve the supplied estimator code, have private row outputs removed, and use repository-relative paths. Their source hashes are recorded in `notebooks/PROVENANCE.json`. The SBC notebook archives the original separate-pool diagnostic; use `code/ppi/sbc.py` or `make sbc` for the corrected labeled-subset design and final reference diagnostics. The command-line Python scripts remain the main reproducibility entry points.
