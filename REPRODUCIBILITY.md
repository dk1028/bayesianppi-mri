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
