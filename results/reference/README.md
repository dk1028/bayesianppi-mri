# Manuscript reference outputs

These files contain cohort-level summaries or synthetic SBC ranks. They contain no participant-level predictions, identifiers, scan dates, or MRI paths.

| Files | Meaning |
| --- | --- |
| `coverage_all_subject_oof_scan_age.csv`, `coverage_6570_subject_oof_scan_age.csv` | Eight aggregate rows per cohort: two priors times four label budgets |
| `table_{full,6570}_{uniform,jeffreys}_summary.csv` | Four full-precision tables used by the manuscript |
| `coverage_full_uniform_reference.csv`, `coverage_6570_uniform_reference.csv` | Identical compatibility copies of the corresponding Uniform summary tables |
| `metrics_overall_age_auc_ci.csv` | Discrimination and fixed-threshold summaries; AUC CIs use scan-level bootstrap |
| `threshold_*.csv` | Fixed, descriptive Youden, held-out subject-grouped threshold, and threshold-bootstrap summaries |
| `pairwise_auc_perm_test_subject_oof.csv` | Exploratory scan-level permutation AUC comparisons with Holm correction |
| `calibration_brier_by_scope.csv`, `exchangeability_auc_random_labeling.csv` | Calibration and random-labeling sanity-check summaries |
| `prevalence_estimators_realdata_subject_oof_scan_age.csv` | Descriptive prevalence estimates using all labels; distinct from repeated-labeling coverage |
| `kbin_sensitivity_*.csv`, `kbin_paper_summary_revised.csv` | Full-precision conjugate K-bin summaries; K=2 is a quantile split, not threshold 0.5 |
| `kbin_paper_summary.csv` | Rounded presentation of the same K-bin results |
| `sbc_ranks_*.csv`, `sbc_*summary.csv` | Corrected labeled-subset SBC ranks and diagnostics |

`subject_oof` in a filename identifies the source prediction workflow, not participant rows in the public CSV. The full-precision CSVs are authoritative for rounding manuscript tables. In particular, avoid rounding a value twice.

The case-study AUC bootstrap uses seeds 2025 (overall), 2026/2027/2028 (age bins), and 3024 (the exact age 65--70 subset), matching the supplied notebook. See [the reproduction guide](../../REPRODUCIBILITY.md).

Reference values are fixed outputs from the supplied authorized OOF tables. Re-training the CNN on different hardware, a different image set, or a different dependency environment need not reproduce them exactly.

