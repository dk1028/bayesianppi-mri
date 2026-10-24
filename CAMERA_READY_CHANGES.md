# Camera-ready changes, TMLR 9119

1. Export both Uniform and Jeffreys repeated-labeling summaries for both cohorts, with complete original-precision reference CSVs.
2. Match the original MRI notebook's resize/z-score pipeline, BCEWithLogitsLoss, sigmoid prediction, and cuDNN settings. No CNN was retrained for this release.
3. Make the SBC labeled examples a subset of the observed prediction pool, matching the paper's likelihood. Update its diagnostics and preserve the original results in manuscript revision notes.
4. Remove private outputs from the supplied notebooks and replace Windows-specific paths with repository-relative paths.
5. Plot the prior-free binary difference baseline once in two-prior coverage plots.
6. Set both the full-cohort and age 65--70 interval-width panels' y-axes to 0.1--0.6, with ticks every 0.1. The full-cohort n=10 PPI width (~0.626) is outside this display range; raw values and reported tables retain the complete values.
7. Include current PDF/PNG paper panels under `figs/paper/` and update the manuscript title and OpenReview link.

The four coverage summary tables were reproduced from the original notebook's estimator and experiment cells on the supplied fixed OOF predictions. Full-precision Uniform outputs agree exactly with the old reference files. See REPRODUCIBILITY.md for checks and scope.

## Alignment audit, 2026-10-24 (software 0.3.1)

- Restore the original case-study AUC bootstrap seed offset `+999` for the exact age 65--70 subset. Primary reference CSV values are unchanged.
- Pin scikit-learn to `1.5.1`, which reproduces the recorded subject-grouped CNN and threshold-audit folds; later versions can change those splits even at the same seed.
- Make the additional subject-cluster bootstrap opt-in, preserving scan-level bootstrap as the manuscript analysis; retain the original `label_frac` column in the exchangeability summary.
- Align auxiliary MRI figure scripts and prediction filenames with direct resize/z-score preprocessing and current subject-OOF outputs.
- Add a manuscript `preferred-citation` without prematurely assigning final journal/DOI metadata, distinguish software and manuscript dates, and document licensing and archived artifacts.
- Reconcile the manuscript-source tables with the full-precision references: five displayed rounding corrections, with no new estimates or changes to the reference CSVs.
- Rebuild the TMLR supplementary code archive from a pinned repository revision, excluding archived manuscript and legacy code. Remove obsolete patch-application instructions from the submission package.
