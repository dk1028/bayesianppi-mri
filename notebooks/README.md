# Notebook roles and provenance

The command-line scripts in `code/` are the supported reproduction entry points. These notebooks preserve the supplied source analyses with saved outputs removed and paths made repository-relative. Original and prepared SHA-256 hashes are recorded in [PROVENANCE.json](PROVENANCE.json).

| Notebook | Role | Current manuscript entry point |
| --- | --- | --- |
| `implementaion_coverage_all_for.ipynb` | MRI training, subject-grouped OOF predictions, scan-age matching | `code/autorater/train_subject_oof.py` and `code/data/attach_scan_age.py` |
| `implementaion_coverage_all_6570_revised.ipynb` | Repeated-labeling comparison for both cohorts and priors | `code/ppi/coverage_experiment.py` |
| `delong_permutation_revised.ipynb` | ROC, calibration, threshold and prevalence audits; permutation AUC comparisons | `code/ppi/case_study_analysis.py` |
| `k_chain_rule_revised.ipynb` | Conjugate quantile-bin sensitivity using all labels | `code/ppi/k_chain_rule.py` |
| `sbc_revised.ipynb` | Archived original separate-pool SBC design | Use `code/ppi/sbc.py` for the corrected final SBC design and reference results |

The historical spelling of the filenames is retained for traceability. The AUC comparisons are permutation tests, despite the `delong` filename. The SBC notebook's original diagnostics are not the final manuscript diagnostics.

The notebooks require authorized local ADNI inputs for the case study. Removing saved outputs does not make those private inputs part of this repository.

