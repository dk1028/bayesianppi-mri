# Data preparation and local file schema

## ADNI access

The real-data case study uses data from the Alzheimer’s Disease Neuroimaging Initiative (ADNI). Raw images and participant-level metadata are not redistributed. Users must obtain ADNI authorization and comply with the ADNI Data Use Agreement.

## Local-only inputs

The following files are expected locally and are ignored by Git:

```text
data/raw_dicom/
data/nifti/
data/csv/adni_mrpage_t1_search.csv
data/csv/matched_cn_ad_labels_all.csv
```

### `matched_cn_ad_labels_all.csv`

Required columns:

| Column | Meaning |
|---|---|
| `subject_id` | ADNI participant identifier |
| `nifti_path` | Absolute path or path relative to `data/nifti/` |
| `label` | `AD` or `CN` |
| `Acq_Date` | Scan acquisition date |

Optional columns are retained.

### ADNI metadata export

By default, `attach_scan_age.py` expects:

| Column | Meaning |
|---|---|
| `Subject` | ADNI participant identifier |
| `Acq Date` | Acquisition date |
| `Age` | Age at acquisition |
| `Group` | ADNI diagnosis group, used only for QC |

Column names can be overridden with command-line options.

## Exact scan-level age matching

Age is joined by normalized subject identifier and normalized acquisition date. Duplicate metadata rows for the same subject-date pair are allowed only if they agree on age. The merge uses `validate="many_to_one"` and fails if any OOF row has a missing age.

The age 65--70 table is created after this exact merge:

```python
full_with_age[full_with_age["Age"].between(65, 70, inclusive="both")]
```

A separate age-specific CNN is not used for the revised manuscript.

## Public synthetic fixture

`data/example/synthetic_oof_example.csv` contains fake identifiers and simulated values that exercise the analysis scripts without ADNI data. It is not intended to reproduce manuscript numbers.
