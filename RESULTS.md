# Authoritative TMLR revision results

These values are summary-level reference outputs. They contain no participant-level data.

## Full cohort QC

- Scans: **2116**
- Subjects: **503**
- Empirical AD prevalence: **0.3081285444**
- `P(A=1)` at threshold 0.5: **0.1417769376**
- OOF AUC: **0.6880143904**
- Accuracy: **0.7136105860**
- TPR: **0.2653374233**
- TNR: **0.9132513661**
- Subjects spanning multiple folds: **0**

## Age 65--70 QC

- Scans: **291**
- Subjects: **93**
- Empirical AD prevalence: **0.2371134021**
- `P(A=1)` at threshold 0.5: **0.1202749141**
- OOF AUC: **0.7717391304**
- Accuracy: **0.8350515464**
- TPR: **0.4057971014**
- TNR: **0.9684684685**
- Subjects spanning multiple folds: **0**

## Repeated-labeling coverage and mean interval width

Both priors use the same 500 subsets at budgets 10, 20, 40, 80. There are 5000 posterior draws and 1000 residual-bootstrap resamples. The prior-free baseline columns are identical across priors.

### Full cohort, Uniform prior

| n | CRE_cov | CRE_w | LabeledOnly_cov | LabeledOnly_w | BinaryDiff_cov | BinaryDiff_w | PPI_cov | PPI_w | PPIpp_cov | PPIpp_w |
|---|---|---|---|---|---|---|---|---|---|---|
| 10 | 0.970 | 0.449 | 0.946 | 0.482 | 0.886 | 0.570 | 0.948 | 0.626 | 0.938 | 0.585 |
| 20 | 0.964 | 0.345 | 0.960 | 0.369 | 0.930 | 0.419 | 0.948 | 0.413 | 0.942 | 0.396 |
| 40 | 0.944 | 0.259 | 0.942 | 0.272 | 0.942 | 0.309 | 0.940 | 0.277 | 0.918 | 0.268 |
| 80 | 0.962 | 0.189 | 0.952 | 0.197 | 0.946 | 0.220 | 0.950 | 0.197 | 0.948 | 0.191 |

### Age 65--70, Uniform prior

| n | CRE_cov | CRE_w | LabeledOnly_cov | LabeledOnly_w | BinaryDiff_cov | BinaryDiff_w | PPI_cov | PPI_w | PPIpp_cov | PPIpp_w |
|---|---|---|---|---|---|---|---|---|---|---|
| 10 | 0.980 | 0.414 | 0.970 | 0.457 | 0.778 | 0.390 | 0.902 | 0.510 | 0.898 | 0.478 |
| 20 | 0.982 | 0.306 | 0.958 | 0.345 | 0.928 | 0.301 | 0.936 | 0.342 | 0.932 | 0.335 |
| 40 | 0.978 | 0.228 | 0.958 | 0.254 | 0.960 | 0.235 | 0.968 | 0.235 | 0.956 | 0.231 |
| 80 | 0.986 | 0.166 | 0.982 | 0.182 | 0.974 | 0.167 | 0.976 | 0.170 | 0.970 | 0.166 |

### Full cohort, Jeffreys prior

| n | CRE_cov | CRE_w | LabeledOnly_cov | LabeledOnly_w | BinaryDiff_cov | BinaryDiff_w | PPI_cov | PPI_w | PPIpp_cov | PPIpp_w |
|---|---|---|---|---|---|---|---|---|---|---|
| 10 | 0.930 | 0.457 | 0.958 | 0.489 | 0.886 | 0.570 | 0.948 | 0.626 | 0.938 | 0.585 |
| 20 | 0.950 | 0.350 | 0.956 | 0.373 | 0.930 | 0.419 | 0.948 | 0.413 | 0.942 | 0.396 |
| 40 | 0.940 | 0.262 | 0.950 | 0.274 | 0.942 | 0.309 | 0.940 | 0.277 | 0.918 | 0.268 |
| 80 | 0.960 | 0.191 | 0.950 | 0.198 | 0.946 | 0.220 | 0.950 | 0.197 | 0.948 | 0.191 |

### Age 65--70, Jeffreys prior

| n | CRE_cov | CRE_w | LabeledOnly_cov | LabeledOnly_w | BinaryDiff_cov | BinaryDiff_w | PPI_cov | PPI_w | PPIpp_cov | PPIpp_w |
|---|---|---|---|---|---|---|---|---|---|---|
| 10 | 0.988 | 0.410 | 0.916 | 0.456 | 0.778 | 0.390 | 0.902 | 0.510 | 0.898 | 0.478 |
| 20 | 0.936 | 0.304 | 0.948 | 0.345 | 0.928 | 0.301 | 0.936 | 0.342 | 0.932 | 0.335 |
| 40 | 0.970 | 0.228 | 0.964 | 0.254 | 0.960 | 0.235 | 0.968 | 0.235 | 0.956 | 0.231 |
| 80 | 0.984 | 0.166 | 0.982 | 0.183 | 0.974 | 0.167 | 0.976 | 0.170 | 0.970 | 0.166 |

## Age-stratified AUC

| Scope | N | Subjects | Prevalence | AUC | 95% bootstrap CI |
|---|---:|---:|---:|---:|---:|
| Overall | 2116 | 503 | 0.308 | 0.688 | [0.664, 0.713] |
| 50--73 | 737 | 205 | 0.294 | 0.722 | [0.682, 0.762] |
| 74--79 | 695 | 218 | 0.291 | 0.700 | [0.655, 0.743] |
| 80--100 | 684 | 204 | 0.341 | 0.638 | [0.593, 0.683] |
| 65--70 | 291 | 93 | 0.237 | 0.772 | [0.699, 0.841] |

The Holm-adjusted age-bin permutation comparison was significant for 50--73 versus 80--100 (`p=0.0138`) and not significant for the other pairs.

## K-bin sensitivity

Full-cohort posterior means changed by at most approximately `0.0005` for K=4 or K=5 versus K=2. In the smaller age 65--70 subset, the largest shift was `0.0050` under the Uniform prior. Interval widths stayed within approximately 3% of the K=2 reference.

See `results/reference/kbin_paper_summary.csv`.

## Corrected simulation-based calibration

The camera-ready SBC draws the labeled examples without replacement from the observed prediction pool. This matches the likelihood in Proposition 1. The previous notebook generated a separate labeled A sample but did not add its A counts to the theta_A update; its archived diagnostics are not the final paper values.

For g, the corrected rank diagnostics are:

- Jeffreys: chi-square p=0.9621, KS p=0.8752.
- Uniform: chi-square p=0.7599, KS p=0.8408.

Full primitive-parameter results are in `results/reference/sbc_paper_summary.csv`. This is a model-based computational diagnostic, not evidence of robustness to longitudinal dependence or arbitrary selection.
