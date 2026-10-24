from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "ppi"))
spec = importlib.util.spec_from_file_location(
    "case_study_analysis", ROOT / "code" / "ppi" / "case_study_analysis.py"
)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_case_study_age_subset_matches_notebook_bootstrap(tmp_path, monkeypatch):
    rng = np.random.default_rng(17)
    data = pd.DataFrame(
        {
            "subject_id": [f"synthetic-{i:03d}" for i in range(64)],
            "H": np.tile([0, 1], 32),
            "autorater_prediction": rng.uniform(0.05, 0.95, 64),
            "Age": np.repeat([60, 67, 76, 85], 16),
            "oof_fold": np.tile(np.arange(1, 5), 16),
        }
    )
    input_csv = tmp_path / "synthetic.csv"
    output = tmp_path / "case-study"
    data.to_csv(input_csv, index=False)

    def unexpected_cluster_bootstrap(*args, **kwargs):
        raise AssertionError("The additional cluster bootstrap must be opt-in")

    monkeypatch.setattr(module, "cluster_bootstrap_auc_ci", unexpected_cluster_bootstrap)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "case_study_analysis.py",
            "--input-csv", str(input_csv),
            "--output-dir", str(output),
            "--bootstrap", "60",
            "--permutations", "8",
            "--threshold-bootstrap", "8",
            "--seed", "2025",
        ],
    )
    module.main()

    # Independent replay of the supplied notebook's exact-subset bootstrap:
    # this scope intentionally uses SEED+999, rather than the age-bin counter.
    sub = data.loc[data.Age.between(65, 70)]
    y = sub.H.to_numpy()
    p = sub.autorater_prediction.to_numpy()
    oracle_rng = np.random.default_rng(2025 + 999)
    aucs = []
    for _ in range(60):
        idx = oracle_rng.integers(0, len(sub), size=len(sub))
        if np.unique(y[idx]).size == 2:
            aucs.append(roc_auc_score(y[idx], p[idx]))
    expected = np.quantile(aucs, [0.025, 0.975])

    metrics = pd.read_csv(output / "metrics_overall_age_auc_ci.csv")
    row = metrics.loc[metrics.scope.eq("Age 65-70")].iloc[0]
    np.testing.assert_allclose([row.AUC_lo, row.AUC_hi], expected, atol=1e-14, rtol=0)
    assert metrics[["AUC_cluster_lo", "AUC_cluster_hi"]].isna().all().all()
    config = json.loads((output / "run_config.json").read_text())
    assert config["auc_bootstrap_seeds"]["Age 65-70"] == 3024
    assert config["cluster_bootstrap"] == 0

