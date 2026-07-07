from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "autorater"))

from fold_utils import make_subject_grouped_folds, validate_subject_fold_assignment


def test_subjects_do_not_cross_folds() -> None:
    rows = []
    for subject_idx in range(30):
        label = subject_idx % 2
        for scan_idx in range(2):
            rows.append(
                {
                    "subject_id": f"S{subject_idx:03d}",
                    "H": label,
                    "scan": scan_idx,
                }
            )
    df = pd.DataFrame(rows)
    folds = make_subject_grouped_folds(df, n_splits=5, seed=42)

    assignment = [-1] * len(df)
    for split in folds:
        for idx in split.valid_idx:
            assignment[int(idx)] = split.fold
    df["oof_fold"] = assignment

    qc = validate_subject_fold_assignment(df)
    assert qc["subjects_spanning_multiple_folds"] == 0
    assert qc["max_folds_per_subject"] == 1
