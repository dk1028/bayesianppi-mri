from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


@dataclass(frozen=True)
class FoldSplit:
    fold: int
    train_idx: np.ndarray
    valid_idx: np.ndarray


def make_subject_grouped_folds(
    df: pd.DataFrame,
    *,
    subject_col: str = "subject_id",
    label_col: str = "H",
    n_splits: int = 5,
    seed: int = 42,
) -> list[FoldSplit]:
    required = {subject_col, label_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns for grouped folds: {sorted(missing)}")

    groups = df[subject_col].astype(str).to_numpy()
    labels = pd.to_numeric(df[label_col], errors="raise").astype(int).to_numpy()

    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )

    folds: list[FoldSplit] = []
    for fold, (train_idx, valid_idx) in enumerate(
        splitter.split(np.zeros(len(df)), labels, groups),
        start=1,
    ):
        train_subjects = set(groups[train_idx])
        valid_subjects = set(groups[valid_idx])
        overlap = train_subjects & valid_subjects
        if overlap:
            raise RuntimeError(
                f"Subject leakage in fold {fold}: {len(overlap)} overlapping subjects"
            )
        folds.append(FoldSplit(fold=fold, train_idx=train_idx, valid_idx=valid_idx))

    return folds


def validate_subject_fold_assignment(
    df: pd.DataFrame,
    *,
    subject_col: str = "subject_id",
    fold_col: str = "oof_fold",
) -> dict[str, int]:
    required = {subject_col, fold_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns for fold validation: {sorted(missing)}")

    fold_counts = df.groupby(subject_col)[fold_col].nunique()
    spanning = int((fold_counts > 1).sum())
    max_folds = int(fold_counts.max()) if len(fold_counts) else 0

    if spanning:
        raise RuntimeError(f"{spanning} subjects span multiple OOF folds")

    return {
        "subjects": int(df[subject_col].nunique()),
        "subjects_spanning_multiple_folds": spanning,
        "max_folds_per_subject": max_folds,
    }
