"""3D CNN autorater for the full cohort.

This revision removes the train/eval leakage issue by writing out-of-fold (OOF)
predictions to ``autorater_predictions_all.csv``.  A legacy holdout prediction file
(``autorater_predictions_test.csv``) is still produced for convenience.
"""

from __future__ import annotations

from pathlib import Path

from _shared import (
    AUTORATER_ROOT,
    CANDIDATE_FULL_CSVS,
    SEED,
    choose_existing,
    load_label_csv,
    oof_predictions,
    set_seed,
    single_holdout_predictions,
)

CSV_PATH = choose_existing(CANDIDATE_FULL_CSVS)
OOF_OUT = AUTORATER_ROOT / "autorater_predictions_all.csv"
TEST_OUT = AUTORATER_ROOT / "autorater_predictions_test.csv"


def main() -> None:
    set_seed(SEED)
    df = load_label_csv(CSV_PATH)
    print(f"Loaded {len(df)} rows from {CSV_PATH}")
    oof_predictions(df, OOF_OUT, seed=SEED)
    single_holdout_predictions(df, TEST_OUT, seed=SEED)


if __name__ == "__main__":
    main()
