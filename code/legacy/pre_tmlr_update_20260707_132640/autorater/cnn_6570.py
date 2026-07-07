"""3D CNN autorater for the fixed 65--70 subset.

The script mirrors ``cnn_all.py`` but targets the 65--70 subset CSV.  To preserve
backward compatibility with older downstream scripts, it writes both the corrected
filename ``autorater_predictions_6570_all.csv`` and the legacy alias
``autorater_predictions_all1.csv``.
"""

from __future__ import annotations

from _shared import (
    AUTORATER_ROOT,
    CANDIDATE_6570_CSVS,
    SEED,
    choose_existing,
    load_label_csv,
    oof_predictions,
    set_seed,
    single_holdout_predictions,
)

CSV_PATH = choose_existing(CANDIDATE_6570_CSVS)
OOF_OUT = AUTORATER_ROOT / "autorater_predictions_6570_all.csv"
LEGACY_OUT = AUTORATER_ROOT / "autorater_predictions_all1.csv"
TEST_OUT = AUTORATER_ROOT / "autorater_predictions_6570_test.csv"


def main() -> None:
    set_seed(SEED)
    df = load_label_csv(CSV_PATH)
    print(f"Loaded {len(df)} rows from {CSV_PATH}")
    oof_df = oof_predictions(df, OOF_OUT, seed=SEED)
    oof_df.to_csv(LEGACY_OUT, index=False)
    print(f"Saved legacy-compatible copy to: {LEGACY_OUT}")
    single_holdout_predictions(df, TEST_OUT, seed=SEED)


if __name__ == "__main__":
    main()
