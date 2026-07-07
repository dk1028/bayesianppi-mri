from __future__ import annotations

import argparse
from pathlib import Path

from _shared import AUTORATER_ROOT, CSV_ROOT, load_label_csv, oof_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the 3D CNN with subject-grouped OOF folds."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=CSV_ROOT / "matched_cn_ad_labels_all.csv",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=AUTORATER_ROOT / "autorater_predictions_all_subject_oof.csv",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_label_csv(args.input_csv)
    oof_predictions(
        df,
        args.output_csv,
        seed=args.seed,
        n_folds=args.folds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )


if __name__ == "__main__":
    main()
