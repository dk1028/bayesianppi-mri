from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _shared import FIGS_ROOT, cre_posterior_draws, load_prediction_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Posterior mean/median resampling figure.")
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("results/autorater/autorater_predictions_all_subject_oof_with_scan_age.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=FIGS_ROOT)
    parser.add_argument("--repetitions", type=int, default=50)
    parser.add_argument("--draws", type=int, default=5000)
    parser.add_argument("--labels", type=int, default=40)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=2025)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = load_prediction_csv(args.input_csv)
    rng = np.random.default_rng(args.seed)
    means, medians = [], []

    for rep in range(args.repetitions):
        idx = rng.choice(len(df), size=args.labels, replace=False)
        draws = cre_posterior_draws(
            df,
            idx,
            args.alpha,
            args.beta,
            args.draws,
            np.random.default_rng(args.seed + rep),
        )
        means.append(float(draws.mean()))
        medians.append(float(np.median(draws)))

    g_true = float(df["H"].mean())
    for values, fname, title in [
        (means, "posterior_mean.png", "Posterior means across repeated labeled subsets"),
        (medians, "posterior_median.png", "Posterior medians across repeated labeled subsets"),
    ]:
        plt.figure(figsize=(6, 4))
        plt.hist(values, bins=12, edgecolor="black")
        plt.axvline(g_true, linestyle="--", linewidth=2, label="Empirical prevalence")
        plt.title(title)
        plt.xlabel("g")
        plt.ylabel("Count")
        plt.legend()
        plt.tight_layout()
        plt.savefig(args.output_dir / fname, dpi=300, bbox_inches="tight")
        plt.close()


if __name__ == "__main__":
    main()
