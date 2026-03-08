from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from _shared import FULL_PRED_CANDIDATES, FIGS_ROOT, choose_existing, cre_posterior_draws, load_prediction_csv

CSV_PATH = choose_existing(FULL_PRED_CANDIDATES)
M = 50
S = 5000
N_LABELS = 40
ALPHA = 0.5
BETA = 0.5
SEED = 2025


def main() -> None:
    df = load_prediction_csv(CSV_PATH)
    rng = np.random.default_rng(SEED)
    means = []
    medians = []
    for m in range(M):
        idx = rng.choice(len(df), size=N_LABELS, replace=False)
        draws = cre_posterior_draws(df, idx, ALPHA, BETA, S, np.random.default_rng(SEED + m))
        means.append(float(draws.mean()))
        medians.append(float(np.median(draws)))

    g_true = float(df["H"].mean())
    for values, fname, title in [
        (means, "posterior_mean.png", "Posterior means across M=50 datasets."),
        (medians, "posterior_median.png", "Posterior medians across M=50 datasets."),
    ]:
        plt.figure(figsize=(6, 4))
        plt.hist(values, bins=12, edgecolor="black")
        plt.axvline(g_true, linestyle="--", linewidth=2)
        plt.title(title)
        plt.xlabel("g")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(FIGS_ROOT / fname, dpi=300)
        plt.close()
        print(f"Saved {FIGS_ROOT / fname}")


if __name__ == "__main__":
    main()
