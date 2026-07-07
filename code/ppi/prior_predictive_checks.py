from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _shared import FIGS_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prior predictive checks for g.")
    parser.add_argument("--output-dir", type=Path, default=FIGS_ROOT)
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2025)
    return parser.parse_args()


def simulate_prior(prior: str, *, draws: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = b = 0.5 if prior == "jeffreys" else 1.0
    theta_a = rng.beta(a, b, size=draws)
    theta_h1 = rng.beta(a, b, size=draws)
    theta_h0 = rng.beta(a, b, size=draws)
    return theta_a * theta_h1 + (1 - theta_a) * theta_h0


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for offset, prior in enumerate(["uniform", "jeffreys"]):
        g = simulate_prior(prior, draws=args.draws, seed=args.seed + offset)
        plt.figure(figsize=(6, 4))
        plt.hist(g, bins=50, density=True, alpha=0.85, edgecolor="black")
        plt.xlabel("g")
        plt.ylabel("Density")
        plt.title(f"Prior predictive for g ({prior})")
        plt.tight_layout()
        plt.savefig(args.output_dir / f"prior_predictive_g_{prior}.png", dpi=300)
        plt.close()


if __name__ == "__main__":
    main()
