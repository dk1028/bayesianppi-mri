from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from _shared import FIGS_ROOT

M = 10000
SEED = 2025


def simulate_prior(prior: str) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    if prior.lower() == "jeffreys":
        a = b = 0.5
    else:
        a = b = 1.0
    theta_a = rng.beta(a, b, size=M)
    theta_h1 = rng.beta(a, b, size=M)
    theta_h0 = rng.beta(a, b, size=M)
    return theta_a * theta_h1 + (1.0 - theta_a) * theta_h0


def main() -> None:
    for prior in ["uniform", "jeffreys"]:
        g = simulate_prior(prior)
        plt.figure(figsize=(6, 4))
        plt.hist(g, bins=50, density=True, alpha=0.85, edgecolor="black")
        plt.xlabel("g")
        plt.ylabel("Density")
        plt.title(f"Prior predictive for g ({prior})")
        plt.tight_layout()
        out = FIGS_ROOT / f"prior_predictive_g_{prior}.png"
        plt.savefig(out, dpi=300)
        plt.close()
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
