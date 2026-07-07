from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chisquare, kstest

from _shared import REPO_ROOT, SBC_ROOT



def draw_beta_post(alpha: float, beta: float, size: int, rng: np.random.Generator) -> np.ndarray:
    return rng.beta(alpha, beta, size=size)



def plot_rank_hist_grouped_with_pvals(ranks: np.ndarray, s_draws: int, out_png: Path, title: str, b: int = 20) -> None:
    u = (ranks + 0.5) / (s_draws + 1.0)
    bins = np.linspace(0.0, 1.0, b + 1)
    counts, _ = np.histogram(u, bins=bins)
    expected = np.full(b, len(u) / b)
    _, chi2_p = chisquare(counts, expected)
    _, ks_p = kstest(u, "uniform")

    plt.figure(figsize=(5.2, 3.4))
    xs = 0.5 * (bins[1:] + bins[:-1])
    width = 1.0 / b
    plt.bar(xs, counts, width=width * 0.9, align="center")
    plt.hlines(expected[0], 0, 1, linestyles="dashed", linewidth=1.2)
    plt.xlim(0, 1)
    plt.xlabel(r"Normalized rank $u \in [0,1)$")
    plt.ylabel("Count per bin")
    plt.title(title)
    txt = (r"$\chi^2$ p={:.2f}; KS p={:.2f}").format(chi2_p, ks_p)
    plt.text(0.99, 0.97, txt, ha="right", va="top", transform=plt.gca().transAxes, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()



def run_sbc_cre(m: int = 500, s: int = 1000, na: int = 1000, nh: int = 100, prior: str = "jeffreys", seed: int = 2025):
    rng = np.random.default_rng(seed)
    if prior.lower() == "jeffreys":
        aA = bA = a1 = b1 = a0 = b0 = 0.5
    else:
        aA = bA = a1 = b1 = a0 = b0 = 1.0

    ranks_thetaA = np.empty(m, dtype=int)
    ranks_thetaH1 = np.empty(m, dtype=int)
    ranks_thetaH0 = np.empty(m, dtype=int)
    ranks_g = np.empty(m, dtype=int)

    for i in range(m):
        thetaA_star = rng.beta(aA, bA)
        thetaH1_star = rng.beta(a1, b1)
        thetaH0_star = rng.beta(a0, b0)
        g_star = thetaA_star * thetaH1_star + (1.0 - thetaA_star) * thetaH0_star

        A_unlab = rng.binomial(1, thetaA_star, size=na)
        A_lab = rng.binomial(1, thetaA_star, size=nh)
        H_lab = np.where(
            A_lab == 1,
            rng.binomial(1, thetaH1_star, size=nh),
            rng.binomial(1, thetaH0_star, size=nh),
        )

        nA = A_unlab.sum()
        n11 = np.sum((A_lab == 1) & (H_lab == 1))
        n10 = np.sum((A_lab == 1) & (H_lab == 0))
        n01 = np.sum((A_lab == 0) & (H_lab == 1))
        n00 = np.sum((A_lab == 0) & (H_lab == 0))

        thetaA_draws = draw_beta_post(aA + nA, bA + (na - nA), s, rng)
        thetaH1_draws = draw_beta_post(a1 + n11, b1 + n10, s, rng)
        thetaH0_draws = draw_beta_post(a0 + n01, b0 + n00, s, rng)
        g_draws = thetaA_draws * thetaH1_draws + (1.0 - thetaA_draws) * thetaH0_draws

        ranks_thetaA[i] = np.sum(thetaA_draws < thetaA_star)
        ranks_thetaH1[i] = np.sum(thetaH1_draws < thetaH1_star)
        ranks_thetaH0[i] = np.sum(thetaH0_draws < thetaH0_star)
        ranks_g[i] = np.sum(g_draws < g_star)

    return {
        "r_thetaA": ranks_thetaA,
        "r_thetaH1": ranks_thetaH1,
        "r_thetaH0": ranks_thetaH0,
        "r_g": ranks_g,
        "S": s,
        "M": m,
    }



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior", default="jeffreys", choices=["uniform", "jeffreys"])
    parser.add_argument("--M", type=int, default=500)
    parser.add_argument("--S", type=int, default=1000)
    args = parser.parse_args()

    SBC_ROOT.mkdir(parents=True, exist_ok=True)
    out = run_sbc_cre(m=args.M, s=args.S, prior=args.prior)
    suffix = f"{args.prior}_b20"
    plot_rank_hist_grouped_with_pvals(out["r_thetaH1"], out["S"], SBC_ROOT / f"sbc_thetaH1_{suffix}.png", r"$\theta_{H\mid1}$")
    plot_rank_hist_grouped_with_pvals(out["r_thetaH0"], out["S"], SBC_ROOT / f"sbc_thetaH0_{suffix}.png", r"$\theta_{H\mid0}$")
    plot_rank_hist_grouped_with_pvals(out["r_thetaA"], out["S"], SBC_ROOT / f"sbc_thetaA_{suffix}.png", r"$\theta_A$")
    plot_rank_hist_grouped_with_pvals(out["r_g"], out["S"], SBC_ROOT / f"sbc_g_{suffix}.png", r"$g$")
    print(f"Saved SBC figures to {SBC_ROOT}")


if __name__ == "__main__":
    main()
