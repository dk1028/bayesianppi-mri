from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chisquare, kstest
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulation-based calibration for conjugate CRE.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--M", type=int, default=500)
    parser.add_argument("--S", type=int, default=1000)
    parser.add_argument("--NA", type=int, default=2116)
    parser.add_argument("--NH", type=int, default=100)
    parser.add_argument("--bins", type=int, default=20)
    parser.add_argument("--priors", nargs="+", default=["jeffreys", "uniform"])
    parser.add_argument("--seed", type=int, default=2025)
    return parser.parse_args()


def prior_params(prior: str) -> dict[str, float]:
    if prior == "jeffreys":
        value = 0.5
    elif prior == "uniform":
        value = 1.0
    else:
        raise ValueError(f"Unknown prior: {prior}")
    return {"aA": value, "bA": value, "a1": value, "b1": value, "a0": value, "b0": value}


def run_sbc_cre(
    *,
    M: int,
    S: int,
    NA: int,
    NH: int,
    prior: str,
    seed: int,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    rng = np.random.default_rng(seed)
    pp = prior_params(prior)
    ranks = {name: np.empty(M, dtype=int) for name in ["thetaA", "thetaH1", "thetaH0", "g"]}
    truth_rows: list[dict[str, float | int]] = []

    for m in tqdm(range(M), desc=f"SBC {prior}"):
        thetaA = rng.beta(pp["aA"], pp["bA"])
        thetaH1 = rng.beta(pp["a1"], pp["b1"])
        thetaH0 = rng.beta(pp["a0"], pp["b0"])
        g_true = thetaA * thetaH1 + (1 - thetaA) * thetaH0

        A_unlab = rng.binomial(1, thetaA, size=NA)
        A_lab = rng.binomial(1, thetaA, size=NH)
        H_lab = np.where(
            A_lab == 1,
            rng.binomial(1, thetaH1, size=NH),
            rng.binomial(1, thetaH0, size=NH),
        )

        nA = int(A_unlab.sum())
        n11 = int(np.sum((A_lab == 1) & (H_lab == 1)))
        n10 = int(np.sum((A_lab == 1) & (H_lab == 0)))
        n01 = int(np.sum((A_lab == 0) & (H_lab == 1)))
        n00 = int(np.sum((A_lab == 0) & (H_lab == 0)))

        dA = rng.beta(pp["aA"] + nA, pp["bA"] + NA - nA, size=S)
        dH1 = rng.beta(pp["a1"] + n11, pp["b1"] + n10, size=S)
        dH0 = rng.beta(pp["a0"] + n01, pp["b0"] + n00, size=S)
        dg = dA * dH1 + (1 - dA) * dH0

        ranks["thetaA"][m] = int(np.sum(dA < thetaA))
        ranks["thetaH1"][m] = int(np.sum(dH1 < thetaH1))
        ranks["thetaH0"][m] = int(np.sum(dH0 < thetaH0))
        ranks["g"][m] = int(np.sum(dg < g_true))

        truth_rows.append(
            {
                "rep": m,
                "thetaA_star": thetaA,
                "thetaH1_star": thetaH1,
                "thetaH0_star": thetaH0,
                "g_star": g_true,
                "nA": nA,
                "n11": n11,
                "n10": n10,
                "n01": n01,
                "n00": n00,
            }
        )

    return ranks, pd.DataFrame(truth_rows)


def summarize_ranks(ranks: np.ndarray, *, S: int, bins: int) -> dict[str, float | int]:
    u = (ranks + 0.5) / (S + 1.0)
    counts, _ = np.histogram(u, bins=np.linspace(0, 1, bins + 1))
    expected = np.full(bins, len(ranks) / bins)
    chi_stat, chi_p = chisquare(counts, expected)
    ks_stat, ks_p = kstest(u, "uniform")
    return {
        "n_ranks": len(ranks),
        "n_bins": bins,
        "expected_per_bin": len(ranks) / bins,
        "chi2_stat": float(chi_stat),
        "chi2_p": float(chi_p),
        "ks_stat": float(ks_stat),
        "ks_p": float(ks_p),
        "min_count": int(counts.min()),
        "max_count": int(counts.max()),
    }


def plot_rank_histogram(
    ranks: np.ndarray,
    *,
    S: int,
    bins: int,
    title: str,
    output: Path,
) -> dict[str, float | int]:
    stats = summarize_ranks(ranks, S=S, bins=bins)
    u = (ranks + 0.5) / (S + 1.0)
    edges = np.linspace(0, 1, bins + 1)
    counts, _ = np.histogram(u, bins=edges)
    centers = 0.5 * (edges[1:] + edges[:-1])

    plt.figure(figsize=(5.5, 3.6))
    plt.bar(centers, counts, width=(1 / bins) * 0.9)
    plt.hlines(len(ranks) / bins, 0, 1, linestyles="dashed", linewidth=1.2)
    plt.xlim(0, 1)
    plt.xlabel("Normalized rank")
    plt.ylabel("Count per bin")
    plt.title(title)
    plt.text(
        0.99,
        0.97,
        f"chi2 p={stats['chi2_p']:.2f}; KS p={stats['ks_p']:.2f}",
        ha="right",
        va="top",
        transform=plt.gca().transAxes,
        fontsize=9,
    )
    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close()
    return stats


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig_root = args.output_dir / "figures"
    fig_root.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, object]] = []
    for prior in args.priors:
        prior_dir = args.output_dir / prior
        prior_fig = fig_root / prior
        prior_dir.mkdir(parents=True, exist_ok=True)
        prior_fig.mkdir(parents=True, exist_ok=True)

        ranks, truths = run_sbc_cre(
            M=args.M,
            S=args.S,
            NA=args.NA,
            NH=args.NH,
            prior=prior,
            seed=args.seed + (0 if prior == "jeffreys" else 100000),
        )
        truths.to_csv(prior_dir / f"sbc_truth_counts_{prior}.csv", index=False)

        for parameter, values in ranks.items():
            fig_path = prior_fig / f"sbc_{parameter}_{prior}_b{args.bins}.png"
            stats = plot_rank_histogram(
                values,
                S=args.S,
                bins=args.bins,
                title=f"SBC rank histogram: {parameter} ({prior})",
                output=fig_path,
            )
            summary_rows.append(
                {
                    **stats,
                    "prior": prior,
                    "parameter": parameter,
                    "M": args.M,
                    "S": args.S,
                    "NA": args.NA,
                    "NH": args.NH,
                    "figure": str(fig_path),
                }
            )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.output_dir / "sbc_uniformity_summary.csv", index=False)
    summary[
        [
            "prior",
            "parameter",
            "chi2_p",
            "ks_p",
            "min_count",
            "max_count",
            "expected_per_bin",
            "M",
            "S",
            "NA",
            "NH",
        ]
    ].to_csv(args.output_dir / "sbc_paper_summary.csv", index=False)

    (args.output_dir / "run_config.json").write_text(
        json.dumps(vars(args), indent=2, default=str), encoding="utf-8"
    )
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
