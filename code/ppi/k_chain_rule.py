from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _shared import load_prediction_csv

PRIORS = [
    {"name": "uniform", "alpha_dirichlet": 1.0, "a_beta": 1.0, "b_beta": 1.0},
    {"name": "jeffreys", "alpha_dirichlet": 0.5, "a_beta": 0.5, "b_beta": 0.5},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run conjugate K-bin CRE sensitivity.")
    parser.add_argument("--full-csv", type=Path, required=True)
    parser.add_argument("--age6570-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--k", type=int, nargs="+", default=[2, 4, 5])
    parser.add_argument("--strategy", choices=["quantile", "uniform"], default="quantile")
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2025)
    return parser.parse_args()


def assign_bins_from_full_pool(
    scores: pd.Series,
    K: int,
    strategy: Literal["quantile", "uniform"],
) -> tuple[pd.Series, np.ndarray, int]:
    scores = pd.to_numeric(scores, errors="coerce")
    if strategy == "quantile":
        codes, edges = pd.qcut(
            scores,
            q=min(K, scores.nunique()),
            labels=False,
            duplicates="drop",
            retbins=True,
        )
    else:
        codes, edges = pd.cut(
            scores,
            bins=K,
            labels=False,
            include_lowest=True,
            retbins=True,
        )
    codes = pd.Series(codes, index=scores.index)
    return codes, np.asarray(edges, dtype=float), int(codes.nunique(dropna=True))


def assign_bins_from_edges(scores: pd.Series, edges: np.ndarray) -> pd.Series:
    edges = np.asarray(edges, dtype=float).copy()
    edges[0] = -np.inf
    edges[-1] = np.inf
    return pd.Series(
        pd.cut(scores, bins=edges, labels=False, include_lowest=True),
        index=scores.index,
    )


def posterior_draws_kbin(
    df: pd.DataFrame,
    *,
    K: int,
    strategy: Literal["quantile", "uniform"],
    alpha_dirichlet: float,
    a_beta: float,
    b_beta: float,
    draws: int,
    rng: np.random.Generator,
) -> dict[str, object]:
    full = df.dropna(subset=["A_prob"]).copy()
    full["bin"], edges, K_eff = assign_bins_from_full_pool(full["A_prob"], K, strategy)

    counts_B = (
        full["bin"].value_counts().sort_index().reindex(range(K_eff), fill_value=0).astype(int)
    )
    alpha_vec = np.full(K_eff, alpha_dirichlet) + counts_B.to_numpy(dtype=float)

    lab = df.dropna(subset=["A_prob", "H"]).copy()
    lab["bin"] = assign_bins_from_edges(lab["A_prob"], edges)
    lab = lab.dropna(subset=["bin"]).copy()
    lab["bin"] = lab["bin"].astype(int)

    n_pos = lab.groupby("bin")["H"].sum().reindex(range(K_eff), fill_value=0).astype(int)
    n_total = lab.groupby("bin")["H"].count().reindex(range(K_eff), fill_value=0).astype(int)

    pi = rng.dirichlet(alpha_vec, size=draws)
    theta = rng.beta(
        a_beta + n_pos.to_numpy(dtype=float),
        b_beta + (n_total - n_pos).to_numpy(dtype=float),
        size=(draws, K_eff),
    )
    g = np.sum(pi * theta, axis=1)
    lo, hi = np.quantile(g, [0.025, 0.975])

    return {
        "K_requested": K,
        "K_effective": K_eff,
        "strategy": strategy,
        "mean": float(g.mean()),
        "sd": float(g.std(ddof=1)),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "width": float(hi - lo),
        "min_bin_count_full": int(counts_B.min()),
        "min_bin_count_labeled": int(n_total.min()),
        "max_bin_count_labeled": int(n_total.max()),
        "bin_edges": ";".join(f"{x:.8g}" for x in edges),
    }


def run_dataset(
    df: pd.DataFrame,
    *,
    dataset_name: str,
    input_csv: Path,
    k_values: list[int],
    strategy: str,
    draws: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for prior in PRIORS:
        for K in k_values:
            local_seed = seed + 1000 * K + (0 if prior["name"] == "uniform" else 100000)
            result = posterior_draws_kbin(
                df,
                K=K,
                strategy=strategy,  # type: ignore[arg-type]
                alpha_dirichlet=float(prior["alpha_dirichlet"]),
                a_beta=float(prior["a_beta"]),
                b_beta=float(prior["b_beta"]),
                draws=draws,
                rng=np.random.default_rng(local_seed),
            )
            result.update(
                {
                    "dataset": dataset_name,
                    "input_csv_path": str(input_csv),
                    "n_rows": len(df),
                    "n_subjects": int(df["subject_id"].nunique()),
                    "g_true_empirical": float(df["H"].mean()),
                    "prior": prior["name"],
                    "alpha_dirichlet": prior["alpha_dirichlet"],
                    "a_beta": prior["a_beta"],
                    "b_beta": prior["b_beta"],
                    "n_draws": draws,
                }
            )
            rows.append(result)

    out = pd.DataFrame(rows)
    out["delta_mean_vs_K2"] = np.nan
    out["width_ratio_vs_K2"] = np.nan
    for prior_name in out["prior"].unique():
        mask = out["prior"] == prior_name
        reference = out[mask & (out["K_requested"] == 2)]
        if len(reference) == 1:
            ref_mean = float(reference["mean"].iloc[0])
            ref_width = float(reference["width"].iloc[0])
            out.loc[mask, "delta_mean_vs_K2"] = out.loc[mask, "mean"] - ref_mean
            out.loc[mask, "width_ratio_vs_K2"] = out.loc[mask, "width"] / ref_width
    return out


def plot_results(summary: pd.DataFrame, dataset_name: str, fig_dir: Path) -> None:
    sub = summary[summary["dataset"] == dataset_name]
    display_name = "Full cohort" if dataset_name == "full" else "Age 65--70 subset"

    plt.figure(figsize=(6, 4))
    for prior in ["uniform", "jeffreys"]:
        psub = sub[sub["prior"] == prior].sort_values("K_requested")
        plt.plot(psub["K_requested"], psub["width"], marker="o", label=prior)
    plt.xlabel("K requested")
    plt.ylabel("95% posterior interval width")
    plt.title(f"K-bin CRE sensitivity: {display_name}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / f"fig_kbin_width_{dataset_name}.png", dpi=300, bbox_inches="tight")
    plt.savefig(fig_dir / f"fig_kbin_width_{dataset_name}.pdf", bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(6, 4))
    for prior in ["uniform", "jeffreys"]:
        psub = sub[sub["prior"] == prior].sort_values("K_requested")
        plt.plot(psub["K_requested"], psub["mean"], marker="o", label=prior)
    plt.axhline(float(sub["g_true_empirical"].iloc[0]), linestyle="--", linewidth=1, label="Empirical H prevalence")
    plt.xlabel("K requested")
    plt.ylabel("Posterior mean of g")
    plt.title(f"K-bin CRE posterior mean: {display_name}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / f"fig_kbin_mean_{dataset_name}.png", dpi=300, bbox_inches="tight")
    plt.savefig(fig_dir / f"fig_kbin_mean_{dataset_name}.pdf", bbox_inches="tight")
    plt.close()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = args.output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    full = load_prediction_csv(args.full_csv)
    age = load_prediction_csv(args.age6570_csv)

    full_result = run_dataset(
        full,
        dataset_name="full",
        input_csv=args.full_csv,
        k_values=args.k,
        strategy=args.strategy,
        draws=args.draws,
        seed=args.seed,
    )
    age_result = run_dataset(
        age,
        dataset_name="age_65_70",
        input_csv=args.age6570_csv,
        k_values=args.k,
        strategy=args.strategy,
        draws=args.draws,
        seed=args.seed,
    )

    summary = pd.concat([full_result, age_result], ignore_index=True)
    summary.to_csv(args.output_dir / "kbin_sensitivity_summary_revised.csv", index=False)
    summary[
        [
            "dataset",
            "prior",
            "K_requested",
            "K_effective",
            "mean",
            "ci_low",
            "ci_high",
            "width",
            "delta_mean_vs_K2",
            "width_ratio_vs_K2",
            "min_bin_count_labeled",
        ]
    ].to_csv(args.output_dir / "kbin_paper_summary_revised.csv", index=False)

    plot_results(summary, "full", fig_dir)
    plot_results(summary, "age_65_70", fig_dir)

    (args.output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "full_csv": str(args.full_csv),
                "age6570_csv": str(args.age6570_csv),
                "k": args.k,
                "strategy": args.strategy,
                "draws": args.draws,
                "seed": args.seed,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
