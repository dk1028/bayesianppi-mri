from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from _shared import (
    binary_difference_estimator,
    coverage_and_width,
    cre_posterior_draws,
    labeled_only_posterior_draws,
    load_prediction_csv,
    posterior_summary,
    ppi_analytic_estimator,
    ppipp_manuscript_estimator,
)

LABEL_SIZES_DEFAULT = [10, 20, 40, 80]
PRIORS = [
    {"name": "uniform", "alpha": 1.0, "beta": 1.0},
    {"name": "jeffreys", "alpha": 0.5, "beta": 0.5},
]
DISPLAY_NAME = {
    "full": "Full cohort",
    "age_65_70": "Age 65--70 subset",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run full-cohort and age 65--70 repeated-labeling experiments."
    )
    parser.add_argument("--full-csv", type=Path, required=True)
    parser.add_argument("--age6570-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--nsim", type=int, default=500)
    parser.add_argument("--label-sizes", type=int, nargs="+", default=LABEL_SIZES_DEFAULT)
    parser.add_argument("--posterior-draws", type=int, default=5000)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=2025)
    return parser.parse_args()


def qc_dataset(df: pd.DataFrame, name: str, threshold: float) -> dict[str, float | int]:
    fold_spanning = None
    max_folds = None
    if "oof_fold" in df.columns:
        fold_counts = df.groupby("subject_id")["oof_fold"].nunique()
        fold_spanning = int((fold_counts > 1).sum())
        max_folds = int(fold_counts.max())
        if fold_spanning:
            raise RuntimeError(f"{fold_spanning} subjects span multiple folds in {name}")

    out = {
        "rows": int(len(df)),
        "subjects": int(df["subject_id"].nunique()),
        "g_true": float(df["H"].mean()),
        "p_A1": float(df["A_class"].mean()),
        "threshold": threshold,
    }
    if fold_spanning is not None:
        out["subjects_spanning_multiple_folds"] = fold_spanning
        out["max_folds_per_subject"] = max_folds

    print("=" * 90)
    print("QC:", name)
    print(json.dumps(out, indent=2))
    return out


def run_coverage_experiment(
    df: pd.DataFrame,
    *,
    dataset_name: str,
    input_csv: Path,
    output_csv: Path,
    nsim: int,
    label_sizes: list[int],
    posterior_draws: int,
    bootstrap: int,
    threshold: float,
    seed: int,
    seed_offset: int,
) -> pd.DataFrame:
    n_rows = len(df)
    g_true = float(df["H"].mean())
    master_rng = np.random.default_rng(seed + seed_offset)

    label_index_sets = {
        n: [master_rng.choice(n_rows, size=n, replace=False) for _ in range(nsim)]
        for n in label_sizes
    }

    rows: list[dict[str, object]] = []

    for prior in PRIORS:
        pname = str(prior["name"])
        alpha = float(prior["alpha"])
        beta = float(prior["beta"])

        for n_labels in label_sizes:
            cov_counts = {
                "chain": 0,
                "labeled_only": 0,
                "diff": 0,
                "ppi": 0,
                "ppipp": 0,
            }
            widths = {key: [] for key in cov_counts}
            lambdas: list[float] = []

            desc = f"{dataset_name} | {pname} | labels={n_labels}"
            for rep, labeled_idx in enumerate(tqdm(label_index_sets[n_labels], desc=desc)):
                rep_seed = seed + seed_offset + 100000 * rep + 1000 * n_labels
                bootstrap_rng = np.random.default_rng(rep_seed)
                posterior_rng = np.random.default_rng(rep_seed + 17)

                cre = posterior_summary(
                    cre_posterior_draws(
                        df,
                        labeled_idx,
                        alpha,
                        beta,
                        posterior_draws,
                        posterior_rng,
                    )
                )
                ok, width = coverage_and_width(cre, g_true)
                cov_counts["chain"] += int(ok)
                widths["chain"].append(width)

                labeled_only = posterior_summary(
                    labeled_only_posterior_draws(
                        df,
                        labeled_idx,
                        alpha,
                        beta,
                        posterior_draws,
                        posterior_rng,
                    )
                )
                ok, width = coverage_and_width(labeled_only, g_true)
                cov_counts["labeled_only"] += int(ok)
                widths["labeled_only"].append(width)

                diff = binary_difference_estimator(
                    df,
                    labeled_idx,
                    bootstrap,
                    bootstrap_rng,
                )
                ok, width = coverage_and_width(diff, g_true)
                cov_counts["diff"] += int(ok)
                widths["diff"].append(width)

                ppi = ppi_analytic_estimator(df, labeled_idx)
                ok, width = coverage_and_width(ppi, g_true)
                cov_counts["ppi"] += int(ok)
                widths["ppi"].append(width)

                ppipp, lam = ppipp_manuscript_estimator(df, labeled_idx)
                ok, width = coverage_and_width(ppipp, g_true)
                cov_counts["ppipp"] += int(ok)
                widths["ppipp"].append(width)
                lambdas.append(lam)

            rows.append(
                {
                    "dataset": dataset_name,
                    "input_csv_path": str(input_csv),
                    "output_csv": str(output_csv),
                    "threshold_policy": f"fixed_{threshold}",
                    "threshold": threshold,
                    "n_rows": n_rows,
                    "n_subjects": int(df["subject_id"].nunique()),
                    "g_true": g_true,
                    "prior": pname,
                    "n_labels": n_labels,
                    "chain_cov": cov_counts["chain"] / nsim,
                    "chain_w": float(np.mean(widths["chain"])),
                    "labeled_only_cov": cov_counts["labeled_only"] / nsim,
                    "labeled_only_w": float(np.mean(widths["labeled_only"])),
                    "diff_cov": cov_counts["diff"] / nsim,
                    "diff_w": float(np.mean(widths["diff"])),
                    "ppi_cov": cov_counts["ppi"] / nsim,
                    "ppi_w": float(np.mean(widths["ppi"])),
                    "ppipp_cov": cov_counts["ppipp"] / nsim,
                    "ppipp_w": float(np.mean(widths["ppipp"])),
                    "ppipp_lambda_mean": float(np.mean(lambdas)),
                    "ppipp_lambda_sd": float(np.std(lambdas, ddof=1)),
                }
            )

    result = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False)
    return result


def make_uniform_summary(results: pd.DataFrame) -> pd.DataFrame:
    sub = results[results["prior"] == "uniform"].sort_values("n_labels")
    out = sub[
        [
            "n_labels",
            "chain_cov",
            "chain_w",
            "labeled_only_cov",
            "labeled_only_w",
            "diff_cov",
            "diff_w",
            "ppi_cov",
            "ppi_w",
            "ppipp_cov",
            "ppipp_w",
        ]
    ].copy()
    return out.rename(
        columns={
            "n_labels": "n",
            "chain_cov": "CRE_cov",
            "chain_w": "CRE_w",
            "labeled_only_cov": "LabeledOnly_cov",
            "labeled_only_w": "LabeledOnly_w",
            "diff_cov": "BinaryDiff_cov",
            "diff_w": "BinaryDiff_w",
            "ppi_cov": "PPI_cov",
            "ppi_w": "PPI_w",
            "ppipp_cov": "PPIpp_cov",
            "ppipp_w": "PPIpp_w",
        }
    )


def plot_dataset(results: pd.DataFrame, dataset_name: str, fig_dir: Path) -> None:
    sub = results[results["dataset"] == dataset_name].copy()
    title = DISPLAY_NAME.get(dataset_name, dataset_name)
    prior_free = (
        sub.groupby("n_labels", as_index=False)[
            ["ppi_cov", "ppi_w", "ppipp_cov", "ppipp_w"]
        ]
        .mean()
        .sort_values("n_labels")
    )

    plt.figure(figsize=(10, 5))
    for prior in ["uniform", "jeffreys"]:
        psub = sub[sub["prior"] == prior].sort_values("n_labels")
        plt.plot(psub["n_labels"], psub["chain_cov"], "o-", label=f"CRE ({prior})")
        plt.plot(
            psub["n_labels"],
            psub["labeled_only_cov"],
            "s--",
            label=f"Labeled-only ({prior})",
        )
        plt.plot(psub["n_labels"], psub["diff_cov"], "d-.", label=f"Binary Diff ({prior})")
    plt.plot(prior_free["n_labels"], prior_free["ppi_cov"], "^:", label="PPI")
    plt.plot(prior_free["n_labels"], prior_free["ppipp_cov"], "v:", label="PPI++")
    plt.axhline(0.95, linestyle="--", linewidth=1, label="Nominal 0.95")
    plt.ylim(0.3, 1.05)
    plt.xlabel("Number of labeled scans")
    plt.ylabel("Empirical coverage")
    plt.title(f"Coverage vs. Label Budget: {title}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / f"fig_coverage_{dataset_name}.png", dpi=300, bbox_inches="tight")
    plt.savefig(fig_dir / f"fig_coverage_{dataset_name}.pdf", bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 5))
    for prior in ["uniform", "jeffreys"]:
        psub = sub[sub["prior"] == prior].sort_values("n_labels")
        plt.plot(psub["n_labels"], psub["chain_w"], "o-", label=f"CRE ({prior})")
        plt.plot(
            psub["n_labels"],
            psub["labeled_only_w"],
            "s--",
            label=f"Labeled-only ({prior})",
        )
        plt.plot(psub["n_labels"], psub["diff_w"], "d-.", label=f"Binary Diff ({prior})")
    plt.plot(prior_free["n_labels"], prior_free["ppi_w"], "^:", label="PPI")
    plt.plot(prior_free["n_labels"], prior_free["ppipp_w"], "v:", label="PPI++")
    plt.xlabel("Number of labeled scans")
    plt.ylabel("Mean 95% interval width")
    plt.title(f"Interval Width vs. Label Budget: {title}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / f"fig_width_{dataset_name}.png", dpi=300, bbox_inches="tight")
    plt.savefig(fig_dir / f"fig_width_{dataset_name}.pdf", bbox_inches="tight")
    plt.close()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = args.output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    full = load_prediction_csv(args.full_csv, threshold=args.threshold)
    age = load_prediction_csv(args.age6570_csv, threshold=args.threshold)

    qc = {
        "full": qc_dataset(full, "Full cohort", args.threshold),
        "age_65_70": qc_dataset(age, "Age 65--70", args.threshold),
    }

    full_out = args.output_dir / "coverage_all_subject_oof_scan_age.csv"
    age_out = args.output_dir / "coverage_6570_subject_oof_scan_age.csv"

    full_res = run_coverage_experiment(
        full,
        dataset_name="full",
        input_csv=args.full_csv,
        output_csv=full_out,
        nsim=args.nsim,
        label_sizes=args.label_sizes,
        posterior_draws=args.posterior_draws,
        bootstrap=args.bootstrap,
        threshold=args.threshold,
        seed=args.seed,
        seed_offset=0,
    )
    age_res = run_coverage_experiment(
        age,
        dataset_name="age_65_70",
        input_csv=args.age6570_csv,
        output_csv=age_out,
        nsim=args.nsim,
        label_sizes=args.label_sizes,
        posterior_draws=args.posterior_draws,
        bootstrap=args.bootstrap,
        threshold=args.threshold,
        seed=args.seed,
        seed_offset=500000,
    )

    combined = pd.concat([full_res, age_res], ignore_index=True)
    combined.to_csv(
        args.output_dir / "coverage_all_6570_subject_oof_scan_age_combined.csv",
        index=False,
    )

    make_uniform_summary(full_res).to_csv(
        args.output_dir / "table_full_uniform_summary.csv", index=False
    )
    make_uniform_summary(age_res).to_csv(
        args.output_dir / "table_6570_uniform_summary.csv", index=False
    )

    plot_dataset(combined, "full", fig_dir)
    plot_dataset(combined, "age_65_70", fig_dir)

    run_config = {
        "full_csv": str(args.full_csv),
        "age6570_csv": str(args.age6570_csv),
        "nsim": args.nsim,
        "label_sizes": args.label_sizes,
        "posterior_draws": args.posterior_draws,
        "bootstrap": args.bootstrap,
        "threshold": args.threshold,
        "seed": args.seed,
        "qc": qc,
        "ppipp_note": "Self-contained power-tuned formula used in manuscript tables",
    }
    (args.output_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2), encoding="utf-8"
    )

    print(combined.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
