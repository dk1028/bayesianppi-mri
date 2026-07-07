from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from _shared import COVERAGE_ROOT, FULL_PRED_CANDIDATES, assign_bins, choose_existing, equal_tailed_interval, load_prediction_csv

CSV_PATH = choose_existing(FULL_PRED_CANDIDATES)
S = 5000
SEED = 2025


def posterior_draws_kbin(
    pred_df: pd.DataFrame,
    labeled_df: pd.DataFrame,
    k: int,
    strategy: Literal["quantile", "uniform"] = "quantile",
    alpha_dirichlet: float = 1.0,
    a_beta: float = 1.0,
    b_beta: float = 1.0,
    s: int = S,
    rng: np.random.Generator = np.random.default_rng(SEED),
) -> dict[str, float]:
    pred_valid = pred_df.dropna(subset=["autorater_prediction"]).copy()
    pred_valid["bin"] = assign_bins(pred_valid["autorater_prediction"], k, strategy=strategy)
    labeled = labeled_df.copy()
    labeled["bin"] = assign_bins(labeled["autorater_prediction"], k, strategy=strategy)

    bins_used = int(pred_valid["bin"].nunique())
    counts = pred_valid["bin"].value_counts().sort_index()
    alpha = np.full(bins_used, alpha_dirichlet, dtype=float)
    for b_idx, count in counts.items():
        alpha[int(b_idx)] += float(count)
    p_bin_draws = rng.dirichlet(alpha, size=s)

    theta_draws = np.zeros((s, bins_used), dtype=float)
    for b in range(bins_used):
        sub = labeled[labeled["bin"] == b]
        successes = float(sub["H"].sum())
        total = float(len(sub))
        theta_draws[:, b] = rng.beta(a_beta + successes, b_beta + total - successes, size=s)

    g_draws = np.sum(p_bin_draws * theta_draws, axis=1)
    lo, hi = equal_tailed_interval(g_draws)
    return {
        "mean": float(g_draws.mean()),
        "sd": float(g_draws.std(ddof=1)),
        "ci_low": lo,
        "ci_high": hi,
        "width": hi - lo,
        "K_effective": bins_used,
    }



def main() -> None:
    pred = load_prediction_csv(CSV_PATH)
    labeled = pred.copy()
    base_uniform = posterior_draws_kbin(pred, labeled, 2, a_beta=1.0, b_beta=1.0)
    base_jeff = posterior_draws_kbin(pred, labeled, 2, a_beta=0.5, b_beta=0.5)
    rows = []
    for prior_name, ab in [("Uniform", (1.0, 1.0)), ("Jeffreys", (0.5, 0.5))]:
        base = base_uniform if prior_name == "Uniform" else base_jeff
        for k in [2, 4, 5]:
            cur = posterior_draws_kbin(pred, labeled, k, a_beta=ab[0], b_beta=ab[1])
            rows.append(
                {
                    "Prior": prior_name,
                    "K": k,
                    "DeltaMean_vs_K2": float(cur["mean"] - base["mean"]),
                    "Width_ratio_K_over_2": float(cur["width"] / base["width"]),
                }
            )
    out_csv = COVERAGE_ROOT / "kbin_sensitivity_summary.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Saved {out_csv}")


if __name__ == "__main__":
    main()
