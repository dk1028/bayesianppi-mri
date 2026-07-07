from __future__ import annotations

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from _shared import (
    AGE6570_PRED_CANDIDATES,
    COVERAGE_ROOT,
    choose_existing,
    coverage_and_width,
    cre_posterior_draws,
    difference_estimator,
    load_prediction_csv,
    naive_posterior_draws,
    posterior_summary,
    ppi_analytic_estimator,
)

CSV_PATH = choose_existing(AGE6570_PRED_CANDIDATES)
NSIM = 500
LABEL_SIZES = [10, 20, 40, 80]
PRIORS = [
    {"name": "uniform", "alpha": 1.0, "beta": 1.0},
    {"name": "jeffreys", "alpha": 0.5, "beta": 0.5},
]
POSTERIOR_DRAWS = 5000
BOOTSTRAP_B = 1000
SEED = 2025


def main() -> None:
    df = load_prediction_csv(CSV_PATH)
    g_true = float(df["H"].mean())
    rng = np.random.default_rng(SEED)
    draws_by_n = {
        n: [rng.choice(len(df), size=n, replace=False) for _ in range(NSIM)]
        for n in LABEL_SIZES
    }

    rows: list[dict[str, float | int | str]] = []
    for prior in PRIORS:
        alpha = float(prior["alpha"])
        beta = float(prior["beta"])
        name = str(prior["name"])
        for n in LABEL_SIZES:
            cov = {"chain": 0, "naive": 0, "diff": 0, "ppi": 0}
            widths = {"chain": [], "naive": [], "diff": [], "ppi": []}
            for sim_id, idx in enumerate(tqdm(draws_by_n[n], desc=f"{name} | labels={n}")):
                post_rng = np.random.default_rng(SEED + 10000 * sim_id + n)

                cre = posterior_summary(
                    cre_posterior_draws(df, idx, alpha, beta, POSTERIOR_DRAWS, post_rng)
                )
                ok, width = coverage_and_width(cre, g_true)
                cov["chain"] += int(ok)
                widths["chain"].append(width)

                naive = posterior_summary(
                    naive_posterior_draws(df, idx, alpha, beta, POSTERIOR_DRAWS, post_rng)
                )
                ok, width = coverage_and_width(naive, g_true)
                cov["naive"] += int(ok)
                widths["naive"].append(width)

                diff = difference_estimator(df, idx, BOOTSTRAP_B, post_rng)
                ok, width = coverage_and_width(diff, g_true)
                cov["diff"] += int(ok)
                widths["diff"].append(width)

                ppi = ppi_analytic_estimator(df, idx)
                ok, width = coverage_and_width(ppi, g_true)
                cov["ppi"] += int(ok)
                widths["ppi"].append(width)

            rows.append(
                {
                    "prior": name,
                    "n_labels": n,
                    "chain_cov": cov["chain"] / NSIM,
                    "chain_w": float(np.mean(widths["chain"])),
                    "naive_cov": cov["naive"] / NSIM,
                    "naive_w": float(np.mean(widths["naive"])),
                    "diff_cov": cov["diff"] / NSIM,
                    "diff_w": float(np.mean(widths["diff"])),
                    "ppi_cov": cov["ppi"] / NSIM,
                    "ppi_w": float(np.mean(widths["ppi"])),
                }
            )

    res_df = pd.DataFrame(rows)
    out_csv = COVERAGE_ROOT / "coverage_6570.csv"
    res_df.to_csv(out_csv, index=False)
    print(res_df)
    print(f"\nSaved 65–70 coverage results to: {out_csv}")


if __name__ == "__main__":
    main()
