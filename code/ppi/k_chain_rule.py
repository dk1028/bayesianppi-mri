import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Literal

# ============================
# 0) Repo-relative paths
# ============================

# This file lives in: repo_root/code/ppi/k_chain_rule.py
REPO_ROOT     = Path(__file__).resolve().parents[2]
DATA_ROOT     = REPO_ROOT / "data"
RESULTS_ROOT  = REPO_ROOT / "results"

AUTORATER_DIR = RESULTS_ROOT / "autorater"
OUT_DIR       = RESULTS_ROOT / "coverage" / "kbin"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Input CSV: autorater predictions + labels
CSV_PRED = AUTORATER_DIR / "autorater_predictions_all4.csv"

# ============================
# 0) Parameters
# ============================

K_LIST = (2, 4, 5)
BINNING_STRATEGY: Literal["quantile", "uniform"] = "quantile"   # "quantile" recommended

ALPHA_DIR = 1.0   # Dirichlet prior for P(B); use 0.5 for Jeffreys-like
A_BETA    = 1.0   # Beta prior for P(H=1|B); use 0.5 for Jeffreys-like
B_BETA    = 1.0
N_DRAWS   = 5000
RNG       = np.random.default_rng(2025)

# ============================
# 1) Load data
# ============================

if not CSV_PRED.exists():
    raise FileNotFoundError(
        f"Input CSV not found: {CSV_PRED}\n"
        "Make sure autorater_predictions_all4.csv exists in results/autorater/."
    )

pred = pd.read_csv(CSV_PRED, dtype=str)
pred.columns = [c.strip() for c in pred.columns]

# Required columns
need_cols = ["autorater_prediction", "H"]
for c in need_cols:
    if c not in pred.columns:
        raise ValueError(
            f"[PRED] Missing column '{c}'. Current columns: {list(pred.columns)}"
        )

# Cast to numeric
pred["autorater_prediction"] = pd.to_numeric(
    pred["autorater_prediction"], errors="coerce"
)
pred["H"] = pd.to_numeric(pred["H"], errors="coerce")

# Sanity filter: keep valid scores in [0,1] and H in {0,1}
pred = pred[
    pred["autorater_prediction"].between(0, 1, inclusive="both")
    & pred["H"].isin([0, 1])
].copy()

# Labeled subset (H available) and full pool
use = pred.dropna(subset=["autorater_prediction", "H"]).copy()

if len(use) == 0:
    raise ValueError(
        "No valid labeled rows found (check 'autorater_prediction' and 'H')."
    )

# ============================
# 2) K-bin utilities
# ============================

def _assign_bins(
    scores: pd.Series,
    K: int,
    strategy: Literal["quantile", "uniform"] = "quantile",
) -> pd.Series:
    """Assign scores to K bins using 'quantile' or 'uniform' binning."""
    scores = pd.to_numeric(scores, errors="coerce")
    if strategy == "quantile":
        # qcut can drop duplicated edges; allow fewer effective bins via duplicates='drop'
        return pd.qcut(
            scores,
            q=min(K, scores.nunique()),
            labels=False,
            duplicates="drop",
        )
    else:
        return pd.cut(scores, bins=K, labels=False, include_lowest=True)


def _posterior_draws_kbin(
    pred_df: pd.DataFrame,      # full pool for scores (unlabeled + labeled OK)
    labeled_df: pd.DataFrame,   # labeled subset with columns: autorater_prediction, H
    K: int,
    strategy: Literal["quantile", "uniform"] = "quantile",
    alpha_dirichlet: float = 1.0,
    a_beta: float = 1.0,
    b_beta: float = 1.0,
    S: int = 5000,
    rng: np.random.Generator = np.random.default_rng(2025),
):
    """
    Conjugate Bayesian CRE for K-bin chain rule:
      g = sum_k P(B = k) * P(H = 1 | B = k)

    Returns:
        dict with summary stats (mean, sd, ci_low, ci_high, width, K_effective, ...).
    """
    # 1) Bin assignment on the full pool (for stable P(B))
    pred_valid = pred_df.dropna(subset=["autorater_prediction"]).copy()
    pred_valid["bin"] = _assign_bins(
        pred_valid["autorater_prediction"], K, strategy=strategy
    )
    bins_used = int(pred_valid["bin"].nunique())
    if bins_used < 2:
        raise ValueError(
            f"Effective number of bins is {bins_used} (<2). "
            "Try 'uniform' or smaller K."
        )

    # Posterior for P(B) ~ Dirichlet(alpha + counts)
    counts_B = pred_valid["bin"].value_counts().sort_index()
    counts_B = counts_B.reindex(range(bins_used), fill_value=0).astype(int)
    alpha_vec = np.full(bins_used, alpha_dirichlet, dtype=float) + counts_B.values

    # 2) Labeled conditional P(H=1|B) Beta posteriors
    lab = labeled_df.dropna(subset=["autorater_prediction", "H"]).copy()
    lab["bin"] = _assign_bins(
        lab["autorater_prediction"], bins_used, strategy=strategy
    )

    n1 = lab.groupby("bin")["H"].sum()
    n = lab.groupby("bin")["H"].count()
    n1 = n1.reindex(range(bins_used), fill_value=0).astype(int)
    n = n.reindex(range(bins_used), fill_value=0).astype(int)

    a_post = a_beta + n1.values
    b_post = b_beta + (n.values - n1.values)

    # 3) Posterior draws
    g_draws = np.empty(S, dtype=float)
    for s in range(S):
        pi = rng.dirichlet(alpha_vec)
        theta = rng.beta(a_post, b_post)  # shape (bins_used,)
        g_draws[s] = float(np.dot(pi, theta))

    # 4) Summaries
    mean = float(np.mean(g_draws))
    sd = float(np.std(g_draws, ddof=1))
    lo, hi = np.percentile(g_draws, [2.5, 97.5])
    width = float(hi - lo)

    return {
        "K_requested": K,
        "K_effective": bins_used,
        "mean": mean,
        "sd": sd,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "width": width,
    }


def run_kbin_sensitivity(
    K_list=(2, 4, 5),
    strategy: Literal["quantile", "uniform"] = "quantile",
    alpha_dirichlet: float = 1.0,
    a_beta: float = 1.0,
    b_beta: float = 1.0,
    S: int = 5000,
    out_dir: Path = OUT_DIR,
    rng: np.random.Generator = RNG,
):
    """
    Runs K-bin CRE for each K in K_list; saves CSV and plots width-vs-K.
    """
    results = []
    for K in K_list:
        res = _posterior_draws_kbin(
            pred_df=pred,
            labeled_df=use,
            K=K,
            strategy=strategy,
            alpha_dirichlet=alpha_dirichlet,
            a_beta=a_beta,
            b_beta=b_beta,
            S=S,
            rng=rng,
        )
        res["strategy"] = strategy
        res["alpha_dirichlet"] = alpha_dirichlet
        res["a_beta"] = a_beta
        res["b_beta"] = b_beta
        results.append(res)

        print(
            f"[K-bin] K={K} (eff={res['K_effective']}): "
            f"mean={res['mean']:.4f}, "
            f"95%CI=[{res['ci_low']:.4f},{res['ci_high']:.4f}] "
            f"(width={res['width']:.4f})"
        )

    df = pd.DataFrame(results)[
        [
            "K_requested",
            "K_effective",
            "strategy",
            "mean",
            "sd",
            "ci_low",
            "ci_high",
            "width",
            "alpha_dirichlet",
            "a_beta",
            "b_beta",
        ]
    ]

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "kbin_sensitivity_summary.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    # Plot: CI width vs K_requested
    plt.figure(figsize=(5.6, 4.2))
    plt.plot(df["K_requested"], df["width"], marker="o")
    for _, r in df.iterrows():
        plt.scatter(r["K_requested"], r["width"])
    plt.xlabel("K (requested)")
    plt.ylabel("95% CI width for g")
    plt.title("K-bin CRE sensitivity: interval width vs K")
    plt.grid(alpha=0.3, linestyle="--")
    plt.tight_layout()

    fig_path = out_dir / "kbin_width_vs_K.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()

    print("\n✅ Saved:")
    print(f"- {csv_path}")
    print(f"- {fig_path}")


# ============================
# 3) Run
# ============================

if __name__ == "__main__":
    run_kbin_sensitivity(
        K_list=K_LIST,
        strategy=BINNING_STRATEGY,   # "quantile" or "uniform"
        alpha_dirichlet=ALPHA_DIR,   # 0.5 for Jeffreys-like
        a_beta=A_BETA,
        b_beta=B_BETA,
        S=N_DRAWS,
        out_dir=OUT_DIR,
        rng=RNG,
    )