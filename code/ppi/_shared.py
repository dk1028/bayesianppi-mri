from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data"
CSV_ROOT = DATA_ROOT / "csv"
RESULTS_ROOT = REPO_ROOT / "results"
FIGS_ROOT = REPO_ROOT / "figs"
AUTORATER_ROOT = RESULTS_ROOT / "autorater"
COVERAGE_ROOT = RESULTS_ROOT / "coverage"
AGE_ANALYSIS_ROOT = RESULTS_ROOT / "age_analysis"
SBC_ROOT = FIGS_ROOT / "sbc"

for path in [FIGS_ROOT, AUTORATER_ROOT, COVERAGE_ROOT, AGE_ANALYSIS_ROOT, SBC_ROOT]:
    path.mkdir(parents=True, exist_ok=True)

Z975 = 1.959963984540054


def choose_existing(paths: list[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    joined = "\n - ".join(str(p) for p in paths)
    raise FileNotFoundError(f"Could not find any expected file. Checked:\n - {joined}")


FULL_PRED_CANDIDATES = [
    AUTORATER_ROOT / "autorater_predictions_all.csv",
    AUTORATER_ROOT / "autorater_predictions_all4.csv",
]

AGE6570_PRED_CANDIDATES = [
    AUTORATER_ROOT / "autorater_predictions_6570_all.csv",
    AUTORATER_ROOT / "autorater_predictions_all1.csv",
]

META_CANDIDATES = [
    CSV_ROOT / "all_people_7_20_2025.csv",
    CSV_ROOT / "all_people.csv",
]


def load_prediction_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path).copy()
    if "H" not in df.columns:
        if "label" not in df.columns:
            raise ValueError(f"{csv_path.name} must contain either 'H' or 'label'.")
        df["H"] = (df["label"].astype(str).str.upper() == "AD").astype(int)
    else:
        df["H"] = pd.to_numeric(df["H"], errors="coerce")
        if df["H"].isna().any() and "label" in df.columns:
            df.loc[df["H"].isna(), "H"] = (
                df.loc[df["H"].isna(), "label"].astype(str).str.upper() == "AD"
            ).astype(int)
        df["H"] = df["H"].astype(int)
    if "autorater_prediction" not in df.columns:
        raise ValueError(f"{csv_path.name} is missing 'autorater_prediction'.")
    df["autorater_prediction"] = pd.to_numeric(df["autorater_prediction"], errors="coerce")
    df = df.dropna(subset=["autorater_prediction", "H"]).copy()
    df = df[df["autorater_prediction"].between(0, 1, inclusive="both")]
    df["A_prob"] = df["autorater_prediction"].astype(float)
    df["A_class"] = (df["A_prob"] >= 0.5).astype(int)
    return df.reset_index(drop=True)


@dataclass
class PosteriorSummary:
    mean: float
    ci_low: float
    ci_high: float



def beta_posterior_draws(alpha: float, beta: float, n_success: float, n_total: float, size: int, rng: np.random.Generator) -> np.ndarray:
    return rng.beta(alpha + n_success, beta + max(n_total - n_success, 0.0), size=size)



def cre_posterior_draws(
    df: pd.DataFrame,
    labeled_idx: np.ndarray,
    alpha: float,
    beta: float,
    size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    sub = df.iloc[labeled_idx]
    n_total = len(df)
    nA1 = float(df["A_class"].sum())
    n11 = float(((sub["A_class"] == 1) & (sub["H"] == 1)).sum())
    n10 = float(((sub["A_class"] == 1) & (sub["H"] == 0)).sum())
    n01 = float(((sub["A_class"] == 0) & (sub["H"] == 1)).sum())
    n00 = float(((sub["A_class"] == 0) & (sub["H"] == 0)).sum())

    theta_a = beta_posterior_draws(alpha, beta, nA1, n_total, size, rng)
    theta_h1 = beta_posterior_draws(alpha, beta, n11, n11 + n10, size, rng)
    theta_h0 = beta_posterior_draws(alpha, beta, n01, n01 + n00, size, rng)
    return theta_a * theta_h1 + (1.0 - theta_a) * theta_h0



def naive_posterior_draws(
    df: pd.DataFrame,
    labeled_idx: np.ndarray,
    alpha: float,
    beta: float,
    size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    sub = df.iloc[labeled_idx]
    return rng.beta(alpha + sub["H"].sum(), beta + len(sub) - sub["H"].sum(), size=size)



def posterior_summary(draws: np.ndarray) -> PosteriorSummary:
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return PosteriorSummary(mean=float(draws.mean()), ci_low=float(lo), ci_high=float(hi))



def difference_estimator(df: pd.DataFrame, labeled_idx: np.ndarray, n_boot: int, rng: np.random.Generator) -> PosteriorSummary:
    a_bar = float(df["A_class"].mean())
    resid = (df.iloc[labeled_idx]["H"] - df.iloc[labeled_idx]["A_class"]).to_numpy(dtype=float)
    g_hat = a_bar + float(resid.mean())
    boots = []
    for _ in range(n_boot):
        sample = rng.choice(resid, size=len(resid), replace=True)
        boots.append(a_bar + float(sample.mean()))
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return PosteriorSummary(mean=g_hat, ci_low=float(lo), ci_high=float(hi))



def ppi_analytic_estimator(df: pd.DataFrame, labeled_idx: np.ndarray) -> PosteriorSummary:
    a_all = df["A_prob"].to_numpy(dtype=float)
    n_all = len(a_all)
    a_bar = float(a_all.mean())
    var_a = float(np.var(a_all, ddof=1)) if n_all > 1 else 0.0
    sub = df.iloc[labeled_idx]
    resid = (sub["H"].to_numpy(dtype=float) - sub["A_prob"].to_numpy(dtype=float))
    n_lab = len(resid)
    r_bar = float(resid.mean())
    var_r = float(np.var(resid, ddof=1)) if n_lab > 1 else 0.0
    g_hat = a_bar + r_bar
    se = float(np.sqrt(var_a / max(n_all, 1) + var_r / max(n_lab, 1)))
    crit = Z975 if n_lab >= 30 else float(student_t.ppf(0.975, df=max(n_lab - 1, 1)))
    return PosteriorSummary(mean=g_hat, ci_low=g_hat - crit * se, ci_high=g_hat + crit * se)



def equal_tailed_interval(draws: np.ndarray) -> tuple[float, float]:
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return float(lo), float(hi)



def coverage_and_width(summary: PosteriorSummary, truth: float) -> tuple[bool, float]:
    return summary.ci_low <= truth <= summary.ci_high, summary.ci_high - summary.ci_low



def assign_bins(scores: pd.Series, k: int, strategy: Literal["quantile", "uniform"] = "quantile") -> pd.Series:
    scores = pd.to_numeric(scores, errors="coerce")
    if strategy == "quantile":
        return pd.qcut(scores, q=min(k, scores.nunique()), labels=False, duplicates="drop")
    return pd.cut(scores, bins=k, labels=False, include_lowest=True)



def wilson_interval(k: int, n: int, z: float = Z975) -> tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return float(center - half), float(center + half)
