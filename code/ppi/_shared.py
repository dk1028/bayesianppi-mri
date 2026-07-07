from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = REPO_ROOT / "results"
FIGS_ROOT = REPO_ROOT / "figs"

Z975 = 1.959963984540054


@dataclass(frozen=True)
class PosteriorSummary:
    mean: float
    ci_low: float
    ci_high: float

    @property
    def width(self) -> float:
        return self.ci_high - self.ci_low


def load_prediction_csv(csv_path: Path, threshold: float = 0.5) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find input CSV: {csv_path}")

    df = pd.read_csv(csv_path).copy()
    df.columns = [c.strip() for c in df.columns]

    required = {"subject_id", "autorater_prediction"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path.name} missing required columns: {sorted(missing)}")

    if "H" in df.columns:
        df["H"] = pd.to_numeric(df["H"], errors="coerce")
    elif "label" in df.columns:
        labels = df["label"].astype(str).str.upper().str.strip()
        df["H"] = (labels == "AD").astype(int)
    else:
        raise ValueError(f"{csv_path.name} must contain H or label")

    df["A_prob"] = pd.to_numeric(df["autorater_prediction"], errors="coerce")
    df = df.dropna(subset=["subject_id", "H", "A_prob"]).copy()
    df = df[df["A_prob"].between(0, 1, inclusive="both")].copy()

    df["H"] = df["H"].astype(int)
    df["subject_id"] = df["subject_id"].astype(str).str.strip()
    df["A_class"] = (df["A_prob"] >= threshold).astype(int)

    if "Age" in df.columns:
        df["Age"] = pd.to_numeric(df["Age"], errors="coerce")

    return df.reset_index(drop=True)


def posterior_summary(draws: np.ndarray) -> PosteriorSummary:
    draws = np.asarray(draws, dtype=float)
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return PosteriorSummary(
        mean=float(draws.mean()),
        ci_low=float(lo),
        ci_high=float(hi),
    )


def beta_posterior_draws(
    alpha: float,
    beta: float,
    n_success: float,
    n_total: float,
    size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    return rng.beta(
        alpha + n_success,
        beta + max(n_total - n_success, 0.0),
        size=size,
    )


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


def labeled_only_posterior_draws(
    df: pd.DataFrame,
    labeled_idx: np.ndarray,
    alpha: float,
    beta: float,
    size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    sub = df.iloc[labeled_idx]
    successes = float(sub["H"].sum())
    return rng.beta(
        alpha + successes,
        beta + len(sub) - successes,
        size=size,
    )


def binary_difference_estimator(
    df: pd.DataFrame,
    labeled_idx: np.ndarray,
    n_boot: int,
    rng: np.random.Generator,
) -> PosteriorSummary:
    a_bar = float(df["A_class"].mean())
    sub = df.iloc[labeled_idx]
    resid = (sub["H"] - sub["A_class"]).to_numpy(dtype=float)
    g_hat = a_bar + float(resid.mean())

    boots = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        sample = rng.choice(resid, size=len(resid), replace=True)
        boots[b] = a_bar + float(sample.mean())

    lo, hi = np.quantile(boots, [0.025, 0.975])
    return PosteriorSummary(g_hat, float(lo), float(hi))


def ppi_analytic_estimator(
    df: pd.DataFrame,
    labeled_idx: np.ndarray,
) -> PosteriorSummary:
    p_all = df["A_prob"].to_numpy(dtype=float)
    n_all = len(p_all)

    sub = df.iloc[labeled_idx]
    resid = sub["H"].to_numpy(dtype=float) - sub["A_prob"].to_numpy(dtype=float)
    n_lab = len(resid)

    g_hat = float(p_all.mean() + resid.mean())
    var_p = float(np.var(p_all, ddof=1)) if n_all > 1 else 0.0
    var_r = float(np.var(resid, ddof=1)) if n_lab > 1 else 0.0
    se = float(np.sqrt(var_p / n_all + var_r / n_lab))

    crit = Z975 if n_lab >= 30 else float(student_t.ppf(0.975, df=max(n_lab - 1, 1)))
    return PosteriorSummary(g_hat, g_hat - crit * se, g_hat + crit * se)


def ppipp_manuscript_estimator(
    df: pd.DataFrame,
    labeled_idx: np.ndarray,
) -> tuple[PosteriorSummary, float]:
    """Power-tuned PPI formula used to reproduce the manuscript tables.

    The full observed prediction pool is treated as the monitoring pool, while
    the selected labeled rows provide the rectifier. The tuning parameter is
    clipped to [0, 1].
    """
    p_all = df["A_prob"].to_numpy(dtype=float)
    n_all = len(p_all)

    sub = df.iloc[labeled_idx]
    y_lab = sub["H"].to_numpy(dtype=float)
    p_lab = sub["A_prob"].to_numpy(dtype=float)
    n_lab = len(y_lab)

    if n_lab < 2 or n_all < 2:
        return ppi_analytic_estimator(df, labeled_idx), 1.0

    var_p_all = float(np.var(p_all, ddof=1))
    var_p_lab = float(np.var(p_lab, ddof=1))
    cov_hp = float(np.cov(y_lab, p_lab, ddof=1)[0, 1])

    denom = var_p_lab / n_lab + var_p_all / n_all
    lam = 0.0 if denom <= 0 or not np.isfinite(denom) else (cov_hp / n_lab) / denom
    lam = float(np.clip(lam, 0.0, 1.0))

    resid = y_lab - lam * p_lab
    g_hat = float(lam * p_all.mean() + resid.mean())

    var_resid = float(np.var(resid, ddof=1)) if n_lab > 1 else 0.0
    se = float(np.sqrt((lam**2) * var_p_all / n_all + var_resid / n_lab))
    crit = Z975 if n_lab >= 30 else float(student_t.ppf(0.975, df=max(n_lab - 1, 1)))

    return PosteriorSummary(g_hat, g_hat - crit * se, g_hat + crit * se), lam


def official_ppipp_mean_ci(
    y_labeled: np.ndarray,
    pred_labeled: np.ndarray,
    pred_unlabeled: np.ndarray,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Optional canonical disjoint PPI++ interval using ppi-python."""
    try:
        from ppi_py import ppi_mean_ci
    except ImportError as exc:
        raise ImportError(
            "Install ppi-python==0.2.3 to use official_ppipp_mean_ci"
        ) from exc

    lo, hi = ppi_mean_ci(
        np.asarray(y_labeled),
        np.asarray(pred_labeled),
        np.asarray(pred_unlabeled),
        alpha=alpha,
    )
    return float(lo), float(hi)


def coverage_and_width(
    summary: PosteriorSummary,
    truth: float,
) -> tuple[bool, float]:
    covered = summary.ci_low <= truth <= summary.ci_high
    return bool(covered), summary.width


def wilson_interval(k: int, n: int, z: float = Z975) -> tuple[float, float]:
    if n <= 0:
        return np.nan, np.nan
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt((p * (1 - p) / n) + z**2 / (4 * n**2)) / denom
    return float(center - half), float(center + half)


def assign_bins(
    scores: pd.Series,
    k: int,
    strategy: Literal["quantile", "uniform"] = "quantile",
) -> pd.Series:
    scores = pd.to_numeric(scores, errors="coerce")
    if strategy == "quantile":
        return pd.qcut(
            scores,
            q=min(k, scores.nunique()),
            labels=False,
            duplicates="drop",
        )
    return pd.cut(scores, bins=k, labels=False, include_lowest=True)
