from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from _shared import Z975, load_prediction_csv, wilson_interval

AGE_BINS = [50, 74, 80, 101]
AGE_LABELS = ["50-73", "74-79", "80-100"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run revised ADNI ROC, threshold, calibration, and prevalence audits."
    )
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--permutations", type=int, default=5000)
    parser.add_argument("--threshold-bootstrap", type=int, default=2000)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=2025)
    return parser.parse_args()


def binary_metrics(y_true: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, float | int]:
    y_true = np.asarray(y_true, dtype=int)
    score = np.asarray(score, dtype=float)
    yhat = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, yhat, labels=[0, 1]).ravel()

    acc_lo, acc_hi = wilson_interval(int((yhat == y_true).sum()), len(y_true))
    tpr_lo, tpr_hi = wilson_interval(int(tp), int(tp + fn))
    tnr_lo, tnr_hi = wilson_interval(int(tn), int(tn + fp))

    return {
        "ACC": float(accuracy_score(y_true, yhat)),
        "ACC_lo": acc_lo,
        "ACC_hi": acc_hi,
        "TPR": float(tp / (tp + fn)) if (tp + fn) else np.nan,
        "TPR_lo": tpr_lo,
        "TPR_hi": tpr_hi,
        "TNR": float(tn / (tn + fp)) if (tn + fp) else np.nan,
        "TNR_lo": tnr_lo,
        "TNR_hi": tnr_hi,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


def bootstrap_auc_ci(y: np.ndarray, p: np.ndarray, *, B: int, seed: int) -> tuple[float, float, float]:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    rng = np.random.default_rng(seed)
    vals: list[float] = []

    for _ in range(B):
        idx = rng.integers(0, len(y), size=len(y))
        if np.unique(y[idx]).size < 2:
            continue
        vals.append(float(roc_auc_score(y[idx], p[idx])))

    lo, hi = np.quantile(vals, [0.025, 0.975])
    return float(roc_auc_score(y, p)), float(lo), float(hi)


def cluster_bootstrap_auc_ci(
    df: pd.DataFrame,
    *,
    B: int,
    seed: int,
) -> tuple[float, float, float]:
    """Optional subject-cluster bootstrap sensitivity."""
    rng = np.random.default_rng(seed)
    subjects = df["subject_id"].drop_duplicates().to_numpy()
    vals: list[float] = []

    for _ in range(B):
        sampled = rng.choice(subjects, size=len(subjects), replace=True)
        pieces = [df[df["subject_id"] == subject] for subject in sampled]
        boot = pd.concat(pieces, ignore_index=True)
        if boot["H"].nunique() < 2:
            continue
        vals.append(float(roc_auc_score(boot["H"], boot["A_prob"])))

    lo, hi = np.quantile(vals, [0.025, 0.975])
    return float(roc_auc_score(df["H"], df["A_prob"])), float(lo), float(hi)


def youden_threshold(y: np.ndarray, p: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(y, p)
    return float(thresholds[int(np.argmax(tpr - fpr))])


def holm_adjust(pvals: np.ndarray) -> np.ndarray:
    pvals = np.asarray(pvals, dtype=float)
    order = np.argsort(pvals)
    adjusted = np.empty(len(pvals), dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(pvals) - rank) * pvals[idx])
        adjusted[idx] = min(running, 1.0)
    return adjusted


def perm_test_auc_diff(
    df: pd.DataFrame,
    bin1: str,
    bin2: str,
    *,
    B: int,
    seed: int,
) -> dict[str, float | int | str]:
    d1 = df[df["age_bin"] == bin1]
    d2 = df[df["age_bin"] == bin2]

    y1, p1 = d1["H"].to_numpy(), d1["A_prob"].to_numpy()
    y2, p2 = d2["H"].to_numpy(), d2["A_prob"].to_numpy()
    auc1 = float(roc_auc_score(y1, p1))
    auc2 = float(roc_auc_score(y2, p2))
    observed = auc1 - auc2

    y_all = np.concatenate([y1, y2])
    p_all = np.concatenate([p1, p2])
    n1 = len(y1)
    rng = np.random.default_rng(seed)
    null_vals: list[float] = []

    for _ in range(B):
        perm = rng.permutation(len(y_all))
        i1, i2 = perm[:n1], perm[n1:]
        if np.unique(y_all[i1]).size < 2 or np.unique(y_all[i2]).size < 2:
            continue
        null_vals.append(
            float(
                roc_auc_score(y_all[i1], p_all[i1])
                - roc_auc_score(y_all[i2], p_all[i2])
            )
        )

    pval = float(np.mean(np.abs(null_vals) >= abs(observed)))
    return {
        "bin1": bin1,
        "bin2": bin2,
        "n1": len(d1),
        "n2": len(d2),
        "auc1": auc1,
        "auc2": auc2,
        "auc_diff": observed,
        "p_raw": pval,
        "B_used": len(null_vals),
    }


def ppi_all_labels(df: pd.DataFrame) -> tuple[float, float, float]:
    p = df["A_prob"].to_numpy(dtype=float)
    y = df["H"].to_numpy(dtype=float)
    n = len(df)
    resid = y - p
    estimate = float(p.mean() + resid.mean())
    se = np.sqrt(np.var(p, ddof=1) / n + np.var(resid, ddof=1) / n)
    return estimate, float(estimate - Z975 * se), float(estimate + Z975 * se)


def ppipp_all_labels(df: pd.DataFrame) -> tuple[float, float, float, float]:
    p = df["A_prob"].to_numpy(dtype=float)
    y = df["H"].to_numpy(dtype=float)
    n = len(df)
    var_p = float(np.var(p, ddof=1))
    cov_hp = float(np.cov(y, p, ddof=1)[0, 1])
    denom = 2 * var_p / n
    lam = 0.0 if denom <= 0 else float(np.clip((cov_hp / n) / denom, 0.0, 1.0))
    resid = y - lam * p
    estimate = float(lam * p.mean() + resid.mean())
    se = np.sqrt((lam**2) * var_p / n + np.var(resid, ddof=1) / n)
    return estimate, float(estimate - Z975 * se), float(estimate + Z975 * se), lam


def cre_all_labels(
    df: pd.DataFrame,
    *,
    alpha: float,
    beta: float,
    draws: int,
    seed: int,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    A = df["A_class"].to_numpy(dtype=int)
    H = df["H"].to_numpy(dtype=int)
    mask1 = A == 1
    mask0 = A == 0

    theta_a = rng.beta(alpha + A.sum(), beta + len(A) - A.sum(), size=draws)
    theta_h1 = rng.beta(
        alpha + H[mask1].sum(),
        beta + mask1.sum() - H[mask1].sum(),
        size=draws,
    )
    theta_h0 = rng.beta(
        alpha + H[mask0].sum(),
        beta + mask0.sum() - H[mask0].sum(),
        size=draws,
    )
    g = theta_a * theta_h1 + (1 - theta_a) * theta_h0
    lo, hi = np.quantile(g, [0.025, 0.975])
    return float(g.mean()), float(lo), float(hi)


def plot_roc(df: pd.DataFrame, title: str, path: Path) -> None:
    fpr, tpr, _ = roc_curve(df["H"], df["A_prob"])
    auc_value = roc_auc_score(df["H"], df["A_prob"])
    plt.figure(figsize=(5.5, 5.0))
    plt.plot(fpr, tpr, label=f"AUC = {auc_value:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Chance")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def threshold_oof_audit(df: pd.DataFrame, *, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    rows: list[dict[str, object]] = []

    for fold, (train_idx, test_idx) in enumerate(
        splitter.split(df, y=df["H"], groups=df["subject_id"]), start=1
    ):
        train = df.iloc[train_idx]
        test = df.iloc[test_idx]
        overlap = set(train["subject_id"]) & set(test["subject_id"])
        if overlap:
            raise RuntimeError("Subject overlap in threshold audit")
        threshold = youden_threshold(train["H"].to_numpy(), train["A_prob"].to_numpy())
        rows.append(
            {
                "fold": fold,
                "threshold_train": threshold,
                "n_train_rows": len(train),
                "n_test_rows": len(test),
                "n_train_subjects": train["subject_id"].nunique(),
                "n_test_subjects": test["subject_id"].nunique(),
                "subject_overlap": 0,
                **binary_metrics(test["H"].to_numpy(), test["A_prob"].to_numpy(), threshold),
            }
        )

    leaky_threshold = youden_threshold(df["H"].to_numpy(), df["A_prob"].to_numpy())
    leaky = pd.DataFrame(
        [
            {
                "fold": "leaky_full",
                "threshold_train": leaky_threshold,
                "n_rows": len(df),
                "n_subjects": df["subject_id"].nunique(),
                **binary_metrics(df["H"].to_numpy(), df["A_prob"].to_numpy(), leaky_threshold),
            }
        ]
    )
    return pd.DataFrame(rows), leaky


def exchangeability_diagnostic(
    df: pd.DataFrame,
    *,
    repetitions: int,
    seed: int,
    figure_path: Path,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int]] = []
    last = None

    for rep in range(repetitions):
        n_labeled = int(round(0.30 * len(df)))
        labeled_idx = rng.choice(len(df), size=n_labeled, replace=False)
        label_indicator = np.zeros(len(df), dtype=int)
        label_indicator[labeled_idx] = 1
        X = df[["Age", "A_prob"]].to_numpy(dtype=float)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            label_indicator,
            test_size=0.35,
            random_state=seed + rep,
            stratify=label_indicator,
        )
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        model.fit(X_train, y_train)
        pred = model.predict_proba(X_test)[:, 1]
        rows.append(
            {
                "rep": rep,
                "n_labeled": n_labeled,
                "N": len(df),
                "propensity_auc": float(roc_auc_score(y_test, pred)),
            }
        )
        last = (y_test, pred)

    if last is not None:
        y_test, pred = last
        plt.figure(figsize=(6, 4))
        plt.hist(pred[y_test == 1], bins=20, alpha=0.6, density=True, label="Labeled")
        plt.hist(pred[y_test == 0], bins=20, alpha=0.6, density=True, label="Unlabeled")
        plt.xlabel("Predicted P(labeled | Age, score)")
        plt.ylabel("Density")
        plt.title("Random-subsampling propensity overlap")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(figure_path, dpi=300, bbox_inches="tight")
        plt.close()

    return pd.DataFrame(rows)


def threshold_bootstrap(
    df: pd.DataFrame,
    *,
    B: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    y = df["H"].to_numpy(dtype=int)
    p = df["A_prob"].to_numpy(dtype=float)
    rows: list[dict[str, float | int]] = []

    for rep in tqdm(range(B), desc="Threshold bootstrap"):
        idx = rng.integers(0, len(df), size=len(df))
        if np.unique(y[idx]).size < 2:
            continue
        threshold = youden_threshold(y[idx], p[idx])
        rows.append({"rep": rep, "threshold": threshold, **binary_metrics(y, p, threshold)})

    samples = pd.DataFrame(rows)
    summary = pd.DataFrame(
        [
            {
                "B_requested": B,
                "B_used": len(samples),
                "threshold_mean": samples["threshold"].mean(),
                "threshold_lo": samples["threshold"].quantile(0.025),
                "threshold_hi": samples["threshold"].quantile(0.975),
                "ACC_mean": samples["ACC"].mean(),
                "ACC_lo": samples["ACC"].quantile(0.025),
                "ACC_hi": samples["ACC"].quantile(0.975),
                "TPR_mean": samples["TPR"].mean(),
                "TPR_lo": samples["TPR"].quantile(0.025),
                "TPR_hi": samples["TPR"].quantile(0.975),
                "TNR_mean": samples["TNR"].mean(),
                "TNR_lo": samples["TNR"].quantile(0.025),
                "TNR_hi": samples["TNR"].quantile(0.975),
            }
        ]
    )
    return samples, summary


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = args.output_dir / "figures"
    cal_dir = args.output_dir / "calibration_curves"
    fig_dir.mkdir(parents=True, exist_ok=True)
    cal_dir.mkdir(parents=True, exist_ok=True)

    df = load_prediction_csv(args.input_csv, threshold=args.threshold)
    required = {"Age", "oof_fold"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Case-study input missing columns: {sorted(missing)}")

    fold_counts = df.groupby("subject_id")["oof_fold"].nunique()
    if int((fold_counts > 1).sum()):
        raise RuntimeError("Subjects span multiple autorater OOF folds")

    df["age_bin"] = pd.cut(
        df["Age"],
        bins=AGE_BINS,
        labels=AGE_LABELS,
        right=False,
        include_lowest=True,
    )
    df["is_age_65_70"] = df["Age"].between(65, 70, inclusive="both")

    scopes: list[tuple[str, pd.DataFrame]] = [("Overall", df)]
    scopes += [(label, df[df["age_bin"] == label]) for label in AGE_LABELS]
    scopes.append(("Age 65-70", df[df["is_age_65_70"]]))

    metrics_rows: list[dict[str, object]] = []
    threshold_rows: list[dict[str, object]] = []
    calibration_rows: list[dict[str, object]] = []
    prevalence_rows: list[dict[str, object]] = []

    for scope_idx, (scope, sub) in enumerate(scopes):
        if len(sub) == 0 or sub["H"].nunique() < 2:
            continue

        auc_value, auc_lo, auc_hi = bootstrap_auc_ci(
            sub["H"].to_numpy(),
            sub["A_prob"].to_numpy(),
            B=args.bootstrap,
            seed=args.seed + scope_idx,
        )
        _, cluster_lo, cluster_hi = cluster_bootstrap_auc_ci(
            sub,
            B=args.bootstrap,
            seed=args.seed + 1000 + scope_idx,
        )
        fixed = binary_metrics(sub["H"].to_numpy(), sub["A_prob"].to_numpy(), args.threshold)
        metrics_rows.append(
            {
                "scope": scope,
                "N": len(sub),
                "subjects": sub["subject_id"].nunique(),
                "prevalence": sub["H"].mean(),
                "AUC": auc_value,
                "AUC_lo": auc_lo,
                "AUC_hi": auc_hi,
                "AUC_cluster_lo": cluster_lo,
                "AUC_cluster_hi": cluster_hi,
                **fixed,
            }
        )

        youden = youden_threshold(sub["H"].to_numpy(), sub["A_prob"].to_numpy())
        threshold_rows.append(
            {
                "scope": scope,
                "threshold_policy": "fixed_0.5",
                "threshold": args.threshold,
                "N": len(sub),
                "subjects": sub["subject_id"].nunique(),
                "prevalence": sub["H"].mean(),
                "AUC": auc_value,
                **fixed,
            }
        )
        threshold_rows.append(
            {
                "scope": scope,
                "threshold_policy": "youden_same_scope_descriptive",
                "threshold": youden,
                "N": len(sub),
                "subjects": sub["subject_id"].nunique(),
                "prevalence": sub["H"].mean(),
                "AUC": auc_value,
                **binary_metrics(sub["H"].to_numpy(), sub["A_prob"].to_numpy(), youden),
            }
        )

        brier = brier_score_loss(sub["H"], sub["A_prob"])
        calibration_rows.append(
            {
                "scope": scope,
                "N": len(sub),
                "subjects": sub["subject_id"].nunique(),
                "Brier": brier,
            }
        )
        frac_pos, mean_pred = calibration_curve(sub["H"], sub["A_prob"], n_bins=10, strategy="uniform")
        plt.figure(figsize=(5.2, 5.0))
        plt.plot(mean_pred, frac_pos, "o-", label="Empirical")
        plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect")
        plt.xlabel("Mean predicted probability")
        plt.ylabel("Fraction positive")
        plt.title(f"Calibration: {scope}, Brier={brier:.3f}")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        safe = scope.replace(" ", "_").replace("-", "_")
        plt.savefig(cal_dir / f"calibration_{safe}.png", dpi=300, bbox_inches="tight")
        plt.close()

        ppi_hat, ppi_lo, ppi_hi = ppi_all_labels(sub)
        pp_hat, pp_lo, pp_hi, lam = ppipp_all_labels(sub)
        cu_hat, cu_lo, cu_hi = cre_all_labels(
            sub, alpha=1.0, beta=1.0, draws=10000, seed=args.seed
        )
        cj_hat, cj_lo, cj_hi = cre_all_labels(
            sub, alpha=0.5, beta=0.5, draws=10000, seed=args.seed + 1
        )
        prevalence_rows.append(
            {
                "scope": scope,
                "N": len(sub),
                "subjects": sub["subject_id"].nunique(),
                "truth_prevalence": sub["H"].mean(),
                "AUC": auc_value,
                "ppi_hat": ppi_hat,
                "ppi_lo": ppi_lo,
                "ppi_hi": ppi_hi,
                "ppipp_hat": pp_hat,
                "ppipp_lo": pp_lo,
                "ppipp_hi": pp_hi,
                "ppipp_lambda": lam,
                "cre_uniform_hat": cu_hat,
                "cre_uniform_lo": cu_lo,
                "cre_uniform_hi": cu_hi,
                "cre_jeffreys_hat": cj_hat,
                "cre_jeffreys_lo": cj_lo,
                "cre_jeffreys_hi": cj_hi,
            }
        )

        plot_roc(sub, f"ROC: {scope}", fig_dir / f"fig_roc_{safe}.png")

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(args.output_dir / "metrics_overall_age_auc_ci.csv", index=False)
    pd.DataFrame(threshold_rows).to_csv(
        args.output_dir / "threshold_metrics_fixed_and_youden.csv", index=False
    )
    pd.DataFrame(calibration_rows).to_csv(
        args.output_dir / "calibration_brier_by_scope.csv", index=False
    )
    pd.DataFrame(prevalence_rows).to_csv(
        args.output_dir / "prevalence_estimators_realdata_subject_oof_scan_age.csv",
        index=False,
    )

    pair_rows = [
        perm_test_auc_diff(
            df,
            b1,
            b2,
            B=args.permutations,
            seed=args.seed + 100 + idx,
        )
        for idx, (b1, b2) in enumerate(itertools.combinations(AGE_LABELS, 2))
    ]
    pair_df = pd.DataFrame(pair_rows)
    pair_df["p_holm"] = holm_adjust(pair_df["p_raw"].to_numpy())
    pair_df["method"] = "Two-sided permutation test with Holm correction"
    pair_df.to_csv(args.output_dir / "pairwise_auc_perm_test_subject_oof.csv", index=False)

    threshold_oof, threshold_leaky = threshold_oof_audit(df, seed=args.seed)
    threshold_oof.to_csv(args.output_dir / "threshold_oof_subject_grouped.csv", index=False)
    threshold_leaky.to_csv(args.output_dir / "threshold_leaky_full.csv", index=False)

    exchange = exchangeability_diagnostic(
        df,
        repetitions=50,
        seed=args.seed,
        figure_path=fig_dir / "fig_exchangeability_propensity_overlap.png",
    )
    exchange.to_csv(args.output_dir / "exchangeability_auc_random_labeling.csv", index=False)

    threshold_samples, threshold_summary = threshold_bootstrap(
        df,
        B=args.threshold_bootstrap,
        seed=args.seed,
    )
    threshold_samples.to_csv(args.output_dir / "threshold_bootstrap_samples.csv", index=False)
    threshold_summary.to_csv(args.output_dir / "threshold_bootstrap_summary.csv", index=False)

    plt.figure(figsize=(6, 4))
    plt.hist(threshold_samples["threshold"], bins=30, density=True, alpha=0.8)
    plt.xlabel("Youden threshold")
    plt.ylabel("Density")
    plt.title("Bootstrap dispersion of Youden threshold")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(fig_dir / "fig_threshold_bootstrap_hist.png", dpi=300, bbox_inches="tight")
    plt.close()

    run_config = {
        "input_csv": str(args.input_csv),
        "rows": len(df),
        "subjects": int(df["subject_id"].nunique()),
        "subjects_spanning_multiple_oof_folds": int((fold_counts > 1).sum()),
        "bootstrap": args.bootstrap,
        "permutations": args.permutations,
        "threshold_bootstrap": args.threshold_bootstrap,
        "threshold": args.threshold,
        "seed": args.seed,
        "subjects_in_multiple_age_bins": int(
            (df.groupby("subject_id")["age_bin"].nunique() > 1).sum()
        ),
    }
    (args.output_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2), encoding="utf-8"
    )

    print(metrics_df.round(4).to_string(index=False))
    print(pair_df.round(4).to_string(index=False))
    print(threshold_summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
