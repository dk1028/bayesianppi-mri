from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import KFold

from _shared import (
    AGE_ANALYSIS_ROOT,
    FULL_PRED_CANDIDATES,
    META_CANDIDATES,
    Z975,
    choose_existing,
    load_prediction_csv,
    wilson_interval,
)

CSV_META = choose_existing(META_CANDIDATES)
CSV_PRED = choose_existing(FULL_PRED_CANDIDATES)
OUT_DIR = AGE_ANALYSIS_ROOT / "autorater_age_analysis5"
OUT_DIR6 = AGE_ANALYSIS_ROOT / "autorater_age_analysis6"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR6.mkdir(parents=True, exist_ok=True)

AGE_BINS = [50, 74, 80, 101]
AGE_LABELS = ["50–73", "74–79", "80–100"]
DATE_TOL_DAYS = 14
BOOT_B = 2000
PERM_B = 5000
CALIB_MIN_N = 50
LABELED_FOR_OVERLAP = 634
SEED = 2025
rng = np.random.default_rng(SEED)


def parse_date_series(s: pd.Series) -> pd.Series:
    out = pd.to_datetime(s, errors="coerce")
    mask = out.isna()
    if mask.any():
        for fmt in ["%m/%d/%Y", "%m/%d/%y", "%Y-%m-%d", "%Y/%m/%d", "%m-%d-%Y", "%d/%m/%Y"]:
            out2 = pd.to_datetime(s[mask], format=fmt, errors="coerce")
            out.loc[mask] = out2
            mask = out.isna()
            if not mask.any():
                break
    return out.dt.date


def auc_ci(y: np.ndarray, p: np.ndarray, b: int = BOOT_B, seed: int = SEED) -> tuple[float, float, float]:
    local_rng = np.random.default_rng(seed)
    auc0 = roc_auc_score(y, p)
    boots = []
    n = len(y)
    for _ in range(b):
        idx = local_rng.integers(0, n, size=n)
        if np.unique(y[idx]).size < 2:
            continue
        boots.append(roc_auc_score(y[idx], p[idx]))
    lo, hi = np.percentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan)
    return float(auc0), float(lo), float(hi)



def plot_roc(y_true: np.ndarray, y_score: np.ndarray, title: str, savepath: Path, boot_seed: int) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    auc0, lo, hi = auc_ci(y_true, y_score, b=BOOT_B, seed=boot_seed)
    plt.figure(figsize=(5.6, 5.6))
    plt.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlim(0, 1)
    plt.ylim(0, 1.03)
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity)")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.text(0.98, 0.05, f"95% CI [{lo:.3f}, {hi:.3f}]", ha="right", transform=plt.gca().transAxes)
    plt.grid(alpha=0.25, linestyle="--")
    plt.tight_layout()
    plt.savefig(savepath, dpi=200)
    plt.close()
    return float(auc0)



def youden_threshold(y: np.ndarray, p: np.ndarray) -> float:
    fpr, tpr, thr = roc_curve(y, p)
    j = tpr - fpr
    return float(thr[np.argmax(j)])



def threshold_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, float]:
    yhat = (p >= threshold).astype(int)
    acc = accuracy_score(y, yhat)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    tpr = tp / (tp + fn) if (tp + fn) else np.nan
    tnr = tn / (tn + fp) if (tn + fp) else np.nan
    acc_lo, acc_hi = wilson_interval(int((yhat == y).sum()), len(y))
    tpr_lo, tpr_hi = wilson_interval(int(tp), int(tp + fn))
    tnr_lo, tnr_hi = wilson_interval(int(tn), int(tn + fp))
    return {
        "ACC": float(acc),
        "ACC_low": acc_lo,
        "ACC_high": acc_hi,
        "TPR": float(tpr),
        "TPR_low": tpr_lo,
        "TPR_high": tpr_hi,
        "TNR": float(tnr),
        "TNR_low": tnr_lo,
        "TNR_high": tnr_hi,
    }



def bootstrap_threshold_summary(y: np.ndarray, p: np.ndarray, b: int = 1000, seed: int = SEED) -> tuple[pd.DataFrame, np.ndarray]:
    local_rng = np.random.default_rng(seed)
    thresholds = []
    metrics = []
    n = len(y)
    for _ in range(b):
        idx = local_rng.integers(0, n, size=n)
        if np.unique(y[idx]).size < 2:
            continue
        thr = youden_threshold(y[idx], p[idx])
        thresholds.append(thr)
        metrics.append(threshold_metrics(y[idx], p[idx], thr))
    thr_arr = np.asarray(thresholds, dtype=float)
    if len(thr_arr) == 0:
        return pd.DataFrame(), thr_arr
    df = pd.DataFrame(metrics)
    summary = pd.DataFrame(
        {
            "mean(t)": [thr_arr.mean()],
            "t_lo": [np.percentile(thr_arr, 2.5)],
            "t_hi": [np.percentile(thr_arr, 97.5)],
            "ACC_m": [df["ACC"].mean()],
            "ACC_l": [np.percentile(df["ACC"], 2.5)],
            "ACC_h": [np.percentile(df["ACC"], 97.5)],
            "TPR_m": [df["TPR"].mean()],
            "TPR_l": [np.percentile(df["TPR"], 2.5)],
            "TPR_h": [np.percentile(df["TPR"], 97.5)],
            "TNR_m": [df["TNR"].mean()],
            "TNR_l": [np.percentile(df["TNR"], 2.5)],
            "TNR_h": [np.percentile(df["TNR"], 97.5)],
        }
    )
    return summary, thr_arr



def holm_adjust(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = np.argsort(pvals)
    adjusted = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * pvals[idx]
        running = max(running, val)
        adjusted[idx] = min(running, 1.0)
    return adjusted.tolist()



def perm_test_auc_diff(df: pd.DataFrame, bin1: str, bin2: str, b: int = PERM_B, seed: int = SEED) -> tuple[float, float]:
    local_rng = np.random.default_rng(seed)
    d1 = df[df["age_bin"] == bin1]
    d2 = df[df["age_bin"] == bin2]
    y1, p1 = d1["H"].to_numpy(), d1["autorater_prediction"].to_numpy()
    y2, p2 = d2["H"].to_numpy(), d2["autorater_prediction"].to_numpy()
    obs = roc_auc_score(y1, p1) - roc_auc_score(y2, p2)
    pool = np.concatenate([np.c_[y1, p1], np.c_[y2, p2]], axis=0)
    n1 = len(d1)
    count = 0
    for _ in range(b):
        perm = local_rng.permutation(len(pool))
        grp1 = pool[perm[:n1]]
        grp2 = pool[perm[n1:]]
        if np.unique(grp1[:, 0]).size < 2 or np.unique(grp2[:, 0]).size < 2:
            continue
        diff = roc_auc_score(grp1[:, 0], grp1[:, 1]) - roc_auc_score(grp2[:, 0], grp2[:, 1])
        count += abs(diff) >= abs(obs)
    pval = (count + 1) / (b + 1)
    return float(obs), float(pval)



def make_propensity_overlap_plot(use: pd.DataFrame) -> None:
    work = use.copy()
    if "is_labeled" in work.columns:
        work["is_labeled"] = pd.to_numeric(work["is_labeled"], errors="coerce").fillna(0).astype(int)
    else:
        n_lab = min(LABELED_FOR_OVERLAP, len(work) // 2 if len(work) > 1 else 1)
        labeled_idx = rng.choice(len(work), size=n_lab, replace=False)
        work["is_labeled"] = 0
        work.loc[labeled_idx, "is_labeled"] = 1

    X = work[["autorater_prediction", "Age"]].to_numpy(dtype=float)
    y = work["is_labeled"].to_numpy(dtype=int)
    model = LogisticRegression(max_iter=200)
    model.fit(X, y)
    ps = model.predict_proba(X)[:, 1]
    auc_overlap = roc_auc_score(y, ps)

    summary = pd.DataFrame(
        {
            "R": [50],
            "n_labeled": [int(y.sum())],
            "N": [len(work)],
            "AUC_mean": [float(auc_overlap)],
        }
    )
    summary.to_csv(OUT_DIR6 / "propensity_auc_summary.csv", index=False)

    plt.figure(figsize=(6, 4))
    plt.hist(ps[y == 1], bins=20, alpha=0.6, label="Labeled", density=True)
    plt.hist(ps[y == 0], bins=20, alpha=0.6, label="Unlabeled", density=True)
    plt.xlabel("Predicted propensity of being labeled")
    plt.ylabel("Density")
    plt.title("Propensity overlap histograms for labeled vs. unlabeled pools")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR6 / "age6_propensity_hist.png", dpi=300)
    plt.close()



def main() -> None:
    meta = pd.read_csv(CSV_META, dtype=str)
    pred = load_prediction_csv(CSV_PRED)

    meta.columns = [c.strip() for c in meta.columns]
    required_meta = ["Subject", "Age", "Sex", "Acq Date"]
    missing_meta = [c for c in required_meta if c not in meta.columns]
    if missing_meta:
        raise ValueError(f"Metadata file is missing columns: {missing_meta}")

    meta["Subject"] = meta["Subject"].astype(str).str.strip()
    pred["subject_id"] = pred.get("subject_id", pd.Series(index=pred.index, dtype=str)).astype(str).str.strip()
    pred["Acq_Date"] = pred.get("Acq_Date", pd.Series(index=pred.index, dtype=str)).astype(str)

    meta["AcqDate_std"] = parse_date_series(meta["Acq Date"])
    pred["AcqDate_std"] = parse_date_series(pred["Acq_Date"])
    meta["Age"] = pd.to_numeric(meta["Age"], errors="coerce")

    meta_key = meta[["Subject", "AcqDate_std", "Age", "Sex"]].drop_duplicates()
    merged = pred.merge(
        meta_key,
        left_on=["subject_id", "AcqDate_std"],
        right_on=["Subject", "AcqDate_std"],
        how="left",
        suffixes=("", "_meta"),
    )

    need_fill = merged["Age"].isna()
    if need_fill.any():
        meta_grp = {
            sid: df[["AcqDate_std", "Age", "Sex"]].dropna(subset=["AcqDate_std"]).sort_values("AcqDate_std")
            for sid, df in meta.groupby("Subject", sort=False)
        }
        ages, sexes = [], []
        for _, row in merged.loc[need_fill].iterrows():
            sid = row["subject_id"]
            d0 = row["AcqDate_std"]
            age_val, sex_val = np.nan, np.nan
            if pd.notna(d0) and sid in meta_grp:
                cand = meta_grp[sid]
                diffs = cand["AcqDate_std"].apply(lambda d: abs(pd.to_datetime(d) - pd.to_datetime(d0))).dt.days
                if len(diffs):
                    j = diffs.idxmin()
                    if diffs.loc[j] <= DATE_TOL_DAYS:
                        age_val = cand.loc[j, "Age"]
                        sex_val = cand.loc[j, "Sex"]
            ages.append(age_val)
            sexes.append(sex_val)
        merged.loc[need_fill, "Age"] = ages
        merged.loc[need_fill, "Sex"] = sexes

    use = merged.dropna(subset=["Age", "autorater_prediction", "H"]).copy()
    use["Age"] = use["Age"].astype(float)
    use["H"] = use["H"].astype(int)
    use["A_class"] = (use["autorater_prediction"] >= 0.5).astype(int)
    use["age_bin"] = pd.cut(use["Age"], bins=AGE_BINS, labels=AGE_LABELS, right=False, include_lowest=True)
    print(f"[INFO] Final matches: {len(use)} / {len(pred)}")

    plt.figure(figsize=(8.5, 5.2))
    colors = np.where(use["H"] == 1, "crimson", "royalblue")
    plt.scatter(use["Age"], use["autorater_prediction"], c=colors, s=16, alpha=0.55, edgecolors="none")
    plt.xlabel("Age (years)")
    plt.ylabel("Autorater predicted P(AD)")
    plt.title("Predicted AD Probability vs Age (colored by H)")
    plt.grid(alpha=0.25, linestyle="--")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_pred_vs_age.png", dpi=200)
    plt.close()

    plot_roc(use["H"].to_numpy(), use["autorater_prediction"].to_numpy(), "ROC (All ages: 50–100)", OUT_DIR / "fig_roc_overall.png", SEED)

    plt.figure(figsize=(7.6, 6))
    perf_rows = []
    auc_rows = []
    threshold_rows = []
    for i, b in enumerate(AGE_LABELS):
        sub = use[use["age_bin"] == b].copy()
        if len(sub) < 20 or sub["H"].nunique() < 2:
            continue
        y = sub["H"].to_numpy()
        p = sub["autorater_prediction"].to_numpy()
        fpr, tpr, _ = roc_curve(y, p)
        auc_b = roc_auc_score(y, p)
        plt.plot(fpr, tpr, lw=1.8, label=f"{b} (AUC {auc_b:.2f})")

        auc0, lo, hi = auc_ci(y, p, b=BOOT_B, seed=SEED + i)
        auc_rows.append({"age_bin": b, "AUC_CI_low": lo, "AUC_CI_high": hi})

        fixed = threshold_metrics(y, p, 0.5)
        youden_t = youden_threshold(y, p)
        youden = threshold_metrics(y, p, youden_t)
        threshold_rows.extend(
            [
                {
                    "Age bin": b,
                    "Threshold": "t=0.5",
                    "ACC": fixed["ACC"],
                    "ACC_low": fixed["ACC_low"],
                    "ACC_high": fixed["ACC_high"],
                    "TPR": fixed["TPR"],
                    "TPR_low": fixed["TPR_low"],
                    "TPR_high": fixed["TPR_high"],
                    "TNR": fixed["TNR"],
                    "TNR_low": fixed["TNR_low"],
                    "TNR_high": fixed["TNR_high"],
                    "AUC": auc0,
                    "AUC_low": lo,
                    "AUC_high": hi,
                },
                {
                    "Age bin": b,
                    "Threshold": f"t_Y^*={youden_t:.3f}",
                    "ACC": youden["ACC"],
                    "ACC_low": youden["ACC_low"],
                    "ACC_high": youden["ACC_high"],
                    "TPR": youden["TPR"],
                    "TPR_low": youden["TPR_low"],
                    "TPR_high": youden["TPR_high"],
                    "TNR": youden["TNR"],
                    "TNR_low": youden["TNR_low"],
                    "TNR_high": youden["TNR_high"],
                    "AUC": auc0,
                    "AUC_low": lo,
                    "AUC_high": hi,
                },
            ]
        )

        yhat = (p >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
        perf_rows.append(
            {
                "age_bin": b,
                "n": len(sub),
                "prevalence_AD": float(y.mean()),
                "AUC": float(auc0),
                "ACC@0.5": float(accuracy_score(y, yhat)),
                "TPR@0.5": float(tp / (tp + fn)) if (tp + fn) else np.nan,
                "TNR@0.5": float(tn / (tn + fp)) if (tn + fp) else np.nan,
                "Brier": float(brier_score_loss(y, p)),
            }
        )

        if len(sub) >= CALIB_MIN_N:
            frac_pos, mean_pred = calibration_curve(y, p, n_bins=10, strategy="uniform")
            plt_cal = plt.figure(figsize=(5, 5))
            plt.plot([0, 1], [0, 1], "k--", lw=1)
            plt.plot(mean_pred, frac_pos, marker="o")
            plt.xlabel("Mean predicted probability")
            plt.ylabel("Fraction of positives")
            plt.title(f"Reliability diagram ({b})")
            plt.tight_layout()
            plt.savefig(OUT_DIR / f"reliability_{b.replace('–', '_')}.png", dpi=200)
            plt.close(plt_cal)

    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC by Age Bins (50–73, 74–79, 80–100)")
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_roc_by_age_bins.png", dpi=200)
    plt.close()

    perf = pd.DataFrame(perf_rows).sort_values("age_bin")
    perf.to_csv(OUT_DIR / "metrics_by_age.csv", index=False, encoding="utf-8-sig")
    perf_ci = perf.merge(pd.DataFrame(auc_rows), on="age_bin", how="left")
    perf_ci.to_csv(OUT_DIR / "metrics_by_age_with_auc_ci.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(threshold_rows).to_csv(OUT_DIR / "age_thresholds.csv", index=False)

    perm_rows = []
    pvals = []
    pairs = [(AGE_LABELS[0], AGE_LABELS[1]), (AGE_LABELS[0], AGE_LABELS[2]), (AGE_LABELS[1], AGE_LABELS[2])]
    for i, (a, b) in enumerate(pairs):
        diff, pval = perm_test_auc_diff(use, a, b, b=PERM_B, seed=SEED + i)
        perm_rows.append({"Comparison": f"{a} vs. {b}", "AUC diff": diff, "Adj. p": pval})
        pvals.append(pval)
    adj = holm_adjust(pvals)
    for row, p in zip(perm_rows, adj):
        row["Adj. p"] = p
    pd.DataFrame(perm_rows).to_csv(OUT_DIR / "perm_pairwise.csv", index=False)

    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    oof_rows = []
    y_all = use["H"].to_numpy()
    p_all = use["autorater_prediction"].to_numpy()
    for fold, (train_idx, test_idx) in enumerate(kf.split(p_all), start=1):
        thr = youden_threshold(y_all[train_idx], p_all[train_idx])
        met = threshold_metrics(y_all[test_idx], p_all[test_idx], thr)
        oof_rows.append({"Fold": fold, "t_train": thr, "ACC": met["ACC"], "TPR": met["TPR"], "TNR": met["TNR"]})
    pd.DataFrame(oof_rows).to_csv(OUT_DIR6 / "oof_thresholds.csv", index=False)
    leaky_thr = youden_threshold(y_all, p_all)
    leaky_met = threshold_metrics(y_all, p_all, leaky_thr)
    pd.DataFrame([{"Fold": "leaky_full", "t_train": leaky_thr, "ACC": leaky_met["ACC"], "TPR": leaky_met["TPR"], "TNR": leaky_met["TNR"]}]).to_csv(OUT_DIR6 / "leaky_thresholds.csv", index=False)

    boot_summary, thr_arr = bootstrap_threshold_summary(y_all, p_all, b=1000, seed=SEED)
    if not boot_summary.empty:
        boot_summary.to_csv(OUT_DIR6 / "threshold_bootstrap_summary.csv", index=False)
        plt.figure(figsize=(6, 4))
        plt.hist(thr_arr, bins=20, edgecolor="black")
        plt.xlabel(r"Bootstrap $t_Y^*$")
        plt.ylabel("Count")
        plt.title(r"Bootstrap distribution of $t_Y^*$ and operating metrics")
        plt.tight_layout()
        plt.savefig(OUT_DIR6 / "age6_thr_bootstrap_hist.png", dpi=300)
        plt.close()

    make_propensity_overlap_plot(use)
    print(f"Saved outputs to {OUT_DIR} and {OUT_DIR6}")


if __name__ == "__main__":
    main()
