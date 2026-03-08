from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from _shared import FIGS_ROOT, load_mid_axial_slice, resolve_nifti_path


def _format_label(h: object) -> str:
    try:
        return "AD" if int(h) == 1 else "CN"
    except (TypeError, ValueError):
        return str(h)


def create_example_figure(pred_csv: Path, out_path: Path, suptitle: str) -> None:
    df = pd.read_csv(pred_csv)
    if "autorater_prediction" not in df.columns:
        raise ValueError(f"Missing 'autorater_prediction' in {pred_csv}")
    if "H" not in df.columns and "label" in df.columns:
        df["H"] = (df["label"].astype(str).str.upper() == "AD").astype(int)
    if "label" not in df.columns and "H" in df.columns:
        df["label"] = df["H"].map({1: "AD", 0: "CN"})

    ad_row = df[df["label"].astype(str).str.upper() == "AD"].sort_values(
        "autorater_prediction", ascending=False
    ).iloc[0]
    cn_row = df[df["label"].astype(str).str.upper() == "CN"].sort_values(
        "autorater_prediction", ascending=True
    ).iloc[0]

    ad_slice = load_mid_axial_slice(resolve_nifti_path(ad_row["nifti_path"]), processed=True)
    cn_slice = load_mid_axial_slice(resolve_nifti_path(cn_row["nifti_path"]), processed=True)

    fig, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)

    axes[0, 0].imshow(ad_slice, cmap="gray")
    axes[0, 0].set_title("(A) AD MRI Slice")
    axes[0, 0].axis("off")
    axes[0, 1].bar(["P(AD)"], [float(ad_row["autorater_prediction"])], color="red")
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].set_title("(B) Autorater Prediction")
    axes[0, 2].text(
        0.5,
        0.5,
        f"Ground-Truth H = {_format_label(ad_row['H'])}",
        fontsize=14,
        ha="center",
        va="center",
    )
    axes[0, 2].axis("off")
    axes[0, 2].set_title("(C) Ground-Truth Label")

    axes[1, 0].imshow(cn_slice, cmap="gray")
    axes[1, 0].set_title("(D) CN MRI Slice")
    axes[1, 0].axis("off")
    axes[1, 1].bar(["P(AD)"], [float(cn_row["autorater_prediction"])], color="blue")
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].set_title("(E) Autorater Prediction")
    axes[1, 2].text(
        0.5,
        0.5,
        f"Ground-Truth H = {_format_label(cn_row['H'])}",
        fontsize=14,
        ha="center",
        va="center",
    )
    axes[1, 2].axis("off")
    axes[1, 2].set_title("(F) Ground-Truth Label")

    plt.suptitle(suptitle, fontsize=16)
    FIGS_ROOT.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved figure to {out_path}")
