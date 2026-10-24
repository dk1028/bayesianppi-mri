"""Illustrate the manuscript preprocessing: direct resize, then volume z-score."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

from _shared import (
    FIGS_ROOT,
    NIFTI_ROOT,
    TARGET_SHAPE,
    load_mid_axial_slice,
    preprocess_volume,
)


root_dir = NIFTI_ROOT
if not root_dir.exists():
    raise FileNotFoundError(f"NIFTI root directory not found: {root_dir}")

subj_dirs = sorted([p for p in root_dir.iterdir() if p.is_dir()])
if not subj_dirs:
    raise RuntimeError(f"No subject folders found under {root_dir}")
subj_folder = subj_dirs[0]
nii_files = [p for p in subj_folder.iterdir() if str(p).endswith((".nii", ".nii.gz"))]
if not nii_files:
    raise RuntimeError(f"No NIfTI files found in {subj_folder}")
nii_path = nii_files[0]

vol = nib.load(str(nii_path)).get_fdata().astype(np.float32)
raw_slice = load_mid_axial_slice(nii_path, processed=False)
resized = F.interpolate(
    torch.from_numpy(vol).unsqueeze(0).unsqueeze(0),
    size=TARGET_SHAPE,
    mode="trilinear",
    align_corners=False,
).squeeze(0).squeeze(0).numpy()
zscored = preprocess_volume(nii_path).numpy()[0]
mid = TARGET_SHAPE[2] // 2

fig, axes = plt.subplots(1, 3, figsize=(9, 3))
axes[0].imshow(raw_slice, cmap="gray")
axes[0].set_title("(A) Original NIfTI slice")
axes[0].axis("off")
axes[1].imshow(resized[:, :, mid], cmap="gray")
axes[1].set_title("(B) Trilinear resize (64³)")
axes[1].axis("off")
axes[2].imshow(zscored[:, :, mid], cmap="gray")
axes[2].set_title("(C) Per-volume z-score")
axes[2].axis("off")
plt.tight_layout()
out_png = FIGS_ROOT / "fig1_pipeline.png"
out_pdf = FIGS_ROOT / "fig1_pipeline.pdf"
plt.savefig(out_png, dpi=300)
plt.savefig(out_pdf, bbox_inches="tight")
plt.close()
print(f"Saved {out_png}")
print(f"Saved {out_pdf}")
