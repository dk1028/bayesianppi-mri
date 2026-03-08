"""Create the preprocessing pipeline figure.

This revision uses the same preprocessing as the training code:
non-zero bounding-box crop/pad, resize to 64^3, and min-max normalization.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

from _shared import FIGS_ROOT, NIFTI_ROOT, crop_or_pad_to_cube, load_mid_axial_slice, preprocess_volume


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
cropped = crop_or_pad_to_cube(vol)
cropped_slice = cropped[:, :, cropped.shape[2] // 2]
proc_slice = preprocess_volume(nii_path).numpy()[0][:, :, 32]

fig, axes = plt.subplots(1, 3, figsize=(9, 3))
axes[0].imshow(raw_slice, cmap="gray")
axes[0].set_title("(A) Raw NIfTI slice")
axes[0].axis("off")
axes[1].imshow(cropped_slice, cmap="gray")
axes[1].set_title("(B) Cropped / padded")
axes[1].axis("off")
axes[2].imshow(proc_slice, cmap="gray")
axes[2].set_title("(C) Resized & normalized (64x64)")
axes[2].axis("off")
plt.tight_layout()
out_png = FIGS_ROOT / "fig1_pipeline.png"
out_pdf = FIGS_ROOT / "fig1_pipeline.pdf"
plt.savefig(out_png, dpi=300)
plt.savefig(out_pdf, bbox_inches="tight")
plt.close()
print(f"Saved {out_png}")
print(f"Saved {out_pdf}")
