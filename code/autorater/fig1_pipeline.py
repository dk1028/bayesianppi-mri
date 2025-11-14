import os
from pathlib import Path

import nibabel as nb
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize

# --- paths relative to repo root ---
REPO_ROOT  = Path(__file__).resolve().parents[2]
DATA_ROOT  = REPO_ROOT / "data"
NIFTI_ROOT = DATA_ROOT / "nifti"
FIGS_ROOT  = REPO_ROOT / "figs"

# 1. Point to your ADNI_NIfTI root folder
root_dir = NIFTI_ROOT

if not root_dir.exists():
    raise FileNotFoundError(f"NIFTI root directory not found: {root_dir}")

# 2. Find one example subject (e.g. the first folder)
subj_dirs = sorted([p for p in root_dir.iterdir() if p.is_dir()])
if not subj_dirs:
    raise RuntimeError(f"No subject folders found under {root_dir}")

subj_folder = subj_dirs[0]

# 3. Find the .nii or .nii.gz file inside that folder
nii_files = [
    p for p in subj_folder.iterdir()
    if str(p).endswith(".nii") or str(p).endswith(".nii.gz")
]
if not nii_files:
    raise RuntimeError(f"No NIfTI files found in {subj_folder}")

nii_path = nii_files[0]

# 4. Load the NIfTI volume
img = nb.load(nii_path)
vol = img.get_fdata()

# 5. Extract the middle axial slice
z_mid = vol.shape[2] // 2
raw_slice = vol[:, :, z_mid]

# 6. Normalize to [0,1] and resize to 64x64
min_val = raw_slice.min()
max_val = raw_slice.max()
denom = max_val - min_val

if denom > 0:
    norm = (raw_slice - min_val) / denom
else:
    # fallback if the slice is constant
    norm = np.zeros_like(raw_slice)

proc_slice = resize(
    norm,
    (64, 64),
    order=1,
    mode="constant",
    cval=0.0,
    anti_aliasing=True,
)

# 7. Plot raw vs. preprocessed
fig, axes = plt.subplots(1, 2, figsize=(6, 3))
axes[0].imshow(raw_slice, cmap="gray")
axes[0].set_title("(A) Raw NIfTI slice")
axes[0].axis("off")

axes[1].imshow(proc_slice, cmap="gray")
axes[1].set_title("(B) Resized & normalized (64x64)")
axes[1].axis("off")

plt.tight_layout()

# 8. Save your pipeline figure
FIGS_ROOT.mkdir(parents=True, exist_ok=True)
out_path = FIGS_ROOT / "fig1_pipeline.png"
plt.savefig(out_path, dpi=300)
plt.close()
