import os
import subprocess
from pathlib import Path

"""
Convert ADNI DICOM folders to NIfTI using dcm2niix.

Configure dcm2niix via the DCM2NIIX_PATH environment variable
(or ensure "dcm2niix" is on your PATH).
"""

# --- paths relative to repo root ---
REPO_ROOT       = Path(__file__).resolve().parents[2]
DATA_ROOT       = REPO_ROOT / "data"
RAW_DICOM_ROOT  = DATA_ROOT / "raw_dicom"
NIFTI_ROOT      = DATA_ROOT / "nifti"

# DICOM root directory (your local DICOM folder)
DICOM_ROOT = RAW_DICOM_ROOT

# Output directory (where converted NIfTI files will be saved)
OUTPUT_ROOT = NIFTI_ROOT

# Path to the dcm2niix executable
#   - If DCM2NIIX_PATH is set, use that
#   - Otherwise, assume "dcm2niix" is on PATH
DCM2NIIX_PATH = os.getenv("DCM2NIIX_PATH", "dcm2niix")


def convert_all_mprage():
    """Convert all MPRAGE series under data/raw_dicom/ to NIfTI in data/nifti/."""
    if not DICOM_ROOT.exists():
        raise FileNotFoundError(
            f"DICOM_ROOT does not exist: {DICOM_ROOT}\n"
            "Create data/raw_dicom/ and place ADNI DICOM folders there."
        )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    # Example match: data/raw_dicom/002_S_0729/MPRAGE/2006-08-02_07_02_00.0
    # subject_folder.parts[-3] = "002_S_0729"
    # subject_folder.parts[-1] = "2006-08-02_07_02_00.0"
    count = 0
    for subject_folder in DICOM_ROOT.glob("*/MPRAGE/*"):
        if not subject_folder.is_dir():
            continue

        subject_id = subject_folder.parts[-3]  # e.g., 002_S_0729
        scan_date  = subject_folder.parts[-1]  # e.g., 2006-08-02_07_02_00.0

        output_dir = OUTPUT_ROOT / subject_id
        output_dir.mkdir(parents=True, exist_ok=True)

        command = [
            DCM2NIIX_PATH,
            "-z", "y",                      # gzip compression
            "-f", f"{subject_id}_{scan_date}",
            "-o", str(output_dir),
            str(subject_folder),
        ]

        print(f"\n[#{count+1}] Converting:")
        print("  subject :", subject_id)
        print("  series  :", subject_folder)
        print("  out dir :", output_dir)
        print("  cmd     :", " ".join(command))

        # check=True → When an error occurs, an exception is thrown immediately to check for failure
        subprocess.run(command, check=True)
        count += 1

    print(f"\nDone. Converted {count} MPRAGE series from {DICOM_ROOT} into {OUTPUT_ROOT}.")


if __name__ == "__main__":
    convert_all_mprage()
