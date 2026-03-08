"""Convert ADNI DICOM folders to NIfTI using dcm2niix."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data"
RAW_DICOM_ROOT = DATA_ROOT / "raw_dicom"
NIFTI_ROOT = DATA_ROOT / "nifti"
DCM2NIIX_PATH = os.getenv("DCM2NIIX_PATH", "dcm2niix")


def convert_all_mprage() -> None:
    if not RAW_DICOM_ROOT.exists():
        raise FileNotFoundError(
            f"RAW_DICOM_ROOT does not exist: {RAW_DICOM_ROOT}\n"
            "Create data/raw_dicom/ and place ADNI DICOM folders there."
        )

    NIFTI_ROOT.mkdir(parents=True, exist_ok=True)
    count = 0
    for subject_folder in RAW_DICOM_ROOT.glob("*/MPRAGE/*"):
        if not subject_folder.is_dir():
            continue
        subject_id = subject_folder.parts[-3]
        scan_date = subject_folder.parts[-1]
        output_dir = NIFTI_ROOT / subject_id
        output_dir.mkdir(parents=True, exist_ok=True)
        command = [
            DCM2NIIX_PATH,
            "-z",
            "y",
            "-f",
            f"{subject_id}_{scan_date}",
            "-o",
            str(output_dir),
            str(subject_folder),
        ]
        print(f"\n[#{count + 1}] Converting {subject_id} | {subject_folder}")
        subprocess.run(command, check=True)
        count += 1

    print(f"\nDone. Converted {count} MPRAGE series into {NIFTI_ROOT}.")


if __name__ == "__main__":
    convert_all_mprage()
