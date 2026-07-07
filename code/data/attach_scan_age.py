from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def normalize_subject(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.upper()


def normalize_date(s: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(s, errors="coerce")
    return parsed.dt.normalize()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Attach scan-level age to subject-grouped OOF predictions."
    )
    parser.add_argument("--oof-csv", type=Path, required=True)
    parser.add_argument("--metadata-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)

    parser.add_argument("--oof-subject-col", default="subject_id")
    parser.add_argument("--oof-date-col", default="Acq_Date")
    parser.add_argument("--metadata-subject-col", default="Subject")
    parser.add_argument("--metadata-date-col", default="Acq Date")
    parser.add_argument("--metadata-age-col", default="Age")
    parser.add_argument("--metadata-group-col", default="Group")

    parser.add_argument(
        "--full-output-name",
        default="autorater_predictions_all_subject_oof_with_scan_age.csv",
    )
    parser.add_argument(
        "--age6570-output-name",
        default="autorater_predictions_6570_subject_oof_scan_age.csv",
    )
    parser.add_argument("--expect-paper-counts", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    oof = pd.read_csv(args.oof_csv).copy()
    meta = pd.read_csv(args.metadata_csv).copy()

    required_oof = {args.oof_subject_col, args.oof_date_col, "H", "oof_fold"}
    missing_oof = required_oof - set(oof.columns)
    if missing_oof:
        raise ValueError(f"OOF table missing columns: {sorted(missing_oof)}")

    required_meta = {
        args.metadata_subject_col,
        args.metadata_date_col,
        args.metadata_age_col,
    }
    missing_meta = required_meta - set(meta.columns)
    if missing_meta:
        raise ValueError(f"Metadata table missing columns: {sorted(missing_meta)}")

    oof["subject_norm"] = normalize_subject(oof[args.oof_subject_col])
    oof["acq_date_norm"] = normalize_date(oof[args.oof_date_col])

    meta["subject_norm"] = normalize_subject(meta[args.metadata_subject_col])
    meta["acq_date_norm"] = normalize_date(meta[args.metadata_date_col])
    meta["Age_numeric"] = pd.to_numeric(meta[args.metadata_age_col], errors="coerce")

    if oof["acq_date_norm"].isna().any():
        raise ValueError("OOF table contains unparseable acquisition dates")
    if meta["acq_date_norm"].isna().any():
        raise ValueError("Metadata table contains unparseable acquisition dates")
    if meta["Age_numeric"].isna().any():
        raise ValueError("Metadata table contains missing/unparseable age")

    agg_spec: dict[str, tuple[str, str]] = {
        "Age": ("Age_numeric", "median"),
        "age_min": ("Age_numeric", "min"),
        "age_max": ("Age_numeric", "max"),
        "age_n": ("Age_numeric", "size"),
    }
    if args.metadata_group_col in meta.columns:
        agg_spec["meta_group"] = (args.metadata_group_col, "first")

    meta_pair = (
        meta.groupby(["subject_norm", "acq_date_norm"], as_index=False)
        .agg(**agg_spec)
        .copy()
    )

    inconsistent_age = meta_pair[~np.isclose(meta_pair["age_min"], meta_pair["age_max"])]
    if len(inconsistent_age):
        example = inconsistent_age.head(10).to_dict(orient="records")
        raise ValueError(
            "Duplicate metadata rows disagree on age for the same subject-date pair. "
            f"Examples: {example}"
        )

    merged = oof.merge(
        meta_pair,
        on=["subject_norm", "acq_date_norm"],
        how="left",
        validate="many_to_one",
    )

    missing_age = int(merged["Age"].isna().sum())
    if missing_age:
        examples = (
            merged.loc[merged["Age"].isna(), [args.oof_subject_col, args.oof_date_col]]
            .drop_duplicates()
            .head(20)
            .to_dict(orient="records")
        )
        raise ValueError(f"{missing_age} OOF rows are missing exact scan-level age: {examples}")

    fold_counts = merged.groupby(args.oof_subject_col)["oof_fold"].nunique()
    spanning = int((fold_counts > 1).sum())
    if spanning:
        raise RuntimeError(f"{spanning} subjects span multiple OOF folds")

    age6570 = merged[merged["Age"].between(65, 70, inclusive="both")].copy()

    full_out = args.output_dir / args.full_output_name
    age_out = args.output_dir / args.age6570_output_name
    merged.to_csv(full_out, index=False)
    age6570.to_csv(age_out, index=False)

    qc = {
        "oof_csv": str(args.oof_csv),
        "metadata_csv": str(args.metadata_csv),
        "matching": "exact subject identifier + acquisition date",
        "full_rows": int(len(merged)),
        "full_subjects": int(merged[args.oof_subject_col].nunique()),
        "full_prevalence": float(merged["H"].mean()),
        "age_65_70_rows": int(len(age6570)),
        "age_65_70_subjects": int(age6570[args.oof_subject_col].nunique()),
        "age_65_70_prevalence": float(age6570["H"].mean()),
        "missing_age_rows": missing_age,
        "subjects_spanning_multiple_folds": spanning,
        "duplicate_metadata_subject_date_pairs": int((meta_pair["age_n"] > 1).sum()),
        "full_output": str(full_out),
        "age_65_70_output": str(age_out),
    }

    if args.expect_paper_counts:
        expected = {
            "full_rows": 2116,
            "full_subjects": 503,
            "age_65_70_rows": 291,
            "age_65_70_subjects": 93,
        }
        mismatches = {
            key: {"expected": value, "observed": qc[key]}
            for key, value in expected.items()
            if qc[key] != value
        }
        if mismatches:
            raise RuntimeError(f"Paper-count QC failed: {mismatches}")

    qc_path = args.output_dir / "scan_age_merge_qc.json"
    qc_path.write_text(json.dumps(qc, indent=2), encoding="utf-8")

    print(json.dumps(qc, indent=2))


if __name__ == "__main__":
    main()
