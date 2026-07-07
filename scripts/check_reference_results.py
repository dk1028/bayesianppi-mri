from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare generated summary files to reference tables.")
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checks = [
        (
            args.results_root / "coverage" / "table_full_uniform_summary.csv",
            args.results_root / "reference" / "coverage_full_uniform_reference.csv",
        ),
        (
            args.results_root / "coverage" / "table_6570_uniform_summary.csv",
            args.results_root / "reference" / "coverage_6570_uniform_reference.csv",
        ),
    ]

    for generated, reference in checks:
        if not generated.exists():
            print(f"SKIP missing generated file: {generated}")
            continue
        if not reference.exists():
            print(f"SKIP missing reference file: {reference}")
            continue

        got = pd.read_csv(generated)
        ref = pd.read_csv(reference)
        if list(got.columns) != list(ref.columns):
            raise RuntimeError(f"Column mismatch for {generated}")

        numeric_cols = got.select_dtypes(include="number").columns
        max_abs = (got[numeric_cols] - ref[numeric_cols]).abs().to_numpy().max()
        print(f"{generated}: max absolute difference = {max_abs:.6g}")


if __name__ == "__main__":
    main()
