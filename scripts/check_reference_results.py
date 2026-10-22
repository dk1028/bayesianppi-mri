from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare generated summary files to reference tables.")
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--atol", type=float, default=1e-6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checks = [
        (args.results_root / "coverage" / f"table_{cohort}_{prior}_summary.csv",
         args.results_root / "reference" / f"table_{cohort}_{prior}_summary.csv")
        for cohort in ["full", "6570"] for prior in ["uniform", "jeffreys"]
    ]

    for generated, reference in checks:
        if not generated.exists():
            raise FileNotFoundError(f"Missing required prior summary: {generated}")
        if not reference.exists():
            raise FileNotFoundError(f"Missing reference file: {reference}")

        got = pd.read_csv(generated)
        ref = pd.read_csv(reference)
        if list(got.columns) != list(ref.columns):
            raise RuntimeError(f"Column mismatch for {generated}")

        numeric_cols = got.select_dtypes(include="number").columns
        pd.testing.assert_frame_equal(
            got, ref, check_dtype=False, check_exact=False,
            atol=args.atol, rtol=0,
        )
        max_abs = (got[numeric_cols] - ref[numeric_cols]).abs().to_numpy().max()
        print(f"{generated}: max absolute difference = {max_abs:.6g}")


if __name__ == "__main__":
    main()
