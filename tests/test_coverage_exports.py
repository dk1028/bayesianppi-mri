from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "ppi"))
spec = importlib.util.spec_from_file_location(
    "coverage_experiment", ROOT / "code" / "ppi" / "coverage_experiment.py"
)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize("cohort,source", [("full", "all"), ("6570", "6570")])
@pytest.mark.parametrize("prior", ["uniform", "jeffreys"])
def test_prior_export_matches_manuscript_reference(
    cohort: str, source: str, prior: str
) -> None:
    reference = ROOT / "results" / "reference"
    results = pd.read_csv(reference / f"coverage_{source}_subject_oof_scan_age.csv")
    expected = pd.read_csv(reference / f"table_{cohort}_{prior}_summary.csv")

    exported = module.make_prior_summary(results, prior)

    pd.testing.assert_frame_equal(
        exported, expected, check_dtype=False, check_exact=False, atol=1e-14, rtol=0
    )
