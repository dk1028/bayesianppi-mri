from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "ppi"))

from _shared import (
    cre_posterior_draws,
    labeled_only_posterior_draws,
    posterior_summary,
    ppi_analytic_estimator,
    ppipp_manuscript_estimator,
)


def example_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "subject_id": [f"S{i}" for i in range(8)],
            "H": [0, 0, 1, 1, 0, 1, 0, 1],
            "A_prob": [0.1, 0.2, 0.8, 0.7, 0.4, 0.9, 0.3, 0.6],
            "A_class": [0, 0, 1, 1, 0, 1, 0, 1],
        }
    )


def test_cre_draws_are_probabilities() -> None:
    df = example_df()
    idx = np.array([0, 2, 4, 6])
    draws = cre_posterior_draws(
        df,
        idx,
        alpha=1.0,
        beta=1.0,
        size=1000,
        rng=np.random.default_rng(1),
    )
    assert draws.shape == (1000,)
    assert np.all((draws >= 0) & (draws <= 1))


def test_labeled_only_and_ppi_summaries_are_finite() -> None:
    df = example_df()
    idx = np.array([0, 2, 4, 6])
    lo = posterior_summary(
        labeled_only_posterior_draws(
            df,
            idx,
            alpha=1.0,
            beta=1.0,
            size=1000,
            rng=np.random.default_rng(2),
        )
    )
    ppi = ppi_analytic_estimator(df, idx)
    ppipp, lam = ppipp_manuscript_estimator(df, idx)

    assert np.isfinite(lo.mean)
    assert np.isfinite(ppi.width)
    assert np.isfinite(ppipp.width)
    assert 0 <= lam <= 1
