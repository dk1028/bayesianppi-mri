from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "code" / "ppi" / "k_chain_rule.py"
spec = importlib.util.spec_from_file_location("k_chain_rule", MODULE_PATH)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_kbin_draws_and_effective_bins() -> None:
    df = pd.DataFrame(
        {
            "A_prob": np.linspace(0.01, 0.99, 100),
            "H": ([0] * 60) + ([1] * 40),
            "subject_id": [f"S{i}" for i in range(100)],
        }
    )
    out = module.posterior_draws_kbin(
        df,
        K=4,
        strategy="quantile",
        alpha_dirichlet=1.0,
        a_beta=1.0,
        b_beta=1.0,
        draws=1000,
        rng=np.random.default_rng(1),
    )
    assert out["K_effective"] == 4
    assert 0 <= out["mean"] <= 1
    assert out["width"] > 0
