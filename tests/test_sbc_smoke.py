from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "code" / "ppi" / "sbc.py"
spec = importlib.util.spec_from_file_location("sbc", MODULE_PATH)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_sbc_smoke() -> None:
    ranks, truth = module.run_sbc_cre(
        M=20,
        S=100,
        NA=100,
        NH=20,
        prior="uniform",
        seed=1,
    )
    assert len(ranks["g"]) == 20
    assert len(truth) == 20


def test_sbc_labels_are_drawn_from_observed_pool() -> None:
    # Labeling every observed unit must partition the pool's A counts exactly.
    _, truth = module.run_sbc_cre(
        M=20, S=50, NA=12, NH=12, prior="uniform", seed=12
    )
    assert ((truth["n11"] + truth["n10"]) == truth["nA"]).all()
    assert ((truth["n01"] + truth["n00"]) == (12 - truth["nA"])).all()
