from __future__ import annotations

from pathlib import Path

from _shared import RESULTS_ROOT, FIGS_ROOT
from _viz_examples import create_example_figure

pred_csv = RESULTS_ROOT / "autorater" / "autorater_predictions_all.csv"
out_path = FIGS_ROOT / "model_ADCN_examples.png"
create_example_figure(
    pred_csv=pred_csv,
    out_path=out_path,
    suptitle="Model prediction quality comparison: AD vs CN (example)",
)
