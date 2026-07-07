from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from _shared import COVERAGE_ROOT, FIGS_ROOT

CSV_PATH = COVERAGE_ROOT / "coverage_all.csv"
if not CSV_PATH.exists():
    raise FileNotFoundError(f"Run implementaion_coverage_all.py first: {CSV_PATH} not found")

df = pd.read_csv(CSV_PATH)
ppi_line = df.groupby("n_labels", as_index=False)[["ppi_cov", "ppi_w"]].mean().sort_values("n_labels")

plt.figure(figsize=(10, 5))
for prior in ["uniform", "jeffreys"]:
    subset = df[df["prior"] == prior].sort_values("n_labels")
    plt.plot(subset["n_labels"], subset["chain_cov"], "o-", label=f"Chain ({prior})")
    plt.plot(subset["n_labels"], subset["naive_cov"], "s--", label=f"Naive ({prior})")
    plt.plot(subset["n_labels"], subset["diff_cov"], "d-.", label=f"Diff ({prior})")
plt.plot(ppi_line["n_labels"], ppi_line["ppi_cov"], "^:", label="PPI")
plt.ylim(0.4, 1.05)
plt.xlabel("Number of Labels")
plt.ylabel("Coverage")
plt.title("Coverage vs. Number of Labels (Full Cohort)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(FIGS_ROOT / "fig_coverage_all.pdf", bbox_inches="tight")
plt.savefig(FIGS_ROOT / "fig_coverage_all.png", dpi=300, bbox_inches="tight")
plt.close()

plt.figure(figsize=(10, 5))
for prior in ["uniform", "jeffreys"]:
    subset = df[df["prior"] == prior].sort_values("n_labels")
    plt.plot(subset["n_labels"], subset["chain_w"], "o-", label=f"Chain ({prior})")
    plt.plot(subset["n_labels"], subset["naive_w"], "s--", label=f"Naive ({prior})")
    plt.plot(subset["n_labels"], subset["diff_w"], "d-.", label=f"Diff ({prior})")
plt.plot(ppi_line["n_labels"], ppi_line["ppi_w"], "^:", label="PPI")
plt.xlabel("Number of Labels")
plt.ylabel("Interval Width")
plt.title("Interval Width vs. Number of Labels (Full Cohort)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(FIGS_ROOT / "fig_interval_all.pdf", bbox_inches="tight")
plt.savefig(FIGS_ROOT / "fig_interval_all.png", dpi=300, bbox_inches="tight")
plt.close()
