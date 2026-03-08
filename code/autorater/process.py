"""Draw the CNN architecture figure used in the paper.

The figure now matches the actual implementation: channels 1->8->16 and Dense 64.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow, Rectangle

REPO_ROOT = Path(__file__).resolve().parents[2]
FIGS_ROOT = REPO_ROOT / "figs"
FIGS_ROOT.mkdir(parents=True, exist_ok=True)

layers = [
    ("Input\n1×64×64×64", 0),
    ("Conv3D\n8×64×64×64", 2),
    ("MaxPool\n8×32×32×32", 4),
    ("Conv3D\n16×32×32×32", 6),
    ("MaxPool\n16×16×16×16", 8),
    ("Flatten", 10),
    ("Dense\n64", 12),
    ("Sigmoid\n1", 14),
]

fig, ax = plt.subplots(figsize=(12, 3))
y = 0.5
width = 1.5
height = 0.7
for idx, (label, x) in enumerate(layers):
    ax.add_patch(
        Rectangle((x, y), width, height, fill=True, edgecolor="black", facecolor="lightgrey")
    )
    ax.text(x + width / 2, y + height / 2, label, ha="center", va="center", fontsize=10)
    if idx < len(layers) - 1:
        ax.add_patch(
            FancyArrow(
                x + width,
                y + height / 2,
                0.5,
                0,
                width=0.05,
                length_includes_head=True,
            )
        )

ax.set_xlim(-0.5, 16)
ax.set_ylim(0, 2)
ax.axis("off")
plt.tight_layout()
out_png = FIGS_ROOT / "fig_cnn_architecture.png"
out_pdf = FIGS_ROOT / "fig_cnn_architecture.pdf"
plt.savefig(out_png, dpi=300)
plt.savefig(out_pdf, bbox_inches="tight")
plt.close()
print(f"Saved {out_png}")
print(f"Saved {out_pdf}")
