import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# --- repo-relative paths ---
REPO_ROOT = Path(__file__).resolve().parents[2]
FIGS_ROOT = REPO_ROOT / "figs"
FIGS_ROOT.mkdir(parents=True, exist_ok=True)

# Re-define data to exactly match the new table (do not alter plotting structure)
data = {
    'prior': ['uniform'] * 4 + ['jeffreys'] * 4,
    'n_labels': [10, 20, 40, 80] * 2,

    # CRE (Chain-rule) coverage and width from the table
    'chain_cov': [1.00, 1.00, 0.98, 1.00, 1.00, 1.00, 0.98, 1.00],
    'chain_w':   [0.379, 0.259, 0.194, 0.144, 0.344, 0.254, 0.182, 0.139],

    # Naive (labels-only) coverage and width from the table
    'naive_cov': [0.98, 1.00, 0.98, 1.00, 0.90, 0.98, 0.94, 0.98],
    'naive_w':   [0.461, 0.336, 0.249, 0.179, 0.434, 0.343, 0.247, 0.179],

    # Difference estimator coverage and width from the table
    'diff_cov':  [0.58, 0.62, 0.94, 1.00, 0.44, 0.76, 0.96, 0.96],
    'diff_w':    [0.202, 0.139, 0.144, 0.110, 0.160, 0.172, 0.138, 0.106],
}

# PPI line (estimator-only; same across priors → plot once)
ppi_n = [10, 20, 40, 80]
ppi_cov_line = [0.96, 0.98, 0.98, 0.98]
ppi_w_line   = [0.527969, 0.345964, 0.233935, 0.165516]

df = pd.DataFrame(data)

# -------------------------------------------------------------------
# 1) Coverage vs. Number of Labels (65–70 subset)
# -------------------------------------------------------------------
plt.figure(figsize=(10, 5))
for prior in ['uniform', 'jeffreys']:
    subset = df[df['prior'] == prior]
    plt.plot(subset['n_labels'], subset['chain_cov'], 'o-', label=f'Chain ({prior})')   # CRE coverage
    plt.plot(subset['n_labels'], subset['naive_cov'], 's--', label=f'Naive ({prior})')  # Naive coverage
    plt.plot(subset['n_labels'], subset['diff_cov'], 'd-.', label=f'Diff ({prior})')    # Diff coverage

# Add PPI coverage as a single line (not tied to prior)
plt.plot(ppi_n, ppi_cov_line, '^:', label='PPI')  # PPI coverage

plt.ylim(0.3, 1.05)
plt.xlabel("Number of Labels")
plt.ylabel("Coverage")
plt.title("Coverage vs. Number of Labels (Age 65–70 subset)")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save camera-ready versions to figs/
plt.savefig(FIGS_ROOT / "fig_coverage_6570.pdf", bbox_inches='tight')
plt.savefig(FIGS_ROOT / "fig_coverage_6570.png", dpi=300, bbox_inches='tight')
plt.close()
# plt.show()  # Uncomment if needed

# -------------------------------------------------------------------
# 2) Interval Width vs. Number of Labels (65–70 subset)
# -------------------------------------------------------------------
plt.figure(figsize=(10, 5))
for prior in ['uniform', 'jeffreys']:
    subset = df[df['prior'] == prior]
    plt.plot(subset['n_labels'], subset['chain_w'], 'o-', label=f'Chain ({prior})')   # CRE width
    plt.plot(subset['n_labels'], subset['naive_w'], 's--', label=f'Naive ({prior})')  # Naive width
    plt.plot(subset['n_labels'], subset['diff_w'], 'd-.', label=f'Diff ({prior})')    # Diff width

# Add PPI width as a single line (not tied to prior)
plt.plot(ppi_n, ppi_w_line, '^:', label='PPI')  # PPI width

plt.xlabel("Number of Labels")
plt.ylabel("Interval Width")
plt.title("Interval Width vs. Number of Labels (Age 65–70 subset)")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save camera-ready versions to figs/
plt.savefig(FIGS_ROOT / "fig_width_6570.pdf", bbox_inches='tight')
plt.savefig(FIGS_ROOT / "fig_width_6570.png", dpi=300, bbox_inches='tight')
plt.close()
# plt.show()  # Uncomment if needed
