"""Redraw the paper's coverage figures from the supplied Uniform summaries.

No estimates or posterior samples are recomputed by this script.
Run from any directory: python scripts/plot_uniform_coverage.py
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "results/reference"
FIGURES = ROOT / "figs" / "paper"
FIGURES.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 8.5,
    'axes.labelsize': 8.5, 'axes.titlesize': 9,
    'legend.fontsize': 7.1, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
})
METHODS = [
    ('CRE', 'CRE', '#0072B2', 'o'),
    ('LabeledOnly', 'Labeled-only', '#D55E00', 's'),
    ('BinaryDiff', 'Binary difference', '#009E73', 'D'),
    ('PPI', 'PPI', '#CC79A7', '^'),
    ('PPIpp', 'Power-tuned PPI', '#6A3D9A', 'v'),
]

def main():
    for cohort, suffix in [('full', 'full'), ('6570', 'age_65_70')]:
        data = pd.read_csv(HERE / f'table_{cohort}_uniform_summary.csv')
        assert data['n'].tolist() == [10, 20, 40, 80]
        for metric, ylabel, kind in [('cov', 'Empirical 95% coverage', 'coverage'),
                                      ('w', 'Mean 95% interval width', 'width')]:
            fig, ax = plt.subplots(figsize=(3.2, 3.05))
            fig.subplots_adjust(left=.19, right=.97, top=.95, bottom=.34)
            for key, label, color, marker in METHODS:
                ax.plot(data['n'], data[f'{key}_{metric}'], color=color,
                        marker=marker, markersize=3.7, linewidth=1.25, label=label)
            if metric == 'cov':
                ax.axhline(.95, color='#555555', linestyle='--', linewidth=.9,
                           label='Nominal 0.95')
                ax.set_ylim(.84 if cohort == 'full' else .72, 1.005)
            else:
                ax.set_ylim(0.1, 0.6)
                ax.set_yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
            ax.set_xlabel('Number of labeled scans')
            ax.set_ylabel(ylabel)
            ax.set_xticks([10, 20, 40, 80])
            ax.grid(True, alpha=.25, linewidth=.6)
            ax.spines[['top', 'right']].set_visible(False)
            ax.legend(loc='upper center', bbox_to_anchor=(.5, -.29), ncol=2,
                      frameon=False, columnspacing=1.0, handlelength=1.8,
                      borderaxespad=0, labelspacing=.5)
            name = f'fig_{kind}_{suffix}'
            fig.savefig(FIGURES / (name+'.png'), dpi=400)
            fig.savefig(FIGURES / (name+'.pdf'))
            plt.close(fig)
            print(name)

if __name__ == '__main__':
    main()
