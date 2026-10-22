"""Recreate manuscript rank panels from the supplied corrected SBC ranks."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
HERE=ROOT/"results/reference"
FIGURES=ROOT/'figs'/'paper'
FIGURES.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,
    'axes.labelsize':8.5,'legend.fontsize':7.2,'pdf.fonttype':42})

def main():
    summary=pd.read_csv(HERE/'sbc_paper_summary.csv')
    for prior in ['jeffreys','uniform']:
        ranks=pd.read_csv(HERE/f'sbc_ranks_{prior}.csv')['g'].to_numpy()
        row=summary.loc[summary.prior.eq(prior)&summary.parameter.eq('g')].iloc[0]
        u=(ranks+.5)/(int(row['S'])+1)
        fig,ax=plt.subplots(figsize=(3.2,2.65),layout='constrained')
        ax.hist(u,bins=np.linspace(0,1,21),rwidth=.9,color='#0072B2')
        ax.axhline(len(u)/20,color='#777777',ls='--',lw=.9)
        ax.set(xlim=(0,1),ylim=(0,40),xlabel='Normalized rank',ylabel='Count per bin')
        ax.set_title(f'{prior.capitalize()} prior',fontsize=9)
        ax.text(.98,.96,f"$\\chi^2$ p={row.chi2_p:.3f}\nKS p={row.ks_p:.3f}",
                ha='right',va='top',transform=ax.transAxes,fontsize=7.3)
        ax.spines[['top','right']].set_visible(False)
        out=FIGURES/f'sbc_g_{prior}_b20'
        fig.savefig(out.with_suffix('.pdf'))
        fig.savefig(out.with_suffix('.png'),dpi=400)
        plt.close(fig)

if __name__=='__main__':main()
