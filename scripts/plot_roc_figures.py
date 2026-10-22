"""Recreate ROC panels from authorized, fixed OOF predictions (no fitting)."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,
    'axes.labelsize':8.5,'legend.fontsize':7.2,'pdf.fonttype':42})

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-csv',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parents[1]/'figs'/'paper')
    args=parser.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    data=pd.read_csv(args.input_csv)
    age=pd.cut(data['Age'],bins=[50,74,80,101],right=False,labels=['50--73','74--79','80--100'])
    scopes=[('overall',[(data,'Overall','#0072B2')]),('by_age_bins',
        [(data.loc[age.eq(label)],label,color) for label,color in
         [('50--73','#D55E00'),('74--79','#009E73'),('80--100','#CC79A7')]])]
    for name,groups in scopes:
        fig,ax=plt.subplots(figsize=(3.2,2.8),layout='constrained')
        for frame,label,color in groups:
            score=frame['autorater_prediction'];human=frame['H']
            fpr,tpr,_=roc_curve(human,score)
            auc=roc_auc_score(human,score)
            ax.plot(fpr,tpr,color=color,lw=1.4,label=f'{label}: AUC {auc:.3f}')
        ax.plot([0,1],[0,1],'--',color='#777777',lw=.8,label='Chance')
        ax.set(xlim=(0,1),ylim=(0,1.02),xlabel='False positive rate',ylabel='True positive rate')
        ax.legend(loc='lower right',frameon=True,framealpha=.85)
        ax.grid(alpha=.22,lw=.5);ax.spines[['top','right']].set_visible(False)
        fig.savefig(args.output_dir/f'fig_roc_{name}_subject_oof.pdf')
        plt.close(fig)

if __name__=='__main__':main()
