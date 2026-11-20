"""Figure factories matched to plots in ring/square/triangular notebook cells.

Uses the same statistical expressions and visual encodings as those notebooks,
with English strings in place of Spanish. Derived from raw rerun modes.csv.
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

REGIME_ORDER=['Regular','Small-world','Shortcut-rich']


def generate_plots(df:pd.DataFrame,topology:str,out:Path,format:str='png')->list[Path]:
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    files=[]
    present=[r for r in REGIME_ORDER if r in set(df['p_region'])]
    palette=dict(zip(REGIME_ORDER,sns.color_palette('tab10',3)))
    def save(fig,name):
        path=out / f'{name}_{topology}.{format}'
        fig.savefig(path,dpi=170,bbox_inches='tight')
        files.append(path)
        plt.close(fig)
    data=df.copy()
    sns.set_theme(style='whitegrid',font_scale=0.90)
    # Notebook kdeplot uses one hue call with common_norm=True. Separate calls
    # normalize each regime independently, giving densities 3x too tall.
    fig,ax=plt.subplots(figsize=(7.6,4.8))
    sns.kdeplot(data=data, x='eigenvalue', hue='p_region',hue_order=present,
                fill=True, common_norm=True, alpha=.23, palette=palette, ax=ax)
    ax.set(xlabel=r'Laplacian eigenvalue, $\lambda_\alpha$',ylabel=r'Spectral density, $p(\lambda)$')
    if ax.get_legend() is not None:
        ax.get_legend().set_title('Topology regime')
    save(fig,'density')

    # Original notebooks: lineplot of IPR versus eigenvalue/k_avg.
    # Implement directly in matplotlib: seaborn.lineplot on >200,000 distinct
    # floating-point x values is prohibitively slow. The actual mode values
    # and line-with-markers encoding remain the same.
    fig,ax=plt.subplots(figsize=(7.6,4.8))
    for region in present:
        sub=data[data.p_region==region].sort_values('eigenvalue_over_k')
        ax.plot(sub.eigenvalue_over_k.to_numpy(),sub.ipr.to_numpy(),
                alpha=.20,linewidth=.35,marker='.',markersize=.3,
                label=region,color=palette[region])
    ax.set(xlabel=r'Normalized Laplacian eigenvalue, $\lambda_\alpha/\langle k\rangle$',
           ylabel=r'Inverse participation ratio, $\mathrm{IPR}(\phi_\alpha)$')
    ax.legend(title='Topology regime')
    save(fig,'ipr')

    # Notebook cells 29/31: OLS polynomial regression of degree/clustering
    # ratio against normalized eigenvalue (order=2, scatter=False).
    # Direct polyfit equals the underlying quadratic model in sns.lmplot;
    # original bootstrap CIs omitted to remove unseeded stochastic variation.
    for metric,ylabel,fname in [
        ('weighted_degree_over_k',r'Mode-weighted degree / $\langle k\rangle$','degree_regression'),
        ('weighted_clustering_over_avg',r'Mode-weighted clustering / $\langle C\rangle$','clustering_regression')]:
        fig,ax=plt.subplots(figsize=(7.6,4.8))
        for region in present:
            sub=data[(data.p_region==region)&np.isfinite(data[metric])]
            if len(sub)>=3 and sub.eigenvalue_over_k.nunique()>2:
                x=sub.eigenvalue_over_k.to_numpy(); y=sub[metric].to_numpy()
                xx=np.linspace(np.min(x),np.max(x),160)
                fit=np.polyfit(x,y,2)
                ax.plot(xx,np.polyval(fit,xx),label=region,color=palette[region],linewidth=2)
        ax.axhline(1,color='black',linestyle='--',linewidth=.8)
        ax.axvline(-1,color='black',linestyle='--',linewidth=.8)
        ax.set(xlabel=r'Normalized Laplacian eigenvalue, $\lambda_\alpha/\langle k\rangle$',ylabel=ylabel)
        ax.legend(title='Topology regime')
        if metric=='weighted_clustering_over_avg' and data[metric].notna().sum()==0:
            ax.text(.5,.5,'Undefined: zero average clustering',ha='center',va='center',transform=ax.transAxes)
        save(fig,fname)
    return files
