"""Build revision tables/plots directly from recorded observations."""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
BASE=Path(__file__).resolve().parent;OUT=BASE/'figures';OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42,'svg.fonttype':'none'})
a=pd.read_csv(BASE/'results/audited/raw_attacks.csv');d=pd.read_csv(BASE/'results/diagnostics/failure_modes.csv');v=pd.read_csv(BASE/'results/diagnostics/vote_thinning.csv');p=pd.read_csv(BASE/'results/diagnostics/pca_conditioning.csv')
names=['Railways','Building','Landuse','Boundary','Road','Lake']
fig,axes=plt.subplots(2,3,figsize=(7.15,4.3),sharey=True)
for ax,name in zip(axes.flat,names):
 for family,levels,label in [('vertex_delete',[30,50],'Deletion'),('vertex_add',[10,30,50],'Addition'),('object_delete',[30,90],'Object removal')]:
  means=[];stds=[]
  for n in levels:
   vals=a[(a.dataset==name)&(a.attack==f'{family}_{n}pct')].nc;means.append(vals.mean());stds.append(vals.std(ddof=1))
  ax.errorbar(levels,means,yerr=stds,marker='o',ms=3,capsize=2,lw=1,label=label)
 ax.set_title(name,fontsize=9);ax.set_ylim(.5,1.025);ax.set_xlabel('Nominal attack (%)');ax.grid(alpha=.2)
axes[0,0].set_ylabel('Cosine NC');axes[1,0].set_ylabel('Cosine NC');fig.legend(*axes[0,0].get_legend_handles_labels(),loc='lower center',ncol=3,frameon=False)
fig.tight_layout(rect=[0,.06,1,1]);fig.savefig(OUT/'revision_robustness.pdf');plt.close(fig)
fig,axes=plt.subplots(1,3,figsize=(7.15,2.5))
for attack,label in [('no_attack','No attack'),('vertex_delete_50pct','Delete 50%'),('object_delete_90pct','Remove 90%')]:
 q=v[v.attack==attack].groupby('cap').nc.mean();axes[0].plot(range(5),q.values,'o-',ms=3,label=label)
axes[0].set_xticks(range(5),['1','3','5','10','All']);axes[0].set_xlabel('Votes retained per bit');axes[0].set_ylabel('Cosine NC');axes[0].legend(fontsize=6,frameon=False);axes[0].set_title('(a) Redundancy',fontsize=9)
q=p[p.removal==.1].groupby('axis_ratio').axis_drift_degrees.mean();axes[1].plot(range(len(q)),q.values,'o-',ms=3);axes[1].set_xticks(range(len(q)),['1','1.001','1.01','1.1','2','5'],rotation=45);axes[1].set_xlabel('Ellipse axis ratio');axes[1].set_ylabel('PCA axis drift (degrees)');axes[1].set_title('(b) 10% random removal',fontsize=9)
q=d[d['case'].str.startswith('round_')].copy();q['digits']=q['case'].str.extract(r'round_(\d+)').astype(int);q=q.groupby('digits').nc.mean();axes[2].plot(q.index,q.values,'o-',ms=3);axes[2].set_xlabel('Decimal places retained');axes[2].set_ylabel('Cosine NC');axes[2].set_title('(c) Coordinate rounding',fontsize=9)
for ax in axes:ax.grid(alpha=.2)
fig.tight_layout();fig.savefig(OUT/'revision_diagnostics.pdf');plt.close(fig)

def tex_table(caption,label,columns,rows):
 return '\n'.join([r'\begin{table*}[!t]',r'\centering\footnotesize',r'\caption{'+caption+'}',r'\label{'+label+'}',r'\begin{tabular}{'+'l'+'r'*(len(columns)-1)+'}',r'\toprule',' & '.join(columns)+r' \\',r'\midrule']+[' & '.join(map(str,row))+r' \\' for row in rows]+[r'\bottomrule',r'\end{tabular}',r'\end{table*}'])
rows=[]
for name in names:
 cells=[name]
 for attack in ['vertex_delete_50pct','object_delete_90pct','noise_50pct_1p4tau_m','compound_metric']:
  x=a[(a.dataset==name)&(a.attack==attack)].nc;cells.append(f'${x.mean():.4f}\\pm{x.std():.4f}$')
 rows.append(cells)
tables=tex_table('Audited cosine NC, mean $\\pm$ sample standard deviation over ten attack seeds per dataset. Embedding is fixed. Noise is in meters; the compound protocol is defined in the text.','tab:revised_attacks',['Dataset','Vertex deletion 50\\%','Object removal 90\\%','Noise 50\\%, $1.4\\tau$','Compound'],rows)
f=pd.read_csv(BASE/'results/audited/fidelity_runtime.csv');rows=[]
for name in names:
 x=f[f.dataset==name].iloc[0];q=a[(a.dataset==name)&(a.attack=='no_attack')].iloc[0]
 rows.append([name,str(int(q.votes_total)),f'{q.votes_mean:.1f}',f'{q.votes_median:.0f}',f'{x.mean_displacement_m:.2e}',f'{x.max_displacement_m:.2e}',str(int(x.unclosed_rings))])
tables+='\n'+tex_table('Carrier redundancy and geodesic fidelity after a shapefile write/read round trip. Votes are distributed over 1024 positions. Displacements are in meters; all revised polygon rings remain exactly closed.','tab:revised_fidelity',['Dataset','Votes','Mean/bit','Median/bit','Mean (m)','Max (m)','Unclosed rings'],rows)
rows=[]
for name in names:
 x=f[f.dataset==name].iloc[0];q=a[(a.dataset==name)&(a.attack=='object_delete_90pct')];rows.append([name,f'{x.embed_seconds:.2f}',f'{x.iterations:.2f}',f'{a[(a.dataset==name)&(a.attack=="no_attack")].extraction_seconds.mean():.3f}',f'{q.votes_mean.mean():.1f}',f'{q.empty_bins.mean():.1f}',int(q.empty_bins.max())])
tables+='\n'+tex_table('Observed CPU runtime and severe-removal coverage. Extraction includes all PCA candidates. Erasures are empty watermark positions under 90\\% object removal.','tab:revised_runtime',['Dataset','Embed (s)','PSO iter.','Extract (s)','Votes/bit','Mean erasures','Max erasures'],rows)
(BASE/'results/revision_tables.tex').write_text(tables)
