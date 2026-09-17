"""Generate vector carrier-pairing diagrams (PDF and SVG)."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42,'svg.fonttype':'none'})
OUT=Path(__file__).resolve().parent/'figures';OUT.mkdir(exist_ok=True)
points=np.array([[0,1],[1,2],[2,1.6],[3,3.5],[4,3],[5,.9],[6,2.5],[7,1.6],[8,1.3]])
for vv in [False,True]:
 fig,axes=plt.subplots(2,3,figsize=(7.15,4.3))
 for k,ax in enumerate(axes.flat):
  ids=list(range(9)); added=[]
  if k in [1,2]:ids.remove(1)
  if k==2:ids.remove(4)
  seq=[(points[i],i) for i in ids]
  if k in [4,5]:seq.insert(3,((points[2]+points[3])/2,9));added.append(9)
  if k==5:seq.insert(7,((points[5]+points[6])/2,10));added.append(10)
  xy=np.array([p for p,i in seq]);ax.plot(xy[:,0],xy[:,1],'-',color='#16659b',lw=1.3)
  for j,(p,i) in enumerate(seq):
   ax.plot(*p,'o',color='#c43c32' if i in added else '#16659b',ms=3)
   ax.text(p[0],p[1]+.17,rf'$x_{{{i+1}}}$' if i<9 else rf'$a_{{{i-8}}}$',ha='center',fontsize=7)
   if vv and j<len(seq)-1:
    mid=(p+seq[j+1][0])/2;ax.plot(*mid,'s',color='#25885b',ms=3)
    changed=i in added or seq[j+1][1] in added or seq[j+1][1]!=i+1
    ax.plot([p[0],mid[0]],[.15,.15],color='#d47b22' if changed else '#25885b',lw=3)
    ax.text((p[0]+mid[0])/2,-.12,rf'$H_{{{j+1}}}$',ha='center',fontsize=6)
   elif not vv and j%2==0 and j+1<len(seq):
    changed=j!=i or seq[j+1][1]!=i+1
    ax.plot([p[0],seq[j+1][0][0]],[.15,.15],color='#d47b22' if changed else '#25885b',lw=3)
    ax.text((p[0]+seq[j+1][0][0])/2,-.12,rf'$H_{{{j//2+1}}}$',ha='center',fontsize=7)
  for i in set(range(9))-set(ids):ax.plot(*points[i],'o',mfc='none',mec='#c43c32',ms=5)
  titles=['Original','Delete one vertex','Delete two vertices','Original','Insert one vertex','Insert two vertices']
  ax.set_title(f'({chr(97+k)}) '+titles[k],fontsize=8)
  ax.set(xlim=(-.5,8.5),ylim=(-.35,4.2));ax.axis('off')
 handles=[Line2D([],[],marker='o',color='#16659b',ls='',label='Real vertex'),Line2D([],[],color='#25885b',lw=3,label='Unchanged pair'),Line2D([],[],color='#d47b22',lw=3,label='Changed pair')]
 if vv:handles.append(Line2D([],[],marker='s',color='#25885b',ls='',label='Virtual midpoint'))
 fig.legend(handles=handles,loc='lower center',ncol=len(handles),frameon=False,fontsize=7)
 fig.tight_layout(rect=[0,.07,1,1]);name=f'Figure_{2 if vv else 1}'
 fig.savefig(OUT/f'{name}.pdf',bbox_inches='tight');fig.savefig(OUT/f'{name}.svg',bbox_inches='tight');plt.close(fig)

for path in OUT.glob("Figure_*.svg"):
 path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines())+"\n")
