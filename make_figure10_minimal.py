#!/usr/bin/env python3
"""Standalone reproduction/style template for IoTJ Figure_12."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "legacy_results"
OUTPUT = HERE / "figures"

DATASET_ORDER = ["Railways", "Building", "Landuse", "Boundary", "Road", "Lake"]
METHODS = ["yan_2017", "li_2021", "zhang_2025", "lin_2018", "xi_2022", "proposed"]
LABELS = {"yan_2017":"Yan et al. (2017)", "li_2021":"Li et al. (2021)",
          "zhang_2025":"Zhang et al. (2025)", "lin_2018":"Lin et al. (2018)",
          "xi_2022":"Xi et al. (2022)", "proposed":"Proposed"}
COLORS = dict(zip(METHODS, ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#D62728"]))
MARKERS = dict(zip(METHODS, ["v", "s", "D", "^", "o", "P"]))

TABLE_METHODS = ["Tan_2024", "Wu_2025", "Zhang_2025", "Lin_2018", "Xi_2022", "Proposed"]
TABLE_LABELS = {"Tan_2024":"Tan et al. (2024)", "Wu_2025":"Wu et al. (2025)",
                "Zhang_2025":"Zhang et al. (2025)", "Lin_2018":"Lin et al. (2018)",
                "Xi_2022":"Xi et al. (2022)", "Proposed":"Proposed"}
TABLE_COLORS = dict(zip(TABLE_METHODS, COLORS.values()))
TABLE_MARKERS = dict(zip(TABLE_METHODS, MARKERS.values()))

ADDITION = {
 "Tan_2024":[.92,.91,.92,.92,.93,.92,.92,.93,.92], "Wu_2025":[1,.99,.97,.99,.99,.97,.99,.99,.96],
 "Zhang_2025":[1,1,1,.60,.56,.54,.62,.55,.55], "Lin_2018":[.56,.53,.53,.56,.52,.53,.57,.53,.48],
 "Xi_2022":[.78,.70,.56,.78,.70,.55,.78,.70,.58], "Proposed":[1,1,1,.99,.99,.95,1,.99,.96]}
NOISE = {
 "Tan_2024":[.99]*9, "Wu_2025":[1,.99,.90,1,.99,.90,1,.99,.89],
 "Zhang_2025":[.66,.63,.58,.66,.63,.58,.66,.63,.59], "Lin_2018":[.55,.52,.51,.55,.51,.51,.55,.52,.51],
 "Xi_2022":[.71,.62,.67,.71,.70,.53,.76,.57,.55], "Proposed":[.99,.98,.89,.99,.98,.90,.99,.98,.90]}
CROP_LABELS = ["Rail.", "Build.", "Land.", "Bound.", "Road", "Lake"]
CROPPING = {
 "Tan_2024":[.79,.98,.94,.95,.97,.89], "Wu_2025":[1,1,.98,1,1,1],
 "Zhang_2025":[.97,1,.87,.80,.86,.70], "Lin_2018":[.67,.54,.54,.54,.54,.54],
 "Xi_2022":[.99,.61,.79,1,1,.90], "Proposed":[1,1,1,.99,.99,.99]}

def configure_matplotlib():
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":7.0, "axes.labelsize":7.0,
        "axes.titlesize":7.8, "legend.fontsize":6.2, "xtick.labelsize":6.4, "ytick.labelsize":6.4,
        "axes.linewidth":.65, "xtick.major.width":.55, "ytick.major.width":.55,
        "xtick.major.size":2.6, "ytick.major.size":2.6, "svg.fonttype":"none",
        "pdf.fonttype":42, "ps.fonttype":42, "savefig.bbox":"tight", "savefig.pad_inches":.04})

def format_axes(ax, y_min=.45):
    ax.set_ylim(y_min, 1.04); ax.set_yticks(np.arange(max(0., y_min), 1.01, .1))
    ax.grid(axis="y", color="#D8D8D8", linewidth=.45, alpha=.9)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

def attack_nc(results, attack, dataset=None):
    sub = results[results.attack == attack]
    if dataset is not None: sub = sub[sub.dataset == dataset]
    if sub.empty: raise KeyError(f"Missing attack={attack}, dataset={dataset}")
    return float(sub.nc.mean())

def apply_latest(deletion, removal, geometry, results):
    deletion=deletion.copy();removal=removal.copy();geometry=geometry.copy()
    for i,row in deletion.iterrows():
        pct=int(row.vertex_deletion_percent)
        deletion.loc[i,'proposed']=attack_nc(results,'no_attack' if pct==0 else f'vertex_delete_{pct}pct',row.dataset)
    for i,row in removal.iterrows():
        pct=int(row.object_removal_percent);ds='Landuse' if row.series=='Landuse' else None
        removal.loc[i,'proposed']=attack_nc(results,'no_attack' if pct==0 else f'object_delete_{pct}pct',ds)
    for i,row in geometry.iterrows():
        x=float(row.semantic_x)
        if row.attack_family=='rotation':label='no_attack' if x in (0,360) else f'rotation_{int(x)}'
        elif row.attack_family=='scaling':label='no_attack' if x==0 else ('scale_0p5' if x==.5 else f'scale_{x:g}')
        else:label='no_attack' if x==0 else f'translation_{int(x)}'
        geometry.loc[i,'proposed']=attack_nc(results,label)
    ADDITION['Proposed']=[attack_nc(results,f'addition_alpha{alpha:g}_pct{pct}') for alpha in [0,.5,1] for pct in [10,30,50]]
    NOISE['Proposed']=[attack_nc(results,f'noise_strength{strength:g}_pct{pct}') for strength in [.6,1.,1.4] for pct in [10,30,50]]
    CROPPING['Proposed']=[attack_nc(results,'crop_half_x',ds) for ds in DATASET_ORDER]
    deletion.to_csv(HERE/'results/minimal_revision/figure10_deletion.csv',index=False)
    removal.to_csv(HERE/'results/minimal_revision/figure10_removal.csv',index=False)
    geometry.to_csv(HERE/'results/minimal_revision/figure10_geometry.csv',index=False)
    pd.DataFrame(ADDITION).to_csv(HERE/'results/minimal_revision/figure10_addition.csv',index=False)
    pd.DataFrame(NOISE).to_csv(HERE/'results/minimal_revision/figure10_noise.csv',index=False)
    pd.DataFrame(CROPPING).to_csv(HERE/'results/minimal_revision/figure10_cropping.csv',index=False)
    return deletion,removal,geometry

def plot_curves(ax, df, xcol, xlabel):
    for method in METHODS:
        d=df[[xcol,method]].dropna()
        if d.empty: continue
        proposed=method=="proposed"
        ax.plot(d[xcol],d[method],color=COLORS[method],marker=MARKERS[method],markersize=3.1 if proposed else 3,
                markeredgewidth=.4,linewidth=1.2 if proposed else .95,alpha=1 if proposed else .88,
                label=LABELS[method],zorder=4 if proposed else 2)
    ax.set(xlabel=xlabel,ylabel="NC"); format_axes(ax)

def plot_table(ax, xpos, data, ticks, xlabel=None, groups=None):
    for method in TABLE_METHODS:
        proposed=method=="Proposed"
        ax.plot(xpos,data[method],color=TABLE_COLORS[method],marker=TABLE_MARKERS[method],markersize=3.1 if proposed else 3,
                markeredgewidth=.4,linewidth=1.2 if proposed else .95,alpha=1 if proposed else .88,
                label=TABLE_LABELS[method],zorder=4 if proposed else 2)
    ax.set_xticks(xpos); ax.set_xticklabels(ticks); ax.set_ylabel("NC"); format_axes(ax)
    if groups:
        for sep in (2.5,5.5): ax.axvline(sep,color="#CBCBCB",linewidth=.5,linestyle=(0,(3,2)),zorder=1)
        for label,xc in groups: ax.text(xc,-.30,label,transform=ax.get_xaxis_transform(),ha="center",va="top",fontsize=6.2)
        ax.set_xlim(-.4,8.4)
    if xlabel: ax.set_xlabel(xlabel)

def compound_data(results):
    proposed={ds:attack_nc(results,"compound",ds) for ds in DATASET_ORDER}
    d=pd.DataFrame({"dataset":DATASET_ORDER,
      "Tan_2024":[.89,.80,.53,.94,.95,.47], "Wu_2025":[.85,.76,.85,.99,.96,.94],
      "Zhang_2025":[.61,.57,.55,.63,.73,.61], "Lin_2018":[.57,.54,.53,.56,.55,.55],
      "Xi_2022":[.49,.53,.49,.48,.50,.47]})
    d["Proposed"]=[proposed[x] for x in d.dataset]; return d

def main():
    configure_matplotlib(); OUTPUT.mkdir(exist_ok=True)
    deletion=pd.read_csv(DATA/"Figure_13_vertex_deletion_nc.csv")
    removal=pd.read_csv(DATA/"Figure_14_object_removal_nc.csv")
    geometry=pd.read_csv(DATA/"Figure_15_geometric_attacks_nc.csv")
    results=pd.read_csv(HERE/"results/minimal_revision/raw.csv"); results=results[results.series=="main"]
    deletion,removal,geometry=apply_latest(deletion,removal,geometry,results)
    compound=compound_data(results); compound.to_csv(HERE/"results/minimal_revision/figure10_compound.csv",index=False)
    fig=plt.figure(figsize=(7.35,9.15),constrained_layout=True)
    fig.get_layout_engine().set(rect=(0,.055,1,.910),h_pad=.040,w_pad=.018,hspace=.055,wspace=.018)
    gs=fig.add_gridspec(5,3); letters=iter("abcdefghijklmn"); del_axes=[]
    def title(ax,text): ax.set_title(f"({next(letters)}) {text}",loc="left",pad=2,fontweight="bold")
    def compact(ax,keep=True):
        if not keep: ax.set_ylabel("")
        ax.tick_params(axis="both",pad=1.2,length=2); ax.xaxis.labelpad=1.5; ax.yaxis.labelpad=1.5
    for i,ds in enumerate(DATASET_ORDER):
        r,c=divmod(i,3); ax=fig.add_subplot(gs[r,c]); plot_curves(ax,deletion[deletion.dataset==ds],"vertex_deletion_percent","Deleted vertices (%)")
        title(ax,f"Vertex deletion-{ds}"); compact(ax,c==0); del_axes.append(ax)
    ax=fig.add_subplot(gs[2,0]); plot_table(ax,np.arange(9),ADDITION,["10","30","50"]*3,groups=[(r"$\alpha=0$",1),(r"$\alpha=0.5$",4),(r"$\alpha=1$",7)])
    title(ax,"Vertex addition"); compact(ax); ax.set_xlabel("Attack intensity (%)"); ax.xaxis.labelpad=7
    ax=fig.add_subplot(gs[2,1]); plot_table(ax,np.arange(9),NOISE,["10","30","50"]*3,groups=[("0.6 degree",1),("1.0 degree",4),("1.4 degree",7)])
    title(ax,"Noise"); compact(ax,False); ax.set_xlabel("Attack intensity (%)"); ax.xaxis.labelpad=7
    ax=fig.add_subplot(gs[2,2]); plot_curves(ax,removal[removal.series=="Average"],"object_removal_percent","Removed objects (%)"); title(ax,"Object removal"); compact(ax,False)
    specs=[("rotation","display_x","Rotation (degree)","Rotation"),("scaling","semantic_x","Scale factor","Scaling"),("translation","display_x","Translation (map unit)","Translation")]
    for c,(family,xcol,xlabel,name) in enumerate(specs):
        ax=fig.add_subplot(gs[3,c]); sub=geometry[geometry.attack_family==family].copy(); sub[xcol]=pd.to_numeric(sub[xcol]); plot_curves(ax,sub,xcol,xlabel); title(ax,name); compact(ax,c==0)
        if family=="scaling": ax.set_xticks([0,.5,1,1.5,2]); ax.set_xlim(-.08,2.18)
        if family=="rotation": ax.set_xticks([0,90,180,270,360])
    crop_ax=fig.add_subplot(gs[4,0]); plot_table(crop_ax,np.arange(6),CROPPING,CROP_LABELS,"Dataset"); title(crop_ax,"Cropping"); compact(crop_ax); crop_ax.tick_params(axis="x",labelrotation=18)
    bar_ax=fig.add_subplot(gs[4,1:3]); x=np.arange(6); width=.8/6
    for j,m in enumerate(TABLE_METHODS): bar_ax.bar(x+(j-2.5)*width,compound[m],width=width*.92,color=TABLE_COLORS[m],edgecolor="#333333",linewidth=.3,label=TABLE_LABELS[m],zorder=3 if m=="Proposed" else 2)
    bar_ax.set_xticks(x); bar_ax.set_xticklabels(CROP_LABELS); bar_ax.set(xlabel="Dataset",ylabel="NC"); format_axes(bar_ax); title(bar_ax,"Compound attack"); compact(bar_ax,False)
    h,l=del_axes[0].get_legend_handles_labels(); fig.legend(h,l,loc="upper center",ncol=6,frameon=False,bbox_to_anchor=(.5,.992),fontsize=7.8,handlelength=1.55,handletextpad=.3,columnspacing=.85,borderaxespad=0)
    h,l=bar_ax.get_legend_handles_labels(); fig.legend(h,l,loc="lower center",ncol=6,frameon=False,bbox_to_anchor=(.5,.014),fontsize=7.8,handlelength=1.55,handletextpad=.3,columnspacing=.85,borderaxespad=0)
    for ext in ("png","pdf","svg"):
        kw={"dpi":600} if ext in ("png","tiff") else {}
        if ext=="tiff": kw["pil_kwargs"]={"compression":"tiff_lzw"}
        fig.savefig(OUTPUT/f"Figure_10_minimal.{ext}",**kw)
    plt.close(fig); print(f"Generated files in {OUTPUT}")

if __name__ == "__main__":
    main()
    for path in OUTPUT.glob("Figure_10_minimal.svg"):
        path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines())+"\n")
