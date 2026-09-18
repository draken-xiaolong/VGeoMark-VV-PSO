"""Fixed-protocol PSO screen; exploratory outputs never overwrite paper records."""
from pathlib import Path
from dataclasses import asdict
import argparse,json,time,platform,hashlib
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from psoqim import EmbedConfig,read_shapefile,write_shapefile,embed_dataset,extract_dataset
from psoqim import attacks
from psoqim.transforms import PCATransform,collect_points
from psoqim.watermark import load_watermark
from run_experiments import save_pca_key
from run_revision import geod
BASE=Path(__file__).resolve().parent
NAMES=['Railways','Building','Landuse','Boundary','Road','Lake']
PROFILES={'baseline':{},'p20':{'num_particles':20},'p40':{'num_particles':40},'long':{'num_particles':20,'max_iter':80,'early_stop_rounds':16},'full40':{'early_stop_rounds':40},'q01':{'q':.1}}
def correlated_noise(d,ratio,strength,seed):
    """Same selection/x noise as current attack; y equals x (paired diagnostic)."""
    rng=np.random.default_rng(seed);out=d.copy()
    for i,s in enumerate(d.shapes):
        parts=[]
        for a,b in attacks._parts(s):
            p=np.asarray(s.points[a:b],float);mask=rng.random(len(p))<ratio
            noise=rng.uniform(-strength,strength,size=p.shape)
            noise[:,1]=noise[:,0];p=p.copy();p[mask]+=noise[mask];parts.append(p)
        out.shapes[i]=attacks._replace_parts(s,parts,d.shape_type)
    return out

def compound(d,seed,noise_mode='independent',main=False):
    seeds=[1,2025,233337,2024] if main else [seed,seed+10000,seed+20000,seed+30000]
    d=attacks.vertex_delete(d,.3,seeds[0]);d=attacks.object_delete(d,.3,seeds[1])
    d=attacks.vertex_add(d,.1,tolerance=.01,seed=seeds[2])
    d=(attacks.vertex_noise if noise_mode=='independent' else correlated_noise)(d,.2,.6,seeds[3])
    d=attacks.geometric(d,x_shift=10,y_shift=10);d=attacks.geometric(d,scale=.5)
    return attacks.reverse_objects(attacks.reverse_vertices(d))

def run(task):
    profile,name,seeds=task;dest=BASE/'results/robustness_screen';dest.mkdir(parents=True,exist_ok=True)
    c=EmbedConfig(n_jobs=1,preserve_parts=False,**PROFILES[profile]);w=load_watermark(BASE/'data/M.png')
    origin=read_shapefile(BASE/'data'/f'{name}.shp')
    path=BASE/'outputs'/('legacy' if profile=='baseline' else f'robustness_{profile}')/f'{name}.shp'
    if not path.exists():
        t=time.perf_counter();e=embed_dataset(origin,w,c);seconds=time.perf_counter()-t
        write_shapefile(e.dataset,path);save_pca_key(e.transform,path.with_suffix('.pca_key.json'),name,c)
        d=read_shapefile(path);a,b=collect_points(origin),collect_points(d)
        delta=np.asarray(geod(origin).inv(a[:,0],a[:,1],b[:,0],b[:,1])[2])
        meta=dict(dataset=name,profile=profile,config=asdict(c),embed_seconds=seconds,fitness=e.average_fitness,iterations=e.average_iterations,max_displacement_m=delta.max(),mean_displacement_m=delta.mean(),mean_displacement_degrees=np.linalg.norm(a-b,axis=1).mean())
        (dest/f'{profile}_{name}_embedding.json').write_text(json.dumps(meta,indent=2))
    d=read_shapefile(path);j=json.loads(path.with_suffix('.pca_key.json').read_text());key=PCATransform(np.array(j['center']),np.array(j['basis']))
    rows=[]
    def add(label,dd,seed,mode='independent'):
        r=extract_dataset(dd,w,c,reference_transform=key)
        rows.append(dict(profile=profile,dataset=name,attack=label,seed=seed,noise_mode=mode,nc=r.nc_value,accuracy=r.bit_accuracy_value,empty_bins=r.empty_bins))
    add('no_attack',d,0)
    for mode in ['independent','correlated']:
        add('compound_main',compound(d,0,mode,True),0,mode)
        for s in seeds:add('compound',compound(d,s,mode),s,mode)
    for s in seeds:
        add('vertex_delete_50pct',attacks.vertex_delete(d,.5,s),s)
        add('object_delete_90pct',attacks.object_delete(d,.9,s),s)
    frame=pd.DataFrame(rows);frame.to_csv(dest/f'{profile}_{name}_{min(seeds)}-{max(seeds)}.csv',index=False)
    return profile,name,frame.groupby(['attack','noise_mode']).nc.mean().to_dict()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--profiles',nargs='+',default=list(PROFILES));p.add_argument('--datasets',nargs='+',default=NAMES);p.add_argument('--seeds',nargs='+',type=int,default=[11,12,13]);p.add_argument('--workers',type=int,default=2);a=p.parse_args()
    dest=BASE/'results/robustness_screen';dest.mkdir(parents=True,exist_ok=True)
    (dest/'screen_plan.json').write_text(json.dumps(dict(profiles=PROFILES,selection_seeds=[11,12,13],reserved_validation_seeds=list(range(21,31)),selection_rule='Maximize mean compound NC under unchanged independent-coordinate noise; retain baseline if gains are inconsistent or single-attack controls regress materially.',attack_changes='None for parameter comparison; correlated noise is a separate diagnostic.',python=platform.python_version(),numpy=np.__version__,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2))
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for r in pool.map(run,[(p,n,a.seeds) for p in a.profiles for n in a.datasets]):print(r,flush=True)
