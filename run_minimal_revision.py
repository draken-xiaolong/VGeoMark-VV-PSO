"""Complete the original Fig.10 grid; keep legacy attack semantics explicit."""
from pathlib import Path
import argparse,json
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from psoqim import EmbedConfig,read_shapefile,embed_dataset,extract_dataset
from psoqim.transforms import PCATransform
from psoqim.watermark import load_watermark
from psoqim import attacks
from run_experiments import attack_suite
BASE=Path(__file__).resolve().parent
NAMES=['Railways','Building','Landuse','Boundary','Road','Lake']

def main_seeds(label):
 if label == 'compound': return [1,2025,233337,2024]
 if label.startswith('vertex_delete'): return [1]
 if label.startswith('object_delete'): return [2025]
 if label.startswith(('vertex_add','addition_')): return [233337]
 if label.startswith('noise_'): return [2024]
 return []

def compound_seed(d,seed):
 d=attacks.vertex_delete(d,.3,seed)
 d=attacks.object_delete(d,.3,seed+10000)
 d=attacks.vertex_add(d,.1,tolerance=.01,seed=seed+20000)
 d=attacks.vertex_noise(d,.2,.6,seed+30000)
 d=attacks.geometric(d,x_shift=10,y_shift=10)
 d=attacks.geometric(d,scale=.5)
 return attacks.reverse_objects(attacks.reverse_vertices(d))

def run(name):
 dest=BASE/'results/minimal_revision';dest.mkdir(parents=True,exist_ok=True)
 c=EmbedConfig(n_jobs=1,preserve_parts=False);w=load_watermark(BASE/'data/M.png')
 saved=BASE/'outputs/legacy'/f'{name}.shp'
 if saved.exists():
  d=read_shapefile(saved);j=json.loads(saved.with_suffix('.pca_key.json').read_text());key=PCATransform(np.array(j['center']),np.array(j['basis']))
 else:
  e=embed_dataset(read_shapefile(BASE/'data'/f'{name}.shp'),w,c);d=e.dataset;key=e.transform
 rows=[]
 def record(label,dd,series='main',seed=None):
  seeds=main_seeds(label) if series=='main' else ([seed,seed+10000,seed+20000,seed+30000] if label=='compound' else [seed])
  seed=seeds[0] if seeds else 0
  r=extract_dataset(dd,w,c,reference_transform=key);v=np.array(r.details['votes_per_bit'])
  rows.append(dict(dataset=name,series=series,attack=label,seed=seed,attack_seeds=json.dumps(seeds),nc=r.nc_value,bit_accuracy=r.bit_accuracy_value,ber=r.ber_value,votes_total=r.votes_total,empty_bins=r.empty_bins,votes_min=int(v.min()),votes_median=float(np.median(v)),votes_mean=float(v.mean()),votes_max=int(v.max())))
 for label,_,dd in attack_suite(d):record(label,dd)
 for alpha in [0,.5,1]:
  for pct in [10,30,50]:record(f'addition_alpha{alpha:g}_pct{pct}',attacks.vertex_add(d,pct/100,strength=alpha,tolerance=1e-6))
 for strength in [.6,1.,1.4]:
  for pct in [10,30,50]:record(f'noise_strength{strength:g}_pct{pct}',attacks.vertex_noise(d,pct/100,strength))
 for pct in [10,20,40]:record(f'vertex_delete_{pct}pct',attacks.vertex_delete(d,pct/100))
 for pct in [10,50,70]:record(f'object_delete_{pct}pct',attacks.object_delete(d,pct/100))
 for scale in [.1,.9,1.3,1.7,2.1]:record(f'scale_{scale:g}',attacks.geometric(d,scale=scale))
 for shift in [20,30,40,50,60]:record(f'translation_{shift}',attacks.geometric(d,x_shift=shift,y_shift=shift))
 for seed in range(1,11):
  for pct in [30,50]:record(f'vertex_delete_{pct}pct',attacks.vertex_delete(d,pct/100,seed),'repeat',seed)
  for pct in [30,90]:record(f'object_delete_{pct}pct',attacks.object_delete(d,pct/100,seed),'repeat',seed)
  record('vertex_add_50pct',attacks.vertex_add(d,.5,seed=seed),'repeat',seed)
  record('noise_50pct_1p4tau',attacks.vertex_noise(d,.5,1.4,seed),'repeat',seed)
  record('compound',compound_seed(d,seed),'repeat',seed)
 pd.DataFrame(rows).to_csv(dest/f'{name}.csv',index=False)
 return name,len(rows)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=2);p.add_argument('--datasets',nargs='+',default=NAMES);args=p.parse_args()
 with ProcessPoolExecutor(max_workers=args.workers) as pool:
  for result in pool.map(run,args.datasets):print(result,flush=True)
 dest=BASE/'results/minimal_revision';f=pd.concat([pd.read_csv(dest/f'{n}.csv') for n in args.datasets],ignore_index=True);f.to_csv(dest/'raw.csv',index=False)
 f[f.series=='repeat'].groupby(['dataset','attack']).agg(n=('nc','count'),nc_mean=('nc','mean'),nc_std=('nc','std'),accuracy_mean=('bit_accuracy','mean'),erasures_mean=('empty_bins','mean')).reset_index().to_csv(dest/'repeated_summary.csv',index=False)
