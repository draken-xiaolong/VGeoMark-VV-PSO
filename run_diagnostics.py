"""Paired PSO ablation, PCA conditioning, vote thinning and rounding diagnostics."""
from pathlib import Path
import json, time
import numpy as np
import pandas as pd
from psoqim import EmbedConfig,embed_dataset,extract_dataset,read_shapefile
from psoqim.algorithm import _extract_votes,_result_from_votes,_pca_candidate_transforms
from psoqim.transforms import PCATransform,collect_points,fit_global_pca
from psoqim.watermark import load_watermark
from psoqim import attacks
from run_revision import NAMES,BASE,crop_features,geod

def main():
 out=BASE/'results/diagnostics';out.mkdir(parents=True,exist_ok=True);w=load_watermark(BASE/'data/M.png');rows=[]
 for name in ([] if __import__('os').environ.get('DIAGNOSTICS_ONLY') == '1' else NAMES):
  d=read_shapefile(BASE/'data'/f'{name}.shp');c=EmbedConfig(n_jobs=1,preserve_parts=True)
  for label,n,t in [('single_candidate',1,1),('pso_10x40',10,40)]:
   c.num_particles=n;c.max_iter=t;tic=time.perf_counter();e=embed_dataset(d,w,c);sec=time.perf_counter()-tic
   original,modified=collect_points(d),collect_points(e.dataset)
   distances=np.asarray(geod(d).inv(original[:,0],original[:,1],modified[:,0],modified[:,1])[2])
   for attack,dd in [('no_attack',e.dataset),('delete50',attacks.vertex_delete(e.dataset,.5,1)),('remove90',attacks.object_delete(e.dataset,.9,1)),('rotate45',attacks.geometric(e.dataset,45))]:
    rr=extract_dataset(dd,w,c,reference_transform=e.transform);rows.append(dict(dataset=name,method=label,attack=attack,nc=rr.nc_value,bit_accuracy=rr.bit_accuracy_value,fitness=e.average_fitness,embed_seconds=sec,mean_displacement_m=float(distances.mean()),max_displacement_m=float(distances.max())))
  print(name,'ablation completed',flush=True)
  pd.DataFrame(rows).to_csv(out/'ablation.csv',index=False)
 if __import__('os').environ.get('ABLATION_ONLY') == '1': return
 # Empirical vote thinning preserves selected frame and samples observed votes without replacement.
 thinning=[];z=np.load(BASE/'results/audited/votes.npz')
 for key in z.files:
  if not any(x in key for x in ['__no_attack__','__vertex_delete_50pct__','__object_delete_90pct__']):continue
  total,one=z[key];name,attack,attackseed=key.split('__')
  for cap in [1,3,5,10,1000000]:
   n=np.minimum(total,cap)
   for seed in range(10):
    rng=np.random.default_rng(seed+40000);sample=rng.hypergeometric(one,total-one,n);r=_result_from_votes(sample,n,w,'fixed_selected_frame')
    thinning.append(dict(dataset=name,attack=attack,attack_seed=int(attackseed),sampling_seed=seed,cap=cap,nc=r.nc_value,bit_accuracy=r.bit_accuracy_value,empty_bins=r.empty_bins))
 pd.DataFrame(thinning).to_csv(out/'vote_thinning.csv',index=False)
 # Conditioning-only experiment: exact isotropy versus separated eigenvalues, no watermark claims.
 stability=[]
 theta=np.linspace(0,2*np.pi,2000,endpoint=False)
 for ratio in [1.,1.001,1.01,1.1,2.,5.]:
  xy=np.column_stack([ratio*np.cos(theta),np.sin(theta)]);ref=np.linalg.eigh(np.cov(xy.T))[1][:,-1]
  for seed in range(30):
   rng=np.random.default_rng(seed)
   for removal in [.01,.1,.5,.9]:
    attacked=xy[rng.random(len(xy))>=removal];ev,u=np.linalg.eigh(np.cov(attacked.T));angle=np.degrees(np.arccos(np.clip(abs(u[:,-1]@ref),0,1)))
    stability.append(dict(axis_ratio=ratio,removal=removal,seed=seed,eigengap=(ev[-1]-ev[0])/ev[-1],axis_drift_degrees=angle))
 pd.DataFrame(stability).to_csv(out/'pca_conditioning.csv',index=False)
 failure=[]
 for name in NAMES:
  d=read_shapefile(BASE/'outputs/audited'/f'{name}.shp');payload=json.loads((BASE/'outputs/audited'/f'{name}.pca_key.json').read_text());key=PCATransform(np.array(payload['center']),np.array(payload['basis']));c=EmbedConfig(n_jobs=1,preserve_parts=True)
  for label,dd in [('rotate45_crop',crop_features(attacks.geometric(d,45))),('rotate45_remove90',attacks.object_delete(attacks.geometric(d,45),.9,1))]:
   r=extract_dataset(dd,w,c,reference_transform=key);failure.append(dict(dataset=name,case=label,nc=r.nc_value,bit_accuracy=r.bit_accuracy_value,empty_bins=r.empty_bins))
  for decimals in [6,7,8,9,10,12,14]:
   dd=d.copy()
   for s in dd.shapes:s.points=np.round(np.array(s.points),decimals).tolist()
   r=extract_dataset(dd,w,c,reference_transform=key);failure.append(dict(dataset=name,case=f'round_{decimals}_decimals',nc=r.nc_value,bit_accuracy=r.bit_accuracy_value,empty_bins=r.empty_bins))
 pd.DataFrame(failure).to_csv(out/'failure_modes.csv',index=False)
if __name__=='__main__':main()
