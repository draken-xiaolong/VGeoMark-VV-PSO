#!/usr/bin/env python3
"""Audited supplementary experiment; no MATLAB dependency. See PROTOCOL.md."""
from pathlib import Path
import argparse, json, time, platform, subprocess, hashlib
from dataclasses import asdict
import numpy as np
import pandas as pd
from pyproj import CRS
from psoqim import EmbedConfig, embed_dataset, extract_dataset, read_shapefile, write_shapefile
from psoqim import attacks
from psoqim.algorithm import _extract_votes, _result_from_votes, _pca_candidate_transforms
from psoqim.transforms import collect_points, fit_global_pca
from psoqim.watermark import load_watermark, Watermark, logistic_encrypt
from run_experiments import attack_suite, save_pca_key

BASE=Path(__file__).resolve().parent
NAMES=['Railways','Building','Landuse','Boundary','Road','Lake']

def geod(dataset):
    return CRS.from_wkt(dataset.source_path.with_suffix('.prj').read_text()).get_geod()

def metric_noise(dataset, ratio, strength_m, seed):
    rng=np.random.default_rng(seed); out=dataset.copy(); g=geod(dataset)
    for shape in out.shapes:
        pts=np.asarray(shape.points,float); mask=rng.random(len(pts))<ratio
        east,north=rng.uniform(-strength_m,strength_m,(2,len(pts)))
        lon,lat,_=g.fwd(pts[:,0],pts[:,1],np.degrees(np.arctan2(east,north)),np.hypot(east,north))
        pts[mask,0]=lon[mask]; pts[mask,1]=lat[mask]
        if dataset.shape_type==5:
            bounds=list(shape.parts)+[len(pts)]
            for a,b in zip(bounds[:-1],bounds[1:]): pts[b-1]=pts[a]
        shape.points=pts.tolist()
    return out

def crop_features(dataset):
    # Remove entire features according to centroid; no fallback restoring removed geometry.
    means=np.array([np.mean(np.asarray(s.points)[:,0]) for s in dataset.shapes])
    keep=means<=np.median(means);out=dataset.copy()
    out.shapes=[s for s,k in zip(out.shapes,keep) if k]
    out.records=[r for r,k in zip(out.records,keep) if k]
    return out

def cases(dataset, seed, tau):
    yield 'vertex_delete_30pct',attacks.vertex_delete(dataset,.3,seed)
    yield 'vertex_delete_50pct',attacks.vertex_delete(dataset,.5,seed)
    yield 'object_delete_30pct',attacks.object_delete(dataset,.3,seed)
    yield 'object_delete_90pct',attacks.object_delete(dataset,.9,seed)
    yield 'vertex_add_10pct',attacks.vertex_add(dataset,.1,seed=seed)
    yield 'vertex_add_30pct',attacks.vertex_add(dataset,.3,seed=seed)
    yield 'vertex_add_50pct',attacks.vertex_add(dataset,.5,seed=seed)
    yield 'noise_10pct_0p6tau_m',metric_noise(dataset,.1,.6*tau,seed)
    yield 'noise_50pct_1p4tau_m',metric_noise(dataset,.5,1.4*tau,seed)
    # Legacy coordinate-unit noise and compound are separate for traceability.
    yield 'legacy_noise_50pct_1p4degrees',attacks.vertex_noise(dataset,.5,1.4,seed)
    d=attacks.vertex_delete(dataset,.3,seed)
    d=attacks.object_delete(d,.3,seed+10000)
    d=attacks.vertex_add(d,.1,tolerance=1e-6,seed=seed+20000)
    d=metric_noise(d,.2,.6*tau,seed+30000)
    d=attacks.geometric(d,scale=.5,x_shift=10,y_shift=10)
    yield 'compound_metric',attacks.reverse_objects(attacks.reverse_vertices(d))

def result_row(name,mode,attack,seed,d,w,c,key):
    t=time.perf_counter();r=extract_dataset(d,w,c,reference_transform=key);elapsed=time.perf_counter()-t
    v=np.array(r.details['votes_per_bit']); one=np.array(r.details['one_votes_per_bit'])
    frame=fit_global_pca(d); a=collect_points(d); ev=np.linalg.eigvalsh(np.cov(a.T))
    gap=float((ev[-1]-ev[0])/max(ev[-1],np.finfo(float).eps))
    row=dict(dataset=name,mode=mode,attack=attack,seed=seed,nc=r.nc_value,bit_accuracy=r.bit_accuracy_value,ber=r.ber_value,extraction_seconds=elapsed,votes_total=r.votes_total,empty_bins=r.empty_bins,votes_min=int(v.min()),votes_median=float(np.median(v)),votes_mean=float(v.mean()),votes_max=int(v.max()),ties=r.details['ties'],variant=r.transform_variant,eigengap=gap,retained_vertices=len(a))
    return row,v,one

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seeds',type=int,default=10);ap.add_argument('--jobs',type=int,default=1);ap.add_argument('--datasets',nargs='+',default=NAMES);ap.add_argument('--mode',choices=['legacy','audited'],default='audited');args=ap.parse_args()
    dest=BASE/'results'/args.mode;dest.mkdir(parents=True,exist_ok=True)
    c=EmbedConfig(n_jobs=args.jobs,preserve_parts=args.mode=='audited');w=load_watermark(BASE/'data/M.png')
    env=dict(config=asdict(c),platform=platform.platform(),python=platform.python_version(),numpy=np.__version__,seeds=list(range(1,args.seeds+1)),processor=(subprocess.getoutput('sysctl -n machdep.cpu.brand_string') if platform.system()=='Darwin' else platform.processor()),cpu_count=__import__('os').cpu_count(),threads={k:__import__('os').environ.get(k) for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']})
    (dest/'environment.json').write_text(json.dumps(env,indent=2))
    rows=[];summ=[];votes={};controls=[]
    for name in args.datasets:
        d=read_shapefile(BASE/'data'/f'{name}.shp');t=time.perf_counter();e=embed_dataset(d,w,c);duration=time.perf_counter()-t
        output=BASE/'outputs'/args.mode/f'{name}.shp';write_shapefile(e.dataset,output);save_pca_key(e.transform,output.with_suffix('.pca_key.json'),name,c)
        embedded=read_shapefile(output);embedded.source_path=d.source_path
        a,b=collect_points(d),collect_points(embedded);dist=np.asarray(geod(d).inv(a[:,0],a[:,1],b[:,0],b[:,1])[2]);native=np.linalg.norm(a-b,axis=1)
        ring_bad=0
        if d.shape_type==5:
            for s in embedded.shapes:
                bounds=list(s.parts)+[len(s.points)]
                ring_bad+=sum(not np.array_equal(s.points[i],s.points[j-1]) for i,j in zip(bounds[:-1],bounds[1:]))
        summary=dict(dataset=name,mode=args.mode,features=len(d.shapes),vertices=len(a),embed_seconds=duration,fitness=e.average_fitness,iterations=e.average_iterations,mean_displacement_m=dist.mean(),max_displacement_m=dist.max(),p95_displacement_m=np.quantile(dist,.95),mean_displacement_degrees=native.mean(),unclosed_rings=ring_bad)
        summ.append(summary);print(json.dumps(summary),flush=True)
        if args.mode=='legacy':
            iterator=((attack,0,attacked) for attack,_,attacked in attack_suite(embedded))
        else:
            def iterator_fn():
                yield 'no_attack',0,embedded
                yield 'crop_half_centroids',0,crop_features(embedded)
                for angle in [45,90,135,180,225,270,315]: yield f'rotation_{angle}',0,attacks.geometric(embedded,angle_deg=angle)
                for label,dd in [('translation_10',attacks.geometric(embedded,x_shift=10,y_shift=10)),('scale_0p5',attacks.geometric(embedded,scale=.5)),('reverse_vertices',attacks.reverse_vertices(embedded)),('reverse_objects',attacks.reverse_objects(embedded))]:yield label,0,dd
                for seed in range(1,args.seeds+1):
                    for label,dd in cases(embedded,seed,2. if name in NAMES[:3] else .5):yield label,seed,dd
            iterator=iterator_fn()
        for label,seed,dd in iterator:
            row,v,one=result_row(name,args.mode,label,seed,dd,w,c,e.transform);rows.append(row);votes[f'{name}__{label}__{seed}']=np.vstack([v,one])
        # Null-host control uses identical candidate search and known target.
        null=extract_dataset(d,w,c,reference_transform=e.transform)
        controls.append(dict(dataset=name,kind='unwatermarked_host',seed=0,nc=null.nc_value,bit_accuracy=null.bit_accuracy_value))
        for seed in range(1,21):
            bits=np.random.default_rng(seed+90000).integers(0,2,w.original.shape,dtype=np.uint8);wrong=Watermark(bits,logistic_encrypt(bits))
            rr=extract_dataset(embedded,wrong,c,reference_transform=e.transform)
            controls.append(dict(dataset=name,kind='wrong_watermark',seed=seed,nc=rr.nc_value,bit_accuracy=rr.bit_accuracy_value))
        pd.DataFrame(rows).to_csv(dest/'raw_attacks.csv',index=False);pd.DataFrame(summ).to_csv(dest/'fidelity_runtime.csv',index=False);pd.DataFrame(controls).to_csv(dest/'controls.csv',index=False)
        np.savez_compressed(dest/'votes.npz',**votes)
        print(name,'completed',len(rows),'attack observations',flush=True)
    frame=pd.DataFrame(rows)
    frame.groupby(['dataset','mode','attack']).agg(n=('nc','count'),nc_mean=('nc','mean'),nc_std=('nc','std'),ba_mean=('bit_accuracy','mean'),ba_std=('bit_accuracy','std'),ber_mean=('ber','mean'),empty_mean=('empty_bins','mean'),votes_mean=('votes_mean','mean'),extraction_seconds_mean=('extraction_seconds','mean')).reset_index().fillna({'nc_std':0.,'ba_std':0.}).to_csv(dest/'attack_summary.csv',index=False)
if __name__=='__main__':main()
