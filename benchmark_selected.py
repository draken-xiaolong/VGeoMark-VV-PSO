"""Benchmark one selected profile in isolation, then save its reusable embedding.
Input: results/robustness_screen/selection.json (frozen before validation).
"""
from pathlib import Path
import json,time,platform,os,subprocess
from dataclasses import asdict
import numpy as np
import pandas as pd
from psoqim import EmbedConfig,embed_dataset,extract_dataset,read_shapefile,write_shapefile
from psoqim.transforms import collect_points
from psoqim.watermark import load_watermark,Watermark,logistic_encrypt
from run_experiments import save_pca_key
from run_revision import geod
from run_robustness_screen import PROFILES,NAMES
BASE=Path(__file__).resolve().parent
if __name__=='__main__':
    selection=json.loads((BASE/'results/robustness_screen/selection.json').read_text());profile=selection['profile']
    c=EmbedConfig(n_jobs=1,preserve_parts=False,**PROFILES[profile]);dest=BASE/'results/optimized_revision';dest.mkdir(parents=True,exist_ok=True)
    (dest/'config.json').write_text(json.dumps(asdict(c),indent=2))
    (dest/'environment.json').write_text(json.dumps(dict(config=asdict(c),platform=platform.platform(),python=platform.python_version(),numpy=np.__version__,processor=subprocess.getoutput('sysctl -n machdep.cpu.brand_string'),threads={k:os.environ.get(k) for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']},timing_scope='One in-memory call per map, no I/O or attack construction; sequential maps; validation/attack jobs not running concurrently.'),indent=2))
    w=load_watermark(BASE/'data/M.png');summ=[];controls=[]
    for name in NAMES:
        d=read_shapefile(BASE/'data'/f'{name}.shp');t=time.perf_counter();e=embed_dataset(d,w,c);duration=time.perf_counter()-t
        output=BASE/'outputs/optimized_revision'/f'{name}.shp';write_shapefile(e.dataset,output);save_pca_key(e.transform,output.with_suffix('.pca_key.json'),name,c)
        embedded=read_shapefile(output);a,b=collect_points(d),collect_points(embedded)
        screen=read_shapefile(BASE/'outputs'/f'robustness_{profile}'/f'{name}.shp')
        assert np.array_equal(b,collect_points(screen)), 'Embedding changed since selection'
        dist=np.asarray(geod(d).inv(a[:,0],a[:,1],b[:,0],b[:,1])[2]);native=np.linalg.norm(a-b,axis=1)
        ring_bad=0
        if d.shape_type==5:
            for s in embedded.shapes:
                bounds=list(s.parts)+[len(s.points)]
                ring_bad+=sum(not np.array_equal(s.points[i],s.points[j-1]) for i,j in zip(bounds[:-1],bounds[1:]))
        t=time.perf_counter();r=extract_dataset(embedded,w,c,reference_transform=e.transform);extraction=time.perf_counter()-t
        row=dict(dataset=name,profile=profile,features=len(d.shapes),vertices=len(a),embed_seconds=duration,extraction_seconds=extraction,fitness=e.average_fitness,iterations=e.average_iterations,mean_displacement_m=dist.mean(),max_displacement_m=dist.max(),p95_displacement_m=np.quantile(dist,.95),mean_displacement_degrees=native.mean(),unclosed_rings=ring_bad,nc=r.nc_value)
        summ.append(row);print(row,flush=True)
        null=extract_dataset(d,w,c,reference_transform=e.transform);controls.append(dict(dataset=name,kind='unwatermarked_host',seed=0,nc=null.nc_value,bit_accuracy=null.bit_accuracy_value))
        for seed in range(1,21):
            bits=np.random.default_rng(seed+90000).integers(0,2,w.original.shape,dtype=np.uint8);wrong=Watermark(bits,logistic_encrypt(bits))
            rr=extract_dataset(embedded,wrong,c,reference_transform=e.transform);controls.append(dict(dataset=name,kind='wrong_watermark',seed=seed,nc=rr.nc_value,bit_accuracy=rr.bit_accuracy_value))
        pd.DataFrame(summ).to_csv(dest/'fidelity_runtime.csv',index=False);pd.DataFrame(controls).to_csv(dest/'controls.csv',index=False)
