"""Validate numerical identities, protocol continuity and held-out improvement."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
B=Path(__file__).resolve().parent;D=B/'results/optimized_revision'
f=pd.read_csv(D/'raw.csv');old=pd.read_csv(B/'results/minimal_revision/raw.csv')
keys=['dataset','series','attack','seed']
assert len(f)==762 and not f.duplicated(keys).any()
assert set(map(tuple,f[keys].values))==set(map(tuple,old[keys].values))
a=f.merge(old,on=keys,suffixes=('_new','_old'),validate='one_to_one')
assert a.attack_seeds_new.equals(a.attack_seeds_old)
assert np.isfinite(f.select_dtypes('number')).all().all()
assert f.nc.between(0,1).all() and np.allclose(f.bit_accuracy+f.ber,1)
assert np.allclose(f.votes_total/1024,f.votes_mean)
r=f[f.series=='repeat'];assert r.groupby(['dataset','attack']).size().eq(10).all()
s=pd.read_csv(D/'repeated_summary.csv');expected=r.groupby(['dataset','attack']).nc.agg(['mean','std']).reset_index();a=s.merge(expected,on=['dataset','attack'])
assert len(a)==42 and np.allclose(a.nc_mean,a['mean']) and np.allclose(a.nc_std,a['std'])
c=json.loads((D/'config.json').read_text());baseline=json.loads((B/'results/legacy/environment.json').read_text())['config']
diff={k:(baseline.get(k),v) for k,v in c.items() if baseline.get(k)!=v}
assert diff=={'early_stop_rounds':(8,40)},diff
v=pd.read_csv(B/'results/robustness_screen/validation_paired.csv')
assert len(v)==60 and set(v.seed)==set(range(21,31))
assert (v.groupby('dataset').gain.mean()>0).all()
assert (v.full40-v.baseline-v.gain).abs().max()<1e-12
metrics=pd.read_csv(D/'fidelity_runtime.csv');assert len(metrics)==6 and metrics.max_displacement_m.max()<.5
controls=pd.read_csv(D/'controls.csv');assert len(controls)==126
print('PASS: 762 observations; unchanged attack labels/stage seeds; 42 ten-seed groups; NC/BER/vote identities; only stagnation-stop setting changed; six-map held-out compound gains; fidelity and 126 controls.')
