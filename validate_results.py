"""Validate the distributed experiment records and original input manifest."""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
r=Path(__file__).resolve().parent
a=pd.read_csv(r/'results/audited/raw_attacks.csv');l=pd.read_csv(r/'results/legacy/raw_attacks.csv');v=np.load(r/'results/audited/votes.npz')
assert len(a)==738 and len(l)==138 and len(v.files)==738
assert not a[['dataset','attack','seed']].duplicated().any()
assert ((a.nc>=0)&(a.nc<=1+1e-12)).all()
assert np.allclose(a.ber+a.bit_accuracy,1)
for row in a.itertuples():
 total,one=v[f'{row.dataset}__{row.attack}__{row.seed}']
 assert total.shape==(1024,) and np.all(one<=total)
 assert total.sum()==row.votes_total and (total==0).sum()==row.empty_bins
for line in (r/'data/SHA256SUMS').read_text().splitlines():
 digest,name=line.split('  ')
 assert hashlib.sha256((r/'data'/name).read_bytes()).hexdigest()==digest
f=pd.read_csv(r/'results/audited/fidelity_runtime.csv')
assert (f.unclosed_rings==0).all()
assert (f.max_displacement_m<.5).all()
print('PASS: 738 audited rows, 138 legacy rows, 738 vote arrays, input hashes, closed rings and fidelity bounds.')
