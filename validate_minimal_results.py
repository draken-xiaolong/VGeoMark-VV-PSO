"""Check original-layout results without rerunning the expensive attack grid."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
raw = pd.read_csv(BASE / 'results/minimal_revision/raw.csv')
main = raw[raw.series == 'main']
repeat = raw[raw.series == 'repeat']
assert (len(main), len(repeat)) == (342, 420)
assert not raw.duplicated(['dataset', 'series', 'attack', 'seed']).any()
assert repeat.groupby(['dataset', 'attack']).size().eq(10).all()
assert repeat.groupby(['dataset', 'attack']).seed.apply(lambda x: set(x) == set(range(1, 11))).all()
assert raw.nc.between(0, 1).all()
assert np.allclose(raw.bit_accuracy + raw.ber, 1)
assert np.allclose(raw.votes_total / 1024, raw.votes_mean)
for row in raw.itertuples():
    seeds = json.loads(row.attack_seeds)
    assert row.seed == (seeds[0] if seeds else 0)
    assert len(seeds) == (4 if row.attack == 'compound' else (1 if row.seed else 0))
old = pd.read_csv(BASE / 'results/legacy/raw_attacks.csv')
joined = main.merge(old, on=['dataset', 'attack'], suffixes=('_new', '_old'))
assert len(joined) == 138
assert np.max(np.abs(joined.nc_new - joined.nc_old)) < 1e-12
summary = pd.read_csv(BASE / 'results/minimal_revision/repeated_summary.csv')
expected = repeat.groupby(['dataset', 'attack']).nc.agg(['mean', 'std']).reset_index()
joined = summary.merge(expected, on=['dataset', 'attack'])
assert len(joined) == 42
assert np.allclose(joined.nc_mean, joined['mean'])
assert np.allclose(joined.nc_std, joined['std'])
for dataset in raw.dataset.unique():
    per_map = pd.read_csv(BASE / 'results/minimal_revision' / f'{dataset}.csv')
    assert len(per_map) == 127
    assert np.allclose(per_map.nc, raw[raw.dataset == dataset].nc)
print('PASS: 762 observations, 42 ten-seed groups, seed metadata, vote/metric identities, original 138 NC values and summaries.')
