# VGeoMark: virtual-vertex PSO-QIM revision artifact

Python-only reproduction and audit for the IEEE IoT Journal major revision.
See [PROTOCOL.md](PROTOCOL.md) before interpreting the numerical results.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m unittest discover -p test_core.py -v
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python run_revision.py --mode legacy
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python run_revision.py --mode audited --seeds 10
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python run_diagnostics.py
python make_diagrams.py
python make_revision_results.py
python validate_results.py
```

* `data/`: six original shapefile datasets, CRS/attribute sidecars and watermark.
* `psoqim/`: NumPy Haar, QIM, PSO, synchronization, votes, and shapefile I/O.
* `results/legacy/`: original-protocol rerun (not MATLAB bitwise equivalence).
* `results/audited/`: raw multi-seed observations, summaries, per-bit vote arrays,
  metric fidelity and negative controls.
* `results/diagnostics/`: paired internal ablation, redundancy, PCA conditioning,
  and compound-geometry/coordinate-rounding failure modes.
* `figures/`: genuine vector PDF/SVG diagrams generated from source.
* `legacy_results/`: historical screenshot-derived tables with provenance caveats.

`run_experiments.py` and `run_single.py` remain compatibility entry points; use
`run_revision.py` for the explicitly audited protocol. All outputs are generated
locally. No MATLAB runtime, GPU, online API, or server credential is required.
The registered-watermark detector must not be described as reference-free.
Original datasets retain their source terms; the code license does not relicense
data. See `data/README.md` and `data/SHA256SUMS`.
