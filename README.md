# VGeoMark: virtual-vertex PSO-QIM revision artifact

Python-only reproduction and audit for the IEEE IoT Journal major revision.
The current manuscript preserves the original figure layout. Start with
[ROBUSTNESS_REVISION.md](ROBUSTNESS_REVISION.md) for the selected stopping rule,
current Figure 10 results, and independent validation.
[MINIMAL_REVISION.md](MINIMAL_REVISION.md) records the preceding grid;
[PROTOCOL.md](PROTOCOL.md) describes the separate diagnostic protocol.

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
* `results/optimized_revision/`: current Fig. 10 grid, ten-seed repetitions and fidelity/runtime.
* `results/robustness_screen/`: all exploratory configurations and paired held-out validation.
* `results/minimal_revision/`: preserved pre-refinement Fig. 10 grid and repetitions.
* `results/legacy/`: original-protocol rerun (not MATLAB bitwise equivalence).
* `results/audited/`: raw multi-seed observations, summaries, per-bit vote arrays,
  metric fidelity and negative controls.
* `results/diagnostics/`: paired internal ablation, redundancy, PCA conditioning,
  and compound-geometry/coordinate-rounding failure modes.
* `figures/`: numerical plots and archived optional diagrams; main manuscript diagrams retain their original artwork.
* `comparisons/`: external-method provenance inventory and missing implementation/tuning evidence.
* `legacy_results/`: historical screenshot-derived tables with provenance caveats.

`run_experiments.py` and `run_single.py` remain compatibility entry points; use
`run_revision.py` for the explicitly audited protocol. All outputs are generated
locally. No MATLAB runtime, GPU, online API, or server credential is required.
The registered-watermark detector must not be described as reference-free.
Original datasets retain their source terms; the code license does not relicense
data. See `data/README.md` and `data/SHA256SUMS`.
