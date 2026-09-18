> Historical pre-refinement protocol. For the current manuscript see [ROBUSTNESS_REVISION.md](ROBUSTNESS_REVISION.md).

# Original-layout revision evidence

The manuscript retains its original figure order and all non-experimental artwork.
The complete fourteen-panel robustness Figure 10 uses the original plotting style,
with updated proposed-method observations. The former redesigned overview plots
and vector diagrams remain archived here but are not used as replacements in the
minimal manuscript revision.

## Reproduce

Use the pinned dependencies in `requirements.txt` and limit BLAS threads:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python run_revision.py --mode legacy
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python run_minimal_revision.py --workers 2
python make_figure10_minimal.py
python validate_minimal_results.py
```

The second driver reuses `outputs/legacy` embeddings and their PCA sidecars when
available, otherwise embeds the original inputs with `preserve_parts=False`.
Do not replace that cache with outputs from another protocol. All main observations
match the original-protocol traversal. Platform and embedding parameters are in
`results/legacy/environment.json`; numeric equivalence across different floating
point implementations is not guaranteed.

`results/minimal_revision/raw.csv` contains 762 observations: 342 main-grid records
and 420 repeated records. Main-grid stochastic seeds are explicitly recorded in
`attack_seeds`; a zero primary seed with an empty list denotes a deterministic
operation. Compound stages have four seeds. `repeated_summary.csv` reports sample
standard deviations over ten attack seeds per map for seven representative cases.
The embedding seed remains fixed at 20260615.

Noise strengths in this grid are **native degrees**, as implemented originally,
not multiples of meter-valued accuracy. Addition uses the original 1e-6 tolerance;
compound addition uses 0.01. Nominal half-map cropping restores a part if fewer than
two vertices remain, as in the archived routine. Repeated compound stages use seeds
s, s+10000, s+20000, and s+30000; main compound uses 1, 2025, 233337, and 2024.
The part-preserving, metric-noise, centroid-crop protocol in `results/audited` is
separate supplementary diagnostic evidence and is not mixed into these curves.

## Figure data and comparison scope

`make_figure10_minimal.py` derives from the original Figure_12 plotting template
(source filenames differ from manuscript numbering). It saves PDF, SVG, and
600-dpi PNG under `figures/Figure_10_minimal.*` and seven `figure10_*.csv` files.
The red proposed curves come from recorded Python observations, without fitted
or interpolated missing results. Other curves retain historical values; their
provenance limitations are documented in `legacy_results/README.md`. For addition
and noise the nine rows are three strengths, each with percentages 10, 30, 50;
for cropping, rows are Railways, Building, Landuse, Boundary, Road, Lake.

The original multi-variant ablation has not been independently reconstructed and
is identified as historical in the manuscript. The new paired single-candidate
versus PSO ablation is separate diagnostic evidence, not an external reproduction.

The original 50% deletion mean NC (0.9231), 90% object-removal mean (0.9110), and
rounded degree-valued mean distortions are reproduced. The compound mean is
0.8680 instead of the originally reported 0.927. That discrepancy is not described
as a minor language-porting difference. The publication materials state the
actual reproducibility scope.
