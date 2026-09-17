# Revision protocol and interpretation

The source of truth is the Python code plus per-observation CSV/NPZ files. No
new result is manually transcribed from a plot. The original manuscript and
historical baseline values are NOT proof of independently reproduced baselines.

## Two explicit protocols

`run_revision.py --mode legacy` retains the original Python carrier traversal,
PSO seed (20260615 + feature_index*1009), original attack seeds and original
coordinate-unit noise. It is the closest executable comparison with the initial
submission. It preserves the original feature-wide periodic Haar boundary,
including multipart concatenation. It is for historical reproduction only.

`run_revision.py --mode audited` preserves the original Q=0.03, M=1e6,
10-particle/40-iteration PSO and registered 32x32 watermark. It changes only:

* multipart boundaries are respected and polygon rings remain exactly closed;
* repeated closing vertices are not separate carriers;
* RNG for first parts is retained; additional parts have deterministic seeds;
* stochastic attacks use seeds 1,...,10, one fixed embedding per dataset;
* noise offsets use east/north meters, with tau=2 m for the three WGS84 layers
  and tau=0.5 m for the CGCS2000 layers, converted by an ellipsoidal direct
  geodesic; the geometric embedding itself remains in original coordinates;
* half cropping removes features by median feature centroid (entire features),
  whereas legacy clipping restored parts that had fewer than two surviving
  vertices and therefore was not a strict half-map crop;
* compound_metric applies vertex deletion 30%, object deletion 30%, vertex
  addition 10% (offset +1e-6,-1e-6 degrees), metric noise on 20% of vertices at
  0.6 tau, scaling 0.5, translation (10,10), and reversal of vertices/objects.
  Stage seeds are s,s+10000,s+20000,s+30000. These same stage definitions and
  seeds must be used for any future paired baseline evaluation.

A nominal vertex-deletion fraction applies to Bernoulli selection of vertices;
endpoints are retained and short parts are protected, so the realized fraction
is not necessarily the nominal fraction. Object removal is Bernoulli sampling,
with one object retained if all would be removed. Insertion uses Bernoulli
selection of segments, uniform interpolation, and a fixed degree offset. It is
not a Poisson process. Legacy degree noise remains separately labelled.
Rotation and scaling are algebraic operations on coordinate arrays, not a
physical geodesic motion on Earth. No coordinate rounding is applied by default.

## Metrics and detector

NC is cosine similarity of binary arrays, matching both archived NC.m and the
original Python implementation. The initial manuscript incorrectly defined NC
as XNOR accuracy. Bit accuracy (XNOR) and BER=1-accuracy are separately reported.
The null cosine NC depends on watermark density; it need not be 0.5.

The detector chooses the maximum registered-watermark NC among identity,
stored PCA frame, and four signs of the attacked PCA basis. This is a known-
watermark ownership verifier, blind only to original coordinates. It does not
recover an unknown watermark independently. Negative controls use the same
candidate selection. Twenty random wrong watermarks per dataset and an
unwatermarked-host test are descriptive controls, not an operationally
calibrated false-positive guarantee.

`votes.npz` records total votes and one-votes for every one of the 1024 positions
for every attack. Empty bins get zero only for metric calculation; erasures
are reported explicitly. Nonempty ties vote one. Vote-thinning diagnostics
sample observed votes without replacement in the selected frame; they isolate
redundancy conditional on frame selection, not an independent detector.

Fidelity is calculated with ellipsoidal geodesic distances using the .prj
ellipsoid (pyproj/PROJ). All six source CRSs use degrees. Mean, p95, maximum
meters and mean degree displacement are reported. This is a metric evaluation
without reprojecting before embedding. Ring closure is checked exactly; it is
not a claim that all map topology is preserved. Tiny degree perturbations can
be lost under coordinate rounding, explicitly tested in diagnostics.

Single-run embedding times in fidelity_runtime.csv include the in-memory
embedding call but not I/O or attacks; extraction time includes all candidates.
Machine metadata, workers and BLAS limits are in environment.json. Timings are
machine/run dependent; isolated three-repeat benchmarks are separate files.
No GPU acceleration is claimed: all numerical work uses CPU NumPy.

## Baselines and attribution

`legacy_results/` contains historical CSV tables previously transcribed from
screenshots, including documented OCR normalization. Original baseline source
code, parameter tuning records, and shared attack realizations are not present.
These numbers cannot establish a new controlled comparison or a speed ranking.
The single-candidate ablation uses one feasible random QIM candidate with the
same carrier/PCA/voting mechanism; it is NOT presented as Xi et al.'s original
implementation. No missing experimental observations are synthesized.
