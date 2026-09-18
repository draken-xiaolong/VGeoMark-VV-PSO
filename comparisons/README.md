# External-method comparison evidence

## Current status

This directory indexes the external methods displayed in the manuscript.
**Their executable implementations and tuning logs are not currently included.**
Creating this directory does not establish reproduction or a controlled comparison.
`methods.json` records missing evidence explicitly, without guessing parameter values.

Available historical numerical material:

- [`../legacy_results/`](../legacy_results/): screenshot-derived comparison tables,
  column mappings and normalization notes.
- [`../make_figure10_minimal.py`](../make_figure10_minimal.py): plotting code and
  historical addition/noise/crop/compound comparison values; not implementations
  of the external algorithms.
- [`../results/optimized_revision/`](../results/optimized_revision/): current
  proposed-method observations, seeds, configuration, fidelity and runtime.

The `baseline_*` files in `../results/robustness_screen/` refer to the proposed
method before its stopping-rule refinement. They are **not external baselines**.

## Evidence required for each external method

When the actual source is available, place it under `comparisons/<method_id>/`
and retain its license and attribution. Record:

1. **Implementation provenance:** author-supplied or reimplemented; original URL,
   immutable version/commit, license, and any deviations from the published method.
2. **Parameters:** published defaults, the exact candidate search space, selection
   criterion and selection datasets/seeds, selected values, and per-candidate logs.
   If no tuning occurred, state that and give the source of the fixed settings.
   Never reconstruct an undocumented historical tuning log as if it were original.
3. **Evaluation:** dataset hashes, watermark/reference assumptions, attack definitions,
   stage seeds and attack realizations, repetitions, and full per-trial outputs.
4. **Fidelity and cost:** mean/p95/max displacement for coordinate-modifying methods,
   topology checks, runtime scope, hardware and environment. Zero-watermarking
   methods must retain their own detector/reference assumptions; zero displacement
   alone does not establish comparable operating conditions.
5. **Execution:** dependencies, a runnable command, expected output files and checks.

Use the common data and explicitly documented attack protocol where compatible.
Retain historical results separately; newly reproduced values must be labeled and
must not silently replace historical observations or be represented as the
original runs. Do not substitute the proposed-method single-candidate ablation
for an external author's implementation.

## Scope of the source audit

The current tracked repository, local original code directory and the original
supplementary code archive were inspected. The inspected scripts implement the
proposed method and its attacks; they do not establish the seven external
implementations or their parameter-selection histories. This finding does not
assert that the original authors never released code elsewhere.
