# Robustness refinement and protocol audit

The selected configuration retains Q=0.03, M=1e6, 10 particles, a maximum of
40 iterations, embedding seed 20260615 and the same carrier/frame/voting rules.
Only `early_stop_rounds` changes from 8 to 40: within a 40-iteration search this
removes stagnation-based termination. The target-fitness stop at 0.995 remains.
The package-wide default is unchanged so historical commands remain reproducible;
`results/optimized_revision/config.json` specifies the new manuscript run.

## Selection and independent validation

Six profiles were evaluated on six maps and attack seeds 11–13. Baseline, 20 or
40 particles, 20 particles/80 iterations, disabling the 8-round stop, and Q=0.1
were all retained in `results/robustness_screen/screen_raw.csv`. The first four
profiles were screened first; the last two were added as an exploratory follow-up.
The best selection mean under independent-coordinate compound noise was the
40-round stopping profile (0.892490). Selection was frozen in `selection.json`
before validation on seeds 21–30. No embedding seeds were tuned.

The validation compound mean rises from 0.868447 to 0.889230. Per-map average
gains are positive on all six maps (0.011888–0.028125). These are 60 paired
map/attack-seed observations conditional on one embedding, not 60 independent
maps or an estimate over embedding random seeds. No significance claim is made.
The original displayed attack realization rises from 0.867970 to 0.889118.
Unrounded records are authoritative; two-decimal manuscript values are display
rounding and do not alter source data.

## Why this does not recover the original 0.927 claim

The initial submission reports 0.927 average compound NC and 0.900 minimum.
The executable pre-refinement Python grid yields 0.867970 and 0.801532. The
archived component figure does not provide sufficient complete raw runs for
independent recovery of the originally claimed compound result.

Inspection also found protocol differences between archived implementations:
original noise adds the same scalar to both coordinate components; the Python
attack uses independent offsets. Original deletion permits endpoint deletion,
whereas the current Python implementation protects endpoints/short parts.
Random-number consumption and generators also differ. Thus language choice alone
is not a scientific explanation. The paired correlated-noise diagnostic changes
only the y offset to equal x, preserving the current selection and x-offset
sequence. It is not a bitwise reproduction of the original generator or of the
complete original geometry handling. It does not fully recover 0.927.

For consistency, the new main grid keeps the previous Python attack definitions,
order, nominal intensities and all stage seeds. The correlated-noise results are
separate diagnostics and are not substituted into the main figure. No attack
intensity or seed was selected to improve the reported score.

## Reproduction

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python run_robustness_screen.py --profiles baseline p20 p40 long --seeds 11 12 13
python run_robustness_screen.py --profiles full40 q01 --seeds 11 12 13
python run_robustness_screen.py --profiles baseline full40 --seeds 21 22 23 24 25 26 27 28 29 30
python benchmark_selected.py
python run_optimized_revision.py --workers 2
python validate_optimized_results.py
python make_figure10_minimal.py --results results/optimized_revision --stem Figure_10_optimized
```

Run the benchmark without concurrent experiment jobs. It writes serialized
embeddings, full configuration, environment, metric displacement, closure counts,
embedding/extraction timing, and 126 negative-control observations. The new grid
contains the same 342 main and 420 repeated observations as the prior grid.
Historical result directories are retained. Original nonexperimental artwork is
not regenerated. External-baseline records remain descriptive historical values;
this refinement does not independently reproduce those methods.
