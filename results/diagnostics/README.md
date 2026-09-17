# Diagnostic execution provenance

ablation.csv was executed on the user-authorized Xeon Platinum 8352V CPU server, Python 3.12.3, NumPy 2.4.6, one Python worker and one BLAS/OpenMP thread. The GPU was not used. CPU quota was 16 cores (cpu.max=1600000 100000), despite 128 host logical CPUs being visible. No server address or credentials are included.

Other diagnostic CSV files were executed on Apple M1 Pro / macOS 14.6.1 / Python 3.11.5 / NumPy 2.4.6. Differences at quantization boundaries across architectures are possible. Ablation uses common nominal Q and M, seeds and attack settings; both conditions' actual achieved geodesic displacement is recorded. It is not a reimplementation of a published baseline.

To reproduce all diagnostics locally: `python run_diagnostics.py`. Optional environment variables `ABLATION_ONLY=1` and `DIAGNOSTICS_ONLY=1` allow independent execution of the two portions.
