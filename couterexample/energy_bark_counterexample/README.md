# Sparse energy-BARK counterexample

This folder contains the Hamiltonian and experiment source extracted from
`energy_bark_counterexample_reimplementation.zip`. The scripts do not read the
archive, so it can be removed after extraction.

For the 10-qubit embedding (`m=9`), regenerate the trajectories with the
repository's `SKQD.full_skqd_run`, `BARK.full_bark_run` (energy selection), and
`BARK.simplified_bark_run` (amplitude selection), then create the
PaperFigures_v2-style infidelity plot:

```bash
MPLCONFIGDIR=/tmp/mplconfig MPLBACKEND=Agg conda run -n bark-counterexample python couterexample/energy_bark_counterexample/repo_implementation_comparison.py
MPLCONFIGDIR=/tmp/mplconfig MPLBACKEND=Agg conda run -n bark-counterexample python couterexample/energy_bark_counterexample/plot_infidelity.py
```

The plot compares SKQD, energy-BARK, and amplitude-BARK. Its x-coordinate is
the subspace dimension divided by the full 10-qubit dimension, `K/2^10`. The
Hamiltonian builder and archive's original standalone runner are also retained
here; the repo-based comparison does not depend on the zip archive.
