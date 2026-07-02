# SlurmOut

This directory holds stdout/stderr log files produced by Slurm when cluster jobs
are submitted via `../run_8qubits.slurm`.

Files are named `<JobID>_<JobName>.out`, e.g. `24972984_Permutations.out`.

These logs capture print output from `../Test_8qubits.py`, including progress
bars, convergence warnings, and any Python tracebacks if the job failed.

To inspect a run:
```bash
cat SlurmOut/24972984_Permutations.out
```
