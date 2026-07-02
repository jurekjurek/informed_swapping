# cluster_studies / BARK — BARK vs SKQD

Systematic cluster study of **BARK vs SKQD** over a 5-dimensional parameter grid.
BARK is `subspace_search.algorithms.BarkBarkBark` (bitstring recursive Krylov).

Each job runs, for one parameter combination:

1. Build a Hamiltonian with controlled ground-state and Hamiltonian sparsity.
2. Pick an initial state with a target overlap with the true ground state.
3. Run SKQD for 12 time steps and compute each energy-vs-#states path.
4. Compute `N` random orderings as a baseline.
5. Run BARK with a fixed `keep_states` budget.
6. Save a comparison figure (and CSV row) to the output dir.

## Setup

```bash
source ../../.SKQD/bin/activate
pip install -e ../../subspace_search      # scripts import from the package
```

Outputs (`results/`, any `data/`) are generated when you run the study. The
original outputs from before the 2026-07-02 restructure are preserved in
`../../backup/ClusterStudy/BARK/`.

## Directory layout

```
BARK/
├── run_experiment.py   # one job (5 CLI flags) -> comparison figure
├── run_chunk.py        # run a slice of the param file sequentially
├── generate_jobs.py    # write params/param_grid.txt from the chosen grid
├── run_array.slurm     # Slurm array job over the chunks
├── params/param_grid.txt
└── SlurmOut/           # Slurm logs
```

## Hyperparameters

| Parameter | Values (full grid) | Description |
|-----------|-------------------|-------------|
| `n_qubits` | 6, 8, 10 | System size; Hilbert-space dim = 2ⁿ |
| `gs_sparsity` | 0.05, 0.1, 0.25 | Ground-state support fraction |
| `ham_sparsity` | (per grid) | Fraction of possible off-diagonal pairs filled in H |
| `overlap` | (per grid) | Target \|⟨ψ\|init⟩\|² overlap |
| `keep_states` | 1%, 5%, 10% of dim | States kept per BARK Hamiltonian application (resolved to an integer per `n_qubits`) |

Run `python generate_jobs.py --preset full --dry_run` for the exact job count.

## How to run

```bash
# 1. parameter file
python generate_jobs.py --preset full      # or --preset small for a quick test

# 2. edit run_array.slurm: --account, --partition, venv path, and the
#    `cd .../cluster_studies/BARK` line if your checkout lives elsewhere

# 3. submit (replace N with the count generate_jobs.py printed)
sbatch --array=0-<N-1> run_array.slurm
sbatch --array=0-<N-1>%50 run_array.slurm  # throttle to 50 at once
```

Figures land in `results/`. Filenames encode the parameters:
```
nq8_gs0.10_ham0.50_ov0.30_ks0010_seed42.png
 ^    ^       ^       ^     ^       ^
 |    |       |       |     |       seed
 |    |       |       |     keep_states (zero-padded to 4 digits)
 |    |       |       target overlap
 |    |       Hamiltonian sparsity
 |    ground-state sparsity
 n_qubits
```

## Running a single job locally

```bash
python run_experiment.py \
    --n_qubits 8 --gs_sparsity 0.1 --ham_sparsity 0.5 \
    --overlap 0.3 --keep_states 10 \
    --n_random_paths 20 --seed 42 \
    --output_dir results
```

## Runtime estimates

| n_qubits | dim | Typical time (100 random paths) |
|----------|-----|---------------------------------|
| 6 | 64 | ~10 s |
| 8 | 256 | ~2 min |
| 10 | 1024 | ~60 min |

The bottleneck is `get_all_paths`: each random path requires `dim` `eigsh` calls
on growing submatrices. Reduce `--n_random_paths` to speed things up.
