# ClusterStudy / KRAB — KRAB vs SKQD

Systematic cluster study comparing the **KRAB** (selected-subspace Krylov)
algorithm in `new_approach/krab_4.py` against **SKQD** (and a random baseline),
over a 6-dimensional hyperparameter grid.

The study is built to answer three questions:

1. **Does KRAB beat SKQD?** — for every Hamiltonian, how many sampled basis
   states does each method need to reach relative energy error `< 10⁻³`?
2. **Where does it work / not work?** — in which parameter regions does KRAB
   reach the target, and where does it stall or lose to SKQD?
3. **How do the resources scale?** — how do the required subspace size,
   number of sampled states, and wall time grow with system size and the
   hyperparameters?

## Common currency: states-to-target

Every method produces an **ordering** of basis states; `get_one_path` then
gives the ground-energy estimate as states are added one at a time. The metric
`n_states_to_target` is the first count of basis states at which the estimate
reaches relative error `< 10⁻³`.

| Method | Ordering used |
|--------|---------------|
| **KRAB** | its surviving basis states, in the order it discovered them (`final_indices`) |
| **SKQD** | the sampling order for each time step `t`; the **best** (fewest states) over the sweep is reported |
| **Random** | random orderings starting from the same initial state |

KRAB "wins" when it reaches the target with fewer states than the best SKQD
time step.

## What each job produces

For one combination `(n_qubits, gs_sparsity, ham_sparsity, overlap, Q, epsilon)`:

- **One combined figure** in `figures/`:
  - top: the **convergence-path comparison** (the BARK-vs-SKQD style plot) —
    random paths in the background, the SKQD time-step sweep, the KRAB
    discovery-order path, and markers where KRAB actually diagonalized;
  - bottom: the **KRAB internal diagnostics** (energy, relative error, subspace
    size, bitstrings added vs Q, residual reach, state sparsity).
- **One row** appended to `data/comparison.csv` with comparative metrics
  (states-to-target for KRAB / best-SKQD / random, convergence flags, subspace
  sizes, wall times, win flag, advantage ratio).

## Hyperparameters scanned

| Parameter | Values | Role |
|-----------|--------|------|
| `n_qubits` | 6, 8, 10 | System size; Hilbert-space dim = 2ⁿ |
| `gs_sparsity` | 0.05, 0.1, 0.25 | Ground-state support fraction |
| `ham_sparsity` | 0.1, 0.25, 0.5 | Hamiltonian off-diagonal density |
| `overlap` | 0.1, 0.3, 0.5, 0.8 | Initial-state overlap with the ground state |
| `Q` | 1%, 3%, 5%, 10% | KRAB: new basis states added per iteration, **as a fraction of 2ⁿ** (resolved to `round(Q·2ⁿ)` per system) |
| `epsilon` | 10⁻⁷, 10⁻⁶, 10⁻⁵, 10⁻⁴ | KRAB: coefficient pruning threshold |

Fixed: `delta=1e-8`, `ground_energy=-5`, `gap=1`,
SKQD time-step sweep `[0.001 … 1.0]` (12 values), `n_random_paths=20`.
KRAB iteration cap = `max(30, round(3/Q))` (so small Q gets more rounds:
1% → 300, 3% → 100, 5% → 60, 10% → 30).

**Total experiments (full grid):** 3 × 3 × 3 × 4 × 4 × 4 = **1728**

## Directory layout

```
KRAB/
├── run_experiment.py   # Single-job: KRAB + SKQD + random, figure + CSV row
├── run_chunk.py        # Runs a slice of param_grid.txt sequentially
├── generate_jobs.py    # Writes params/param_grid.txt
├── run_array.slurm     # Slurm array job (30 tasks)
├── plot_scaling.py     # Aggregates comparison.csv -> analysis/*.png
├── params/param_grid.txt
├── figures/            # Per-experiment comparison + diagnostics PNGs
├── data/comparison.csv # Accumulated during the cluster run
├── analysis/           # Output of plot_scaling.py
└── SlurmOut/           # Slurm logs
```

> The legacy KRAB-only artifacts (`results/`, `data/results.csv`) from the
> earlier non-comparative run are left untouched for reference.

## Workflow

```bash
cd ClusterStudy/KRAB
source /home/erosanow_hpc/informed_swapping/.SKQD/bin/activate

# 1. parameter file
python generate_jobs.py --preset full      # 1728 jobs

# 2. submit (edit --account / paths in run_array.slurm first)
sbatch run_array.slurm                      # 30 array tasks

# 3. analysis once jobs finish
python plot_scaling.py                      # data/comparison.csv -> analysis/
```

Jobs are spread across the 30 array tasks by **cost-balanced (greedy LPT)**
assignment, not contiguous slicing: the expensive experiments (large `n_qubits`
and large `Q`) are dealt out across distinct chunks so every task takes roughly
the same wall time. The assignment is deterministic, so re-submitting is safe —
experiments whose figure already exists in `figures/` are skipped.

## Analysis plots (`analysis/`)

| File | Question | What it shows |
|------|----------|---------------|
| `head_to_head.png` | Q1 | KRAB vs SKQD states-to-target scatter + win/loss summary |
| `win_rate_by_param.png` | Q1 | fraction of cases KRAB wins, per hyperparameter |
| `convergence_rate.png` | Q2 | KRAB vs SKQD reach-target rate per hyperparameter |
| `region_heatmaps.png` | Q2 | win-rate & KRAB reach-rate over parameter pairs |
| `cost_scaling.png` | Q3 | states-to-target vs n_qubits (KRAB / SKQD / random) |
| `subspace_scaling.png` | Q3 | KRAB subspace size vs n_qubits vs full dim 2ⁿ |
| `walltime_scaling.png` | Q3 | KRAB vs SKQD wall time vs n_qubits |
| `krab_hyperparams.png` | — | KRAB reach-rate & cost vs Q and ε |

## Running a single experiment locally

```bash
python run_experiment.py \
    --n_qubits 8 --gs_sparsity 0.1 --ham_sparsity 0.25 \
    --overlap 0.3 --Q_frac 0.03 --epsilon 1e-6 \
    --n_random_paths 20 --seed 42 \
    --figure_dir figures --data_dir data
```

## Runtime note

Each experiment now runs the **full SKQD time-step sweep** plus a random
baseline, both reconstructed with `get_one_path` (one `eigsh` per added state).
This is the dominant cost and grows with `2ⁿ`: seconds at n=6, tens of seconds
at n=8, minutes at n=10. Reduce `--n_random_paths` to speed things up.

## Dependencies

```
numpy scipy matplotlib pandas tqdm
```
(virtual environment: `../../.SKQD`)
