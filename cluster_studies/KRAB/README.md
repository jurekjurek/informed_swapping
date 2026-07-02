# cluster_studies / KRAB — KRAB vs SKQD

Systematic cluster study comparing the **KRAB** (selected-subspace Krylov)
algorithm — `subspace_search.algorithms.selected_krylov_ground_state` — against
**SKQD** (and a random baseline), over a 6-dimensional hyperparameter grid.

The study answers three questions:

1. **Does KRAB beat SKQD?** — for every Hamiltonian, how many sampled basis
   states does each method need to reach relative energy error `< 10⁻³`?
2. **Where does it work / not work?** — in which parameter regions does KRAB
   reach the target, and where does it stall or lose to SKQD?
3. **How do the resources scale?** — how do the required subspace size,
   number of sampled states, and wall time grow with system size and the
   hyperparameters?

## Setup

```bash
source ../../.SKQD/bin/activate
pip install -e ../../subspace_search      # scripts import from the package
```

Outputs (`figures/`, `figures_decay/`, `data/`, `analysis/`, `analysis_decay/`)
are generated when you run the study. The original outputs from before the
2026-07-02 restructure are preserved in `../../backup/ClusterStudy/KRAB/`.

## Common currency: states-to-target

Every method produces an **ordering** of basis states;
`subspace_search.paths.get_one_path` then gives the ground-energy estimate as
states are added one at a time. The metric `n_states_to_target` is the first
count of basis states at which the estimate reaches relative error `< 10⁻³`.

| Method | Ordering used |
|--------|---------------|
| **KRAB** | its surviving basis states, in discovery order (`result.final_indices`) |
| **SKQD** | the sampling order for each time step `t`; the **best** (fewest states) over the sweep is reported |
| **Random** | random orderings starting from the same initial state |

KRAB "wins" when it reaches the target with fewer states than the best SKQD
time step.

## What each job produces

For one combination `(n_qubits, gs_sparsity, ham_sparsity, overlap, Q, epsilon)`:

- **One combined figure** in `figures/`:
  - top: the **convergence-path comparison** — random paths in the background,
    the SKQD time-step sweep, the KRAB discovery-order path, and markers where
    KRAB actually diagonalized;
  - bottom: the **KRAB internal diagnostics** (energy, relative error, subspace
    size, bitstrings added vs Q, residual reach, state sparsity).
- **One row** appended to `data/comparison.csv` (states-to-target for
  KRAB/best-SKQD/random, convergence flags, subspace sizes, wall times, win
  flag, advantage ratio).

## Hyperparameters scanned

| Parameter | Values | Role |
|-----------|--------|------|
| `n_qubits` | 6, 8, 10 | System size; Hilbert-space dim = 2ⁿ |
| `gs_sparsity` | 0.05, 0.1, 0.25 | Ground-state support fraction |
| `ham_sparsity` | 0.1, 0.25, 0.5 | Hamiltonian off-diagonal density |
| `overlap` | 0.1, 0.3, 0.5, 0.8 | Initial-state overlap with the ground state |
| `Q` | 1%, 3%, 5%, 10% | KRAB: new basis states per iteration, **as a fraction of 2ⁿ** (resolved to `round(Q·2ⁿ)`) |
| `epsilon` | 10⁻⁷, 10⁻⁶, 10⁻⁵, 10⁻⁴ | KRAB: coefficient pruning threshold |

Fixed: `delta=1e-8`, `ground_energy=-5`, `gap=1`, SKQD sweep `[0.001 … 1.0]`
(12 values), `n_random_paths=20`. KRAB iteration cap = `max(30, round(3/Q))`
(1% → 300, 3% → 100, 5% → 60, 10% → 30).

**Total experiments (full grid):** 3 × 3 × 3 × 4 × 4 × 4 = **1728**

## Second study: decaying Q (`--Q_mode decay`)

A parallel study replaces KRAB's **constant** `Q` with a **geometric decaying
schedule** — start wide, narrow as the subspace fills:

```
Q(app) = max( 1% · 2ⁿ ,  int( 20% · 2ⁿ · 0.75**app ) )
```

KRAB accepts a callable `Q` schedule
(`subspace_search.algorithms.krab.resolve_Q`), so the experiment just passes this
lambda. Schedule constants live in `run_experiment.py` (`DECAY_Q_START_FRAC=0.20`,
`DECAY_Q_FLOOR_FRAC=0.01`, `DECAY_Q_FACTOR=0.75`). Because `Q` is now fixed, its
axis collapses: **3 × 3 × 3 × 4 × 4 = 432** experiments.

The decay study is fully separate (own param file, figures, CSV, analysis dir):

| Artifact | Fixed-Q study | Decaying-Q study |
|----------|---------------|------------------|
| param file | `params/param_grid.txt` | `params/param_grid_decay.txt` |
| figures | `figures/` (`…_Qf0.030_…`) | `figures_decay/` (`…_Qdecay_…`) |
| CSV | `data/comparison.csv` | `data/comparison_decay.csv` |
| Slurm | `run_array.slurm` | `run_array_decay.slurm` |
| analysis | `analysis/` | `analysis_decay/` |

### Decay-study workflow

```bash
python generate_jobs.py --preset full --mode decay   # -> params/param_grid_decay.txt (432)
sbatch run_array_decay.slurm                          # 30 array tasks, --Q_mode decay
python plot_scaling.py --data_file comparison_decay.csv --out_dir analysis_decay
```

## Directory layout

```
KRAB/
├── run_experiment.py     # one job: KRAB + SKQD + random, figure + CSV row (--Q_mode fixed|decay)
├── run_chunk.py          # a slice of a param file, sequentially (--Q_mode fixed|decay)
├── generate_jobs.py      # write the param file (--mode fixed|decay)
├── run_array.slurm       # Slurm array job (30 tasks), fixed-Q study
├── run_array_decay.slurm # Slurm array job (30 tasks), decaying-Q study
├── plot_scaling.py       # aggregate a comparison CSV -> an analysis dir
├── params/param_grid.txt        # fixed-Q grid
├── params/param_grid_decay.txt  # decaying-Q grid
└── SlurmOut/             # Slurm logs
```

## Workflow

```bash
python generate_jobs.py --preset full      # 1728 jobs -> params/param_grid.txt
sbatch run_array.slurm                      # 30 array tasks (edit --account/paths first)
python plot_scaling.py                      # data/comparison.csv -> analysis/
```

Jobs are spread across the 30 array tasks by **cost-balanced (greedy LPT)**
assignment, not contiguous slicing, so every task takes roughly the same wall
time. The assignment is deterministic and re-submitting is safe — experiments
whose figure already exists in `figures/` are skipped.

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

Each experiment runs the **full SKQD time-step sweep** plus a random baseline,
both reconstructed with `get_one_path` (one `eigsh` per added state). This
dominates the cost and grows with `2ⁿ`: seconds at n=6, tens of seconds at n=8,
minutes at n=10. Reduce `--n_random_paths` to speed things up.
