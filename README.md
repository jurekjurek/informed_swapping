# informed_swapping

Research code investigating whether **classical, subspace-selecting heuristics**
can reproduce or beat **SKQD** (Subspace-based Krylov Quantum Diagonalization) at
finding sparse ground states of a Hamiltonian.

The central question: SKQD builds a Krylov-like subspace by time-evolving a state
and sampling high-amplitude computational-basis states. Can we get the same (or
better) convergence *classically* — by using the structure of the Hamiltonian
and the initial-state overlap to choose which basis states to keep? The
dog-named algorithms **KRAB** and **BARK** are the attempts.

> **This repository was restructured on 2026-07-02.** Everything that existed
> before is preserved verbatim in [`backup/`](backup/). The layout below is the
> clean rebuild.

## Layout

```
informed_swapping/
├── subspace_search/      # pip-installable package — the reusable core
│   ├── pyproject.toml
│   ├── README.md
│   └── src/subspace_search/
│       ├── hamiltonians/ # sparse-ground-state generators (controlled-sparsity + planted projector)
│       ├── skqd/         # SKQD reference routine (+ power-iteration sampler)
│       ├── algorithms/   # KRAB, BARK, and space for new algorithms
│       ├── paths.py      # subspace projection + convergence paths
│       └── plotting.py   # Hamiltonian / spectrum / convergence plots
│
├── experiments/          # quick local test & exploration scripts
├── cluster_studies/      # Slurm grid studies of the algorithms
│   ├── KRAB/             # KRAB-vs-SKQD systematic study
│   └── BARK/             # BARK-vs-SKQD systematic study
│
├── docs/
│   └── LOGBOOK.md        # running log of documentation updates
│
├── backup/               # everything from before the 2026-07-02 restructure
├── CLAUDE.md             # repo guide + the "Update Documentation" routine
└── README.md            # this file
```

## Getting started

The project lives in the `.SKQD/` virtual environment. Install the package once,
in editable mode, so every experiment and study picks up code changes
automatically:

```bash
source .SKQD/bin/activate
pip install -e subspace_search
```

Then, from anywhere:

```python
from subspace_search.hamiltonians import make_controlled_sparse_ground_state_hamiltonian_fast
from subspace_search.skqd import do_skqd
from subspace_search.algorithms import selected_krylov_ground_state   # KRAB
from subspace_search.algorithms import BarkBarkBark                   # BARK
from subspace_search.paths import get_one_path, get_all_paths
from subspace_search.plotting import plot_hamiltonian, plot_convergence_paths
```

## Where things are

| I want to... | Go to |
|--------------|-------|
| Understand / import the core routines | [`subspace_search/README.md`](subspace_search/README.md) |
| Add a new algorithm (a new BARK/KRAB) | [`subspace_search/src/subspace_search/algorithms/README.md`](subspace_search/src/subspace_search/algorithms/README.md) |
| Try something quickly / prototype | [`experiments/README.md`](experiments/README.md) |
| Run a systematic study on the cluster | [`cluster_studies/README.md`](cluster_studies/README.md) |
| See what changed and when | [`docs/LOGBOOK.md`](docs/LOGBOOK.md) |
| Find the old code | [`backup/`](backup/) |

## Keeping docs in sync

Type **"Update Documentation"** in a Claude Code session and the assistant will
review the changes since the last logbook entry, update every affected README,
and append a dated entry to [`docs/LOGBOOK.md`](docs/LOGBOOK.md). The routine is
defined in [`CLAUDE.md`](CLAUDE.md).
