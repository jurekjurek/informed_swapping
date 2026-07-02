# subspace_search

The reusable core of the *informed-swapping* project, packaged so it can be
`pip install`-ed once and imported everywhere (experiments, cluster studies,
notebooks). It bundles the things that are needed all the time:

- **controlled-sparsity Hamiltonian generators**,
- the **SKQD** reference routine,
- the classical search **algorithms** (KRAB, BARK) — and room for new ones,
- **convergence-path** utilities (the common currency for comparisons),
- shared **plotting** functions.

## Install

```bash
source ../.SKQD/bin/activate      # the project venv
pip install -e .                  # editable — code edits take effect immediately
```

Requires Python ≥ 3.10. Dependencies (numpy, scipy, matplotlib, tqdm, qiskit)
are installed automatically; `pip install -e ".[dev]"` adds pytest/jupyter/pandas.

## Package map

```
src/subspace_search/
├── __init__.py            # version + overview docstring
├── hamiltonians/          # controlled-sparsity Hamiltonian generators
│   └── controlled_sparsity.py
├── skqd/                  # SKQD reference routine + power-iteration sampler
│   ├── skqd.py            # do_skqd, get_exponential
│   └── power.py           # do_power
├── algorithms/            # classical subspace-search algorithms (see its README)
│   ├── krab.py            # selected_krylov_ground_state (KRAB) + diagnostics
│   └── bark.py            # BarkBarkBark (BARK)
├── paths.py               # project_down, get_one_path, get_all_paths, get_permutations
└── plotting.py            # plot_hamiltonian, plot_ground_state, plot_convergence_paths
```

## Public API by module

### `subspace_search.hamiltonians`
Build a computational-basis Hamiltonian whose exact ground state has a
*controlled* number of nonzero amplitudes, with independently tunable
off-diagonal fill.

- `make_controlled_sparse_ground_state_hamiltonian_fast(...)` — efficient
  version (samples candidate couplings once, builds `K = AᴴA` once). Use this.
- `make_controlled_sparse_ground_state_hamiltonian_from_qubits(...)` — exact but
  slow reference version (rebuilds `K` after each candidate).

Both return `(H_csr, H_pauli, psi, support, info)`; `H_pauli` is optional
(`make_pauli_op=False`) and `info` carries full diagnostics (actual gap,
densities, support size, ...). Key knobs: `ground_state_sparsity`,
`hamiltonian_sparsity`, `ground_energy`, `gap`, `max_amplitude` (fixes the
dominant basis-state probability = initial-state overlap), `seed`.

### `subspace_search.skqd`
- `do_skqd(H, num_steps, t, initial=None)` — the SKQD reference: evolve with
  `U = exp(-iHt)`, sample high-amplitude basis states each step, return the
  discovery ordering (a length-`dim` array with `-1` sentinels for unfilled
  slots).
- `do_power(H, num_steps, initial=None)` — same sampling loop but using the
  inverse of a shifted `H` (power iteration) instead of a real-time propagator.
- `get_exponential(H, t)` — `expm(-iHt)`.

### `subspace_search.algorithms`
- `selected_krylov_ground_state(...)` → `SelectedKrylovResult` — **KRAB**, with
  per-iteration `Q` budget, candidate/coefficient pruning, optional subspace
  cap, and rich diagnostics.
- `BarkBarkBark` — **BARK**, Pauli-string-propagation classical analogue of SKQD.
- `plot_selected_krylov_diagnostics(result)` — KRAB diagnostics figure.

See [`src/subspace_search/algorithms/README.md`](src/subspace_search/algorithms/README.md)
for details and for how to add a new algorithm.

### `subspace_search.paths`
The **common currency** for comparing methods: every method produces an
*ordering* of basis states, and these turn an ordering into an energy curve.

- `project_down(H, indices)` — restrict `H` to a subspace.
- `get_one_path(H, indices)` — ground-energy estimate as states are added one at
  a time (`path[i]` uses the first `i+1` states).
- `get_all_paths(H, number_of_paths, start)` — random-ordering baseline bundle.
- `get_permutations(n, k, first=None)` — random orderings with a fixed start.

### `subspace_search.plotting`
- `plot_hamiltonian(H, ...)` — sparsity pattern, `|H|` magnitude heatmap, and
  spectrum in one figure.
- `plot_ground_state(psi, ...)` — `|ψᵢ|²` bar plot (optionally top-k states).
- `plot_convergence_paths({label: path}, true_energy=...)` — overlay KRAB / BARK
  / SKQD / random convergence paths on one axis (2-D arrays are drawn as a faint
  bundle, e.g. the random baseline).

## Minimal example

```python
import numpy as np
from subspace_search.hamiltonians import make_controlled_sparse_ground_state_hamiltonian_fast
from subspace_search.skqd import do_skqd
from subspace_search.algorithms import selected_krylov_ground_state
from subspace_search.paths import get_one_path

H, _, psi, support, info = make_controlled_sparse_ground_state_hamiltonian_fast(
    n_qubits=6, ground_state_sparsity=0.1, hamiltonian_sparsity=0.3,
    seed=42, ground_energy=-5.0, gap=1.0, make_pauli_op=False, max_amplitude=0.3,
)
init = int(np.argmax(np.abs(psi) ** 2))

order = do_skqd(H, num_steps=8, t=0.5, initial=init)
skqd_path = get_one_path(H, [int(i) for i in order if i >= 0])

krab = selected_krylov_ground_state(
    H=H, initial_bitstring=init, Q=8, delta=1e-8, epsilon=1e-6, n_iterations=30,
)
print("SKQD ->", skqd_path[-1], " KRAB ->", krab.final_energy)
```
