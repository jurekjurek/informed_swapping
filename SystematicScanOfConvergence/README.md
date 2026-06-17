# SystematicScanOfConvergence

This folder performs a **parameter grid scan** to study how the convergence of
BARK depends on three axes simultaneously:

- **Number of qubits** (Hilbert-space dimension = 2^n)
- **Ground-state sparsity** (fraction of computational-basis states in the
  ground-state support)
- **Initial-state overlap** with the true ground state

## Files

| File | Purpose |
|------|---------|
| `MakeHam.py` | Generates random Hermitian Hamiltonians with a *known*, *controlled* sparse ground state and a guaranteed spectral gap. |
| `JBARK.py` | Simplified matrix-level BARK: applies H iteratively, keeps the top-amplitude states (`keepstates` per step), projects the cumulative subspace, and fills remaining states at the end. |
| `SKQD.py` | Classical SKQD reference: computes `U = exp(-iHt)`, evolves the state, samples proportional to `|ψ|²`, and builds the subspace over `num_steps` steps. |
| `GridScan.py` | Orchestrates the full parameter sweep. Constructs Hamiltonians, runs BARK for each parameter combination, and records stopping times into a `pandas.DataFrame`. |
| `Debug.ipynb` | Interactive notebook for debugging and quick exploration of individual runs. |
| `dog.png` | Image asset (used in print statements for fun). |
| `interesting_thing.pdf` | Output plot saved from a previous run. |

## How it works

### Hamiltonian generation (`MakeHam.py`)

`make_sparse_ground_state_hamiltonian_from_qubits(n_qubits, ground_state_sparsity, seed, ...)`

Builds `H = E₀|ψ⟩⟨ψ| + excited_part` where:
- `|ψ⟩` has exactly `k` non-zero computational-basis amplitudes
- All other eigenvalues are ≥ `E₀ + gap`

This gives full control over what the ground state looks like.

### JBARK (`JBARK.py`)

```python
bark = BARK(H_sparse, initial_state=idx, keepstates=k)
bark.run()
# bark.samples  -- list of lists; bark.samples[t] = cumulative set of indices seen by step t
```

Each call to `step()`:
1. Applies `H` to the current state vector.
2. Looks back through *all* stored amplitude snapshots and picks the top
   `keepstates` indices not yet in the subspace.
3. The new state is projected onto those indices and renormalized.

### GridScan (`GridScan.py`)

```python
scan = GridScan(
    n_qubits=[4, 5, 6],
    sparsity_values=[0.1, 0.25, 0.5],
    overlaps=[0.01, 0.1, 0.5],
    seeds=[0, 1, 2, 3, 4],
    keepstates=[1, 5],
)
df = scan.run()
scan.plot_mean_stopping_time_vs_sparsity()
scan.plot_keepstates_effect(n_qubits=5, wanted_overlap=0.1)
```

The key metric recorded is **stopping time**: the first step at which all
ground-state support states have appeared in the sampled subspace.

### SKQD reference (`SKQD.py`)

```python
from SKQD import SKQD
model = SKQD(H_sparse, num_steps=100, t=0.1, initial_state=idx)
model.run()
# model.samples  -- same format as BARK.samples
```

## Running on a cluster

The `JBARK.py` + `GridScan.py` stack is self-contained and has no cluster-specific
code. For large sweeps, serialise the `GridScan` call inside a Slurm script
(see `../Permutations/run_8qubits.slurm` for an example).

## Dependencies

```
numpy scipy pandas matplotlib tqdm
```

Run from the parent directory (or from within the `SystematicScanOfConvergence/`
directory after adding `.` to `PYTHONPATH`) so that the relative imports
(`from MakeHam import ...`, `from JBARK import ...`) resolve correctly.
