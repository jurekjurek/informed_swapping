# Permutations

This folder studies how the **ordering** (permutation) in which basis states are
added to the Krylov subspace affects convergence of ground-state energy estimation.
The central question: given that SKQD/BARK produce a sequence of sampled indices,
does the order matter — and if so, how much?

## Files

| File | Purpose |
|------|---------|
| `MakeHam.py` | Identical to `SystematicScanOfConvergence/MakeHam.py` — random Hermitian Hamiltonians with controlled sparse ground states. |
| `SKQD.py` | Classical SKQD: time-evolves with `U = exp(-iHt)`, samples proportional to `|ψ|²`. |
| `PowerSampling.py` | Alternative sampler: uses **inverse power iteration** (`U = H⁻¹`) instead of unitary time evolution, which amplifies the ground state differently. |
| `BARK.py` | BARK variant using Qiskit `SparsePauliOp` with an optional `keep_states` truncation and 75th-percentile coefficient pruning to speed up the sweep. |
| `BARKlouder.py` | Extended/noisier BARK variant (louder print output). |
| `Bark_2_0.py` | Updated BARK version (second generation). |
| `Helpers.py` | Utility functions: `project_down`, `get_permutations`, `get_one_path`, `get_all_paths`. |
| `CSRtoSPO.py` | Converts a Hermitian `scipy.sparse.csr_matrix` to a Qiskit `SparsePauliOp` via explicit Pauli decomposition. |
| `Test_8qubits.py` | Script that runs the permutation experiment for 8 qubits and saves results. |
| `compare_random_barkbarkbark.ipynb` | Notebook comparing random-order baselines against BARK-guided ordering. |
| `Test.ipynb` | Interactive test notebook. |
| `run_8qubits.slurm` | Slurm job script for running `Test_8qubits.py` on a compute cluster. |
| `SlurmOut/` | Directory holding Slurm stdout/stderr logs from cluster runs. |
| `*.pdf` | Result plots: `SparseAndOverlap`, `SparseNoOverlap`, `NonSparseButOverlap`, `NonSparseNoOverlap`. |
| `dog_ascii.py` | ASCII art of a dog (used in BARK print statements). |

## Core utility: `Helpers.py`

### `get_permutations(n, k, first=None)`
Generates `k` random full permutations of `{0, …, n-1}`, optionally fixing the
first element. Used to sample random orderings as baselines.

### `get_one_path(H, indices) -> np.ndarray`
Given an ordered list of indices, computes the **ground-state energy as a
function of subspace size** by projecting `H` onto `{indices[0], …, indices[i]}`
at each step `i` and computing the lowest eigenvalue. Returns an array of length
`len(indices)`.

### `get_all_paths(H, number_of_paths, start) -> np.ndarray`
Calls `get_one_path` over many random permutations (all starting from `start`)
and returns a 2D array of shape `(number_of_paths, n)`.

## Scripts

### `Test_8qubits.py`
Runs the full permutation experiment for an 8-qubit Hamiltonian:
1. Generates `H` via `MakeHam.py`.
2. Runs SKQD/BARK to get their proposed ordering.
3. Samples many random orderings via `Helpers.get_all_paths`.
4. Compares convergence curves.

### `run_8qubits.slurm`
Submits `Test_8qubits.py` to a Slurm cluster:
```bash
sbatch run_8qubits.slurm
```
Output files go to `SlurmOut/`.

## Sampling methods compared

| Method | Update rule | Key parameter |
|--------|------------|---------------|
| SKQD (`SKQD.py`) | `ψ ← U ψ`, sample ∝ `\|ψ\|²` | time step `t` |
| Power sampling (`PowerSampling.py`) | `ψ ← H⁻¹ ψ`, normalize | num_steps |
| BARK (`BARK.py`) | Pauli-string propagation on bitstrings | `keep_states`, `tolerance` |
| Random baseline | Uniform random permutation | — |

## PDF outputs

The four PDFs correspond to the four quadrants of the (sparsity × overlap) plane:

| File | Condition |
|------|-----------|
| `SparseAndOverlap.pdf` | Sparse ground state, large initial overlap |
| `SparseNoOverlap.pdf` | Sparse ground state, small initial overlap |
| `NonSparseButOverlap.pdf` | Dense ground state, large initial overlap |
| `NonSparseNoOverlap.pdf` | Dense ground state, small initial overlap |

## Dependencies

```
numpy scipy qiskit matplotlib tqdm
```

Run `Test_8qubits.py` from within this directory:
```bash
python Test_8qubits.py
```
