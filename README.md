# informed_swapping

This repository investigates whether quantum-classical hybrid algorithms based on
**SKQD** (Subspace-based Krylov Quantum Diagonalization) can be "debunked" —
i.e. whether their advantage can be reproduced or surpassed by purely classical
heuristics.  The central idea is **informed swapping**: using information already
available (structure of the Hamiltonian, overlap with the initial state) to build
a Krylov-like basis without running a quantum circuit.

## Core algorithms

| Algorithm | File | Description |
|-----------|------|-------------|
| **BARK** | `BARK.py` | *Bitstring Algorithm for Recursive Krylov* — a purely classical analogue of SKQD that works directly on computational-basis bitstrings via Pauli-string propagation. |
| **SKQD** (reference) | `SystematicScanOfConvergence/SKQD.py`, `Permutations/SKQD.py` | Classical simulation of SKQD: time-evolves the state with `U = exp(-iHt)` and samples the highest-amplitude basis states at each step to build the subspace. |
| **JBARK** | `SystematicScanOfConvergence/JBARK.py` | Simpler matrix-level BARK variant: applies `H` directly (no Pauli decomposition), picks the top-amplitude states at each step, projects, and fills missing states at the end. |

## Key ideas

* **Krylov subspace construction** — both BARK and SKQD grow a subspace of
  computational-basis states iteratively; the hope is that the ground-state
  support is covered early.
* **Stopping time** — the primary figure of merit: how many basis states must
  be sampled before the full support of the ground state is covered?
* **Ground-state sparsity vs Hamiltonian sparsity** — two independent knobs
  that control how hard the problem is. These are studied systematically in
  `SystematicScanOfConvergence/` and `new_approach/`.

## Repository layout

```
informed_swapping/
├── BARK.py                          # Root-level BARK (Pauli/Qiskit version)
├── schwingermodel.py                # Schwinger model Hamiltonian builder
├── Debug.py                         # Ad-hoc debugging helpers
├── TestSchwinger.py                 # Quick tests on Schwinger model
├── compare_bark_skqd*.ipynb         # Head-to-head BARK vs SKQD comparisons
├── informed_swapping.ipynb          # Original exploration notebook
├── hamiltonian_from_paper.ipynb     # Reproduces Hamiltonians from the SKQD paper
├── random_perms*.ipynb              # Random-permutation baselines
├── straight_forward.ipynb           # Straightforward baseline approach
├── test_bark_skqd_new_ham.ipynb     # Tests with new Hamiltonian generator
│
├── SystematicScanOfConvergence/     # Grid scan of (sparsity, overlap, qubits)
├── Permutations/                    # SKQD ordering / permutation study
├── UnitaryVsPower/                  # Unitary vs power-iteration comparison
└── new_approach/                    # Hamiltonian generator with controlled sparsity
```

## Dependencies

```
pip install numpy scipy qiskit matplotlib pandas tqdm
```

The `.BARK/` directory is a local Python virtual environment — activate it with:
```bash
source .BARK/bin/activate
```

## Quick start

Open any of the `compare_bark_skqd*.ipynb` notebooks in Jupyter to see BARK and
SKQD running side-by-side on the same Hamiltonian.

For a systematic numerical experiment see `SystematicScanOfConvergence/` and its
`GridScan.py`.
