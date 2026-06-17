# UnitaryVsPower

This folder compares two different operators for generating the Krylov subspace
used in quantum diagonalization:

1. **Unitary evolution** — `U = exp(-iHt)`, as used in SKQD.
2. **Power / inverse iteration** — repeated application of `H⁻¹` (or a
   polynomial of `H`), which amplifies the ground-state component instead of
   rotating uniformly.

The comparison is done both in terms of convergence rate and in terms of
practical cost (how many matrix-vector products are needed).

## Files

| File | Purpose |
|------|---------|
| `Comparison.py` | Core module: Hamiltonian generators, unitary/power operator builders, Lanczos and Arnoldi ground-energy estimators. |
| `ComparisonWithSchwinger.py` | Same comparison applied to the Schwinger model (from `../schwingermodel.py`). |
| `Test.ipynb` | Interactive notebook driving the comparison experiment. |

## Key functions in `Comparison.py`

### Hamiltonian generators
- `make_hermitian_sparse_random(size, density, seed)` — random Hermitian sparse matrix.
- `make_random_state(size, seed)` — random normalized initial state.

### Operators
- `make_unitary_operator(H, dt)` — matrix-free `U = exp(-iHt)` via
  `scipy.sparse.linalg.expm_multiply`.
- `make_cooked_unitary_operator(H, dt)` — modified unitary using
  `exp(-i(-H + ½H²)dt)` as an alternative ansatz.

### Krylov estimators
- `lanczos_ground_energy(H, initial_state, max_iter, ...)` — standard Lanczos
  iteration (with optional reorthogonalization) applied directly to `H`.
  Returns ground energy estimates at each step.
- `arnoldi_ground_energy(H, U, initial_state, max_iter, ...)` — Arnoldi
  iteration using operator `U` (unitary or power), with Ritz energies extracted
  from the small projected Hamiltonian `Q*HQ`.

## Conceptual difference

| | Lanczos (H) | Arnoldi (U) |
|--|-------------|-------------|
| Basis generator | `H` itself | `U = exp(-iHt)` or `H⁻¹` |
| Basis orthogonality | Short 3-term recurrence | Full Gram-Schmidt |
| Ground-state amplification | Directly via Krylov | Via spectral mapping |
| SKQD connection | Classical limit | Direct analogue |

The Arnoldi approach with a unitary `U` mirrors what happens on a quantum
computer in SKQD: the circuit generates `U|ψ⟩, U²|ψ⟩, …` and measurements
project onto computational-basis states.

## How to use

```python
from Comparison import (
    make_hermitian_sparse_random,
    make_random_state,
    make_unitary_operator,
    lanczos_ground_energy,
    arnoldi_ground_energy,
)

H = make_hermitian_sparse_random(size=256, density=0.05, seed=42)
psi0 = make_random_state(size=256, seed=0)
U = make_unitary_operator(H, dt=0.1)

# Classical Lanczos (ground truth)
energies_lanczos, iters = lanczos_ground_energy(H, psi0, max_iter=50)

# Arnoldi with unitary operator (SKQD-like)
energies_arnoldi, iters = arnoldi_ground_energy(H, U, psi0, max_iter=50)
```

## Dependencies

```
numpy scipy matplotlib
```

The Schwinger comparison additionally requires `../schwingermodel.py`.
