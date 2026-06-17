# new_approach

This folder introduces an improved Hamiltonian generator that gives **independent
control over two sparsity parameters**:

1. **Ground-state sparsity** — how many computational-basis states appear in
   the ground state `|ψ⟩`.
2. **Hamiltonian sparsity** — how many off-diagonal matrix elements `H` has.

The earlier `MakeHam.py` (in `SystematicScanOfConvergence/`) only controlled
ground-state sparsity; the Hamiltonian was always effectively dense. This
folder fixes that.

## Files

| File | Purpose |
|------|---------|
| `Hamiltonian_controlled_sparsity.py` | Two generators: `make_controlled_sparse_ground_state_hamiltonian_from_qubits` (exact, slow) and `make_controlled_sparse_ground_state_hamiltonian_fast` (efficient, scales better). |
| `CheckHamiltonian.ipynb` | Notebook that verifies the generators: checks Hermiticity, confirms ground-state energy, checks the spectral gap, and inspects sparsity patterns. |

## Construction method

Both generators use the same algebraic idea:

1. **Choose a sparse ground state `|ψ⟩`** with `k` nonzero computational-basis
   amplitudes (the *support*).

2. **Build a sparse annihilator `A`** such that `A|ψ⟩ = 0`.
   - Support-support edges: rows of the form `ψ[b]|a⟩ − ψ[a]|b⟩`, which
     vanish when applied to `|ψ⟩`.
   - Complement singletons: rows `|c⟩` for each `c` outside the support
     (since `ψ[c] = 0`).
   - Cross edges (support ↔ complement): controlled rows that stay in the
     null space of `ψ`.

3. **Form `K = A†A`** (positive semi-definite, `K|ψ⟩ = 0`).

4. **Set `H = E₀·I + scale·K`** where `scale` is chosen so that the first
   excited eigenvalue is `E₀ + gap`.

The *number* of optional edge types added to `A` (before building `K`) controls
the Hamiltonian sparsity.

### Fast vs exact version

| | `make_controlled_sparse_ground_state_hamiltonian_from_qubits` | `make_controlled_sparse_ground_state_hamiltonian_fast` |
|--|---|---|
| Sparsity control | Iterative: rebuilds K after each candidate | Batch: samples candidates once, builds K once |
| Pauli output | `SparsePauliOp.from_operator` (dense) | Same, but optional (`make_pauli_op=False`) |
| Large-n scaling | Slow | Better (avoids repeated K builds) |

## Usage

```python
from Hamiltonian_controlled_sparsity import (
    make_controlled_sparse_ground_state_hamiltonian_fast,
)

H_csr, H_pauli, psi, support, info = make_controlled_sparse_ground_state_hamiltonian_fast(
    n_qubits=6,
    ground_state_sparsity=0.1,   # 10% of basis states in ground state
    hamiltonian_sparsity=0.3,    # 30% of possible off-diagonal pairs filled
    seed=42,
    ground_energy=0.0,
    gap=1.0,
)

print(info["hamiltonian_density"])      # fraction of nonzero entries
print(info["ground_state_support_size"])  # = k
print(info["actual_gap"])               # should be close to gap=1.0
```

`info` contains full diagnostics including actual spectral gap, number of Pauli
terms, off-diagonal count, etc.

## Verification

Open `CheckHamiltonian.ipynb` and run all cells to:
- Confirm `H @ psi ≈ ground_energy * psi`
- Verify the spectral gap
- Plot the sparsity structure of `H`
- Compare requested vs actual sparsity

## Dependencies

```
numpy scipy qiskit matplotlib
```
