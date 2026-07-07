"""Hamiltonian generators with controlled sparsity.

Exposes Hamiltonian generators with controlled sparse planted ground states.
"""

from .controlled_sparsity import (
    make_controlled_sparse_ground_state_hamiltonian_from_qubits,
    make_controlled_sparse_ground_state_hamiltonian_fast,
)
from .new_hamiltonian_approach import diagnostics, make_planted_hamiltonian

__all__ = [
    "diagnostics",
    "make_planted_hamiltonian",
    "make_controlled_sparse_ground_state_hamiltonian_from_qubits",
    "make_controlled_sparse_ground_state_hamiltonian_fast",
]
