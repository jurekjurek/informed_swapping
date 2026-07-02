"""Hamiltonian generators with controlled sparsity.

Currently exposes the two generators from :mod:`controlled_sparsity`, which
build a computational-basis Hamiltonian whose exact ground state has a
*controlled* number of nonzero amplitudes and whose off-diagonal fill can be
tuned independently.
"""

from .controlled_sparsity import (
    make_controlled_sparse_ground_state_hamiltonian_from_qubits,
    make_controlled_sparse_ground_state_hamiltonian_fast,
)

__all__ = [
    "make_controlled_sparse_ground_state_hamiltonian_from_qubits",
    "make_controlled_sparse_ground_state_hamiltonian_fast",
]
