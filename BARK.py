"""Implementation of BARK - Bitstring Algorithm for Recursive Krylov.

This is a purely classical (bitstring) variant of SKQD.
"""

from typing import List, Optional, Sequence, Tuple

import numpy as np
from qiskit.quantum_info import SparsePauliOp
from scipy.sparse import csr_matrix

from dog_ascii import DOG


class BARK:
    def __init__(
        self,
        H: SparsePauliOp,
        initial_state: str,
        max_iterations: int = 10,
        time_step: Optional[float] = None,
        tolerance: Optional[float] = None,
    ) -> None:
        self.print_angry_dog()

        self.H: SparsePauliOp = H
        self.initial_state: str = initial_state
        self.max_iterations: int = max_iterations
        self.time_step: Optional[float] = time_step
        self.tolerance: Optional[float] = tolerance
        self.basis: List[List[str]] = [[initial_state]]

        # Derived / precomputed data
        self.num_qubits: int = len(initial_state)
        self.strings: List[str] = []
        self.coeffs: np.ndarray | List[float] = []
        self._flip_indices: List[List[int]] = []

        self.H_map()
        self.do_time_evolution()

    def H_map(self) -> None:
        """Precompute effective Pauli strings and their sine-coefficients.

        The mapping follows the original implementation:
        I -> I, X -> X, Y -> X, Z -> I.
        """

        pauli_strings: Sequence[Tuple[str, complex]] = self.H.to_list()

        self.strings = []
        self._flip_indices = []
        coeffs: List[float] = []

        mapping = {"I": "I", "X": "X", "Y": "X", "Z": "I"}

        for pauli, coeff in pauli_strings:
            mapped = "".join(mapping[p] for p in pauli)
            self.strings.append(mapped)

            # Precompute bit positions to flip for this term
            flip = [idx for idx, symbol in enumerate(mapped) if symbol == "X"]
            self._flip_indices.append(flip)

            angle = (self.time_step * coeff) if self.time_step is not None else coeff
            coeffs.append(float(np.sin(angle)))

        # Store as numpy array for efficient numerical operations
        self.coeffs = np.asarray(coeffs, dtype=float)

    def apply_time_step(self, bitstring: str, coeff: float) -> List[Tuple[str, float]]:
        """Apply a single time step to one basis state.

        Returns a list of (new_bitstring, new_coeff) pairs.
        """

        new_states: List[Tuple[str, float]] = []

        for flip_indices, op_coeff in zip(self._flip_indices, self.coeffs):
            chars = list(bitstring)
            for idx in flip_indices:
                chars[idx] = "1" if chars[idx] == "0" else "0"
            new_state = "".join(chars)

            new_coeff = coeff * op_coeff
            if self.tolerance is None or abs(new_coeff) > self.tolerance:
                new_states.append((new_state, new_coeff))

        return new_states

    @staticmethod
    def _combine_duplicates(states: Sequence[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """Combine duplicate bitstrings by summing their coefficients."""

        combined: dict[str, float] = {}
        for bitstring, coeff in states:
            combined[bitstring] = combined.get(bitstring, 0.0) + coeff
        return list(combined.items())

    def do_time_evolution(self) -> None:
        current_basis: List[Tuple[str, float]] = []

        for i in range(self.max_iterations):
            basis_length = len(self.basis[-1])

            if i == 0:
                current_basis = [(self.initial_state, 1.0)]

            new_basis: List[Tuple[str, float]] = []
            for bitstring, coeff in current_basis:
                new_states = self.apply_time_step(bitstring, coeff)
                new_basis.extend(new_states)

            # Combine duplicates by adding their coefficients
            current_basis = self._combine_duplicates(new_basis)

            basis_in_this_step = [bitstring for bitstring, _ in current_basis]

            # Add old basis states to new basis
            for previous_basis in self.basis:
                for bitstring in previous_basis:
                    if bitstring not in basis_in_this_step:
                        basis_in_this_step.append(bitstring)

            if len(basis_in_this_step) == basis_length:
                print(f"Converged at iteration {i}")
                break

            self.basis.append(basis_in_this_step)

    def compute_matrix_element(self, bitstring_i: str, bitstring_j: str, Hamiltonian: tuple) -> float:
        """Compute the matrix element <bitstring_i|H|bitstring_j>."""

        element = 0.0
        for string, coeff in Hamiltonian:
            pref = coeff
            for j, (va1, va2) in enumerate(zip(bitstring_i, bitstring_j)):
                if string[j] == "X":
                    if va1 != va2:
                        pref *= 1
                    else:
                        pref *= 0
                elif string[j] == "I":
                    if va1 == va2:
                        pref *= 1
                    else:
                        pref *= 0
                elif string[j] == "Y":
                    if va1 == "1" and va2 == "0":
                        pref *= 1j
                    elif va1 == "0" and va2 == "1":
                        pref *= -1j
                    else:
                        pref *= 0
                elif string[j] == "Z":
                    if va1 == "1" and va2 == "1":
                        pref *= -1
                    elif va1 == "0" and va2 == "0":
                        pref *= 1
                    else:                        
                        pref *= 0
            element += pref
        return element

    def project_to_subspace(self, basis: List[str]) -> np.ndarray:
        """Project the Hamiltonian onto the subspace spanned by the given basis."""

        Hamiltonian = self.H.to_list()
        data = []
        row_indices = []
        col_indices = []

        for i, bitstring_i in enumerate(basis):
            for j, bitstring_j in enumerate(basis):
                data.append(self.compute_matrix_element(bitstring_i, bitstring_j, Hamiltonian))
                row_indices.append(i)
                col_indices.append(j)
        
        return csr_matrix((data, (row_indices, col_indices)), shape=(len(basis), len(basis)))

    def print_angry_dog(self) -> None:
            print(DOG)

