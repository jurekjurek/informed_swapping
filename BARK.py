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
        even_numbers: Optional[bool] = None,
    ) -> None:
        self.print_angry_dog()

        self.H: SparsePauliOp = H
        self.initial_state: str = initial_state
        self.max_iterations: int = max_iterations
        self.time_step: Optional[float] = time_step
        self.tolerance: Optional[float] = tolerance
        self.basis: List[List[str]] = [[initial_state]]
        self.even_numbers: Optional[bool] = even_numbers

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
                if self.even_numbers is not None:
                    num_ones = new_state.count("1")
                    if num_ones == len(new_state)//2:
                        new_states.append((new_state, new_coeff))
                else:
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
        """Project the Hamiltonian onto the subspace spanned by ``basis``.

        This implementation avoids the explicit double loop over all basis
        pairs (i, j). Instead, for each basis vector |j> and each Pauli term
        in the Hamiltonian, it applies the Pauli string to |j>, obtains the
        resulting bitstring |i>, and, if |i> is also in the basis, updates the
        corresponding matrix entry H_ij. This reduces the complexity from
        O(|basis|^2 * n_terms) to roughly O(|basis| * n_terms).
        """

        # Map basis bitstrings to their indices for O(1) lookups
        basis_index: dict[str, int] = {bitstring: idx for idx, bitstring in enumerate(basis)}

        # (pauli_string, coefficient) pairs from the original Hamiltonian
        h_terms = self.H.to_list()

        data: list[complex] = []
        row_indices: list[int] = []
        col_indices: list[int] = []

        for j, ket in enumerate(basis):
            for pauli, coeff in h_terms:
                # Apply the Pauli string to |ket>
                bits = list(ket)
                amplitude = complex(coeff)

                for q, op in enumerate(pauli):
                    b = bits[q]
                    if op == "I":
                        continue
                    elif op == "X":
                        # X flips the bit
                        bits[q] = "1" if b == "0" else "0"
                    elif op == "Y":
                        # Y flips the bit and introduces a phase
                        if b == "0":
                            bits[q] = "1"
                            amplitude *= 1j
                        else:  # b == "1"
                            bits[q] = "0"
                            amplitude *= -1j
                    elif op == "Z":
                        # Z leaves the bit but may add a sign
                        if b == "1":
                            amplitude *= -1

                bra_state = "".join(bits)
                i = basis_index.get(bra_state)
                if i is not None and amplitude != 0:
                    row_indices.append(i)
                    col_indices.append(j)
                    data.append(amplitude)

        size = len(basis)
        return csr_matrix((data, (row_indices, col_indices)), shape=(size, size))

    def print_angry_dog(self) -> None:
            print(DOG)

