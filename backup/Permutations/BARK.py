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
        keep_states: Optional[int] = None,
    ) -> None:
        # self.print_angry_dog()

        self.H: SparsePauliOp = H
        self.initial_state: str = initial_state
        self.max_iterations: int = max_iterations
        self.time_step: Optional[float] = time_step
        self.tolerance: Optional[float] = tolerance
        self.basis: List[List[str]] = [[initial_state]]
        self.even_numbers: Optional[bool] = even_numbers
        self.keep_states: Optional[int] = keep_states

        # Derived / precomputed data
        self.num_qubits: int = len(initial_state)
        self.strings: List[str] = []
        self.coeffs: np.ndarray | List[float] = []
        self._flip_indices: List[List[int]] = []

        self.H_map()
        # Cut 75% of the terms with the smallest coefficients to speed up the time evolution
        cut_value = np.percentile(np.abs(self.coeffs), 75)
        self.reduce_flip_indices(tol=cut_value)
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
            #Check if the flip is not empty, otherwise we will have a zero coefficient and can skip it
            if flip:
                if flip not in self._flip_indices:
                    self._flip_indices.append(flip)

                    angle = (self.time_step * coeff) if self.time_step is not None else coeff
                    coeffs.append(float(np.sin(angle)))
                else:
                    # If the flip pattern already exists, we need to add the new coefficient to the existing one
                    idx = self._flip_indices.index(flip)
                    angle = (self.time_step * coeff) if self.time_step is not None else coeff
                    coeffs[idx] += float(np.sin(angle))

        # Store as numpy array for efficient numerical operations
        self.coeffs = np.asarray(coeffs, dtype=float)

    def reduce_flip_indices(self, tol: float) -> None:
        """Remove flip indices with coefficients below the tolerance."""

        mask = np.abs(self.coeffs) > tol
        self._flip_indices = [flip for flip, keep in zip(self._flip_indices, mask) if keep]
        self.coeffs = self.coeffs[mask]

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
                if self.even_numbers is True:
                    num_ones = new_state.count("1")
                    if num_ones == len(new_state)//2:
                        new_states.append((new_state, new_coeff))
                else:
                    new_states.append((new_state, new_coeff))

        if self.keep_states is not None:
            new_states.sort(key=lambda x: abs(x[1]), reverse=True)
            new_states = new_states[:self.keep_states]

        return new_states

    @staticmethod
    def _combine_duplicates(states: Sequence[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """Combine duplicate bitstrings by summing their coefficients."""

        combined: dict[str, float] = {}
        for bitstring, coeff in states:
            combined[bitstring] = combined.get(bitstring, 0.0) + coeff
        return list(combined.items())

    def do_time_evolution(self) -> None:
        self.basis = [[self.initial_state]]
        current_basis: List[Tuple[str, float]] = [(self.initial_state, 1.0)]

        for i in range(self.max_iterations):
            basis_length = len(self.basis[-1])

            new_basis: List[Tuple[str, float]] = []
            for bitstring, coeff in current_basis:
                new_states = self.apply_time_step(bitstring, coeff)
                new_basis.extend(new_states)

            # Combine duplicates by adding their coefficients
            current_basis = self._combine_duplicates(new_basis)

            basis_in_this_step = [bitstring for bitstring, _ in current_basis]

            # Add old basis states to new basis
            for bitstring in self.basis[-1]:
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

    def project_to_subspace(self, basis: List[str]) -> csr_matrix:
        """Project the Hamiltonian onto the subspace spanned by ``basis``.

        Fast version using integer bitmasks instead of string/list operations
        in the inner loop.

        For each Pauli term and basis ket |j>:
        - X and Y flip bits
        - Z and Y contribute a phase/sign

        The matrix element is computed by:
        bra = ket ^ flip_mask

        and the phase is:
        coeff
        * (-1)^(popcount(ket & z_mask))
        * i^(popcount(y_mask))
        * (-1)^(popcount(ket & y_mask))

        Returns:
            csr_matrix: Hamiltonian projected into the given basis.
        """
        size = len(basis)
        if size == 0:
            return csr_matrix((0, 0), dtype=complex)

        # Convert basis strings to integers once
        basis_ints = [int(b, 2) for b in basis]
        basis_index = {b: i for i, b in enumerate(basis_ints)}

        # Precompute masks for each Pauli term
        # Bit convention:
        # pauli[q] acts on the same position as basis string character q
        # so the leftmost character corresponds to the highest-order bit.
        proj_terms: list[tuple[int, int, int, complex]] = []
        for pauli, coeff in self.H.to_list():
            flip_mask = 0  # qubits flipped by X or Y
            z_mask = 0     # qubits contributing a -1 phase when ket bit is 1
            y_mask = 0     # qubits with Y

            for q, op in enumerate(pauli):
                bit = 1 << (self.num_qubits - 1 - q)

                if op == "X":
                    flip_mask |= bit
                elif op == "Y":
                    flip_mask |= bit
                    y_mask |= bit
                elif op == "Z":
                    z_mask |= bit

            proj_terms.append((flip_mask, z_mask, y_mask, complex(coeff)))

        data: list[complex] = []
        row_indices: list[int] = []
        col_indices: list[int] = []

        for j, ket in enumerate(basis_ints):
            for flip_mask, z_mask, y_mask, coeff in proj_terms:
                bra = ket ^ flip_mask
                i = basis_index.get(bra)
                if i is None:
                    continue

                amp = coeff

                # Z contributes (-1) for each occupied qubit under Z
                if (ket & z_mask).bit_count() & 1:
                    amp = -amp

                # Y contributes:
                #   Y|0> = i|1>
                #   Y|1> = -i|0>
                #
                # For all Y positions together:
                #   i^(n_y) * (-1)^(number of Y positions where ket bit = 1)
                n_y = y_mask.bit_count()
                if n_y:
                    amp *= (1j) ** n_y
                    if (ket & y_mask).bit_count() & 1:
                        amp = -amp

                if amp != 0:
                    row_indices.append(i)
                    col_indices.append(j)
                    data.append(amp)

        mat = csr_matrix((data, (row_indices, col_indices)), shape=(size, size), dtype=complex)
        mat.sum_duplicates()
        return mat

    def print_angry_dog(self) -> None:
            print(DOG)

