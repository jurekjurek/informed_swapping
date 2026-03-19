"""Implementation of BARK - Bitstring Algorithm for Recursive Krylov.

This is a purely classical (bitstring) variant of SKQD.
"""

from typing import List, Optional, Sequence, Tuple

import numpy as np
from qiskit.quantum_info import SparsePauliOp
from scipy.sparse import csr_matrix

from collections import defaultdict
from typing import List, Tuple
import numpy as np

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
        keep_states: Optional[int] = None
    ) -> None:
        
        self.H: SparsePauliOp = H
        self.H_paulis: List[str] = H.to_list()
        self.initial_state: str = initial_state
        self.max_iterations: int = max_iterations
        self.time_step: Optional[float] = time_step
        self.tolerance: Optional[float] = tolerance
        self.basis: List[List[Tuple[str, float]]] = [[(initial_state, 1.0)]]
        self.even_numbers: Optional[bool] = even_numbers
        self.keep_states: Optional[int] = keep_states

        self._compile_teo_terms()

    def apply_pauli_string(self, state: str, pauli_string: str) -> str:
        """Apply a Pauli string to a bitstring state.

        The mapping is as follows:
        I -> no change
        X -> flip the bit
        Y -> flip the bit + i/-i
        Z -> no change + (-1) if bit is 1

        Parameters
        ----------
        state : str
            The input bitstring state (e.g. '0101').
        pauli_string : str
            The Pauli string to apply (e.g. 'IXYZ').

        Returns
        -------
        tuple[str, float]
            The resulting bitstring state and the corresponding coefficient after applying the Pauli string.
        """
        new_state = list(state)
        coeff = 1.0
        for i, p in enumerate(pauli_string):
            if p == 'I':
                continue
            elif p == 'X':
                new_state[i] = '1' if new_state[i] == '0' else '0'
            elif p == 'Y':
                new_state[i] = '1' if new_state[i] == '0' else '0'
                coeff *= 1j if new_state[i] == '1' else -1j
            elif p == 'Z':
                coeff *= -1 if new_state[i] == '1' else 1
            else:
                raise ValueError(f"Invalid Pauli character: {p}")
            
        return ''.join(new_state), coeff

    def _compile_teo_terms(self):
        """Precompute masks and trig factors for apply_teo_fast()."""
        if hasattr(self, "_teo_terms"):
            return self._teo_terms

        n = len(self.initial_state)
        terms = []

        for pauli_string, coeff in self.H_paulis:
            flip_mask = 0   # bits flipped by X or Y
            phase_mask = 0  # bits contributing a (-1)^bit phase from Y or Z
            y_count = 0

            for i, p in enumerate(pauli_string):
                bit = 1 << (n - 1 - i)   # match string indexing: state[0] is leftmost bit
                if p == "X":
                    flip_mask |= bit
                elif p == "Y":
                    flip_mask |= bit
                    phase_mask |= bit
                    y_count += 1
                elif p == "Z":
                    phase_mask |= bit
                elif p == "I":
                    pass
                else:
                    raise ValueError(f"Invalid Pauli character: {p}")

            # coeff may be complex from SparsePauliOp.to_list()
            theta = coeff * self.time_step
            sin_theta = np.sin(theta)
            cos_theta = np.cos(theta)

            # Product of all fixed i factors from Y operators
            y_prefactor = (1j) ** y_count

            terms.append((flip_mask, phase_mask, y_prefactor, sin_theta, cos_theta))

        self._teo_terms = terms
        return terms


    def apply_teo_fast(self, state: str, prev_amp: float) -> List[Tuple[str, complex]]:
        """Fast version of apply_teo with the same functionality/output type."""
        n = len(state)
        if not hasattr(self, "_teo_terms"):
            self._compile_teo_terms()

        # Internal representation: {integer_bitstring: amplitude}
        states = {int(state, 2): prev_amp + 0.0j}

        for flip_mask, phase_mask, y_prefactor, sin_theta, cos_theta in self._teo_terms:
            new_states = defaultdict(complex)

            for bits, amp in states.items():
                # Phase from Y/Z acting on occupied bits:
                # each occupied Y or Z contributes a factor -1
                sign = -1.0 if ((bits & phase_mask).bit_count() & 1) else 1.0

                evolved_bits = bits ^ flip_mask
                evolved_amp = amp * y_prefactor * sign * sin_theta

                new_states[evolved_bits] += evolved_amp
                new_states[bits] += amp * cos_theta

            states = new_states

        return [(format(bits, f"0{n}b"), amp) for bits, amp in states.items()]
    
    def run(self):
        """Run the BARK algorithm."""
        for iteration in range(self.max_iterations):
            new_basis = []
            for state, coeff in self.basis[-1]:
                evolved_states = self.apply_teo_fast(state, coeff)
                new_basis.extend(evolved_states)

            # Remove dublicates by summing coefficients of identical states
            states = defaultdict(complex)
            for state, coeff in new_basis:
                states[state] += coeff

            # Convert back to the original string format once, at the end, if keep_states is set, keep only the states with the largest amplitudes, same for tolerance. If even_numbers, only keep states with even number of 1s and 0s.
            if self.keep_states is not None:
                states = dict(sorted(states.items(), key=lambda item: abs(item[1]), reverse=True)[:self.keep_states])
            if self.tolerance is not None:
                states = {bits: amp for bits, amp in states.items() if abs(amp) >= self.tolerance}
            if self.even_numbers:
                states = {bits: amp for bits, amp in states.items() if (bits.bit_count() % 2) == 0}

            new_basis = list(states.items())
            self.basis.append(new_basis)
            print(f"Iteration {iteration+1}/{self.max_iterations} completed. Basis size: {len(new_basis)}")

            