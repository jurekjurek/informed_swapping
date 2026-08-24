'''
Author: Emil Rosanowski
This will be a simple implementation of the BARK protocol.
We will start with an initial state (given as an index in the basis), then the Hamiltonian will be applied once in each iteration.
After each application, we will have a set of basis states with non-zero amplitudes.
We will use Johann's methods to rate them according to their potential to lower the energy of the system.
This proxy will be stored as well as memory. 
The state with the highest potential will be selected and added to the pool.
Then the next iteration will be performed. If the potential of another state in the memory is higher, it will be selected instead of the new state.
Every n times, a full diagonalization of the projected Hamiltonian will be performed to check the current performance.
Note: for test purposes, we choose n=1, effectively tracking how well Johann's method performs at the same time.
'''

import numpy as np
from scipy.linalg import eigh
from scipy.sparse import issparse, csr_matrix
from scipy.sparse.linalg import lobpcg


class BARK:
    def __init__(self, hamiltonian: np.ndarray, warm_start_threshold: int = 150):
        """
        Initialize the BARK protocol with a given Hamiltonian and initial state.

        The Hamiltonian may be a dense array or any scipy sparse matrix. For the
        sparse case a CSR copy is kept so single rows can be read off cheaply.
        ``warm_start_threshold`` is the pool size above which the projected
        Hamiltonian is diagonalized with a warm-started iterative solver instead
        of a full ``eigh`` (see ``lowest_eigenpair``).
        """
        self.hamiltonian = hamiltonian
        self.is_sparse = issparse(hamiltonian)
        self.warm_start_threshold = warm_start_threshold
        if self.is_sparse:
            self._csr = csr_matrix(hamiltonian)
            self._csr.eliminate_zeros()
        else:
            self._csr = None

    def _row(self, index: int):
        """Non-zero column indices and values of row ``index`` of the Hamiltonian."""
        if self.is_sparse:
            start, end = self._csr.indptr[index], self._csr.indptr[index + 1]
            return self._csr.indices[start:end], self._csr.data[start:end]
        row = self.hamiltonian[index]
        columns = np.nonzero(row)[0]
        return columns, row[columns]

    def _column(self, index: int) -> np.ndarray:
        """
        Column ``index`` of the Hamiltonian as a dense 1-D vector. H is Hermitian,
        so the column is the conjugate of the corresponding row, which is the cheap
        direction to read for both dense (contiguous) and CSR storage.
        """
        if not self.is_sparse:
            return self.hamiltonian[:, int(index)]
        columns, values = self._row(int(index))
        phi = np.zeros(self.hamiltonian.shape[0], dtype=self._csr.dtype)
        phi[columns] = np.conj(values)
        return phi

    def johanns_method(self, last_approximation: np.ndarray, energy: float, index: int) -> float:
        """
        Johann's method: rate a basis state by the energy the estimator would reach if
        it were added to it, i.e. the lower eigenvalue of the 2x2 problem in
        span{v_i, psi_{i+1}}. Assumes v_i . psi_{i+1} = 0. Lower is better.
        """
        phi = self._column(index)
        diagonal = np.real(phi[int(index)])
        overlap = np.vdot(last_approximation, phi)
        gamma = np.sqrt((energy - diagonal) ** 2 + 4.0 * np.abs(overlap) ** 2)
        return 0.5 * (energy + diagonal - gamma)

    def apply_hamiltonian(self, index: int) -> np.ndarray:
        """
        Apply the Hamiltonian to the state corresponding to the given index.
        Return a list of new states.
        """
        columns, _ = self._row(int(index))
        new_states = columns.tolist()
        return new_states

    def rank_states(self, last_approximation: np.ndarray, energy: float, new_states: np.ndarray) -> dict:
        """
        Rank the new states based on their potential to lower the energy of the system.
        """
        potentials = {}
        for state in new_states:
            potential = self.johanns_method(last_approximation, energy, state)
            potentials[state] = potential
        return potentials
    
    def project_hamiltonian(self, pool: np.ndarray) -> np.ndarray:
        """
        Project the Hamiltonian onto the subspace spanned by the states in the pool.
        The result is always a small dense block, in the order given by ``pool``.
        """
        if self.is_sparse:
            pool = np.asarray(pool, dtype=int)
            return self._csr[pool][:, pool].toarray()
        projected_hamiltonian = self.hamiltonian[np.ix_(pool, pool)]
        return projected_hamiltonian

    def lowest_eigenpair(self, projected_hamiltonian: np.ndarray,
                         previous_vector: np.ndarray | None = None):
        """
        Lowest eigenpair of the projected Hamiltonian.

        Purely a speed-up for the metric -- the eigenpair returned is the one
        ``eigh`` would give, so the protocol itself is unchanged. Two routes:

        * Small pools: ``eigh`` restricted to the single lowest eigenpair.
        * Large pools: since the pool grows by one basis state appended at the
          end, the previous eigenvector padded with one entry is an excellent
          starting guess, so a warm-started LOBPCG solve converges in a few
          iterations instead of costing a full O(k^3) diagonalization.

        The warm-started result is accepted only if its residual is tight;
        otherwise we silently fall back to the exact ``eigh``.
        """
        size = projected_hamiltonian.shape[0]

        # LOBPCG falls back to a dense solve (and warns) on very small blocks,
        # so the warm route needs a floor regardless of the configured threshold.
        if previous_vector is not None and size >= max(self.warm_start_threshold, 8):
            guess = np.zeros((size, 1), dtype=complex)
            guess[: size - 1, 0] = previous_vector
            # The freshly added direction is what we are testing, so seed it weakly
            # rather than with zero, which LOBPCG could never rotate into.
            guess[size - 1, 0] = 1e-3
            try:
                # Keep tol comfortably tighter than the 1e-8 acceptance gate below.
                # A looser tol is a false economy: LOBPCG then fails the gate and we
                # pay for the iterative solve *and* the exact fallback (measured:
                # tol=1e-6 is slower than never warm-starting at all).
                values, vectors = lobpcg(projected_hamiltonian, guess, largest=False,
                                         tol=1e-9, maxiter=200)
                value = float(np.real(values[0]))
                vector = vectors[:, 0]
                residual = np.linalg.norm(projected_hamiltonian @ vector - value * vector)
                if np.isfinite(residual) and residual < 1e-8:
                    return value, vector
            except Exception:
                pass  # fall through to the exact solve

        if size < 3:
            eigenvalues, eigenvectors = eigh(projected_hamiltonian)
            index = int(np.argmin(eigenvalues))
            return float(np.real(eigenvalues[index])), eigenvectors[:, index]

        eigenvalues, eigenvectors = eigh(projected_hamiltonian, check_finite=False,
                                         subset_by_index=[0, 0], driver="evr")
        return float(np.real(eigenvalues[0])), eigenvectors[:, 0]

    def test_run(self, target_fidelity: float, correct_state: np.ndarray, initial_state_index: int) -> int:
        """
        Run the BARK protocol until the target fidelity is reached.
        """

        memory = {}
        pool = [initial_state_index]
        last_approximation = np.zeros(self.hamiltonian.shape[0])
        last_approximation[initial_state_index] = 1.0  # Start with the initial state
        energy = np.real(self.hamiltonian[initial_state_index, initial_state_index])

        last_state = initial_state_index
        previous_vector = None  # warm start for the next diagonalization

        iteration = 0
        while True:
            iteration += 1

            new_states = self.apply_hamiltonian(last_state)

            # Remove states that are already in the pool from the new states
            new_states = [state for state in new_states if state not in pool]

            potentials = self.rank_states(last_approximation, energy, new_states)

            # Update memory with new states and their potentials
            for state, potential in potentials.items():
                if state not in memory or potential < memory[state]:
                    memory[state] = potential

            # Check if memory is empty, which means no new states were found
            if not memory:
                return np.inf
            
            # Take the state with the lowest potential as potential is a proxy for energy lowering
            last_state = min(memory, key=memory.get)

            # Delete the selected state from memory to avoid re-selection
            del memory[last_state]

            pool.append(last_state)

            # Project the Hamiltonian onto the current pool of states
            projected_hamiltonian = self.project_hamiltonian(pool)

            # Diagonalize the projected Hamiltonian to find the ground state.
            # The pool grows by one state appended at the end, so the previous
            # ground state is a valid warm start for the enlarged block.
            energy, ground_state_vector = self.lowest_eigenpair(projected_hamiltonian,
                                                               previous_vector)
            previous_vector = ground_state_vector
            last_approximation = np.zeros(self.hamiltonian.shape[0], dtype=complex)
            last_approximation[pool] = ground_state_vector

            # Calculate fidelity with the correct state
            fidelity = np.abs(np.dot(last_approximation.conj(), correct_state)) ** 2

            if fidelity >= target_fidelity:
                return len(pool)  # Return the number of states in the pool when target fidelity is reached