"""
Author: Emil Rosanowski
This will be the implementation of the SKQD protocol.
As this is only a simulation, we will simulate the quantum part.

We will start with an initial state (given as an index in the basis).
For a given time parameter t, we will precompute U(t)=e^{-iHt}.
Then, U(t) will be applied to the current state, resulting in a new state.
Then we will draw n_shots samples according to the probability distribution of the new state.
This is one iteration and at the end of the iteration, we will add the sampled states to the pool.
In every iteration, the Hamiltonian is projected onto the pool and diagonalized to find the ground state of the projected Hamiltonian.
"""

import numpy as np
from scipy.linalg import expm, eigh
from scipy.sparse import issparse, csr_matrix
from scipy.sparse.linalg import expm as sparse_expm, expm_multiply

class SKQD:
    def __init__(self, hamiltonian: np.ndarray,
                 eigenvalues: np.ndarray | None = None,
                 eigenvectors: np.ndarray | None = None):
        """
        Initialize the SKQD protocol with a given Hamiltonian and initial state.

        The Hamiltonian may be dense or any scipy sparse matrix. If the caller
        already has the eigendecomposition H = V diag(E) V^dagger, passing
        ``eigenvalues`` and ``eigenvectors`` lets U(t) be assembled from it
        directly instead of going through a matrix exponential.
        """
        self.hamiltonian = hamiltonian
        self.is_sparse = issparse(hamiltonian)
        self._csr = csr_matrix(hamiltonian) if self.is_sparse else None
        self.eigenvalues = None if eigenvalues is None else np.asarray(eigenvalues)
        self.eigenvectors = None if eigenvectors is None else np.asarray(eigenvectors)
        self._eigenvectors_h = None if self.eigenvectors is None else self.eigenvectors.conj().T
        self._unitary_cache = {}
        self._phase_cache = {}
        self._generator_cache = {}

    def compute_unitary(self, t: float) -> np.ndarray:
        """
        Compute the unitary evolution operator U(t) = exp(-i * H * t).

        If the eigendecomposition was supplied, U(t) = V exp(-i E t) V^dagger is
        assembled from it, which is far cheaper than a matrix exponential and
        exact to machine precision. Otherwise ``expm`` is used, as before.
        Results are cached per t, since a run reuses the same t throughout.
        """
        key = float(t)
        if key not in self._unitary_cache:
            if self.eigenvalues is not None and self.eigenvectors is not None:
                phases = np.exp(-1j * self.eigenvalues * key)
                unitary = (self.eigenvectors * phases) @ self.eigenvectors.conj().T
            elif self.is_sparse:
                # sparse_expm works in CSC; note the result is dense in content, so
                # for large systems supplying the eigendecomposition is the only
                # affordable route.
                unitary = sparse_expm(-1j * self._csr.tocsc() * key)
            else:
                unitary = expm(-1j * self.hamiltonian * key)
            self._unitary_cache[key] = unitary
        return self._unitary_cache[key]

    def apply_unitary(self, state: np.ndarray, unitary: np.ndarray) -> np.ndarray:
        """
        Apply the unitary evolution operator U(t) to the given state.
        """
        return unitary @ state

    def evolve(self, state: np.ndarray, t: float) -> np.ndarray:
        """
        Return U(t) @ state, avoiding the construction of U(t) where possible.

        Mathematically identical to ``apply_unitary(state, compute_unitary(t))``,
        only cheaper. Three routes:

        * eigendecomposition supplied: apply the phases in the eigenbasis, which
          costs two matrix-vector products instead of assembling U;
        * sparse Hamiltonian: Krylov ``expm_multiply``, which never forms U at
          all -- ``sparse_expm`` would return a matrix that is dense in content,
          so it is both slow and no cheaper in memory;
        * dense Hamiltonian: the cached ``expm``, as before.
        """
        key = float(t)

        if self.eigenvalues is not None and self.eigenvectors is not None:
            if key not in self._phase_cache:
                self._phase_cache[key] = np.exp(-1j * self.eigenvalues * key)
            return self.eigenvectors @ (self._phase_cache[key] * (self._eigenvectors_h @ state))

        if self.is_sparse:
            if key not in self._generator_cache:
                self._generator_cache[key] = (-1j * key) * self._csr
            return expm_multiply(self._generator_cache[key], state)

        return self.apply_unitary(state, self.compute_unitary(key))

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

    def test_run(self, initial_state_index: int, t: float, n_shots: int, correct_state: np.ndarray, target_fidelity: float) -> int:
        """
        Run the SKQD protocol for a given number of iterations.
        """
        # Initialize the pool with the initial state
        pool = np.array([initial_state_index])
        last_approximation = np.zeros(self.hamiltonian.shape[0], dtype=complex)
        last_approximation[initial_state_index] = 1.0  # Start with the initial state

        pool_size = 1

        while True:
            new_states = self.evolve(last_approximation, t)

            last_approximation = new_states

            # Sample n_shots states according to the probability distribution of the new state
            probabilities = np.abs(np.asarray(new_states).ravel()) ** 2
            sampled_indices = np.random.choice(len(probabilities), size=n_shots, p=probabilities / probabilities.sum())
            # Add the sampled states to the pool which are not already in the pool
            pool = np.unique(np.concatenate((pool, sampled_indices)))

            if len(pool) == pool_size:
                return np.inf
            pool_size = len(pool)

            # Project the Hamiltonian onto the current pool of states
            projected_hamiltonian = self.project_hamiltonian(pool)

            # Diagonalize the projected Hamiltonian to find the ground state
            eigenvalues, eigenvectors = eigh(projected_hamiltonian)
            ground_state_index = np.argmin(eigenvalues)
            ground_state = np.zeros(self.hamiltonian.shape[0], dtype=complex)
            ground_state[pool] = eigenvectors[:, ground_state_index]

            fidelity = np.abs(np.dot(ground_state.conj(), correct_state)) ** 2

            if fidelity >= target_fidelity:
                return pool_size  # Return the number of states in the pool when target fidelity is reached

    @staticmethod
    def _parabolic_minimum(values: tuple, best, evaluate) -> float:
        """
        Vertex of the parabola through the best grid point and its two neighbours.

        Returns the grid point itself when it sits on the edge of the grid or when
        the three points are not convex, i.e. when there is no interior minimum to
        extrapolate to.
        """
        values = list(values)
        position = values.index(best)
        if position == 0 or position == len(values) - 1:
            return float(best)

        x0, x1, x2 = (float(values[position - 1]), float(values[position]),
                      float(values[position + 1]))
        y0, y1, y2 = (evaluate(values[position - 1]), evaluate(values[position]),
                      evaluate(values[position + 1]))

        denominator = (x0 - x1) * (x0 - x2) * (x1 - x2)
        if denominator == 0.0:
            return x1
        a = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / denominator
        b = (x2 * x2 * (y0 - y1) + x1 * x1 * (y2 - y0) + x0 * x0 * (y1 - y2)) / denominator
        if a <= 0.0:
            return x1  # concave or flat: no interior minimum
        return float(np.clip(-b / (2.0 * a), x0, x2))

    def optimize(self, initial_state_index: int, correct_state: np.ndarray, target_fidelity: float,
                 t_values: tuple = (0.1, 0.25, 0.5, 1.0, 2.0, 4.0),
                 shot_values: tuple = (10, 25, 50, 100),
                 n_repeats: int = 3):
        """
        Optimize the two free parameters t and n_shots such that the final pool size is minimal.

        The pool size is a noisy, piecewise-constant function of (t, n_shots), which
        gives a simplex method no descent direction to follow. Instead we always scan
        the full fixed 6 x 4 grid, then extrapolate to a refined optimum by fitting a
        parabola through the best grid point and its neighbours along each axis. The
        extrapolated point is evaluated and only kept if it actually beats the grid.

        Returns (best_t, best_n_shots, best_pool_size).
        """
        # A failed run must score strictly worse than the worst possible success,
        # which is a pool spanning the whole Hilbert space.
        penalty = self.hamiltonian.shape[0] + 1

        def objective(t: float, n_shots: int) -> float:
            pool_sizes = [self.test_run(initial_state_index, float(t), int(n_shots),
                                        correct_state, target_fidelity)
                          for _ in range(n_repeats)]
            pool_sizes = [penalty if not np.isfinite(p) else p for p in pool_sizes]
            return float(np.mean(pool_sizes))

        # Scan the full grid
        grid = {(t, n_shots): objective(t, n_shots)
                for t in t_values for n_shots in shot_values}
        best_t, best_shots = min(grid, key=grid.get)
        best_pool_size = grid[(best_t, best_shots)]

        # Extrapolate along each axis through the grid optimum
        refined_t = self._parabolic_minimum(t_values, best_t,
                                            lambda x: grid[(x, best_shots)])
        refined_shots = self._parabolic_minimum(shot_values, best_shots,
                                                lambda x: grid[(best_t, x)])
        refined_shots = max(1, int(round(refined_shots)))

        if (refined_t, refined_shots) != (best_t, best_shots):
            refined_pool_size = objective(refined_t, refined_shots)
            if refined_pool_size < best_pool_size:
                best_t, best_shots = float(refined_t), refined_shots
                best_pool_size = refined_pool_size

        return float(best_t), int(best_shots), best_pool_size