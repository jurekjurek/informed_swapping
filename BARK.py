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

import heapq

import numpy as np
from scipy.sparse.linalg import lobpcg

from Subspace import GrowingProjection, lowest_eigenpair


class BARK:
    def __init__(self, hamiltonian: np.ndarray, warm_start_threshold: int = 150,
                 warm_start_patience: int = 5):
        """
        Initialize the BARK protocol with a given Hamiltonian and initial state.

        The Hamiltonian may be a dense array or any scipy sparse matrix; a CSR
        copy is kept so rows can be read off cheaply and the projected block can
        be grown incrementally.

        ``warm_start_threshold`` is the pool size above which the projected
        Hamiltonian is diagonalized with a warm-started iterative solver instead
        of an exact ``eigh``. ``warm_start_patience`` is how many consecutive
        warm-start failures are tolerated before the route is abandoned for the
        rest of the run (see ``lowest_eigenpair``).
        """
        self.projection = GrowingProjection(hamiltonian)
        self.hamiltonian = hamiltonian
        self.dimension = self.projection.dimension
        self.warm_start_threshold = warm_start_threshold
        self.warm_start_patience = warm_start_patience

        self._csr = self.projection.matrix
        # Read off once instead of per candidate: Johann's method needs H[s, s]
        # for every candidate s, and pulling it out of CSR row by row was one
        # sparse lookup per candidate per iteration.
        self._diagonal = np.real(self._csr.diagonal())

        self._warm_start_enabled = True
        self._warm_start_failures = 0
        self.warm_starts_attempted = 0
        self.warm_starts_accepted = 0

    # ------------------------------------------------------------------ #
    # Ranking
    # ------------------------------------------------------------------ #

    def apply_hamiltonian(self, index: int) -> np.ndarray:
        """
        Apply the Hamiltonian to the state corresponding to the given index.
        Return the basis states with non-zero amplitude.
        """
        begin, end = self._csr.indptr[int(index)], self._csr.indptr[int(index) + 1]
        return self._csr.indices[begin:end]

    def rank_states(self, last_approximation: np.ndarray, energy: float,
                    new_states: np.ndarray) -> np.ndarray:
        """
        Johann's method, evaluated for every candidate at once.

        Rate a basis state by the energy the estimator would reach if it were
        added to the current approximation, i.e. the lower eigenvalue of the 2x2
        problem in span{v_i, psi_{i+1}}. Assumes v_i . psi_{i+1} = 0. Lower is
        better.

        The scalar formulation materialized a dense length-N column vector per
        candidate and took a full-length inner product with it. Only the
        non-zeros of that column contribute, and every candidate's column is a
        row of H, so the whole batch is one sparse mat-vec against the current
        approximation.
        """
        states = np.asarray(new_states, dtype=np.int64)
        if states.size == 0:
            return np.empty(0, dtype=float)

        # phi_s is column s of H, i.e. conj(row s). The scalar version computed
        # vdot(v, phi_s) = conj(row_s . v); only |overlap|^2 is used below, so
        # the conjugation is irrelevant and dropped.
        overlaps = self._csr[states] @ last_approximation
        diagonal = self._diagonal[states]
        gamma = np.sqrt((energy - diagonal) ** 2 + 4.0 * np.abs(overlaps) ** 2)
        return 0.5 * (energy + diagonal - gamma)

    def johanns_method(self, last_approximation: np.ndarray, energy: float, index: int) -> float:
        """Single-candidate form of ``rank_states``, kept for direct callers."""
        return float(self.rank_states(last_approximation, energy, [index])[0])

    def project_hamiltonian(self, pool) -> np.ndarray:
        """
        Project the Hamiltonian onto the subspace spanned by the states in the pool.

        Only used by external callers now; the protocol itself grows the block
        incrementally through ``self.projection``.
        """
        pool = np.asarray(pool, dtype=np.int64)
        return self._csr[pool][:, pool].toarray()

    # ------------------------------------------------------------------ #
    # Diagonalization
    # ------------------------------------------------------------------ #

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
        otherwise we fall back to the exact solve. A rejection costs the
        iterative solve *and* the exact one, so after ``warm_start_patience``
        consecutive rejections the warm route is switched off for the remainder
        of the run rather than paying that penalty on every iteration -- whether
        LOBPCG converges here depends on the spectral gap of this particular
        Hamiltonian, which is not known in advance but shows itself immediately.
        """
        size = projected_hamiltonian.shape[0]

        # LOBPCG falls back to a dense solve (and warns) on very small blocks,
        # so the warm route needs a floor regardless of the configured threshold.
        if (previous_vector is not None
                and self._warm_start_enabled
                and previous_vector.shape[0] == size - 1
                and size >= max(self.warm_start_threshold, 8)):
            self.warm_starts_attempted += 1
            guess = np.zeros((size, 1), dtype=projected_hamiltonian.dtype)
            guess[: size - 1, 0] = previous_vector
            # The freshly added direction is what we are testing, so seed it weakly
            # rather than with zero, which LOBPCG could never rotate into.
            guess[size - 1, 0] = 1e-3
            try:
                # Keep tol comfortably tighter than the acceptance gate below.
                # A looser tol is a false economy: LOBPCG then fails the gate and we
                # pay for the iterative solve *and* the exact fallback (measured:
                # tol=1e-6 is slower than never warm-starting at all). maxiter is
                # deliberately modest for the same reason -- a run that has not
                # converged by then is not about to, and the fallback is waiting.
                values, vectors = lobpcg(projected_hamiltonian, guess, largest=False,
                                         tol=1e-9, maxiter=50)
                value = float(np.real(values[0]))
                vector = vectors[:, 0]
                residual = np.linalg.norm(projected_hamiltonian @ vector - value * vector)
                # Relative gate: an absolute 1e-8 is unreachable once the energy
                # scale of the block grows, which turned every large-pool
                # iteration into a guaranteed double solve.
                if np.isfinite(residual) and residual <= 1e-8 * max(1.0, abs(value)):
                    self.warm_starts_accepted += 1
                    self._warm_start_failures = 0
                    return value, vector
            except Exception:
                pass  # fall through to the exact solve

            self._warm_start_failures += 1
            if self._warm_start_failures >= self.warm_start_patience:
                self._warm_start_enabled = False

        return lowest_eigenpair(projected_hamiltonian)

    # ------------------------------------------------------------------ #
    # Running
    # ------------------------------------------------------------------ #

    def sweep(self, target_fidelities, correct_state: np.ndarray,
              initial_state_index: int) -> np.ndarray:
        """
        Pool size at which each of ``target_fidelities`` is first reached.

        BARK is deterministic and its trajectory does not depend on the target:
        the sequence of pooled states for a given initial state is fixed, and the
        target only decides where to stop reading it. Running the protocol once
        per fidelity therefore recomputed the same prefix over and over. This
        walks the sequence once, up to the strictest target, and records every
        crossing on the way.

        Returns one pool size per entry of ``target_fidelities``, in the order
        given, with ``np.inf`` for any target that was never reached.
        """
        targets = np.atleast_1d(np.asarray(target_fidelities, dtype=float))
        order = np.argsort(targets, kind="stable")
        sorted_targets = targets[order]
        sorted_results = np.full(sorted_targets.size, np.inf)
        if sorted_targets.size == 0:
            return sorted_results

        initial_state_index = int(initial_state_index)
        projection = self.projection
        projection.reset()
        projection.extend([initial_state_index])

        # Re-enable the warm-start route for each run: whether it pays off is a
        # property of this Hamiltonian and this trajectory, not of the object.
        self._warm_start_enabled = True
        self._warm_start_failures = 0

        pool_states = {initial_state_index}
        last_approximation = np.zeros(self.dimension, dtype=complex)
        last_approximation[initial_state_index] = 1.0  # Start with the initial state
        energy = float(self._diagonal[initial_state_index])

        memory = {}          # state -> best (lowest) potential seen so far
        queue = []           # heap of (potential, state), lazily invalidated
        last_state = initial_state_index
        previous_vector = None  # warm start for the next diagonalization
        next_target = 0

        while next_target < sorted_targets.size:
            # Remove states that are already in the pool from the new states.
            # ``pool_states`` is a set: the list membership test this replaces was
            # O(pool) per candidate, i.e. O(P^2 * nnz_per_row) over a whole run.
            candidates = [state for state in self.apply_hamiltonian(last_state)
                          if state not in pool_states]

            potentials = self.rank_states(last_approximation, energy, candidates)

            # Update memory with new states and their potentials
            for state, potential in zip(candidates, potentials):
                state = int(state)
                potential = float(potential)
                if state not in memory or potential < memory[state]:
                    memory[state] = potential
                    heapq.heappush(queue, (potential, state))

            # Take the state with the lowest potential as potential is a proxy for
            # energy lowering. The heap replaces a linear scan over a dict that
            # grows to O(N) entries; stale pushes are discarded on the way out.
            last_state = None
            while queue:
                potential, state = heapq.heappop(queue)
                if memory.get(state) == potential:
                    del memory[state]   # avoid re-selection
                    last_state = state
                    break

            # No new states were found: every remaining target is unreachable.
            if last_state is None:
                break

            pool_states.add(last_state)
            projection.extend([last_state])

            # Diagonalize the projected Hamiltonian to find the ground state.
            # The pool grows by one state appended at the end, so the previous
            # ground state is a valid warm start for the enlarged block.
            energy, ground_state_vector = self.lowest_eigenpair(projection.block,
                                                                previous_vector)
            previous_vector = ground_state_vector
            # The pool only grows, so its previous entries are a prefix of the
            # current ones and this assignment overwrites the entire support --
            # no need to re-zero a length-N vector every iteration.
            last_approximation[projection.pool] = ground_state_vector

            # Calculate fidelity with the correct state. Only the pooled entries
            # of the approximation are non-zero, so this is an O(k) inner product
            # rather than an O(N) one.
            fidelity = np.abs(np.vdot(ground_state_vector,
                                      correct_state[projection.pool])) ** 2

            while (next_target < sorted_targets.size
                   and fidelity >= sorted_targets[next_target]):
                sorted_results[next_target] = projection.size
                next_target += 1

        results = np.empty_like(sorted_results)
        results[order] = sorted_results
        return results

    def test_run(self, target_fidelity: float, correct_state: np.ndarray,
                 initial_state_index: int):
        """
        Run the BARK protocol until the target fidelity is reached.

        Returns the pool size at that point, or ``np.inf`` if the protocol ran
        out of reachable states first.
        """
        pool_size = self.sweep([target_fidelity], correct_state, initial_state_index)[0]
        return pool_size if not np.isfinite(pool_size) else int(pool_size)
