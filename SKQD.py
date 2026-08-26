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
from scipy.linalg import expm
from scipy.sparse import issparse
from scipy.sparse.linalg import expm as sparse_expm, expm_multiply

from Subspace import GrowingProjection, lowest_eigenpair

# ``expm_multiply`` recomputes the whole Al-Mohy & Higham setup on every call:
# the trace, the shifted operator, the exact 1-norm and -- because ||tH||_1 is
# far above the cheap-path cutoff of condition (3.13) for these Hamiltonians --
# a ``onenormest`` of A^p for p up to 8, which is a few hundred extra sparse
# mat-vecs. All of it depends only on (H, t), which is fixed for an entire run,
# so it is computed once per t and cached. The private imports are guarded; if a
# future scipy renames them we fall back to the public entry point.
try:
    from scipy.sparse.linalg._expm_multiply import (
        LazyOperatorNormInfo,
        _exact_1_norm,
        _expm_multiply_simple_core,
        _fragment_3_1,
        _ident_like,
        _trace,
    )
    _EXPM_PLAN_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on the installed scipy
    _EXPM_PLAN_AVAILABLE = False

_EXPM_TOL = 2.0 ** -53


class SKQD:
    def __init__(self, hamiltonian: np.ndarray,
                 eigenvalues: np.ndarray | None = None,
                 eigenvectors: np.ndarray | None = None):
        """
        Initialize the SKQD protocol with a given Hamiltonian and initial state.

        The Hamiltonian may be dense or any scipy sparse matrix. If the caller
        already has the eigendecomposition H = V diag(E) V^dagger, passing
        ``eigenvalues`` and ``eigenvectors`` lets U(t) be applied from it
        directly instead of going through a matrix exponential. For any Hilbert
        space small enough to diagonalize densely -- which at these system sizes
        means everything up to about 4096 states -- that is by far the cheapest
        route, and the caller is expected to supply it.
        """
        self.projection = GrowingProjection(hamiltonian)
        self.hamiltonian = hamiltonian
        self.dimension = self.projection.dimension
        self.is_sparse = issparse(hamiltonian)
        self._csr = self.projection.matrix
        self.eigenvalues = None if eigenvalues is None else np.asarray(eigenvalues)
        self.eigenvectors = None if eigenvectors is None else np.asarray(eigenvectors)
        self._eigenvectors_h = None if self.eigenvectors is None else self.eigenvectors.conj().T
        self._unitary_cache = {}
        self._phase_cache = {}
        self._generator_cache = {}
        self._expm_plan_cache = {}

    # ------------------------------------------------------------------ #
    # Time evolution
    # ------------------------------------------------------------------ #

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

    def _expm_plan(self, t: float):
        """
        The scaling-and-squaring parameters of ``expm_multiply`` for a fixed t.

        Returns ``(shifted_generator, mu, m_star, s)``, everything the Taylor
        core needs, so that repeated evolutions at the same t skip the trace,
        the 1-norm and the norm estimates of the operator powers entirely.
        """
        key = float(t)
        plan = self._expm_plan_cache.get(key)
        if plan is not None:
            return plan

        generator = (-1j * key) * self._csr
        mu = _trace(generator) / float(self.dimension)
        shifted = generator - mu * _ident_like(generator)
        one_norm = _exact_1_norm(shifted)
        if one_norm == 0:
            m_star, s = 0, 1
        else:
            norm_info = LazyOperatorNormInfo(shifted, A_1_norm=one_norm, ell=2)
            m_star, s = _fragment_3_1(norm_info, 1, _EXPM_TOL, ell=2)

        plan = (shifted, mu, m_star, s)
        self._expm_plan_cache[key] = plan
        return plan

    def evolve(self, state: np.ndarray, t: float) -> np.ndarray:
        """
        Return U(t) @ state, avoiding the construction of U(t) where possible.

        Mathematically identical to ``apply_unitary(state, compute_unitary(t))``,
        only cheaper. Three routes:

        * eigendecomposition supplied: apply the phases in the eigenbasis, which
          costs two matrix-vector products instead of assembling U;
        * sparse Hamiltonian: Krylov ``expm_multiply``, which never forms U at
          all -- ``sparse_expm`` would return a matrix that is dense in content,
          so it is both slow and no cheaper in memory. The per-t setup is cached
          (see ``_expm_plan``), leaving only the Taylor mat-vecs per call;
        * dense Hamiltonian: the cached ``expm``, as before.
        """
        key = float(t)

        if self.eigenvalues is not None and self.eigenvectors is not None:
            if key not in self._phase_cache:
                self._phase_cache[key] = np.exp(-1j * self.eigenvalues * key)
            return self.eigenvectors @ (self._phase_cache[key] * (self._eigenvectors_h @ state))

        if self.is_sparse:
            if _EXPM_PLAN_AVAILABLE:
                shifted, mu, m_star, s = self._expm_plan(key)
                return _expm_multiply_simple_core(shifted, state, 1.0, mu, m_star, s,
                                                  _EXPM_TOL)
            if key not in self._generator_cache:
                self._generator_cache[key] = (-1j * key) * self._csr
            return expm_multiply(self._generator_cache[key], state)

        return self.apply_unitary(state, self.compute_unitary(key))

    def project_hamiltonian(self, pool) -> np.ndarray:
        """
        Project the Hamiltonian onto the subspace spanned by the states in the pool.

        Only used by external callers now; the protocol itself grows the block
        incrementally through ``self.projection``.
        """
        pool = np.asarray(pool, dtype=np.int64)
        return self._csr[pool][:, pool].toarray()

    # ------------------------------------------------------------------ #
    # Running
    # ------------------------------------------------------------------ #

    def sweep(self, initial_state_index: int, t: float, n_shots: int,
              correct_state: np.ndarray, target_fidelities,
              correct_probabilities: np.ndarray | None = None) -> np.ndarray:
        """
        Pool size at which each of ``target_fidelities`` is first reached.

        The sampled trajectory does not depend on the target fidelity: the
        diagonalization is only read to test the stopping condition and never
        feeds back into the pool, so one trajectory answers every target at
        once. Targets are checked in ascending order and each is recorded at the
        first iteration whose fidelity clears it, which is exactly what running
        the protocol separately per target used to produce.

        Returns one pool size per entry of ``target_fidelities``, in the order
        given, with ``np.inf`` for any target not reached before the pool stopped
        growing.
        """
        targets = np.atleast_1d(np.asarray(target_fidelities, dtype=float))
        order = np.argsort(targets, kind="stable")
        sorted_targets = targets[order]
        sorted_results = np.full(sorted_targets.size, np.inf)
        if sorted_targets.size == 0:
            return sorted_results

        if correct_probabilities is None:
            correct_probabilities = np.abs(correct_state) ** 2

        initial_state_index = int(initial_state_index)
        n_shots = int(n_shots)

        projection = self.projection
        projection.reset()
        projection.extend([initial_state_index])

        state_vector = np.zeros(self.dimension, dtype=complex)
        state_vector[initial_state_index] = 1.0  # Start with the initial state

        # Probability mass of the true ground state that the pool already covers.
        # For *any* unit vector v supported on the pool, |<g|v>|^2 is bounded by
        # this mass (it is the squared norm of the projection of the ground state
        # onto the pool). So when the mass is below the lowest target still open,
        # no eigenvector of the projected block can clear it and the whole
        # diagonalization can be skipped -- an exact shortcut, not a heuristic,
        # and the one that removes most of the cost: with a dense ground state,
        # a high target needs a pool spanning nearly the whole Hilbert space, and
        # every iteration before that used to pay a full O(k^3) solve to learn
        # what this bound settles in O(shots).
        covered_mass = float(correct_probabilities[initial_state_index])

        next_target = 0
        while next_target < sorted_targets.size:
            state_vector = np.asarray(self.evolve(state_vector, t)).ravel()

            # Sample n_shots states according to the probability distribution of
            # the new state. Sampling straight from the cumulative distribution
            # skips the validation and the internal copies of np.random.choice
            # while drawing from the same stream.
            probabilities = np.abs(state_vector) ** 2
            cumulative = np.cumsum(probabilities)
            total = cumulative[-1]
            if not (total > 0.0):
                break
            draws = np.random.random_sample(n_shots) * total
            sampled_indices = np.minimum(np.searchsorted(cumulative, draws, side="right"),
                                         self.dimension - 1)

            # Add the sampled states to the pool which are not already in the pool
            fresh = projection.extend(sampled_indices)
            if fresh.size == 0:
                break  # pool stagnated; every remaining target is unreachable

            covered_mass += float(correct_probabilities[fresh].sum())
            # The slack absorbs the rounding drift of the running sum, so the
            # bound can never skip a diagonalization that would have cleared the
            # target by a hair.
            if covered_mass + 1e-12 < sorted_targets[next_target]:
                continue

            # Diagonalize the projected Hamiltonian to find the ground state.
            _, ground_state_vector = lowest_eigenpair(projection.block)

            # Only the pooled entries of the embedded ground state are non-zero,
            # so the overlap is an O(k) inner product rather than an O(N) one.
            fidelity = np.abs(np.vdot(ground_state_vector,
                                      correct_state[projection.pool])) ** 2

            while (next_target < sorted_targets.size
                   and fidelity >= sorted_targets[next_target]):
                sorted_results[next_target] = projection.size
                next_target += 1

        results = np.empty_like(sorted_results)
        results[order] = sorted_results
        return results

    def test_run(self, initial_state_index: int, t: float, n_shots: int,
                 correct_state: np.ndarray, target_fidelity: float):
        """
        Run the SKQD protocol until the target fidelity is reached.

        Returns the pool size at that point, or ``np.inf`` if the pool stopped
        growing first.
        """
        pool_size = self.sweep(initial_state_index, t, n_shots, correct_state,
                               [target_fidelity])[0]
        return pool_size if not np.isfinite(pool_size) else int(pool_size)

    # ------------------------------------------------------------------ #
    # Parameter search
    # ------------------------------------------------------------------ #

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

    def optimize_many(self, initial_state_index: int, correct_state: np.ndarray,
                      target_fidelities,
                      t_values: tuple = (0.1, 0.25, 0.5, 1.0, 2.0, 4.0),
                      shot_values: tuple = (10, 25, 50, 100),
                      n_repeats: int = 3,
                      correct_probabilities: np.ndarray | None = None) -> list:
        """
        Optimize (t, n_shots) for every target fidelity in one grid scan.

        The pool size is a noisy, piecewise-constant function of (t, n_shots), which
        gives a simplex method no descent direction to follow. Instead we always scan
        the full fixed 6 x 4 grid, then extrapolate to a refined optimum by fitting a
        parabola through the best grid point and its neighbours along each axis. The
        extrapolated point is evaluated and only kept if it actually beats the grid.

        Because one sampled trajectory yields the pool size for every target at
        once (see ``sweep``), the grid is scanned once instead of once per target
        -- previously a five-fidelity study paid for five identical 72-run scans
        per initial state. The per-fidelity choice of (t, n_shots) is unaffected.

        Note that the fidelities now share their trajectories rather than each
        getting an independent draw. The estimator is unchanged in distribution;
        the fidelity columns of a row are now paired instead of independent.

        Returns one ``(best_t, best_n_shots, best_pool_size)`` triple per entry of
        ``target_fidelities``, in the order given.
        """
        targets = np.atleast_1d(np.asarray(target_fidelities, dtype=float))
        if correct_probabilities is None:
            correct_probabilities = np.abs(correct_state) ** 2

        # A failed run must score strictly worse than the worst possible success,
        # which is a pool spanning the whole Hilbert space.
        penalty = float(self.dimension + 1)

        cache = {}

        def objective(t: float, n_shots: int) -> np.ndarray:
            """Mean pool size over the repeats, one entry per target fidelity."""
            key = (float(t), int(n_shots))
            if key not in cache:
                runs = np.stack([
                    self.sweep(initial_state_index, float(t), int(n_shots),
                               correct_state, targets,
                               correct_probabilities=correct_probabilities)
                    for _ in range(n_repeats)
                ])
                runs = np.where(np.isfinite(runs), runs, penalty)
                cache[key] = runs.mean(axis=0)
            return cache[key]

        # Scan the full grid once, for all fidelities together.
        grid = {(t, n_shots): objective(t, n_shots)
                for t in t_values for n_shots in shot_values}

        results = []
        for position in range(targets.size):
            scores = {key: float(value[position]) for key, value in grid.items()}
            best_t, best_shots = min(scores, key=scores.get)
            best_pool_size = scores[(best_t, best_shots)]

            # Extrapolate along each axis through the grid optimum
            refined_t = self._parabolic_minimum(t_values, best_t,
                                                lambda x: scores[(x, best_shots)])
            refined_shots = self._parabolic_minimum(shot_values, best_shots,
                                                    lambda x: scores[(best_t, x)])
            refined_shots = max(1, int(round(refined_shots)))

            if (refined_t, refined_shots) != (best_t, best_shots):
                refined_pool_size = float(objective(refined_t, refined_shots)[position])
                if refined_pool_size < best_pool_size:
                    best_t, best_shots = float(refined_t), refined_shots
                    best_pool_size = refined_pool_size

            results.append((float(best_t), int(best_shots), best_pool_size))

        return results

    def optimize(self, initial_state_index: int, correct_state: np.ndarray,
                 target_fidelity: float,
                 t_values: tuple = (0.1, 0.25, 0.5, 1.0, 2.0, 4.0),
                 shot_values: tuple = (10, 25, 50, 100),
                 n_repeats: int = 3):
        """
        Single-fidelity form of ``optimize_many``.

        Returns (best_t, best_n_shots, best_pool_size).
        """
        return self.optimize_many(initial_state_index, correct_state,
                                  [target_fidelity], t_values=t_values,
                                  shot_values=shot_values, n_repeats=n_repeats)[0]
