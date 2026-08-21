"""Shot-based SKQD with one diagonalisation per Krylov iteration.

This is deliberately *not* ``subspace_search.skqd.do_skqd``: that routine returns
an ordering of the whole Hilbert space and leaves the caller to diagonalise once
per bitstring. For this study we need the real experimental protocol,

    for k = 1 .. K:
        |psi_k> = exp(-i H dt) |psi_{k-1}>
        draw `shots` measurements from |<i|psi_k>|^2
        S <- S union {sampled indices}
        diagonalise H|_S            <- exactly once per iteration

so that the subspace dimension grows in measurable steps and ``dt`` and ``shots``
are the two knobs the study optimises over.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import scipy.sparse as sp
from scipy.linalg import expm
from scipy.sparse.linalg import expm_multiply

from hamiltonian_cases import HamiltonianCase, InitialState
from subspace_metrics import timed


@dataclass
class SkqdConfig:
    dt: float
    shots: int
    max_iterations: int
    repeat: int = 0
    seed: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "skqd_dt": self.dt,
            "skqd_shots": self.shots,
            "skqd_max_iterations": self.max_iterations,
            "repeat": self.repeat,
            "run_seed": self.seed,
        }


class Propagator:
    """``exp(-i H dt)`` applied to a vector, cached per ``dt`` within a case.

    Small Hilbert spaces get a dense matrix exponential (built once, reused for
    every ``shots`` value and every repeat); larger ones use ``expm_multiply``
    which never forms the propagator. The one-off construction cost is reported
    separately from the per-iteration application cost so it can be amortised
    honestly in the analysis.
    """

    def __init__(self, H: sp.csr_matrix, dt: float, dense_max_dim: int) -> None:
        self.dt = dt
        self.dim = H.shape[0]
        self.dense = self.dim <= dense_max_dim
        self.setup_seconds = 0.0
        self._generator = (-1j * dt) * H

        if self.dense:
            timings: dict[str, float] = {}
            with timed(timings, "setup"):
                self._matrix = expm(np.asarray(self._generator.todense()))
            self.setup_seconds = timings["setup"]
        else:
            self._matrix = None

    def apply(self, state: np.ndarray) -> np.ndarray:
        if self.dense:
            return self._matrix @ state
        return expm_multiply(self._generator, state)


def make_propagator_cache() -> dict[float, Propagator]:
    return {}


def get_propagator(
    cache: dict[float, Propagator],
    H: sp.csr_matrix,
    dt: float,
    dense_max_dim: int,
) -> tuple[Propagator, bool]:
    """Return ``(propagator, was_cached)``."""
    if dt in cache:
        return cache[dt], True
    propagator = Propagator(H, dt, dense_max_dim)
    cache[dt] = propagator
    return propagator, False


def run_skqd(
    case: HamiltonianCase,
    initial: InitialState,
    config: SkqdConfig,
    stop_fidelity: float,
    max_subspace_fraction: float = 1.0,
    dense_propagator_max_dim: int = 2048,
    propagator_cache: dict[float, Propagator] | None = None,
) -> list[dict[str, Any]]:
    """Run one SKQD trajectory and return one record per Krylov iteration."""
    rng = np.random.default_rng(config.seed)
    dim = case.dim
    max_subspace_dim = max(1, int(round(max_subspace_fraction * dim)))

    cache = propagator_cache if propagator_cache is not None else make_propagator_cache()
    propagator, was_cached = get_propagator(
        cache, case.H, config.dt, dense_propagator_max_dim
    )
    setup_seconds = 0.0 if was_cached else propagator.setup_seconds

    state = np.zeros(dim, dtype=np.complex128)
    state[initial.index] = 1.0

    subspace: list[int] = [initial.index]
    in_subspace = np.zeros(dim, dtype=bool)
    in_subspace[initial.index] = True

    # Iteration 0: the bare initial state, before any time evolution.
    evaluation = case.evaluator.evaluate(subspace)
    records: list[dict[str, Any]] = [
        _record(
            iteration=0,
            config=config,
            evaluation=evaluation,
            subspace_dim=1,
            dim=dim,
            shots_cumulative=0,
            shots_this_iteration=0,
            unique_sampled=1,
            new_indices=1,
            diagonalized=True,
            timings={"t_propagator_setup": setup_seconds},
            propagator_dense=propagator.dense,
            propagator_cached=was_cached,
        )
    ]

    best_fidelity = evaluation.fidelity
    if best_fidelity >= stop_fidelity:
        return records

    for iteration in range(1, config.max_iterations + 1):
        timings: dict[str, float] = {}

        with timed(timings, "t_propagate"):
            state = propagator.apply(state)

        with timed(timings, "t_sample"):
            probabilities = np.abs(state) ** 2
            probabilities = np.nan_to_num(probabilities)
            total = probabilities.sum()
            if total <= 0:
                probabilities = np.full(dim, 1.0 / dim)
            else:
                probabilities = probabilities / total
            draws = rng.choice(dim, size=config.shots, replace=True, p=probabilities)
            sampled = np.unique(draws)

        with timed(timings, "t_subspace_update"):
            fresh = sampled[~in_subspace[sampled]]
            if fresh.size:
                room = max_subspace_dim - len(subspace)
                if fresh.size > room:
                    # Respect the subspace cap: keep the most probable newcomers.
                    order = np.argsort(probabilities[fresh])[::-1]
                    fresh = fresh[order[:room]]
                in_subspace[fresh] = True
                subspace.extend(int(i) for i in fresh)

        if fresh.size:
            evaluation = case.evaluator.evaluate(subspace)
            diagonalized = True
        else:
            # Nothing new was sampled -- an unchanged subspace has an unchanged
            # solution, so no algorithm would pay to re-diagonalise it.
            evaluation = _carry_forward(evaluation)
            diagonalized = False

        records.append(
            _record(
                iteration=iteration,
                config=config,
                evaluation=evaluation,
                subspace_dim=len(subspace),
                dim=dim,
                shots_cumulative=iteration * config.shots,
                shots_this_iteration=config.shots,
                unique_sampled=int(sampled.size),
                new_indices=int(fresh.size),
                diagonalized=diagonalized,
                timings=timings,
                propagator_dense=propagator.dense,
                propagator_cached=was_cached,
            )
        )

        best_fidelity = max(best_fidelity, evaluation.fidelity)
        if best_fidelity >= stop_fidelity:
            break
        if len(subspace) >= max_subspace_dim:
            break

    return records


# ----------------------------------------------------------------------
def _carry_forward(evaluation):
    """Copy an evaluation with zeroed timings (subspace did not change)."""
    from dataclasses import replace

    return replace(evaluation, t_project=0.0, t_diagonalize=0.0, t_fidelity=0.0)


def _record(
    iteration: int,
    config: SkqdConfig,
    evaluation,
    subspace_dim: int,
    dim: int,
    shots_cumulative: int,
    shots_this_iteration: int,
    unique_sampled: int,
    new_indices: int,
    diagonalized: bool,
    timings: dict[str, float],
    propagator_dense: bool,
    propagator_cached: bool,
) -> dict[str, Any]:
    t_propagator_setup = timings.get("t_propagator_setup", 0.0)
    t_propagate = timings.get("t_propagate", 0.0)
    t_sample = timings.get("t_sample", 0.0)
    t_subspace_update = timings.get("t_subspace_update", 0.0)

    t_algorithm = t_propagator_setup + t_propagate + t_sample + t_subspace_update
    t_solve = evaluation.t_project + evaluation.t_diagonalize

    return {
        "method": "SKQD",
        "iteration": iteration,
        "krylov_time": iteration * config.dt,
        "subspace_dim": subspace_dim,
        "subspace_fraction": subspace_dim / dim,
        "shots_this_iteration": shots_this_iteration,
        "shots_cumulative": shots_cumulative,
        "unique_sampled": unique_sampled,
        "new_indices": new_indices,
        "applied_dim": subspace_dim,
        "encountered_dim": subspace_dim,
        "frontier_size": 0,
        "diagonalized": diagonalized,
        "internal_diagonalization": False,
        "energy": evaluation.energy,
        "energy_error": evaluation.energy_error,
        "fidelity": evaluation.fidelity,
        "captured_weight": evaluation.captured_weight,
        # --- timing breakdown (seconds) ---
        "t_propagator_setup": t_propagator_setup,
        "t_propagate": t_propagate,
        "t_sample": t_sample,
        "t_subspace_update": t_subspace_update,
        "t_select": 0.0,
        "t_expand": 0.0,
        "t_ground_update": 0.0,
        "t_project": evaluation.t_project,
        "t_diagonalize": evaluation.t_diagonalize,
        "t_fidelity": evaluation.t_fidelity,
        "t_algorithm": t_algorithm,
        "t_solve": t_solve,
        "t_iteration": t_algorithm + t_solve + evaluation.t_fidelity,
        "propagator_dense": propagator_dense,
        "propagator_cached": propagator_cached,
    }
