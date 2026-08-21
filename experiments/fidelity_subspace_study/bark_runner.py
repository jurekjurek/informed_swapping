"""Instrumented BARK driver built on ``bark_best_first_baab.BarkBarkBark``.

The packaged class runs to completion and returns an ordering. This study needs
the run broken into *rounds* so that, exactly like SKQD, the subspace is
diagonalised once per iteration and the run stops as soon as a target fidelity
is reached.

:class:`InstrumentedBark` subclasses the packaged implementation and re-expresses
its two selection strategies as generators that yield one :class:`BarkRound` per
Hamiltonian application, with the wall-clock cost split into

``t_ground_update``  BARK's own ground-state refresh (a full subspace
                     diagonalisation for ``score_mode`` in {coupling,
                     perturbative}; a cheap 2x2 solve for ``two_dimensional``)
``t_select``         candidate scoring + top-M / importance selection
``t_expand``         applying H to the accepted bitstrings
``t_frontier_update``bookkeeping of the frontier / candidate pools

Note on double counting: for ``coupling``/``perturbative`` scoring, BARK's
internal refresh diagonalises the same subspace the study diagonalises for the
energy readout (``t_project`` + ``t_diagonalize``). Both numbers are stored
separately and unmodified; a production implementation would share the work, so
analyses that want a "shared" cost should not simply add the two columns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Set

import numpy as np

from subspace_search.algorithms import BarkBestFirstPerturbative
from subspace_search.algorithms.bark_best_first_baab import CandidatePool

from hamiltonian_cases import HamiltonianCase, InitialState
from subspace_metrics import timed


@dataclass
class BarkConfig:
    score_mode: str
    selection_strategy: str
    keep_states: int
    max_iterations: int
    mode: str = "top_m"
    sample_size: int | None = None
    perturbative_epsilon: float = 1e-12
    ground_update_interval: int = 1
    aggregate_frontier_amplitudes: bool = True
    restrict_equal_ones_zeros: bool = False
    subspace: str = "applied"          # "applied" or "encountered"
    seed: int = 0
    repeat: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "bark_score_mode": self.score_mode,
            "bark_selection_strategy": self.selection_strategy,
            "bark_keep_states": self.keep_states,
            "bark_max_iterations": self.max_iterations,
            "bark_mode": self.mode,
            "bark_sample_size": (
                -1 if self.sample_size is None else int(self.sample_size)
            ),
            "bark_ground_update_interval": self.ground_update_interval,
            "bark_subspace": self.subspace,
            "repeat": self.repeat,
            "run_seed": self.seed,
        }


@dataclass
class BarkRound:
    round_index: int
    accepted: List[str]
    applied_sequence: List[str]
    encountered_order: List[str]
    frontier_size: int
    timings: Dict[str, float] = field(default_factory=dict)


class InstrumentedBark(BarkBestFirstPerturbative):
    """BARK, yielding one timed round per Hamiltonian application."""

    def iterate(self) -> Iterator[BarkRound]:
        if self.selection_strategy == "pool":
            yield from self._iterate_pool()
        else:
            yield from self._iterate_best_first()

    # ------------------------------------------------------------------
    def _seed_round(self, timings: Dict[str, float]) -> Dict[str, complex]:
        """Set up the greedy ground state (if needed) and expand the root."""
        if self.score_mode == "two_dimensional":
            with timed(timings, "t_ground_update"):
                self.initialize_two_dimensional_ground()
        with timed(timings, "t_expand"):
            children = self.apply_hamiltonian({self.initial_state: 1.0 + 0.0j})
        return children

    # ------------------------------------------------------------------
    def _iterate_pool(self) -> Iterator[BarkRound]:
        encountered: Set[str] = {self.initial_state}
        encountered_order: List[str] = [self.initial_state]
        applied_set: Set[str] = {self.initial_state}
        applied_sequence: List[str] = [self.initial_state]

        timings: Dict[str, float] = {}
        first_pool_amps = self._seed_round(timings)
        with timed(timings, "t_frontier_update"):
            for b in first_pool_amps:
                if b not in encountered:
                    encountered.add(b)
                    encountered_order.append(b)
            pools: List[CandidatePool] = [
                CandidatePool(
                    amplitudes=first_pool_amps,
                    unexpanded=set(first_pool_amps.keys()),
                    parent_round=0,
                )
            ]

        yield BarkRound(
            round_index=0,
            accepted=[self.initial_state],
            applied_sequence=list(applied_sequence),
            encountered_order=list(encountered_order),
            frontier_size=len(first_pool_amps),
            timings=timings,
        )

        current_pool_idx: int | None = 0
        applications_done = 0
        timings = {}

        while applications_done < self.max_applications:
            with timed(timings, "t_ground_update"):
                self._maybe_update_ground(applied_sequence, applications_done)

            with timed(timings, "t_select"):
                if current_pool_idx is None or current_pool_idx >= len(pools):
                    backtrack_idx = self._find_backtrack_pool(pools, applied_set)
                    if backtrack_idx is None:
                        return
                    current_pool_idx = backtrack_idx
                current_pool = pools[current_pool_idx]
                chosen = self._choose_from_pool(current_pool, applied_set)
                if not chosen:
                    backtrack_idx = self._find_backtrack_pool(pools, applied_set)
                    if backtrack_idx is None:
                        return
                    current_pool_idx = backtrack_idx
                    retry = True
                else:
                    retry = False
            if retry:
                # Backtracking consumed no Hamiltonian application; its cost is
                # rolled into the next emitted round.
                continue

            accepted_now = list(chosen.keys())
            with timed(timings, "t_frontier_update"):
                for b in accepted_now:
                    applied_sequence.append(b)
                    applied_set.add(b)
                    current_pool.unexpanded.discard(b)

            with timed(timings, "t_ground_update"):
                self._update_two_dimensional_ground_after_acceptance(accepted_now)

            with timed(timings, "t_expand"):
                next_pool_amps = self.apply_hamiltonian(chosen)

            with timed(timings, "t_frontier_update"):
                for b in next_pool_amps:
                    if b not in encountered:
                        encountered.add(b)
                        encountered_order.append(b)
                pools.append(
                    CandidatePool(
                        amplitudes=next_pool_amps,
                        unexpanded=set(next_pool_amps.keys()),
                        parent_round=applications_done + 1,
                    )
                )

            current_pool_idx = len(pools) - 1
            applications_done += 1

            yield BarkRound(
                round_index=applications_done,
                accepted=accepted_now,
                applied_sequence=list(applied_sequence),
                encountered_order=list(encountered_order),
                frontier_size=sum(len(p.unexpanded) for p in pools),
                timings=timings,
            )
            timings = {}

    # ------------------------------------------------------------------
    def _iterate_best_first(self) -> Iterator[BarkRound]:
        encountered: Set[str] = {self.initial_state}
        encountered_order: List[str] = [self.initial_state]
        applied_set: Set[str] = {self.initial_state}
        applied_sequence: List[str] = [self.initial_state]

        timings: Dict[str, float] = {}
        first_children = self._seed_round(timings)
        frontier: Dict[str, complex] = {}
        with timed(timings, "t_frontier_update"):
            for b in first_children:
                if b not in encountered:
                    encountered.add(b)
                    encountered_order.append(b)
            self._add_to_frontier(frontier, first_children, applied_set)

        yield BarkRound(
            round_index=0,
            accepted=[self.initial_state],
            applied_sequence=list(applied_sequence),
            encountered_order=list(encountered_order),
            frontier_size=len(frontier),
            timings=timings,
        )

        applications_done = 0
        timings = {}

        while applications_done < self.max_applications and frontier:
            with timed(timings, "t_ground_update"):
                self._maybe_update_ground(applied_sequence, applications_done)

            with timed(timings, "t_select"):
                eligible = [b for b in frontier if b not in applied_set]
                chosen = self._choose_from_candidates(frontier, eligible) if eligible else {}
            if not chosen:
                return

            accepted_now = list(chosen.keys())
            with timed(timings, "t_frontier_update"):
                for b in accepted_now:
                    frontier.pop(b, None)
                    applied_set.add(b)
                    applied_sequence.append(b)

            with timed(timings, "t_ground_update"):
                self._update_two_dimensional_ground_after_acceptance(accepted_now)

            with timed(timings, "t_expand"):
                next_children = self.apply_hamiltonian(chosen)

            with timed(timings, "t_frontier_update"):
                for b in next_children:
                    if b not in encountered:
                        encountered.add(b)
                        encountered_order.append(b)
                self._add_to_frontier(frontier, next_children, applied_set)

            applications_done += 1

            yield BarkRound(
                round_index=applications_done,
                accepted=accepted_now,
                applied_sequence=list(applied_sequence),
                encountered_order=list(encountered_order),
                frontier_size=len(frontier),
                timings=timings,
            )
            timings = {}


# ----------------------------------------------------------------------
def run_bark(
    case: HamiltonianCase,
    initial: InitialState,
    config: BarkConfig,
    stop_fidelity: float,
    max_subspace_fraction: float = 1.0,
) -> list[dict[str, Any]]:
    """Run one BARK trajectory and return one record per round."""
    bark = InstrumentedBark(
        H=case.H_spo,
        initial_state=initial.bitstring,
        keep_states=config.keep_states,
        max_applications=config.max_iterations,
        mode=config.mode,
        sample_size=config.sample_size,
        restrict_equal_ones_zeros=config.restrict_equal_ones_zeros,
        return_only_applied_bitstrings=True,
        selection_strategy=config.selection_strategy,
        score_mode=config.score_mode,
        perturbative_epsilon=config.perturbative_epsilon,
        ground_update_interval=config.ground_update_interval,
        aggregate_frontier_amplitudes=config.aggregate_frontier_amplitudes,
        random_seed=config.seed,
    )

    dim = case.dim
    max_subspace_dim = max(1, int(round(max_subspace_fraction * dim)))
    internal_diagonalization = config.score_mode in {"coupling", "perturbative"}

    records: list[dict[str, Any]] = []
    previous_dim = 0

    for round_info in bark.iterate():
        bitstrings = (
            round_info.applied_sequence
            if config.subspace == "applied"
            else round_info.encountered_order
        )
        subspace = [int(b, 2) for b in bitstrings][:max_subspace_dim]
        subspace_dim = len(subspace)

        if subspace_dim != previous_dim:
            evaluation = case.evaluator.evaluate(subspace)
            diagonalized = True
        else:
            from dataclasses import replace

            evaluation = replace(
                evaluation, t_project=0.0, t_diagonalize=0.0, t_fidelity=0.0
            )
            diagonalized = False
        previous_dim = subspace_dim

        records.append(
            _record(
                round_info=round_info,
                evaluation=evaluation,
                subspace_dim=subspace_dim,
                applied_dim=len(round_info.applied_sequence),
                encountered_dim=len(round_info.encountered_order),
                dim=dim,
                diagonalized=diagonalized,
                internal_diagonalization=internal_diagonalization,
            )
        )

        if evaluation.fidelity >= stop_fidelity:
            break
        if subspace_dim >= max_subspace_dim:
            break

    return records


def _record(
    round_info: BarkRound,
    evaluation,
    subspace_dim: int,
    applied_dim: int,
    encountered_dim: int,
    dim: int,
    diagonalized: bool,
    internal_diagonalization: bool,
) -> dict[str, Any]:
    t = round_info.timings
    t_ground_update = t.get("t_ground_update", 0.0)
    t_select = t.get("t_select", 0.0)
    t_expand = t.get("t_expand", 0.0)
    t_frontier_update = t.get("t_frontier_update", 0.0)

    t_algorithm = t_ground_update + t_select + t_expand + t_frontier_update
    t_solve = evaluation.t_project + evaluation.t_diagonalize

    return {
        "method": "BARK",
        "iteration": round_info.round_index,
        "krylov_time": float("nan"),
        "subspace_dim": subspace_dim,
        "subspace_fraction": subspace_dim / dim,
        "shots_this_iteration": 0,
        "shots_cumulative": 0,
        "unique_sampled": len(round_info.accepted),
        "new_indices": len(round_info.accepted),
        "applied_dim": applied_dim,
        "encountered_dim": encountered_dim,
        "frontier_size": round_info.frontier_size,
        "diagonalized": diagonalized,
        "internal_diagonalization": internal_diagonalization,
        "energy": evaluation.energy,
        "energy_error": evaluation.energy_error,
        "fidelity": evaluation.fidelity,
        "captured_weight": evaluation.captured_weight,
        # --- timing breakdown (seconds) ---
        "t_propagator_setup": 0.0,
        "t_propagate": 0.0,
        "t_sample": 0.0,
        "t_subspace_update": t_frontier_update,
        "t_select": t_select,
        "t_expand": t_expand,
        "t_ground_update": t_ground_update,
        "t_project": evaluation.t_project,
        "t_diagonalize": evaluation.t_diagonalize,
        "t_fidelity": evaluation.t_fidelity,
        "t_algorithm": t_algorithm,
        "t_solve": t_solve,
        "t_iteration": t_algorithm + t_solve + evaluation.t_fidelity,
        "propagator_dense": False,
        "propagator_cached": False,
    }
