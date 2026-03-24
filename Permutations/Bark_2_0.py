from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from qiskit.quantum_info import SparsePauliOp


@dataclass(frozen=True)
class PauliTerm:
    pauli_string: str
    coeff: complex
    flip_mask: int
    phase_mask: int
    y_count: int


@dataclass
class CandidatePool:
    """
    A pool of bitstrings reached after one Hamiltonian application.

    amplitudes:
        aggregated amplitudes for all bitstrings in this pool
    unexpanded:
        bitstrings from this pool that have not yet been selected for H-application
    parent_round:
        round index that created this pool
    """
    amplitudes: Dict[str, complex]
    unexpanded: Set[str]
    parent_round: int


class BarkBarkBark:
    """
    Hamiltonian application with:
      - per-step candidate generation from the current pool
      - selection of up to M bitstrings from that pool only
      - backtracking to the earliest older pool when the current pool is exhausted
      - skipping bitstrings that already had H applied to them

    Selection modes:
      - mode="top_m": deterministic top-M by |amplitude|
      - mode="importance_sample": sample from the pool, then keep up to M sampled states

    Returns all encountered bitstrings as a list of strings.
    """

    def __init__(
        self,
        H: SparsePauliOp,
        initial_state: str,
        keep_states: int,
        max_applications: int,
        mode: str = "top_m",
        sample_size: Optional[int] = None,
        sampling_score: str = "amplitude",
        restrict_equal_ones_zeros: bool = False,
        return_only_applied_bitstrings: bool = False,
        random_seed: Optional[int] = None,
    ) -> None:
        if keep_states <= 0:
            raise ValueError("keep_states must be positive")
        if max_applications < 0:
            raise ValueError("max_applications must be non-negative")
        if mode not in {"top_m", "importance_sample"}:
            raise ValueError("mode must be 'top_m' or 'importance_sample'")
        if sampling_score not in {"amplitude", "probability"}:
            raise ValueError("sampling_score must be 'amplitude' or 'probability'")

        self.H = H
        self.initial_state = initial_state
        self.n = len(initial_state)
        self.keep_states = keep_states
        self.max_applications = max_applications
        self.mode = mode
        self.sample_size = sample_size
        self.sampling_score = sampling_score
        self.rng = np.random.default_rng(random_seed)

        self.return_only_applied_bitstrings = return_only_applied_bitstrings

        self.restrict_equal_ones_zeros = restrict_equal_ones_zeros
        if self.restrict_equal_ones_zeros and (self.n % 2 != 0):
            raise ValueError("Equal numbers of zeros and ones is only possible for an even number of qubits.")

        # check if initial state passes zero charge criterion
        if not self._passes_bitstring_filter(initial_state):
            raise ValueError("Initial state does not have equal numbers of zeros and ones.")
        self.terms = self._compile_terms()

    # ------------------------------------------------------------------
    # Pauli / Hamiltonian internals
    # ------------------------------------------------------------------
    def _compile_terms(self) -> List[PauliTerm]:
        terms: List[PauliTerm] = []

        for pauli_string, coeff in self.H.to_list():
            flip_mask = 0
            phase_mask = 0
            y_count = 0

            for i, p in enumerate(pauli_string):
                bit = 1 << (self.n - 1 - i)

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

            terms.append(
                PauliTerm(
                    pauli_string=pauli_string,
                    coeff=complex(coeff),
                    flip_mask=flip_mask,
                    phase_mask=phase_mask,
                    y_count=y_count,
                )
            )

        return terms

    # helper to ensure zero charge sector
    def _passes_bitstring_filter(self, bitstring: str) -> bool:
        if not self.restrict_equal_ones_zeros:
            return True
        return bitstring.count("1") == len(bitstring) // 2

    def _phase(self, bits: int, term: PauliTerm) -> complex:
        sign = -1 if ((bits & term.phase_mask).bit_count() % 2) else 1
        return (1j ** term.y_count) * sign

    def apply_hamiltonian(
        self,
        state: Dict[str, complex],
    ) -> Dict[str, complex]:
        """
        Apply H once to a sparse batch of basis states.

        Returns aggregated amplitudes:
            child_bitstring -> total amplitude
        """
        children: Dict[str, complex] = defaultdict(complex)

        for parent_str, parent_amp in state.items():
            parent_bits = int(parent_str, 2)

            for term in self.terms:
                child_bits = parent_bits ^ term.flip_mask
                child_str = format(child_bits, f"0{self.n}b")

                # Discard immediately if it fails the bitstring constraint
                if not self._passes_bitstring_filter(child_str):
                    continue

                contribution = term.coeff * self._phase(parent_bits, term) * parent_amp
                children[child_str] += contribution

        return dict(children)

    # ------------------------------------------------------------------
    # Selection logic
    # ------------------------------------------------------------------
    def _weights(self, amps: Dict[str, complex], keys: List[str]) -> np.ndarray:
        vals = np.array([abs(amps[k]) for k in keys], dtype=float)
        if self.sampling_score == "probability":
            vals = vals ** 2

        if not np.any(vals > 0):
            vals = np.ones_like(vals)

        return vals / vals.sum()

    def _choose_from_pool(
        self,
        pool: CandidatePool,
        applied_set: Set[str],
    ) -> Dict[str, complex]:
        """
        Choose up to M bitstrings from this pool that:
          - are still unexpanded in this pool
          - have never had H applied to them globally
        """
        eligible = [b for b in pool.unexpanded if b not in applied_set]
        if not eligible:
            return {}

        if self.mode == "top_m":
            eligible.sort(key=lambda b: abs(pool.amplitudes[b]), reverse=True)
            chosen = eligible[:self.keep_states]
            return {b: pool.amplitudes[b] for b in chosen}

        # importance_sample
        draw_size = self.sample_size if self.sample_size is not None else self.keep_states
        draw_size = min(draw_size, len(eligible))

        probs = self._weights(pool.amplitudes, eligible)

        # Only sample from entries with strictly positive probability
        positive_idx = np.flatnonzero(probs > 0)

        if len(positive_idx) == 0:
            sampled = []
        elif draw_size >= len(positive_idx):
            sampled = [eligible[i] for i in positive_idx]
        else:
            positive_probs = probs[positive_idx]
            positive_probs = positive_probs / positive_probs.sum()

            chosen_rel = self.rng.choice(
                len(positive_idx),
                size=draw_size,
                replace=False,
                p=positive_probs,
            )
            sampled = [eligible[positive_idx[i]] for i in chosen_rel]

        sampled.sort(key=lambda b: abs(pool.amplitudes[b]), reverse=True)
        chosen = sampled[:self.keep_states]
        return {b: pool.amplitudes[b] for b in chosen}

    def _find_backtrack_pool(
        self,
        pools: List[CandidatePool],
        applied_set: Set[str],
    ) -> Optional[int]:
        """
        Find the earliest pool with at least one bitstring that:
          - remains unexpanded in that pool
          - has not had H applied to it globally
        """
        for idx, pool in enumerate(pools):
            for b in pool.unexpanded:
                if b not in applied_set:
                    return idx
        return None

    # ------------------------------------------------------------------
    # Main run
    # ------------------------------------------------------------------
    def run(self) -> List[str]:
        """
        Returns
        -------
        List[str]
            All encountered bitstrings, as strings.
        """
        encountered: Set[str] = {self.initial_state}
        applied_set: Set[str] = set()

        # Step 0: apply H to the initial state to create the first pool.
        first_pool_amps = self.apply_hamiltonian({self.initial_state: 1.0 + 0.0j})
        encountered.update(first_pool_amps.keys())

        pools: List[CandidatePool] = [
            CandidatePool(
                amplitudes=first_pool_amps,
                unexpanded=set(first_pool_amps.keys()),
                parent_round=0,
            )
        ]

        current_pool_idx = 0
        applications_done = 0

        while applications_done < self.max_applications:
            if current_pool_idx is None or current_pool_idx >= len(pools):
                backtrack_idx = self._find_backtrack_pool(pools, applied_set)
                if backtrack_idx is None:
                    break
                current_pool_idx = backtrack_idx

            current_pool = pools[current_pool_idx]
            chosen = self._choose_from_pool(current_pool, applied_set)

            if not chosen:
                backtrack_idx = self._find_backtrack_pool(pools, applied_set)
                if backtrack_idx is None:
                    break
                current_pool_idx = backtrack_idx
                continue

            # Mark chosen states as expanded globally and within this pool.
            for b in chosen:
                applied_set.add(b)
                current_pool.unexpanded.discard(b)

            # Apply H to the chosen batch and create the next "just reached" pool.
            next_pool_amps = self.apply_hamiltonian(chosen)
            encountered.update(next_pool_amps.keys())

            pools.append(
                CandidatePool(
                    amplitudes=next_pool_amps,
                    unexpanded=set(next_pool_amps.keys()),
                    parent_round=applications_done + 1,
                )
            )

            # Continue from the newest pool first.
            current_pool_idx = len(pools) - 1
            applications_done += 1

        # I only want to consider some of the bitstrings in this case...
        if self.return_only_applied_bitstrings:
            return sorted(applied_set)
        return sorted(encountered)