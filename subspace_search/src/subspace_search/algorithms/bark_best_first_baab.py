from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

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
        Aggregated amplitudes for all bitstrings in this pool.
    unexpanded:
        Bitstrings from this pool that have not yet been selected for H-application.
    parent_round:
        Round index that created this pool.
    """
    amplitudes: Dict[str, complex]
    unexpanded: Set[str]
    parent_round: int


class BarkBarkBark:
    """
    Classical bitstring expansion under a Pauli Hamiltonian.

    Two selection strategies are supported:

    1. selection_strategy="pool"
       Old behavior. Select from the newest candidate pool first, and backtrack
       to older pools only when the newest pool is exhausted.

    2. selection_strategy="best_first"
       New behavior. Maintain one global frontier of all discovered-but-unexpanded
       bitstrings and always select from the globally best candidates.

    Candidate scoring is controlled by score_mode:

    - score_mode="amplitude"
        score(b) = |candidate_amplitude(b)|

    - score_mode="probability"
        score(b) = |candidate_amplitude(b)|^2

    - score_mode="coupling"
        score(b) = |<b|H|g>|, where |g> is the current approximate ground state
        in the already-expanded subspace. Falls back to amplitude scoring before
        a ground approximation is available.

    - score_mode="perturbative"
        score(b) = |<b|H|g>|^2 / (|H_bb - E_g| + perturbative_epsilon)
        where E_g and |g> are obtained from the current projected subspace.
        Falls back to amplitude scoring before a ground approximation is available.

    - score_mode="two_dimensional"
        score(b) = E_g - E_-(b), where E_-(b) is the lower eigenvalue of
        the 2D Hamiltonian in span{|g>, |b>}. Here |g> is updated greedily
        by 2D diagonalizations when new bitstrings are accepted, so this mode
        does not require repeated full-subspace diagonalization.

    The default options reproduce the old pool-based top-M behavior.
    """

    def __init__(
        self,
        H: SparsePauliOp,
        initial_state: str,
        keep_states: int,
        max_applications: int,
        mode: str = "top_m",                         # "top_m" or "importance_sample"
        sample_size: Optional[int] = None,
        sampling_score: str = "amplitude",            # kept for backward compatibility
        restrict_equal_ones_zeros: bool = False,
        return_only_applied_bitstrings: bool = False,
        selection_strategy: str = "pool",              # "pool" or "best_first"
        score_mode: str = "amplitude",                 # "amplitude", "probability", "coupling", "perturbative", "two_dimensional"
        perturbative_epsilon: float = 1e-12,
        ground_update_interval: int = 1,
        aggregate_frontier_amplitudes: bool = True,
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
        if selection_strategy not in {"pool", "best_first"}:
            raise ValueError("selection_strategy must be 'pool' or 'best_first'")
        if score_mode not in {"amplitude", "probability", "coupling", "perturbative", "two_dimensional"}:
            raise ValueError("score_mode must be 'amplitude', 'probability', 'coupling', 'perturbative', or 'two_dimensional'")
        if perturbative_epsilon <= 0:
            raise ValueError("perturbative_epsilon must be positive")
        if ground_update_interval <= 0:
            raise ValueError("ground_update_interval must be positive")

        self.H = H
        self.initial_state = initial_state
        self.n = len(initial_state)
        self.keep_states = keep_states
        self.max_applications = max_applications
        self.mode = mode
        self.sample_size = sample_size
        self.sampling_score = sampling_score
        self.restrict_equal_ones_zeros = restrict_equal_ones_zeros
        self.return_only_applied_bitstrings = return_only_applied_bitstrings
        self.selection_strategy = selection_strategy
        self.score_mode = score_mode
        self.perturbative_epsilon = perturbative_epsilon
        self.ground_update_interval = ground_update_interval
        self.aggregate_frontier_amplitudes = aggregate_frontier_amplitudes
        self.rng = np.random.default_rng(random_seed)

        if self.restrict_equal_ones_zeros and (self.n % 2 != 0):
            raise ValueError("Equal numbers of zeros and ones is only possible for an even number of qubits.")
        if not self._passes_bitstring_filter(initial_state):
            raise ValueError("Initial state does not have equal numbers of zeros and ones.")

        self.terms = self._compile_terms()

        # Current projected-ground-state data, refreshed during run() when needed.
        self._ground_basis: List[str] = []
        self._ground_energy: Optional[float] = None
        self._ground_state: Optional[Dict[str, complex]] = None
        self._ground_couplings: Dict[str, complex] = {}

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

    def _passes_bitstring_filter(self, bitstring: str) -> bool:
        if not self.restrict_equal_ones_zeros:
            return True
        return bitstring.count("1") == len(bitstring) // 2

    def _phase(self, bits: int, term: PauliTerm) -> complex:
        sign = -1 if ((bits & term.phase_mask).bit_count() % 2) else 1
        return (1j ** term.y_count) * sign

    def apply_hamiltonian(self, state: Dict[str, complex]) -> Dict[str, complex]:
        """
        Apply H once to a sparse batch of basis states.

        Returns
        -------
        Dict[str, complex]
            child_bitstring -> aggregated amplitude
        """
        children: Dict[str, complex] = defaultdict(complex)

        for parent_str, parent_amp in state.items():
            parent_bits = int(parent_str, 2)

            for term in self.terms:
                child_bits = parent_bits ^ term.flip_mask
                child_str = format(child_bits, f"0{self.n}b")

                if not self._passes_bitstring_filter(child_str):
                    continue

                contribution = term.coeff * self._phase(parent_bits, term) * parent_amp
                children[child_str] += contribution

        return dict(children)

    def diagonal_element(self, bitstring: str) -> float:
        """Return H_bb = <b|H|b>."""
        bits = int(bitstring, 2)
        value = 0.0 + 0.0j

        for term in self.terms:
            if term.flip_mask != 0:
                continue
            value += term.coeff * self._phase(bits, term)

        return float(np.real_if_close(value).real)

    # ------------------------------------------------------------------
    # Projected ground-state helper for coupling / perturbative scoring
    # ------------------------------------------------------------------
    def project_hamiltonian_to_subspace(self, basis: Sequence[str]) -> np.ndarray:
        """
        Build dense projected Hamiltonian in the supplied bitstring basis.
        H_sub[i, j] = <basis[i]|H|basis[j]>.
        """
        basis = list(basis)
        dim = len(basis)
        index = {b: i for i, b in enumerate(basis)}
        H_sub = np.zeros((dim, dim), dtype=np.complex128)

        for j, ket in enumerate(basis):
            column = self.apply_hamiltonian({ket: 1.0 + 0.0j})
            for bra, amp in column.items():
                i = index.get(bra)
                if i is not None:
                    H_sub[i, j] += amp

        # Numerical safety. SparsePauliOp should represent a Hermitian H for this use-case.
        H_sub = 0.5 * (H_sub + H_sub.conj().T)
        return H_sub

    def update_ground_approximation(self, basis: Sequence[str]) -> None:
        """
        Diagonalize the currently expanded bitstring subspace and cache:
          - E_g
          - |g> as a sparse dict over bitstrings
          - H|g>, so <b|H|g> is available for candidate b
        """
        unique_basis = list(dict.fromkeys(basis))
        if not unique_basis:
            self._ground_basis = []
            self._ground_energy = None
            self._ground_state = None
            self._ground_couplings = {}
            return

        H_sub = self.project_hamiltonian_to_subspace(unique_basis)
        evals, evecs = np.linalg.eigh(H_sub)
        ground_idx = int(np.argmin(evals))

        Eg = float(np.real(evals[ground_idx]))
        vec = evecs[:, ground_idx]

        g_state = {
            b: complex(vec[i])
            for i, b in enumerate(unique_basis)
            if abs(vec[i]) > 1e-14
        }

        self._ground_basis = unique_basis
        self._ground_energy = Eg
        self._ground_state = g_state
        self._ground_couplings = self.apply_hamiltonian(g_state)

    # ------------------------------------------------------------------
    # Greedy 2D ground-state helper
    # ------------------------------------------------------------------
    def initialize_two_dimensional_ground(self) -> None:
        """
        Initialize the greedy 2D ground approximation with the initial bitstring.

        This is used by score_mode="two_dimensional". It avoids full-subspace
        diagonalization; subsequent accepted bitstrings update this state through
        2D diagonalizations only.
        """
        self._ground_basis = [self.initial_state]
        self._ground_energy = self.diagonal_element(self.initial_state)
        self._ground_state = {self.initial_state: 1.0 + 0.0j}
        self._ground_couplings = self.apply_hamiltonian(self._ground_state)

    def two_dimensional_lower_energy(self, bitstring: str) -> float:
        """
        Return the lower eigenvalue in span{|g>, |bitstring>}.

        Assumes |g> and |bitstring> are orthogonal, which is true for a new
        computational basis bitstring not already in the greedy ground support.
        """
        if self._ground_energy is None or self._ground_state is None:
            self.initialize_two_dimensional_ground()

        Eg = float(self._ground_energy)
        Hbb = self.diagonal_element(bitstring)
        coupling = self._ground_couplings.get(bitstring, 0.0 + 0.0j)  # <b|H|g>

        center = 0.5 * (Eg + Hbb)
        half_gap = 0.5 * (Hbb - Eg)
        return float(center - np.sqrt(half_gap * half_gap + abs(coupling) ** 2))

    def two_dimensional_energy_improvement(self, bitstring: str) -> float:
        """
        Score candidate bitstring by the variational energy decrease obtained
        from the 2D subspace span{|g>, |bitstring>}.
        """
        if self._ground_energy is None or self._ground_state is None:
            self.initialize_two_dimensional_ground()

        e_lower = self.two_dimensional_lower_energy(bitstring)
        return float(max(0.0, self._ground_energy - e_lower))

    def update_two_dimensional_ground_with_bitstring(self, bitstring: str) -> None:
        """
        Greedily update |g> by diagonalizing the 2D subspace span{|g>, |bitstring>}.

        This function does not diagonalize the full accumulated selected subspace.
        It only mixes the current best approximation with one newly accepted
        computational-basis bitstring.
        """
        if self._ground_energy is None or self._ground_state is None:
            self.initialize_two_dimensional_ground()

        # If the candidate already has support in |g>, the 2D orthogonal-basis
        # assumption is invalid. This should not happen in the standard run logic,
        # but skipping is safer than applying an invalid update.
        if bitstring in self._ground_state and abs(self._ground_state[bitstring]) > 1e-14:
            return

        Eg = float(self._ground_energy)
        Hbb = self.diagonal_element(bitstring)
        coupling_bg = self._ground_couplings.get(bitstring, 0.0 + 0.0j)  # <b|H|g>

        H2 = np.array(
            [
                [Eg, np.conj(coupling_bg)],
                [coupling_bg, Hbb],
            ],
            dtype=np.complex128,
        )
        evals, evecs = np.linalg.eigh(H2)
        idx = int(np.argmin(evals))
        new_Eg = float(np.real(evals[idx]))
        alpha = complex(evecs[0, idx])
        beta = complex(evecs[1, idx])

        old_ground = self._ground_state
        old_couplings = self._ground_couplings
        Hb = self.apply_hamiltonian({bitstring: 1.0 + 0.0j})

        new_ground: Dict[str, complex] = {}
        for b, amp in old_ground.items():
            val = alpha * amp
            if abs(val) > 1e-14:
                new_ground[b] = val
        if abs(beta) > 1e-14:
            new_ground[bitstring] = new_ground.get(bitstring, 0.0 + 0.0j) + beta

        # Numerical normalization guard.
        norm = np.sqrt(sum(abs(v) ** 2 for v in new_ground.values()))
        if norm > 0:
            new_ground = {b: amp / norm for b, amp in new_ground.items()}
            alpha = alpha / norm
            beta = beta / norm

        new_couplings: Dict[str, complex] = defaultdict(complex)
        for b, amp in old_couplings.items():
            new_couplings[b] += alpha * amp
        for b, amp in Hb.items():
            new_couplings[b] += beta * amp

        self._ground_energy = new_Eg
        self._ground_state = dict(new_ground)
        self._ground_basis = list(new_ground.keys())
        self._ground_couplings = dict(new_couplings)

    def _update_two_dimensional_ground_after_acceptance(self, chosen: Sequence[str]) -> None:
        if self.score_mode != "two_dimensional":
            return
        for b in chosen:
            self.update_two_dimensional_ground_with_bitstring(b)

    def _maybe_update_ground(self, applied_sequence: Sequence[str], applications_done: int) -> None:
        if self.score_mode not in {"coupling", "perturbative"}:
            return
        if applications_done % self.ground_update_interval != 0:
            return
        self.update_ground_approximation(applied_sequence)

    # ------------------------------------------------------------------
    # Scoring and selection
    # ------------------------------------------------------------------
    def _fallback_score(self, bitstring: str, candidate_amplitudes: Dict[str, complex]) -> float:
        amp = candidate_amplitudes.get(bitstring, 0.0 + 0.0j)
        if self.score_mode == "probability":
            return float(abs(amp) ** 2)
        return float(abs(amp))

    def _candidate_score(self, bitstring: str, candidate_amplitudes: Dict[str, complex]) -> float:
        """
        Score a candidate bitstring according to score_mode.

        For coupling/perturbative scoring, this uses the latest cached ground
        approximation. If no ground approximation exists yet, it falls back to
        amplitude scoring.
        """
        if self.score_mode in {"amplitude", "probability"}:
            return self._fallback_score(bitstring, candidate_amplitudes)

        if self.score_mode == "two_dimensional":
            return self.two_dimensional_energy_improvement(bitstring)

        if self._ground_energy is None or self._ground_state is None:
            return float(abs(candidate_amplitudes.get(bitstring, 0.0 + 0.0j)))

        coupling = self._ground_couplings.get(bitstring, 0.0 + 0.0j)

        if self.score_mode == "coupling":
            return float(abs(coupling))

        # Perturbative selected-CI-style energy-lowering proxy.
        Hbb = self.diagonal_element(bitstring)
        denom = abs(Hbb - self._ground_energy) + self.perturbative_epsilon
        return float((abs(coupling) ** 2) / denom)

    def _weights_from_scores(self, scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(scores, dtype=float)
        scores = np.maximum(scores, 0.0)

        if not np.any(scores > 0):
            return np.ones_like(scores) / len(scores)

        return scores / scores.sum()

    def _choose_from_candidates(
        self,
        candidate_amplitudes: Dict[str, complex],
        eligible: List[str],
    ) -> Dict[str, complex]:
        """
        Choose up to keep_states candidates from eligible bitstrings.

        mode="top_m": deterministic top-M by current score.
        mode="importance_sample": sample by current score, then keep top-M within sampled set.
        """
        if not eligible:
            return {}

        scores = np.array(
            [self._candidate_score(b, candidate_amplitudes) for b in eligible],
            dtype=float,
        )

        if self.mode == "top_m":
            order = np.argsort(-scores, kind="stable")
            chosen = [eligible[i] for i in order[: self.keep_states]]
            return {b: candidate_amplitudes[b] for b in chosen}

        # importance_sample mode
        draw_size = self.sample_size if self.sample_size is not None else self.keep_states
        draw_size = min(draw_size, len(eligible))

        if draw_size >= len(eligible):
            sampled = list(eligible)
        else:
            probs = self._weights_from_scores(scores)
            sampled_idx = self.rng.choice(
                len(eligible),
                size=draw_size,
                replace=False,
                p=probs,
            )
            sampled = [eligible[i] for i in sampled_idx]

        sampled.sort(
            key=lambda b: self._candidate_score(b, candidate_amplitudes),
            reverse=True,
        )
        chosen = sampled[: self.keep_states]
        return {b: candidate_amplitudes[b] for b in chosen}

    def _choose_from_pool(self, pool: CandidatePool, applied_set: Set[str]) -> Dict[str, complex]:
        eligible = [b for b in pool.unexpanded if b not in applied_set]
        return self._choose_from_candidates(pool.amplitudes, eligible)

    def _find_backtrack_pool(self, pools: List[CandidatePool], applied_set: Set[str]) -> Optional[int]:
        for idx, pool in enumerate(pools):
            for b in pool.unexpanded:
                if b not in applied_set:
                    return idx
        return None

    # ------------------------------------------------------------------
    # Frontier bookkeeping
    # ------------------------------------------------------------------
    def _add_to_frontier(
        self,
        frontier: Dict[str, complex],
        children: Dict[str, complex],
        applied_set: Set[str],
    ) -> None:
        for b, amp in children.items():
            if b in applied_set:
                continue
            if self.aggregate_frontier_amplitudes:
                frontier[b] = frontier.get(b, 0.0 + 0.0j) + amp
            else:
                frontier[b] = amp

    # ------------------------------------------------------------------
    # Main run methods
    # ------------------------------------------------------------------
    def run(self) -> List[str]:
        if self.selection_strategy == "pool":
            return self._run_pool_strategy()
        return self._run_best_first_strategy()

    def _run_pool_strategy(self) -> List[str]:
        """Old newest-pool-first behavior, now with optional improved scoring."""
        encountered: Set[str] = {self.initial_state}
        encountered_order: List[str] = [self.initial_state]

        # The initial state is expanded immediately to create the first pool.
        applied_set: Set[str] = {self.initial_state}
        applied_sequence: List[str] = [self.initial_state]

        if self.score_mode == "two_dimensional":
            self.initialize_two_dimensional_ground()

        first_pool_amps = self.apply_hamiltonian({self.initial_state: 1.0 + 0.0j})
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

        current_pool_idx: Optional[int] = 0
        applications_done = 0

        while applications_done < self.max_applications:
            self._maybe_update_ground(applied_sequence, applications_done)

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

            accepted_now = list(chosen.keys())
            for b in accepted_now:
                applied_sequence.append(b)
                applied_set.add(b)
                current_pool.unexpanded.discard(b)

            self._update_two_dimensional_ground_after_acceptance(accepted_now)

            next_pool_amps = self.apply_hamiltonian(chosen)
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

        if self.return_only_applied_bitstrings:
            return applied_sequence
        return encountered_order

    def _run_best_first_strategy(self) -> List[str]:
        """
        New global best-first behavior.

        Maintains one global frontier of discovered-but-unexpanded bitstrings.
        At every round, it selects the globally best keep_states candidates.
        """
        encountered: Set[str] = {self.initial_state}
        encountered_order: List[str] = [self.initial_state]

        applied_set: Set[str] = {self.initial_state}
        applied_sequence: List[str] = [self.initial_state]

        if self.score_mode == "two_dimensional":
            self.initialize_two_dimensional_ground()

        frontier: Dict[str, complex] = {}

        first_children = self.apply_hamiltonian({self.initial_state: 1.0 + 0.0j})
        for b in first_children:
            if b not in encountered:
                encountered.add(b)
                encountered_order.append(b)
        self._add_to_frontier(frontier, first_children, applied_set)

        applications_done = 0

        while applications_done < self.max_applications and frontier:
            self._maybe_update_ground(applied_sequence, applications_done)

            eligible = [b for b in frontier if b not in applied_set]
            if not eligible:
                break

            chosen = self._choose_from_candidates(frontier, eligible)
            if not chosen:
                break

            accepted_now = list(chosen.keys())
            for b in accepted_now:
                frontier.pop(b, None)
                applied_set.add(b)
                applied_sequence.append(b)

            self._update_two_dimensional_ground_after_acceptance(accepted_now)

            next_children = self.apply_hamiltonian(chosen)
            for b in next_children:
                if b not in encountered:
                    encountered.add(b)
                    encountered_order.append(b)
            self._add_to_frontier(frontier, next_children, applied_set)

            applications_done += 1

        if self.return_only_applied_bitstrings:
            return applied_sequence
        return encountered_order
