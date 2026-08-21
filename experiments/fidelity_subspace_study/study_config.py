"""Scan definition for the BARK vs SKQD fidelity/subspace-dimension study.

A :class:`StudyConfig` is the complete description of one scan: which random
spin Hamiltonians to sample, which initial overlaps to start from, which target
fidelities to hit, and the SKQD / BARK parameter grids to optimise over.

Presets are provided so the same script can be used for a two-minute smoke test
and for an overnight production scan.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, field, asdict
from typing import Any, Iterator


@dataclass
class StudyConfig:
    # --- Hamiltonian scan ------------------------------------------------
    num_qubits: list[int] = field(default_factory=lambda: [6, 8, 10])
    # -1 means "no cap" (all-to-all coupling).
    max_interactions: list[int] = field(default_factory=lambda: [1, 2, -1])
    num_hamiltonians: int = 3
    seed_offset: int = 0

    # --- Random spin model parameters -----------------------------------
    J_max: float = 1.0
    B_max: float = 1.0
    J_components: tuple[str, ...] = ("x", "y", "z")
    B_components: tuple[str, ...] = ("x", "y", "z")
    coupling_distribution: str = "uniform"
    field_distribution: str = "uniform"
    spin: float = 0.5

    # --- Shared study parameters ----------------------------------------
    initial_overlaps: list[str] = field(default_factory=lambda: ["max", "0.05"])
    target_fidelities: list[float] = field(default_factory=lambda: [0.5, 0.9, 0.99])
    max_subspace_fraction: float = 1.0

    # --- SKQD grid -------------------------------------------------------
    skqd_dt: list[float] = field(
        default_factory=lambda: [0.05, 0.1, 0.25, 0.5, 1.0, 2.0]
    )
    skqd_shots: list[int] = field(default_factory=lambda: [64, 256, 1024])
    skqd_repeats: int = 3
    skqd_max_iterations: int = 30
    skqd_dense_propagator_max_dim: int = 2048

    # --- BARK grid -------------------------------------------------------
    bark_score_modes: list[str] = field(
        default_factory=lambda: ["amplitude", "perturbative", "two_dimensional"]
    )
    bark_selection_strategies: list[str] = field(
        default_factory=lambda: ["pool", "best_first"]
    )
    bark_keep_states: list[int] = field(default_factory=lambda: [1, 2, 4])
    bark_max_iterations: int = 64
    bark_mode: str = "top_m"
    bark_ground_update_interval: int = 1
    bark_subspace: str = "applied"
    bark_restrict_equal_ones_zeros: bool = False
    bark_repeats: int = 1

    # --- Numerics --------------------------------------------------------
    dense_subspace_max_dim: int = 600
    degeneracy_tol: float = 1e-9
    num_reference_eigenvalues: int = 8

    # ------------------------------------------------------------------
    @property
    def seeds(self) -> list[int]:
        return [self.seed_offset + i for i in range(self.num_hamiltonians)]

    @property
    def stop_fidelity(self) -> float:
        """Runs stop once the hardest requested target is met."""
        return max(self.target_fidelities)

    @property
    def model_kwargs(self) -> dict[str, Any]:
        return {
            "J_max": self.J_max,
            "B_max": self.B_max,
            "J_components": tuple(self.J_components),
            "B_components": tuple(self.B_components),
            "coupling_distribution": self.coupling_distribution,
            "field_distribution": self.field_distribution,
            "spin": self.spin,
        }

    def case_specs(self) -> list[tuple[int, int | None, int]]:
        """All ``(num_qubits, max_interactions, seed)`` triples in the scan."""
        specs = []
        for n, mi, seed in itertools.product(
            self.num_qubits, self.max_interactions, self.seeds
        ):
            specs.append((n, None if mi is None or mi < 0 else int(mi), seed))
        return specs

    def skqd_grid(self) -> list[tuple[float, int, int]]:
        return [
            (dt, shots, repeat)
            for dt, shots, repeat in itertools.product(
                self.skqd_dt, self.skqd_shots, range(self.skqd_repeats)
            )
        ]

    def bark_grid(self) -> list[tuple[str, str, int, int]]:
        return [
            (score, strategy, keep, repeat)
            for score, strategy, keep, repeat in itertools.product(
                self.bark_score_modes,
                self.bark_selection_strategies,
                self.bark_keep_states,
                range(self.bark_repeats),
            )
        ]

    def run_count(self) -> dict[str, int]:
        cases = len(self.case_specs())
        units = cases * len(self.initial_overlaps)
        return {
            "hamiltonians": cases,
            "case_overlap_units": units,
            "skqd_runs": units * len(self.skqd_grid()),
            "bark_runs": units * len(self.bark_grid()),
            "total_runs": units * (len(self.skqd_grid()) + len(self.bark_grid())),
        }

    def to_json(self) -> str:
        payload = asdict(self)
        payload["J_components"] = list(self.J_components)
        payload["B_components"] = list(self.B_components)
        return json.dumps(payload, indent=2, sort_keys=True)


PRESETS: dict[str, dict[str, Any]] = {
    "smoke": {
        "num_qubits": [4, 6],
        "max_interactions": [1, -1],
        "num_hamiltonians": 1,
        "initial_overlaps": ["max"],
        "target_fidelities": [0.5, 0.9],
        "skqd_dt": [0.25, 1.0],
        "skqd_shots": [64, 256],
        "skqd_repeats": 1,
        "skqd_max_iterations": 12,
        "bark_score_modes": ["amplitude", "perturbative"],
        "bark_selection_strategies": ["pool", "best_first"],
        "bark_keep_states": [1, 2],
        "bark_max_iterations": 24,
    },
    "default": {},
    "full": {
        "num_qubits": [6, 8, 10, 12],
        "max_interactions": [1, 2, 3, -1],
        "num_hamiltonians": 5,
        "initial_overlaps": ["max", "0.05", "0.005"],
        "target_fidelities": [0.5, 0.75, 0.9, 0.99],
        "skqd_dt": [0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0],
        "skqd_shots": [32, 128, 512, 2048],
        "skqd_repeats": 3,
        "skqd_max_iterations": 40,
        "bark_score_modes": [
            "amplitude",
            "probability",
            "coupling",
            "perturbative",
            "two_dimensional",
        ],
        "bark_selection_strategies": ["pool", "best_first"],
        "bark_keep_states": [1, 2, 4, 8],
        "bark_max_iterations": 128,
    },
}


def build_config(preset: str = "default", **overrides: Any) -> StudyConfig:
    """Build a :class:`StudyConfig` from a preset plus explicit overrides."""
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset {preset!r}; expected one of {sorted(PRESETS)}")
    params: dict[str, Any] = dict(PRESETS[preset])
    params.update({k: v for k, v in overrides.items() if v is not None})
    return StudyConfig(**params)
