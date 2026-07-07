from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp


@dataclass
class EnergyTrackingStep:
    """Energy diagnostics for one SKQD sampling iteration."""

    step: int
    new_indices: np.ndarray
    energy_before: float
    energy_after: float
    improvement: float
    best_single_index: int | None
    best_single_energy: float | None
    best_single_improvement: float | None
    correlation_benefit: float | None
    amplitudes: np.ndarray


def expectation_value(H: sp.csr_matrix, state: np.ndarray) -> float:
    """Return <state|H|state> as a real scalar."""
    return float(np.real(np.vdot(state, H @ state)))


def update_ground_state_proxy(
    H: sp.csr_matrix,
    state: np.ndarray,
    new_indices: np.ndarray,
    step: int,
) -> tuple[np.ndarray, EnergyTrackingStep]:
    """
    Update the iterative ground-state proxy in span{state, |new_indices>}.

    If the current state has no support on the new basis states, this is the
    ordinary eigenproblem described by the 2x2/small projected Hamiltonian.
    Otherwise, the same variational update is solved as a generalized
    eigenproblem with the basis-overlap matrix.
    """
    new_indices = np.asarray(new_indices, dtype=int)
    energy_before = expectation_value(H, state)

    if len(new_indices) == 0:
        return state, EnergyTrackingStep(
            step=step,
            new_indices=new_indices,
            energy_before=energy_before,
            energy_after=energy_before,
            improvement=0.0,
            best_single_index=None,
            best_single_energy=None,
            best_single_improvement=None,
            correlation_benefit=None,
            amplitudes=np.array([1.0], dtype=np.complex128),
        )

    H_state = H @ state
    H_new_state = H_state[new_indices]
    H_new_new = H[new_indices, :][:, new_indices].toarray()

    projected_H = np.empty((len(new_indices) + 1, len(new_indices) + 1), dtype=np.complex128)
    projected_H[0, 0] = energy_before
    projected_H[0, 1:] = np.conj(H_new_state)
    projected_H[1:, 0] = H_new_state
    projected_H[1:, 1:] = H_new_new

    overlap = np.eye(len(new_indices) + 1, dtype=np.complex128)
    overlap[0, 1:] = np.conj(state[new_indices])
    overlap[1:, 0] = state[new_indices]

    eigenvalues, eigenvectors = la.eigh(projected_H, overlap)
    ground_idx = int(np.argmin(eigenvalues))
    energy_after = float(np.real(eigenvalues[ground_idx]))
    amplitudes = eigenvectors[:, ground_idx]

    updated_state = amplitudes[0] * state.copy()
    updated_state[new_indices] += amplitudes[1:]
    norm = np.linalg.norm(updated_state)
    if norm > 0:
        updated_state /= norm

    single_results = [
        _single_bitstring_energy(
            energy_before,
            H_new_state[i],
            H_new_new[i, i],
            state[new_indices[i]],
        )
        for i in range(len(new_indices))
    ]
    best_single_pos = int(np.argmin(single_results))
    best_single_energy = float(single_results[best_single_pos])
    best_single_improvement = energy_before - best_single_energy

    return updated_state, EnergyTrackingStep(
        step=step,
        new_indices=new_indices,
        energy_before=energy_before,
        energy_after=energy_after,
        improvement=energy_before - energy_after,
        best_single_index=int(new_indices[best_single_pos]),
        best_single_energy=best_single_energy,
        best_single_improvement=best_single_improvement,
        correlation_benefit=best_single_energy - energy_after,
        amplitudes=amplitudes,
    )


def _single_bitstring_energy(
    energy_before: float,
    coupling: complex,
    diagonal: complex,
    overlap_with_state: complex,
) -> float:
    projected_H = np.array(
        [
            [energy_before, np.conj(coupling)],
            [coupling, diagonal],
        ],
        dtype=np.complex128,
    )
    overlap = np.array(
        [
            [1.0, np.conj(overlap_with_state)],
            [overlap_with_state, 1.0],
        ],
        dtype=np.complex128,
    )
    return float(np.real(la.eigh(projected_H, overlap, eigvals_only=True)[0]))
