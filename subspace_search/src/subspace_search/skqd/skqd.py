from __future__ import annotations

import scipy.sparse as sp
import scipy.sparse.linalg as spla
import numpy as np
import tqdm

from .energy_tracking import EnergyTrackingStep, update_ground_state_proxy


def get_exponential(H: sp.csr_matrix, t: float) -> sp.csr_matrix:
    """
    Get the exponential of the Hamiltonian H, i.e. exp(-iHt).
    """
    return spla.expm(-1j * H * t)


def do_skqd(H: sp.csr_matrix, num_steps: int, t: float, initial: int = None) -> np.ndarray:
    skqd_list, _ = _do_skqd(H, num_steps, t, initial, track_energy=False)
    return skqd_list


def do_skqd_with_energy_tracking(
    H: sp.csr_matrix,
    num_steps: int,
    t: float,
    initial: int = None,
) -> tuple[np.ndarray, list[EnergyTrackingStep]]:
    """
    Run SKQD and track an iterative ground-state proxy after each iteration.

    Returns the usual SKQD ordering together with per-iteration diagnostics.
    """
    return _do_skqd(H, num_steps, t, initial, track_energy=True)


def _do_skqd(
    H: sp.csr_matrix,
    num_steps: int,
    t: float,
    initial: int = None,
    track_energy: bool = False,
) -> tuple[np.ndarray, list[EnergyTrackingStep]]:
    samples_per_step = H.shape[0] // num_steps
    skqd_list = np.ones(H.shape[0])*-1
    skqd_list[0] = initial

    leftover = H.shape[0] % num_steps
    if leftover > 0:
        print(f"Warning: H.shape[0] is not perfectly divisible by num_steps. Last {leftover} samples will be ignored.")

    initial_state = np.zeros(H.shape[0], dtype=np.complex128)
    if initial is None:
        initial_state = np.ones_like(initial_state) / np.sqrt(len(initial_state))
    else:
        initial_state[initial] = 1.0
    U = get_exponential(H, t)
    # Get the current state
    current_state = initial_state.copy()
    ground_state_proxy = initial_state.copy()
    energy_tracking_steps = []
    leftover_indices = range(H.shape[0])
    mask = np.ones(H.shape[0], dtype=bool)
    mask[initial] = False
    leftover_indices = np.where(mask)[0]

    for step in tqdm.tqdm(range(num_steps)):

        current_state = U @ current_state

        # Sample from the distribution defined by the current state
        probabilities = np.abs(current_state) ** 2
        # Make NaN to zero
        probabilities = np.nan_to_num(probabilities)

        i = 0
        sampled_indices = []
        while i<samples_per_step:
            if len(leftover_indices) == 0:
                print("No more indices left to sample.")
                break
            probabilities_here = probabilities[mask]
            probability_sum = np.sum(probabilities_here)
            if probability_sum > 0:
                probabilities_here /= probability_sum
            if np.sum(probabilities_here) != 1:
                # Make an even distribution if all probabilities are zero
                probabilities_here = np.ones_like(probabilities_here) / len(probabilities_here)
            sampled_index = np.random.choice(leftover_indices, p=probabilities_here)
            if sampled_index not in skqd_list:
                skqd_list[step * samples_per_step + i+1] = sampled_index
                mask[sampled_index] = False
                leftover_indices = np.where(mask)[0]
                sampled_indices.append(sampled_index)
                i += 1

        if track_energy:
            ground_state_proxy, diagnostics = update_ground_state_proxy(
                H,
                ground_state_proxy,
                np.array(sampled_indices, dtype=int),
                step,
            )
            energy_tracking_steps.append(diagnostics)

    return skqd_list, energy_tracking_steps
