import scipy.sparse as sp
import numpy as np
import tqdm

def get_exponential(H: sp.csr_matrix, t: float) -> sp.csr_matrix:
    """
    Get the exponential of the Hamiltonian H, i.e. exp(-iHt).
    """
    return sp.linalg.expm(-1j * H * t)

def do_skqd(H: sp.csr_matrix, num_steps: int, t: float, initial: int) -> sp.csr_matrix:
    samples_per_step = H.shape[0] // num_steps
    skqd_list = np.ones(H.shape[0])*-1
    skqd_list[0] = initial

    leftover = H.shape[0] % num_steps
    if leftover > 0:
        print(f"Warning: H.shape[0] is not perfectly divisible by num_steps. Last {leftover} samples will be ignored.")

    initial_state = np.zeros(H.shape[0], dtype=np.complex128)
    initial_state[initial] = 1.0
    U = get_exponential(H, t)
    # Get the current state
    current_state = initial_state.copy()
    leftover_indices = range(H.shape[0])
    mask = np.ones(H.shape[0], dtype=bool)
    mask[initial] = False
    leftover_indices = np.where(mask)[0]

    for step in tqdm.tqdm(range(num_steps)):

        current_state = U @ current_state

        # Sample from the distribution defined by the current state
        probabilities = np.abs(current_state) ** 2

        i = 0
        while i<samples_per_step:
            probabilities_here = probabilities[mask]
            probabilities_here /= np.sum(probabilities_here)  # normalize
            if len(leftover_indices) == 0:
                print("No more indices left to sample.")
                break
            sampled_index = np.random.choice(leftover_indices, p=probabilities_here)
            if sampled_index not in skqd_list:
                skqd_list[step * samples_per_step + i+1] = sampled_index
                mask[sampled_index] = False
                leftover_indices = np.where(mask)[0]
                i += 1

    return skqd_list