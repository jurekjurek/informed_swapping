import scipy.sparse as sp
import numpy as np
import tqdm

def do_power(H: sp.csr_matrix, num_steps: int, initial: int = None) -> sp.csr_matrix:
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
    # Shift H to make it positive definite by the smallest eigenvalue
    min_eig = sp.linalg.eigsh(H, k=1, which='SA', return_eigenvectors=False)[0]
    if min_eig < 0:
        H_shifted = H - min_eig * sp.eye(H.shape[0])
    else:
        H_shifted = H
    U = sp.linalg.inv(H_shifted)  # Use the inverse of the shifted Hamiltonian for power iteration
    
    # Get the current state
    current_state = initial_state.copy()
    leftover_indices = range(H.shape[0])
    mask = np.ones(H.shape[0], dtype=bool)
    mask[initial] = False
    leftover_indices = np.where(mask)[0]

    for step in tqdm.tqdm(range(num_steps)):

        current_state = U @ current_state

        #Normalize
        current_state /= np.linalg.norm(current_state)

        # Sample from the distribution defined by the current state
        probabilities = np.abs(current_state) ** 2
        # Make NaN to zero
        probabilities = np.nan_to_num(probabilities)

        i = 0
        while i<samples_per_step:
            probabilities_here = probabilities[mask]
            probabilities_here /= np.sum(probabilities_here) 
            if np.sum(probabilities_here) != 1:
                # Make an even distribution if all probabilities are zero
                probabilities_here = np.ones_like(probabilities_here) / len(probabilities_here)
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