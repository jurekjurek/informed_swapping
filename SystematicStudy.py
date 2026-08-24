"""
This file compares SKQD.py and BARK.py for random spin models in a systematic study.
It will create many random spin Hamiltonians and charaterize them for ground state density and Hamiltonian density.
For a given array of Fidelities, it will run both protocols and record the number of states in the pool when the target fidelity is reached.
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from RandomSpinModel import make_random_spin_hamiltonian
from BARK import BARK
from SKQD import SKQD
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh
from tqdm import tqdm

def run_study(num_Hamiltonians: int, sparse: bool = True):
    """
    Run the comparison study.

    With ``sparse=True`` the Hamiltonian is kept in CSR form and only the ground
    state is solved for, which is what is needed once the Hilbert space is too
    large to diagonalize densely. With ``sparse=False`` the full dense
    eigendecomposition is computed and handed to SKQD, which then builds U(t)
    from it instead of calling a matrix exponential -- much faster, but only
    affordable for small systems.
    """

    final_data = pd.DataFrame(columns=["Number_of_Sites", "Ground_State_Density", "Hamiltonian_Density", "Max_Interactions", "Fidelity", "Overlap", "BARK_Pool_Size", "SKQD_Pool_Size"])

    num_sites = [6, 8]

    fidelities = [0.8, 0.85, 0.9]

    max_interactions = [1, 2]

    for hamiltonian_index in tqdm(range(num_Hamiltonians), desc="Hamiltonians"):
        for n_sites in tqdm(num_sites, desc="Number of Sites", leave=False):
            for max_interaction in tqdm(max_interactions, desc="Max Interactions", leave=False):
                # Create a random spin Hamiltonian
                hamiltonian = make_random_spin_hamiltonian(num_sites=n_sites, max_interactions=max_interaction)[0].to_matrix(sparse=sparse)

                # Calculate ground state and its density. In the dense case we keep
                # the full eigendecomposition so SKQD can build U(t) from it.
                if sparse:
                    eigenvalues, eigenvectors = eigsh(hamiltonian, k=1, which='SA')
                    all_eigenvalues, all_eigenvectors = None, None
                else:
                    eigenvalues, eigenvectors = eigh(hamiltonian)
                    all_eigenvalues, all_eigenvectors = eigenvalues, eigenvectors
                ground_state_index = np.argmin(eigenvalues)
                ground_state = eigenvectors[:, ground_state_index]
                # Calculate ground state density as the ratio of non-zero (>10^-3) elements to total elements
                ground_state_density = np.count_nonzero(np.abs(ground_state) > 1e-3) / ground_state.shape[0]

                # Calculate Hamiltonian density as the ratio of non-zero elements to total elements
                stored_nonzeros = hamiltonian.nnz if sparse else np.count_nonzero(hamiltonian)
                hamiltonian_density = stored_nonzeros / hamiltonian.shape[0]**2

                # Take the five largest and five smallest probabilities of the ground state and save their indices so we can use them as initial states for the protocols
                probabilities = np.abs(ground_state) ** 2
                largest_indices = np.argsort(probabilities)[-5:]
                smallest_indices = np.argsort(probabilities)[:5]
                indices_to_test = np.concatenate((largest_indices, smallest_indices))
                
                # Now save the overlaps, which is the probability of this index
                overlaps = probabilities[indices_to_test]

                for fidelity in tqdm(fidelities, desc="Fidelities", leave=False):
                    for initial_state_index, overlap in zip(indices_to_test, overlaps):
                        # Run BARK protocol
                        bark_protocol = BARK(hamiltonian)
                        bark_pool_size = bark_protocol.test_run(target_fidelity=fidelity, correct_state=ground_state, initial_state_index=initial_state_index)

                        # Run SKQD protocol

                        skqd_protocol = SKQD(hamiltonian, eigenvalues=all_eigenvalues, eigenvectors=all_eigenvectors)
                        t, shots, _ = skqd_protocol.optimize(initial_state_index=initial_state_index, correct_state=ground_state, target_fidelity=fidelity)

                        skqd_pool_size = skqd_protocol.test_run(initial_state_index=initial_state_index, t=t, n_shots=shots, correct_state=ground_state, target_fidelity=fidelity)

                        # Append results to the DataFrame
                        final_data = pd.concat([final_data, pd.DataFrame({
                            "Number_of_Sites": [n_sites],
                            "Ground_State_Density": [ground_state_density],
                            "Hamiltonian_Density": [hamiltonian_density],
                            "Max_Interactions": [max_interaction],
                            "Fidelity": [fidelity],
                            "Overlap": [overlap],
                            "BARK_Pool_Size": [bark_pool_size],
                            "SKQD_Pool_Size": [skqd_pool_size]
                        })], ignore_index=True)

    # Save the final DataFrame to a CSV file
    final_data.to_csv("systematic_study_results.csv", index=False)

if __name__ == "__main__":
    run_study(num_Hamiltonians=5)  # Adjust the number of Hamiltonians as needed