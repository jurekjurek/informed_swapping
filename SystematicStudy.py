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
from ClusterStudy import DEFAULT_DENSE_LIMIT, solve_ground_state
from scipy.sparse import issparse
from tqdm import tqdm

def run_study(num_Hamiltonians: int, sparse: bool = True,
              dense_limit: int = DEFAULT_DENSE_LIMIT):
    """
    Run the comparison study.

    ``sparse=True`` keeps the Hamiltonian in CSR form, which is what both
    protocols want for reading rows and growing the projected block.

    ``dense_limit`` is the Hilbert-space dimension up to which the full dense
    eigendecomposition is computed and handed to SKQD, which then applies U(t)
    from it instead of calling a matrix exponential. That is much faster and
    affordable well beyond the sizes studied here, so it is on by default; set
    it to 0 to fall back to solving for the ground state alone.
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

                # Calculate ground state and its density. Whenever the Hilbert
                # space is small enough we keep the full eigendecomposition so
                # SKQD can apply U(t) from it.
                ground_state, all_eigenvalues, all_eigenvectors = solve_ground_state(
                    hamiltonian, dense_limit)
                # Calculate ground state density as the ratio of non-zero (>10^-3) elements to total elements
                ground_state_density = np.count_nonzero(np.abs(ground_state) > 1e-3) / ground_state.shape[0]

                # Calculate Hamiltonian density as the ratio of non-zero elements to total elements
                stored_nonzeros = hamiltonian.nnz if issparse(hamiltonian) else np.count_nonzero(hamiltonian)
                hamiltonian_density = stored_nonzeros / hamiltonian.shape[0]**2

                # Take the five largest and five smallest probabilities of the ground state and save their indices so we can use them as initial states for the protocols
                probabilities = np.abs(ground_state) ** 2
                largest_indices = np.argsort(probabilities)[-5:]
                smallest_indices = np.argsort(probabilities)[:5]
                indices_to_test = np.concatenate((largest_indices, smallest_indices))
                
                # Now save the overlaps, which is the probability of this index
                overlaps = probabilities[indices_to_test]

                # Both protocol objects are built once per Hamiltonian: they hold
                # the per-t caches and the growing projection buffer, all of which
                # were thrown away every iteration when they lived inside the loop.
                bark_protocol = BARK(hamiltonian)
                skqd_protocol = SKQD(hamiltonian, eigenvalues=all_eigenvalues,
                                     eigenvectors=all_eigenvectors)

                for initial_state_index, overlap in tqdm(list(zip(indices_to_test, overlaps)),
                                                         desc="Initial states", leave=False):
                    # BARK is deterministic and its trajectory does not depend on
                    # the target, so one walk answers every fidelity at once.
                    bark_pool_sizes = bark_protocol.sweep(target_fidelities=fidelities,
                                                          correct_state=ground_state,
                                                          initial_state_index=initial_state_index)

                    # One grid scan likewise answers every fidelity at once,
                    # though each fidelity still gets its own (t, n_shots).
                    optima = skqd_protocol.optimize_many(initial_state_index=initial_state_index,
                                                         correct_state=ground_state,
                                                         target_fidelities=fidelities,
                                                         correct_probabilities=probabilities)

                    skqd_pool_sizes = [
                        skqd_protocol.sweep(initial_state_index=initial_state_index,
                                            t=t, n_shots=shots,
                                            correct_state=ground_state,
                                            target_fidelities=[fidelity],
                                            correct_probabilities=probabilities)[0]
                        for fidelity, (t, shots, _) in zip(fidelities, optima)
                    ]

                    # Append results to the DataFrame
                    for fidelity, bark_pool_size, skqd_pool_size in zip(
                            fidelities, bark_pool_sizes, skqd_pool_sizes):
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