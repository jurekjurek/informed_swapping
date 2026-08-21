"""
Author: Emil Rosanowski
This will be the implementation of the SKQD protocol.
As this is only a simulation, we will simulate the quantum part.

We will start with an initial state (given as an index in the basis).
For a given time parameter t, we will precompute U(t)=e^{-iHt}.
Then, U(t) will be applied to the current state, resulting in a new state.
Then we will draw n_shots samples according to the probability distribution of the new state.
This is one iteration and at the end of the iteration, we will add the sampled states to the pool.
In every iteration, the Hamiltonian is projected onto the pool and diagonalized to find the ground state of the projected Hamiltonian.
"""

import numpy as np
from scipy.linalg import expm, eigh

class SKQD:
    def __init__(self, hamiltonian: np.ndarray):
        """
        Initialize the SKQD protocol with a given Hamiltonian and initial state.
        """
        self.hamiltonian = hamiltonian

    def compute_unitary(self, t: float) -> np.ndarray:
        """
        Compute the unitary evolution operator U(t) = exp(-i * H * t).
        """
        return expm(-1j * self.hamiltonian * t)
    
    def apply_unitary(self, state: np.ndarray, unitary: np.ndarray) -> np.ndarray:
        """
        Apply the unitary evolution operator U(t) to the given state.
        """
        return unitary @ state
    
    def project_hamiltonian(self, pool: np.ndarray) -> np.ndarray:
        """
        Project the Hamiltonian onto the subspace spanned by the states in the pool.
        """
        projected_hamiltonian = self.hamiltonian[np.ix_(pool, pool)]
        return projected_hamiltonian
    
    def test_run(self, initial_state_index: int, t: float, n_shots: int, correct_state: np.ndarray, target_fidelity: float) -> int:
        """
        Run the SKQD protocol for a given number of iterations.
        """
        # Initialize the pool with the initial state
        pool = np.array([initial_state_index])
        last_approximation = np.zeros(self.hamiltonian.shape[0])
        last_approximation[initial_state_index] = 1.0  # Start with the initial state

        unitary = self.compute_unitary(t)

        pool_size = 1

        while True:
            new_states = self.apply_unitary(last_approximation, unitary)

            last_approximation = new_states

            # Sample n_shots states according to the probability distribution of the new state
            probabilities = np.abs(new_states) ** 2
            sampled_indices = np.random.choice(len(probabilities), size=n_shots, p=probabilities / probabilities.sum())
            # Add the sampled states to the pool which are not already in the pool
            pool = np.unique(np.concatenate((pool, sampled_indices)))

            if len(pool) == pool_size:
                print("No new states added to the pool. Stopping.")
                return None
            pool_size = len(pool)

            # Project the Hamiltonian onto the current pool of states
            projected_hamiltonian = self.project_hamiltonian(pool)

            # Diagonalize the projected Hamiltonian to find the ground state
            eigenvalues, eigenvectors = eigh(projected_hamiltonian)
            ground_state_index = np.argmin(eigenvalues)
            ground_state = np.zeros(self.hamiltonian.shape[0])
            ground_state[pool] = eigenvectors[:, ground_state_index]

            fidelity = np.abs(np.dot(ground_state.conj(), correct_state)) ** 2

            if fidelity >= target_fidelity:
                print(f"Target fidelity reached: {fidelity}")
                return pool_size  # Return the number of states in the pool when target fidelity is reached