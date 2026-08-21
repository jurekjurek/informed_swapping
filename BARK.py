'''
Author: Emil Rosanowski
This will be a simple implementation of the BARK protocol.
We will start with an initial state (given as an index in the basis), then the Hamiltonian will be applied once in each iteration.
After each application, we will have a set of basis states with non-zero amplitudes.
We will use Johann's methods to rate them according to their potential to lower the energy of the system.
This proxy will be stored as well as memory. 
The state with the highest potential will be selected and added to the pool.
Then the next iteration will be performed. If the potential of another state in the memory is higher, it will be selected instead of the new state.
Every n times, a full diagonalization of the projected Hamiltonian will be performed to check the current performance.
Note: for test purposes, we choose n=1, effectively tracking how well Johann's method performs at the same time.
'''

from functools import Placeholder

import numpy as np
from scipy.linalg import eigh


class BARK:
    def __init__(self, hamiltonian: np.ndarray):
        """
        Initialize the BARK protocol with a given Hamiltonian and initial state.
        """
        self.hamiltonian = hamiltonian

    def johanns_method(self, last_approximation: np.ndarray, energy: float, index: int) -> float:
        """
        Johann's method: rate a basis state by the energy the estimator would reach if
        it were added to it, i.e. the lower eigenvalue of the 2x2 problem in
        span{v_i, psi_{i+1}}. Assumes v_i . psi_{i+1} = 0. Lower is better.
        """
        phi = self.hamiltonian[:, int(index)]
        diagonal = np.real(phi[int(index)])
        overlap = np.vdot(last_approximation, phi)
        gamma = np.sqrt((energy - diagonal) ** 2 + 4.0 * np.abs(overlap) ** 2)
        return 0.5 * (energy + diagonal - gamma)
    
    def apply_hamiltonian(self, index: int) -> np.ndarray:
        """
        Apply the Hamiltonian to the state corresponding to the given index.
        Return a list of new states.
        """
        applied_state = self.hamiltonian[int(index)]
        new_states = [i for i, amp in enumerate(applied_state) if amp != 0]
        return new_states
    
    def rank_states(self, last_approximation: np.ndarray, energy: float, new_states: np.ndarray) -> dict:
        """
        Rank the new states based on their potential to lower the energy of the system.
        """
        potentials = {}
        for state in new_states:
            potential = self.johanns_method(last_approximation, energy, state)
            potentials[state] = potential
        return potentials
    
    def project_hamiltonian(self, pool: np.ndarray) -> np.ndarray:
        """
        Project the Hamiltonian onto the subspace spanned by the states in the pool.
        """
        projected_hamiltonian = self.hamiltonian[np.ix_(pool, pool)]
        return projected_hamiltonian
    
    def test_run(self, target_fidelity: float, correct_state: np.ndarray, initial_state_index: int) -> int:
        """
        Run the BARK protocol until the target fidelity is reached.
        """

        memory = {}
        pool = [initial_state_index]
        last_approximation = np.zeros(self.hamiltonian.shape[0])
        last_approximation[initial_state_index] = 1.0  # Start with the initial state
        energy = np.real(self.hamiltonian[initial_state_index, initial_state_index])

        last_state = initial_state_index

        iteration = 0
        while True:
            iteration += 1

            new_states = self.apply_hamiltonian(last_state)

            # Remove states that are already in the pool from the new states
            new_states = [state for state in new_states if state not in pool]

            potentials = self.rank_states(last_approximation, energy, new_states)

            # Update memory with new states and their potentials
            for state, potential in potentials.items():
                if state not in memory or potential > memory[state]:
                    memory[state] = potential
            
            # Take the state with the lowest potential as potential is a proxy for energy lowering
            last_state = min(memory, key=memory.get)

            # Delete the selected state from memory to avoid re-selection
            del memory[last_state]

            pool.append(last_state)

            # Project the Hamiltonian onto the current pool of states
            projected_hamiltonian = self.project_hamiltonian(pool)

            # Diagonalize the projected Hamiltonian to find the ground state
            eigenvalues, eigenvectors = eigh(projected_hamiltonian)
            ground_state_index = np.argmin(eigenvalues)
            energy = eigenvalues[ground_state_index]
            last_approximation = np.zeros(self.hamiltonian.shape[0])
            last_approximation[pool] = eigenvectors[:, ground_state_index]

            # Calculate fidelity with the correct state
            fidelity = np.abs(np.dot(last_approximation.conj(), correct_state)) ** 2

            if fidelity >= target_fidelity:
                print(f"Target fidelity reached: {fidelity} at iteration {iteration}")
                return len(pool)  # Return the number of states in the pool when target fidelity is reached