import numpy as np
import scipy.sparse as sp

class BARK:
    def __init__(self, hamiltonian: sp.csr_matrix, initial_state: int):
        self.H = hamiltonian
        self.psi = np.zeros(self.H.shape[0], dtype=complex)
        self.psi[initial_state] = 1.0
        
        self.samples = [[initial_state]]
        self.amplitudes = [self.psi.copy()]

    def step(self):
        # Compute the new state by applying the Hamiltonian
        new_psi = self.H @ self.psi
        self.amplitudes.append(new_psi.copy())
        new_index = None
        current_sample_set = set(self.samples[-1])
        
        max_iters = len(self.amplitudes)
        finished = False

        i = 0
        while new_index is None:
            i -= 1
            if i < -max_iters:
                finished = True
                break
            amps = np.abs(self.amplitudes[i])**2
            sorted_indices = np.argsort(amps)[::-1]  # Sort in descending order
            
            for idx in sorted_indices:
                if idx not in current_sample_set:
                    new_index = idx
                    break

        if finished:
            missing_indices = set(range(self.H.shape[0])) - current_sample_set
            for idx in missing_indices:
                self.samples.append(self.samples[-1] + [idx])
        else:            
            self.samples.append(self.samples[-1] + [new_index])
            self.psi = np.zeros_like(self.psi)
            self.psi[new_index] = 1.0
        return finished
    
    def run(self):
        while True:
            finished = self.step()
            if finished or len(self.samples[-1]) >= self.H.shape[0]:
                break