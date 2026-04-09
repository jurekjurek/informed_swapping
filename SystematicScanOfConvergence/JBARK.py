import numpy as np
import scipy.sparse as sp

class BARK:
    def __init__(self, hamiltonian: sp.csr_matrix, initial_state: int, keepstates: int = 1):
        self.H = hamiltonian
        self.psi = np.zeros(self.H.shape[0], dtype=complex)
        self.psi[initial_state] = 1.0
        self.keepstates = keepstates if keepstates < hamiltonian.shape[0] else hamiltonian.shape[0]
        self.samples = [[initial_state]]
        self.amplitudes = [self.psi.copy()]

    def step(self):
        # Compute the new state by applying the Hamiltonian
        new_psi = self.H @ self.psi
        self.amplitudes.append(new_psi.copy())
        new_index = []
        current_sample_set = set(self.samples[-1])
        
        max_iters = len(self.amplitudes)
        finished = False

        i = 0
        kept_states = 0
        while len(new_index) < self.keepstates:
            i -= 1
            if i < -max_iters:
                finished = True
                break
            amps = np.abs(self.amplitudes[i])**2
            sorted_indices = np.argsort(amps)[::-1]  # Sort in descending order
            for idx in sorted_indices:
                if idx not in set(current_sample_set.union(set(new_index))):
                    new_index.append(idx)
                    kept_states += 1
                    if kept_states >= self.keepstates:
                        break

        if finished:
            self.samples.append(self.samples[-1] + new_index)
            missing_indices = set(range(self.H.shape[0])) - set(self.samples[-1])
            for idx in missing_indices:
                self.samples.append(self.samples[-1] + [idx])
        else:            
            self.samples.append(self.samples[-1] + new_index)
            self.psi = np.zeros_like(self.psi)
            self.psi[new_index] = new_psi[new_index]
            self.psi[new_index] /= np.linalg.norm(self.psi[new_index])  # Normalize the new state
        return finished
    
    def run(self):
        while True:
            finished = self.step()
            if finished or len(self.samples[-1]) >= self.H.shape[0]:
                break