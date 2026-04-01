# Import Bark here
import numpy as np
import matplotlib.pyplot as plt
from MakeHam import make_sparse_ground_state_hamiltonian_from_qubits
from JBARK import BARK

class GridScan:
    def __init__(self, n_qubits: list[int], sparsity_values: list[float | int], overlaps: list[float], seeds: list[int]):
        self.n_qubits = n_qubits
        self.sparsity_values = sparsity_values
        self.wanted_overlaps = overlaps
        self.seeds = seeds
        
        self.hamiltonians = []
        self.ground_states = []
        self.supports = []
        self.real_sparsity_values = []

        self.generate_hamiltonians()

        self.stopping_times = np.zeros((len(n_qubits), len(sparsity_values), len(seeds),len(overlaps)))
        self.real_overlaps = np.zeros((len(n_qubits), len(sparsity_values), len(seeds),len(overlaps)))


    def generate_hamiltonians(self):
        for n_qubits in self.n_qubits:
            hams = []
            gs = []
            sups = []
            rspars = []
            for sparsity in self.sparsity_values:
                sparse_hams = []
                sparse_gs = []
                sparse_sups = []
                real_sp = []
                for seed in self.seeds:
                    H, psi, sup = make_sparse_ground_state_hamiltonian_from_qubits(
                        n_qubits=n_qubits,
                        ground_state_sparsity=sparsity,
                        seed=seed,
                        ground_energy=-5.0,
                        gap=1.0,
                        return_sparse=True,
                    )
                    sparse_hams.append(H)
                    sparse_gs.append(psi)
                    sparse_sups.append(sup)
                    real_sp.append(len(sup) / (2 ** n_qubits))
                hams.append(sparse_hams)
                gs.append(sparse_gs)
                sups.append(sparse_sups)
                rspars.append(real_sp)
            self.hamiltonians.append(hams)
            self.ground_states.append(gs)
            self.supports.append(sups)
            self.real_sparsity_values.append(rspars)

    
    def get_stopping_time(self):
        for i, n_qubits in enumerate(self.n_qubits):
            for j, sparsity in enumerate(self.sparsity_values):
                for k, seed in enumerate(self.seeds):
                    H = self.hamiltonians[i][j][k]
                    psi = self.ground_states[i][j][k]
                    sup = self.supports[i][j][k]

                    for l, wanted_overlap in enumerate(self.wanted_overlaps):
                        
                        # Get the element of psi where el^2 is closest to wanted_overlap
                        amps = np.abs(psi)**2
                        idx = np.argmin(np.abs(amps - wanted_overlap))
                        self.real_overlaps[i,j,k,l] = amps[idx]

                        bark = BARK(H, idx)
                        bark.run()

                        for t, sublist in enumerate(bark.samples):
                            if all(s in sublist for s in sup):
                                self.stopping_times[i,j,k,l] = t + 1
                                break