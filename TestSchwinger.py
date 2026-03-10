from qiskit.quantum_info import SparsePauliOp
import numpy as np
from BARK import BARK
from schwingermodel import *
import primme
from scipy.sparse import csr_matrix

# Random Hamiltonian

N = 14
x = (N/30)**2
lam = 1000
l0 = 2
m_lat = 10.0
g = 1.0
H = generateSchwingerHamiltonian(N, x, lam, l0, m_lat, g)

initial_state = '10' * (N//2)

BK = BARK(H, initial_state, max_iterations=100, time_step=1, tolerance=0.001, even_numbers=True)

bases = BK.basis

energy_est = []

correct_energy = primme.eigsh(csr_matrix(H.to_matrix()), k=1, which='SA', tol=1e-10)[0][0]

print("Exact ground state energy:", correct_energy)

for basis in bases:
    print("Size of basis:", len(basis))
    H_p = BK.project_to_subspace(basis)
    w,v = primme.eigsh(H_p, k=1, which='SA', tol=1e-10)
    energy_est.append(w[0])
    print("Estimated ground state energy:", w[0])
    print("Error:", abs(w[0] - correct_energy))