from qiskit.quantum_info import SparsePauliOp
import numpy as np
from BARK import BARK
from schwingermodel import *
import primme
from scipy.sparse import csr_matrix

# Random Hamiltonian

K = 5
U = 7.0
t = 1.0
V = 1.0

H, eps, Vk = siam_diagonal_bath_sparse_pauli(K=K, U=U, t=t, V=V)

psi0, occ = siam_bitstring_initial_state(K=K)

initial_state = ''.join(str(b) for b in occ)

BK = BARK(H, initial_state, max_iterations=7, time_step=1, tolerance=0.1)

bases = BK.basis

energy_est = []

correct_energy = primme.eigsh(csr_matrix(H.to_matrix()), k=1, which='SA', tol=1e-6)[0][0]

for basis in bases:
    print("Size of basis:", len(basis))
    H_p = BK.project_to_subspace(basis)
    w,v = primme.eigsh(H_p, k=1, which='SA', tol=1e-6)
    energy_est.append(w[0])
    print("Estimated ground state energy:", w[0])
    print("Error:", abs(w[0] - correct_energy))