from qiskit.quantum_info import SparsePauliOp
import math

def generateSchwingerHamiltonian(N, x, lam, l0, m_lat, g):
    """
    Construct the SparsePauliOp for

        W_S =  (x/2) * sum_{n=0}^{N-2} (X_n X_{n+1} + Y_n Y_{n+1})
             + (1/2) * sum_{n=0}^{N-2} sum_{k=n+1}^{N-1} (N - k - 1 + lam) * Z_n Z_k
             + sum_{n=0}^{N-2} ( N/4 - (1/2) * floor(n/2) + l0*(N - n - 1) ) * Z_n
             + (m_lat/g) * sqrt(x) * sum_{n=0}^{N-1} (-1)^n * Z_n
             + l0*2 * (N - 1) + (1/2) * l0 * N + (1/8) * N*2 + (lam/4) * N

    Args:
        N (int): number of qubits (sites).
        x (float)
        lam (float): lambda
        l0 (float): ell_0
        m_lat (float)
        g (float)

    Returns:
        SparsePauliOp on N qubits.
    """
    terms = []

    def add_1q(op, i, coeff):
        # Qiskit labels are little-endian: qubit 0 is rightmost
        s = ['I'] * N
        s[N - 1 - i] = op
        terms.append((''.join(s), complex(coeff)))

    def add_2q(op, i, j, coeff):
        s = ['I'] * N
        s[N - 1 - i] = op
        s[N - 1 - j] = op
        terms.append((''.join(s), complex(coeff)))

    # (x/2) * sum (X_i X_{i+1} + Y_i Y_{i+1})
    for n in range(N - 1):
        add_2q('X', n, n + 1, x / 2.0)
        add_2q('Y', n, n + 1, x / 2.0)

    # (1/2) * sum_{n<k} (N - k - 1 + lam) * Z_n Z_k
    for n in range(N - 1):
        for k in range(n + 1, N):
            coeff = 0.5 * (N - k - 1 + lam)
            add_2q('Z', n, k, coeff)

    # sum_{n=0}^{N-2} (N/4 - 1/2 floor(n/2) + l0*(N - n - 1)) * Z_n
    for n in range(N - 1):
        coeff = (N / 4.0) - 0.5 * math.ceil(n / 2) + l0 * (N - n - 1)
        add_1q('Z', n, coeff)

    # (m_lat/g) * sqrt(x) * sum_{n=0}^{N-1} (-1)^n * Z_n
    pref = (m_lat / g) * math.sqrt(x)
    for n in range(N):
        add_1q('Z', n, pref * ((-1) ** n))

    # Constant (identity) term
    const = (l0 ** 2) * (N - 1) + 0.5 * l0 * N + 0.125 * (N ** 2) + (lam / 4.0) * N
    terms.append(('I' * N, complex(const)))

    return SparsePauliOp.from_list(terms).simplify()


import numpy as np
from scipy.sparse.linalg import eigsh
def exact_ground_state_energy(H):
    """
    Compute the exact ground-state energy of a SparsePauliOp Hamiltonian
    using sparse diagonalization.

    Parameters
    ----------
    H : SparsePauliOp

    Returns
    -------
    float
        Ground-state energy.
    """
    # Convert to sparse matrix (CSR format)
    H_sparse = H.to_matrix(sparse=True)

    # Compute smallest algebraic eigenvalue
    evals, _ = eigsh(H_sparse, k=1, which='SA')
    return float(evals[0].real)



from qiskit.quantum_info import SparsePauliOp
import math

def WS_sparse_pauli_op(N, x, lam, l0, m_lat, g):
    """
    Construct the SparsePauliOp for

        W_S =  (x/2) * sum_{n=0}^{N-2} (X_n X_{n+1} + Y_n Y_{n+1})
             + (1/2) * sum_{n=0}^{N-2} sum_{k=n+1}^{N-1} (N - k - 1 + lam) * Z_n Z_k
             + sum_{n=0}^{N-2} ( N/4 - (1/2) * floor(n/2) + l0*(N - n - 1) ) * Z_n
             + (m_lat/g) * sqrt(x) * sum_{n=0}^{N-1} (-1)^n * Z_n
             + l0**2 * (N - 1) + (1/2) * l0 * N + (1/8) * N**2 + (lam/4) * N

    Args:
        N (int): number of qubits (sites).
        x (float)
        lam (float): lambda
        l0 (float): ell_0
        m_lat (float)
        g (float)

    Returns:
        SparsePauliOp on N qubits.
    """
    terms = []

    def add_1q(op, i, coeff):
        # Qiskit labels are little-endian: qubit 0 is rightmost
        s = ['I'] * N
        s[N - 1 - i] = op
        terms.append((''.join(s), complex(coeff)))

    def add_2q(op, i, j, coeff):
        s = ['I'] * N
        s[N - 1 - i] = op
        s[N - 1 - j] = op
        terms.append((''.join(s), complex(coeff)))

    # (x/2) * sum (X_i X_{i+1} + Y_i Y_{i+1})
    for n in range(N - 1):
        add_2q('X', n, n + 1, x / 2.0)
        add_2q('Y', n, n + 1, x / 2.0)

    # (1/2) * sum_{n<k} (N - k - 1 + lam) * Z_n Z_k
    for n in range(N - 1):
        for k in range(n + 1, N):
            coeff = 0.5 * (N - k - 1 + lam)
            add_2q('Z', n, k, coeff)

    # sum_{n=0}^{N-2} (N/4 - 1/2 floor(n/2) + l0*(N - n - 1)) * Z_n
    for n in range(N - 1):
        coeff = (N / 4.0) - 0.5 * math.ceil(n / 2) + l0 * (N - n - 1)
        add_1q('Z', n, coeff)

    # (m_lat/g) * sqrt(x) * sum_{n=0}^{N-1} (-1)^n * Z_n
    pref = (m_lat / g) * math.sqrt(x)
    for n in range(N):
        add_1q('Z', n, pref * ((-1) ** n))

    # Constant (identity) term
    const = (l0 ** 2) * (N - 1) + 0.5 * l0 * N + 0.125 * (N ** 2) + (lam / 4.0) * N
    terms.append(('I' * N, complex(const)))

    return SparsePauliOp.from_list(terms).simplify()





# and hamiltonian from skqd paper
import numpy as np

from qiskit.quantum_info import SparsePauliOp
from qiskit_nature.second_q.operators import FermionicOp
from qiskit_nature.second_q.mappers import JordanWignerMapper


def siam_diagonal_bath_sparse_pauli(
    K: int,
    U: float,
    t: float = 1.0,
    V: float = 1.0,
):
    """
    Single-impurity Anderson model in the bath-diagonal basis.

    Modes per spin sector:
      0        -> impurity orbital d
      1..K     -> bath orbitals k=0..K-1

    Total spin orbitals / qubits = 2 * (K + 1), with ordering:
      [up impurity, up bath0, ..., up bath_{K-1},
       dn impurity, dn bath0, ..., dn bath_{K-1}]

    Returns
    -------
    ham_pauli : SparsePauliOp
        Jordan-Wigner mapped qubit Hamiltonian.
    eps : np.ndarray
        Bath single-particle energies.
    Vk : np.ndarray
        Hybridizations in the diagonal-bath basis.
    """
    n_orb = K + 1                    # spatial orbitals per spin
    n_spin_orb = 2 * n_orb

    # Open-chain bath hopping matrix T, Eq. (19)
    T = np.zeros((K, K), dtype=float)
    for j in range(K - 1):
        T[j, j + 1] = -t
        T[j + 1, j] = -t

    # Diagonalize bath: eps_k and Xi
    eps, Xi = np.linalg.eigh(T)
    Vk = V * Xi[0, :]                # Eq. (21)

    def idx(spin: str, orb: int) -> int:
        # orb = 0 is impurity, 1..K are bath orbitals
        if spin == "up":
            return orb
        elif spin == "dn":
            return n_orb + orb
        raise ValueError("spin must be 'up' or 'dn'")

    terms = {}

    def add_term(label: str, coeff: complex):
        terms[label] = terms.get(label, 0.0) + coeff

    # Impurity one-body term: U/2 (n_d↑ + n_d↓), Eq. (15)
    add_term(f"+_{idx('up', 0)} -_{idx('up', 0)}", U / 2)
    add_term(f"+_{idx('dn', 0)} -_{idx('dn', 0)}", U / 2)

    # Impurity interaction: U n_d↑ n_d↓, Eq. (15)
    add_term(
        f"+_{idx('up', 0)} -_{idx('up', 0)} +_{idx('dn', 0)} -_{idx('dn', 0)}",
        U,
    )

    # Bath diagonal term: sum_{k,sigma} eps_k n_{k sigma}, Eq. (20)
    for k in range(K):
        add_term(f"+_{idx('up', k + 1)} -_{idx('up', k + 1)}", eps[k])
        add_term(f"+_{idx('dn', k + 1)} -_{idx('dn', k + 1)}", eps[k])

    # Hybridization: sum_{k,sigma} Vk (d† c_k + c_k† d), Eq. (21)
    for k in range(K):
        vk = Vk[k]
        # spin up
        add_term(f"+_{idx('up', 0)} -_{idx('up', k + 1)}", vk)
        add_term(f"+_{idx('up', k + 1)} -_{idx('up', 0)}", vk)
        # spin down
        add_term(f"+_{idx('dn', 0)} -_{idx('dn', k + 1)}", vk)
        add_term(f"+_{idx('dn', k + 1)} -_{idx('dn', 0)}", vk)

    ferm_op = FermionicOp(terms, num_spin_orbitals=n_spin_orb)
    mapper = JordanWignerMapper()
    ham_pauli = mapper.map(ferm_op).simplify()

    return ham_pauli, eps, Vk





import numpy as np

from qiskit.quantum_info import Statevector
from qiskit_nature.second_q.circuit.library import SlaterDeterminant


def siam_quadratic_initial_state(
    K: int,
    U: float,
    t: float = 1.0,
    V: float = 1.0,
    N_up: int | None = None,
    N_dn: int | None = None,
):
    """
    Reasonable initial vector for the SIAM in the bath-diagonal basis:
    the Slater determinant obtained by filling the lowest-energy orbitals
    of the quadratic part H1.

    Assumed orbital ordering per spin:
      0      -> impurity
      1..K   -> bath orbitals in the diagonal-bath basis

    Total qubit ordering:
      [up sector][down sector]

    Returns
    -------
    psi0 : Statevector
        Initial statevector on 2*(K+1) qubits.
    qc : QuantumCircuit
        Circuit preparing the same state.
    evals : np.ndarray
        One-body eigenvalues for one spin sector.
    evecs : np.ndarray
        One-body eigenvectors for one spin sector (columns).
    """
    n_orb = K + 1

    # Default: half filling, zero magnetization
    if N_up is None:
        N_up = n_orb // 2
    if N_dn is None:
        N_dn = n_orb // 2

    # Bath energies in the diagonal-bath basis from the open chain
    T = np.zeros((K, K), dtype=float)
    for j in range(K - 1):
        T[j, j + 1] = -t
        T[j + 1, j] = -t

    eps, Xi = np.linalg.eigh(T)
    Vk = V * Xi[0, :]  # impurity couples to all bath modes in this basis

    # One-body Hamiltonian h1 for a single spin sector
    # Matches the earlier SIAM convention: impurity onsite = U/2
    h1 = np.zeros((n_orb, n_orb), dtype=float)
    h1[0, 0] = U / 2
    h1[1:, 1:] = np.diag(eps)
    h1[0, 1:] = Vk
    h1[1:, 0] = Vk

    # Diagonalize h1 and occupy the lowest orbitals
    evals, evecs = np.linalg.eigh(h1)

    Q_up = evecs[:, :N_up].T   # shape (N_up, n_orb), rows orthonormal
    Q_dn = evecs[:, :N_dn].T   # same for down spin

    # Block-diagonal transformation matrix on all spin orbitals
    Q = np.zeros((N_up + N_dn, 2 * n_orb), dtype=complex)
    Q[:N_up, :n_orb] = Q_up
    Q[N_up:, n_orb:] = Q_dn

    qc = SlaterDeterminant(Q)
    psi0 = Statevector.from_instruction(qc)

    return psi0, qc, evals, evecs


# simple bitstring initial state
import numpy as np
from qiskit.quantum_info import Statevector


def siam_bitstring_initial_state(
    K: int,
    N_up: int | None = None,
    N_dn: int | None = None,
    occupy_impurity: bool = False,
):
    """
    Very simple computational-basis initial state in the diagonal-bath basis.

    Ordering:
      [up impurity, up bath0, ..., up bath_{K-1},
       dn impurity, dn bath0, ..., dn bath_{K-1}]
    """
    n_orb = K + 1
    n_qubits = 2 * n_orb

    if N_up is None:
        N_up = n_orb // 2
    if N_dn is None:
        N_dn = n_orb // 2

    occ = np.zeros(n_qubits, dtype=int)

    # up sector
    start_up = 0
    bath_up = list(range(1, n_orb))
    filled_up = 0

    if occupy_impurity and filled_up < N_up:
        occ[start_up + 0] = 1
        filled_up += 1

    for q in bath_up:
        if filled_up >= N_up:
            break
        occ[start_up + q] = 1
        filled_up += 1

    # down sector
    start_dn = n_orb
    bath_dn = list(range(1, n_orb))
    filled_dn = 0

    if occupy_impurity and filled_dn < N_dn:
        occ[start_dn + 0] = 1
        filled_dn += 1

    for q in bath_dn:
        if filled_dn >= N_dn:
            break
        occ[start_dn + q] = 1
        filled_dn += 1

    # Qiskit basis index: qubit 0 is least-significant bit
    index = sum(int(bit) << i for i, bit in enumerate(occ))
    psi0 = Statevector.from_int(index, dims=2**n_qubits)

    return psi0, occ