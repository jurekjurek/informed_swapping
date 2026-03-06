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