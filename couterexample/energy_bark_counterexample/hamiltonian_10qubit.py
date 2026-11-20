"""Explicit sparse Hamiltonian for the energy-BARK vs SKQD counterexample.

Hamiltonian has 2**m+4 active computational-basis configurations.
Set embed_qubits=True to pad it to 2**(m+1) basis states, so m=9 is 10 qubits.
No dependence on the counterexample's SKQD/BARK runners.
"""
from scipy.sparse import block_diag, diags, lil_matrix


def make_counterexample(m=9, U=None, embed_qubits=True):
    """Return (CSR Hamiltonian, dict of named basis indices)."""
    if m < 2:
        raise ValueError("m must be >= 2")
    U = float(20 * m**2 if U is None else U)
    M = 1 << m
    s, a, y, d, c = 0, M, M + 1, M + 2, M + 3
    H = lil_matrix((M + 4, M + 4), dtype=float)

    def edge(i, j, value):
        H[i, j] = value
        H[j, i] = value

    for z in range(M):
        H[z, z] = U * z.bit_count()
        for i in range(m):
            other = z ^ (1 << i)
            if z < other:
                edge(z, other, -1.0)

    edge(s, a, -2.0)
    for i in range(m):
        edge(a, 1 << i, -1.0)
    edge(s, y, +1.8)
    edge(a, y, -1.8)
    edge(y, d, -1.0)
    edge(d, c, -1.0)
    H[c, c] = -2.2
    H = H.tocsr()

    if embed_qubits:
        padding = (1 << (m + 1)) - (M + 4)
        if padding:
            H = block_diag((H, diags([float((m + 2) * U + 10)] * padding)), format="csr")

    return H, {"s": s, "a": a, "y": y, "d": d, "c": c}


if __name__ == "__main__":
    from scipy.sparse.linalg import eigsh, expm_multiply
    import numpy as np

    m = 9
    H, ids = make_counterexample(m)
    eigs = np.sort(eigsh(H, k=2, which="SA", return_eigenvectors=False))
    psi0 = np.zeros(H.shape[0], dtype=complex)
    psi0[ids["s"]] = 1.0
    evolved = expm_multiply(-1.5j * H, psi0)
    print("shape:", H.shape, "nnz:", H.nnz, "basis indices:", ids)
    print("E0:", eigs[0], "gap:", eigs[1] - eigs[0])
    print("t=1.5 probabilities:", {k: float(abs(evolved[v])**2) for k, v in ids.items()})
