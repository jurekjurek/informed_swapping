import numpy as np
from scipy.sparse import csr_matrix, isspmatrix_csr
from qiskit.quantum_info import SparsePauliOp


def hermitian_csr_to_sparse_pauli_op(mat: csr_matrix, tol: float = 1e-12) -> SparsePauliOp:
    """
    Convert a Hermitian scipy.sparse.csr_matrix of size (2^n, 2^n)
    into a qiskit.quantum_info.SparsePauliOp.

    Parameters
    ----------
    mat
        Hermitian operator in CSR format.
    tol
        Numerical tolerance used for Hermiticity checking and coefficient pruning.

    Returns
    -------
    SparsePauliOp
        Pauli decomposition of `mat`.

    Notes
    -----
    - The matrix dimension must be a power of two.
    - This is still exponential in the number of qubits in the worst case:
      a generic n-qubit operator can require up to 4^n Pauli terms.
    - Qiskit Pauli labels are returned in its usual string order:
      leftmost character is the highest-index qubit.
    """
    if not isspmatrix_csr(mat):
        raise TypeError("Input must be a scipy.sparse.csr_matrix.")

    if mat.shape[0] != mat.shape[1]:
        raise ValueError("Matrix must be square.")

    dim = mat.shape[0]
    n = int(round(np.log2(dim)))
    if 2**n != dim:
        raise ValueError("Matrix dimension must be a power of 2.")

    # Hermiticity check
    diff = (mat - mat.getH()).tocoo()
    if diff.nnz and np.max(np.abs(diff.data)) > tol:
        raise ValueError("Input matrix is not Hermitian within tolerance.")

    # Clean canonical CSR structure
    mat = mat.copy()
    mat.sum_duplicates()
    mat.eliminate_zeros()

    coo = mat.tocoo()
    rows = coo.row.astype(np.uint64)
    cols = coo.col.astype(np.uint64)
    data = coo.data.astype(complex)

    # Group by X-mask = row XOR col
    x_masks = rows ^ cols

    def parity_u64(arr: np.ndarray) -> np.ndarray:
        """Return popcount(arr) mod 2 as uint8."""
        return np.fromiter(
            ((int(v).bit_count() & 1) for v in arr),
            count=len(arr),
            dtype=np.uint8,
        )

    def pauli_label(xmask: int, zmask: int, nqubits: int) -> str:
        chars = []
        for q in range(nqubits):
            xb = (xmask >> q) & 1
            zb = (zmask >> q) & 1
            if xb == 0 and zb == 0:
                chars.append("I")
            elif xb == 1 and zb == 0:
                chars.append("X")
            elif xb == 0 and zb == 1:
                chars.append("Z")
            else:
                chars.append("Y")
        # Qiskit label order: q_{n-1} ... q_0
        return "".join(reversed(chars))

    pauli_terms = []

    # Iterate over all possible X-supports
    for xmask in range(1 << n):
        sel = (x_masks == xmask)
        if not np.any(sel):
            continue

        r = rows[sel]
        vals = data[sel]

        # Precompute phase from overlapping X/Z bits: i^{|x & z|}
        # Then coefficient is:
        # c_{x,z} = 2^{-n} sum_r H[r, r xor x] * i^{|x&z|} * (-1)^{z·r}
        for zmask in range(1 << n):
            overlap = (xmask & zmask).bit_count()
            global_phase = (1j) ** overlap

            signs = 1 - 2 * parity_u64(r & np.uint64(zmask))   # +1 or -1
            coeff = global_phase * np.dot(signs.astype(complex), vals) / dim

            if abs(coeff) > tol:
                pauli_terms.append((pauli_label(xmask, zmask, n), coeff))

    if not pauli_terms:
        pauli_terms = [("I" * n, 0.0)]

    return SparsePauliOp.from_list(pauli_terms).simplify(atol=tol)