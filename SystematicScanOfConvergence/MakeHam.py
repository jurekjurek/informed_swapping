import numpy as np
import scipy.sparse as sp


def make_sparse_ground_state_hamiltonian_from_qubits(
    n_qubits: int,
    ground_state_sparsity: float | int,
    seed: int = 42,
    ground_energy: float = 0.0,
    gap: float = 1.0,
    random_strength: float = 1.0,
    return_sparse: bool = False,
) -> tuple[np.ndarray | sp.csr_matrix, np.ndarray, np.ndarray]:
    """
    Construct a random Hermitian Hamiltonian H such that:

      - H is generally non-diagonal and dense
      - psi is an exact ground state of H
      - psi has the requested sparsity in the computational basis
      - all other eigenvalues are >= ground_energy + gap

    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    ground_state_sparsity : float | int
        If float in (0,1], interpreted as fraction of basis states in support.
        If int, interpreted as exact number of nonzero amplitudes.
    seed : int
        Random seed.
    ground_energy : float
        Ground-state energy.
    gap : float
        Minimum spectral gap above the ground state.
    random_strength : float
        Strength of the random excited-space term.
    return_sparse : bool
        If True, return csr_matrix. Note: the matrix will usually be dense in content.

    Returns
    -------
    H : np.ndarray | sp.csr_matrix
        Hermitian Hamiltonian.
    psi : np.ndarray
        Ground-state vector.
    support : np.ndarray
        Indices where psi is nonzero.
    """
    size = 2 ** n_qubits
    rng = np.random.default_rng(seed)

    # Interpret sparsity
    if isinstance(ground_state_sparsity, float):
        if not (0 < ground_state_sparsity <= 1):
            raise ValueError("Float sparsity must satisfy 0 < sparsity <= 1.")
        k = int(np.ceil(ground_state_sparsity * size))
    else:
        k = int(ground_state_sparsity)

    if not (1 <= k <= size):
        raise ValueError(f"k must satisfy 1 <= k <= {size}")

    # Random sparse ground state
    support = np.sort(rng.choice(size, size=k, replace=False))
    amps = rng.standard_normal(k) + 1j * rng.standard_normal(k)
    amps /= np.linalg.norm(amps)

    psi = np.zeros(size, dtype=np.complex128)
    psi[support] = amps

    # Rank-1 projector onto ground state
    P = np.outer(psi, psi.conj())
    I = np.eye(size, dtype=np.complex128)
    Q = I - P

    # Random Hermitian matrix
    X = rng.standard_normal((size, size)) + 1j * rng.standard_normal((size, size))
    X = 0.5 * (X + X.conj().T)

    # Make excited-space part positive semidefinite so spectrum is controlled
    # R = X^† X is PSD
    R = X.conj().T @ X
    R /= max(size, 1)

    # Excited-space Hamiltonian:
    # on the orthogonal complement, eigenvalues are >= ground_energy + gap
    excited_part = (ground_energy + gap) * Q + np.abs((ground_energy + gap))*random_strength * (Q @ R @ Q)

    # Full Hamiltonian
    H = ground_energy * P + excited_part

    # Numerical Hermiticity cleanup
    H = 0.5 * (H + H.conj().T)

    if return_sparse:
        return sp.csr_matrix(H), psi, support
    return H, psi, support