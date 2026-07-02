import itertools
import math
from typing import Iterable, List, Tuple, Optional

import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt


# ============================================================
# Hamiltonian construction
# ============================================================

def make_hermitian_sparse_random_from_qubits(
    n_qubits: int,
    density: float,
    seed: int = 42,
) -> sp.csr_matrix:
    """
    Build a random sparse Hermitian Hamiltonian on the computational basis
    of n_qubits, so the matrix size is 2**n_qubits.
    """
    size = 2 ** n_qubits
    rng = np.random.default_rng(seed)

    A = sp.random(
        size,
        size,
        density=density,
        format="csr",
        dtype=np.complex128,
        random_state=rng,
        data_rvs=lambda n: rng.standard_normal(n) + 1j * rng.standard_normal(n),
    )

    H = A + A.conj().T
    H = 0.5 * (H + H.getH())  # enforce exact Hermiticity numerically
    return H.tocsr()


def make_sparse_ground_state_hamiltonian_from_qubits(
    n_qubits: int,
    ground_state_sparsity: float | int,
    seed: int = 42,
    ground_energy: float = 0.0,
    gap: float = 1.0,
    outside_energy: float | None = None,
    epsilon: float = 1e-12,
    add_excited_randomness: bool = False,
    excited_random_strength: float = 0.1,
) -> tuple[sp.csr_matrix, np.ndarray]:
    """
    Same as before, but avoids exact zeros by replacing them with ~epsilon.
    """

    size = 2 ** n_qubits
    rng = np.random.default_rng(seed)

    if outside_energy is None:
        outside_energy = ground_energy + 2.0 * gap

    # Interpret sparsity
    if isinstance(ground_state_sparsity, float):
        k = int(np.ceil(ground_state_sparsity * size))
    else:
        k = int(ground_state_sparsity)

    if not (1 <= k <= size):
        raise ValueError(f"k must satisfy 1 <= k <= {size}")

    # Random support
    support = np.sort(rng.choice(size, size=k, replace=False))

    # Sparse ground state
    amps = rng.standard_normal(k) + 1j * rng.standard_normal(k)
    amps /= np.linalg.norm(amps)

    psi = np.zeros(size, dtype=np.complex128)
    psi[support] = amps

    # --- Support block ---
    I_k = np.eye(k, dtype=np.complex128)
    projector = np.outer(amps, amps.conj())

    H_block = ground_energy * I_k + gap * (I_k - projector)

    # Optional randomness (preserves ground state)
    if add_excited_randomness and k > 1:
        X = rng.standard_normal((k, k)) + 1j * rng.standard_normal((k, k))
        X = 0.5 * (X + X.conj().T)

        P_perp = I_k - projector
        Y = P_perp @ X @ P_perp

        evals, evecs = np.linalg.eigh(Y)
        evals = np.clip(evals, 0.0, None)
        Y_psd = (evecs * evals) @ evecs.conj().T

        H_block += excited_random_strength * Y_psd

    # --- Replace exact zeros in support block ---
    H_block[np.abs(H_block) < epsilon] = epsilon

    # --- Outside diagonal ---
    diag = np.full(size, outside_energy, dtype=np.complex128)

    # Instead of exact zero shift, use epsilon
    diag[support] = epsilon

    H = sp.diags(diag, offsets=0, format="lil")

    # Insert support block
    for a, ia in enumerate(support):
        for b, ib in enumerate(support):
            val = H_block[a, b]

            # Ensure no exact zeros
            if abs(val) < epsilon:
                val = epsilon

            H[ia, ib] = H[ia, ib] + val

    H = H.tocsr()
    H = 0.5 * (H + H.getH())  # enforce Hermiticity

    return H, psi