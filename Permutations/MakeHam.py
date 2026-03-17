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