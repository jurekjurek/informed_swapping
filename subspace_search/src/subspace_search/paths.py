import itertools
import math
from typing import Iterable, List, Tuple, Optional

import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

from scipy.sparse.linalg import eigsh, LinearOperator
from scipy.linalg import eigh
from typing import List, Tuple, Optional
import tqdm

# Function to project down in a subspace given by a list of indices
def project_down(H: sp.csr_matrix, indices: List[int]) -> sp.csr_matrix:
    """
    Project down the Hamiltonian H to the subspace given by the list of indices.
    """
    return H[indices, :][:, indices]

def get_permutations(n: int, k: int, first: Optional[int] = None) -> List[Tuple[int, ...]]:
    """
    Get k unique random permutations (length k) from numbers 0 to n-1.
    
    Parameters:
        n (int): range of numbers (0 to n-1)
        k (int): number of permutations AND length of each permutation
        first (Optional[int]): if set, fixes the first element of each permutation
    
    Returns:
        List of unique permutations (tuples)
    """
    if first is not None and not (0 <= first < n):
        raise ValueError("first must be in range [0, n-1]")

    rng = np.random.default_rng()
    permutations = set()

    while len(permutations) < k:
        perm = list(rng.permutation(n))

        if first is not None:
            # Move `first` to the front
            perm.remove(first)
            perm = [first] + perm

        # Take first k elements
        perm_tuple = tuple(perm)

        permutations.add(perm_tuple)

    return list(permutations)

def get_one_path(H: sp.csr_matrix, indices: List[int]) -> np.ndarray:
    """
    Get one path through the subspace defined by the indices, starting from the first index.
    """
    energies = np.empty(len(indices))
    for i, idx in enumerate(indices):
        H_proj = project_down(H, indices[:i+1])
        # Compute the smallest eigenvalue of the projected Hamiltonian
        # Check size before calling eigsh
        if H_proj.shape[0] <= 2:
            # For very small matrices, use dense eigenvalue solver
            energies[i] = eigh(H_proj.toarray(), eigvals_only=True)[0]
        else:
            energies[i] = eigsh(H_proj, k=1, which='SA', return_eigenvectors=False)[0]
    return energies

def get_all_paths(H: sp.csr_matrix, number_of_paths: int, start: int) -> np.ndarray:
    """
    Get all paths through the subspace defined by the permutations of indices.
    """
    n = H.shape[0]
    k = number_of_paths
    permutations = get_permutations(n, k, first=start)
    
    all_energies = np.empty((k, n))
    for i, perm in tqdm.tqdm(enumerate(permutations), total=k, desc="Computing paths"):
        all_energies[i] = get_one_path(H, list(perm))
    
    return all_energies