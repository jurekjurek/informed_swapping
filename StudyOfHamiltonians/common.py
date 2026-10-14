"""
Shared pieces of the Hamiltonian study: building one Hamiltonian exactly as
EqualNumberOfBitstrings.run_cell does, and measuring the quantities that decide
whether it is useful for the fixed-budget study.

Nothing here is imported *into* the main folder; the main folder is only read.
"""

import os
import sys

import numpy as np
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from RandomSpinModel import make_heisenberg_hamiltonian  # noqa: E402

# The definitions of EqualNumberOfBitstrings.py, copied rather than imported so
# that this study does not pull in matplotlib styling, BARK and SKQD.
DENSITY_AMPLITUDE = 1e-3          # |psi_i| above this counts towards the density
OVERLAP_FLOOR = 1e-12             # p_i above this is a valid initial state
MIN_GAP = 1e-4                    # drop_small_gaps
TRIVIAL_GROUND_STATE = 1.0 - 1e-10  # drop_trivial_ground_states
DENSE_LIMIT = 256                 # below this a dense eigh is faster than eigsh

# Quantiles of log10(overlap) over the support. Initial states are drawn
# uniformly from the support, so these are quantiles of the overlap a run starts
# from.
OVERLAP_QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)


def build(num_sites, dimension, delta, J, field):
    """Sparse matrix of the study's Hamiltonian (spin = 1, i.e. Pauli operators)."""
    matrix = make_heisenberg_hamiltonian(
        num_sites=num_sites, dimension=dimension, delta=delta, J=[J],
        h=tuple(field), spin=1, max_distance=1)[0].to_matrix(sparse=True)
    # By = 0 everywhere in this study, so the matrix is real.
    return matrix.real.tocsr()


def lowest_two(matrix):
    if matrix.shape[0] <= DENSE_LIMIT:
        values, vectors = eigh(matrix.toarray())
        return values[:2], vectors[:, 0]
    values, vectors = eigsh(matrix, k=2, which="SA", tol=1e-10)
    order = np.argsort(values)
    return values[order], vectors[:, order[0]]


def measure(num_sites, dimension, delta, J, field):
    """All per-Hamiltonian quantities the study's filters and facets look at."""
    values, ground_state = lowest_two(build(num_sites, dimension, delta, J, field))
    probabilities = np.abs(ground_state) ** 2
    support = probabilities[probabilities > OVERLAP_FLOOR]
    log_support = np.log10(support)
    row = {
        "density": np.count_nonzero(np.abs(ground_state) > DENSITY_AMPLITUDE) / ground_state.size,
        "gap": float(values[1] - values[0]),
        "top_weight": float(probabilities.max()),
        "support_fraction": support.size / ground_state.size,
    }
    for q, value in zip(OVERLAP_QUANTILES, np.quantile(log_support, OVERLAP_QUANTILES)):
        row[f"log_overlap_q{int(q * 100):02d}"] = float(value)
    row["kept"] = (row["gap"] >= MIN_GAP) and (row["top_weight"] < TRIVIAL_GROUND_STATE)
    return row, probabilities
