"""Random-spin Hamiltonian cases plus every static statistic we log for them.

A :class:`HamiltonianCase` bundles one sampled Hamiltonian with the reference
data both algorithms need (exact ground space, evaluator) and with the static
descriptors we want available at analysis time: Hamiltonian sparsity, ground
state sparsity/localisation, spectral gap, interaction-graph degrees, and so on.

The initial computational basis state is a *scanned parameter*: the study picks
the basis index whose ground-state weight is closest to a requested initial
overlap, and hands the very same index to SKQD and to BARK.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import scipy.sparse as sp
from qiskit.quantum_info import SparsePauliOp
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh

from subspace_search.hamiltonians import make_random_spin_hamiltonian

from subspace_metrics import SubspaceEvaluator, timed


# Eigen-decomposition of the full Hamiltonian is done densely up to this
# dimension; above it we fall back to ARPACK for the low end of the spectrum.
DENSE_SPECTRUM_MAX_DIM = 1024


@dataclass
class InitialState:
    """One scanned starting point, shared by SKQD and BARK."""

    spec: str                 # what the user asked for ("max", "0.05", ...)
    index: int                # computational basis index
    bitstring: str
    overlap: float            # realised |<b|ground space|b>|^2
    rank: int                 # rank of that basis state by ground-state weight


@dataclass
class HamiltonianCase:
    ham_id: str
    num_qubits: int
    dim: int
    max_interactions: int | None
    seed: int
    H_spo: SparsePauliOp
    H: sp.csr_matrix
    info: dict[str, Any]
    ground_energy: float
    first_excited_energy: float
    gap: float
    degeneracy: int
    ground_space: np.ndarray
    ground_weights: np.ndarray          # <i|P_gs|i> / degeneracy, sums to 1
    spectral_range: float
    evaluator: SubspaceEvaluator
    stats: dict[str, Any] = field(default_factory=dict)
    build_timings: dict[str, float] = field(default_factory=dict)

    # ------------------------------------------------------------------
    def initial_state(self, spec: str) -> InitialState:
        """Resolve an initial-overlap specification into a basis state."""
        return resolve_initial_state(self, spec)

    def row(self) -> dict[str, Any]:
        """Flat record for the ``hamiltonians`` table."""
        record: dict[str, Any] = {
            "ham_id": self.ham_id,
            "num_qubits": self.num_qubits,
            "hilbert_dim": self.dim,
            "max_interactions": (
                -1 if self.max_interactions is None else int(self.max_interactions)
            ),
            "max_interactions_label": (
                "all-to-all" if self.max_interactions is None
                else str(self.max_interactions)
            ),
            "seed": self.seed,
            "ground_energy": self.ground_energy,
            "first_excited_energy": self.first_excited_energy,
            "gap": self.gap,
            "degeneracy": self.degeneracy,
            "spectral_range": self.spectral_range,
        }
        record.update(self.stats)
        record.update({f"build_{k}": v for k, v in self.build_timings.items()})
        return record


# ----------------------------------------------------------------------
# Spectrum
# ----------------------------------------------------------------------
def _low_spectrum(
    H: sp.csr_matrix,
    num_eigenvalues: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (eigenvalues, eigenvectors, spectral_range) for the low end."""
    dim = H.shape[0]
    k = min(num_eigenvalues, dim)

    if dim <= DENSE_SPECTRUM_MAX_DIM:
        dense = np.asarray(H.todense())
        dense = 0.5 * (dense + dense.conj().T)
        vals, vecs = eigh(dense)
        return vals[:k], vecs[:, :k], float(vals[-1] - vals[0])

    vals, vecs = eigsh(H.tocsc(), k=k, which="SA", maxiter=50_000)
    order = np.argsort(vals)
    vals, vecs = vals[order], vecs[:, order]
    top = eigsh(H.tocsc(), k=1, which="LA", maxiter=50_000, return_eigenvectors=False)
    return vals, vecs, float(float(top[0]) - vals[0])


# ----------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------
def hamiltonian_sparsity_stats(H: sp.csr_matrix, H_spo: SparsePauliOp) -> dict[str, Any]:
    dim = H.shape[0]
    nnz = int(H.nnz)
    row_nnz = np.diff(H.indptr)
    off_diagonal = int(nnz - np.count_nonzero(H.diagonal()))
    return {
        "ham_nnz": nnz,
        "ham_density": float(nnz / (dim * dim)),
        "ham_row_nnz_mean": float(row_nnz.mean()),
        "ham_row_nnz_max": int(row_nnz.max()),
        "ham_row_nnz_min": int(row_nnz.min()),
        "ham_offdiagonal_nnz": off_diagonal,
        "ham_num_pauli_terms": int(len(H_spo.paulis)),
        "ham_coeff_abs_mean": float(np.mean(np.abs(H_spo.coeffs))),
        "ham_coeff_abs_max": float(np.max(np.abs(H_spo.coeffs))),
        "ham_frobenius_norm": float(sp.linalg.norm(H, "fro")),
    }


def ground_state_sparsity_stats(weights: np.ndarray) -> dict[str, Any]:
    """Localisation descriptors of the ground-state weight distribution."""
    dim = weights.size
    p = np.asarray(weights, dtype=float)
    p = p / p.sum()

    nonzero = p[p > 1e-12]
    ipr = float(np.sum(p**2))
    participation_ratio = 1.0 / ipr if ipr > 0 else float(dim)
    entropy = float(-np.sum(nonzero * np.log(nonzero)))

    ordered = np.sort(p)[::-1]
    cumulative = np.cumsum(ordered)

    def dim_for(fraction: float) -> int:
        hit = np.searchsorted(cumulative, fraction) + 1
        return int(min(hit, dim))

    stats = {
        "gs_support": int(np.count_nonzero(p > 1e-12)),
        "gs_support_fraction": float(np.count_nonzero(p > 1e-12) / dim),
        "gs_ipr": ipr,
        "gs_participation_ratio": participation_ratio,
        "gs_participation_fraction": float(participation_ratio / dim),
        "gs_entropy": entropy,
        "gs_entropy_normalized": float(entropy / math.log(dim)) if dim > 1 else 0.0,
        "gs_max_weight": float(ordered[0]),
    }
    for fraction in (0.5, 0.9, 0.99):
        needed = dim_for(fraction)
        tag = f"{int(fraction * 100)}"
        stats[f"gs_dim_for_{tag}pct"] = needed
        stats[f"gs_dim_fraction_for_{tag}pct"] = float(needed / dim)
    return stats


# ----------------------------------------------------------------------
# Construction
# ----------------------------------------------------------------------
def build_case(
    num_qubits: int,
    max_interactions: int | None,
    seed: int,
    model_kwargs: dict[str, Any],
    degeneracy_tol: float = 1e-9,
    num_eigenvalues: int = 8,
    dense_subspace_max_dim: int = 600,
) -> HamiltonianCase:
    """Sample one random spin Hamiltonian and precompute all reference data."""
    timings: dict[str, float] = {}

    with timed(timings, "t_sample_hamiltonian"):
        H_spo, info = make_random_spin_hamiltonian(
            num_sites=num_qubits,
            max_interactions=max_interactions,
            seed=seed,
            **model_kwargs,
        )

    with timed(timings, "t_to_matrix"):
        H = sp.csr_matrix(H_spo.to_matrix(sparse=True))

    with timed(timings, "t_exact_diagonalization"):
        values, vectors, spectral_range = _low_spectrum(H, num_eigenvalues)

    ground_energy = float(values[0])
    degeneracy = int(np.count_nonzero(values - ground_energy <= degeneracy_tol))
    degeneracy = max(1, degeneracy)
    ground_space = vectors[:, :degeneracy]
    first_excited = (
        float(values[degeneracy]) if degeneracy < values.size else float("nan")
    )
    gap = float(first_excited - ground_energy)

    weights = np.sum(np.abs(ground_space) ** 2, axis=1) / degeneracy

    stats = {}
    stats.update(hamiltonian_sparsity_stats(H, H_spo))
    stats.update(ground_state_sparsity_stats(weights))
    degrees = np.asarray(info["degrees"], dtype=int)
    stats.update(
        {
            "num_edges": len(info["edges"]),
            "edge_density": float(
                len(info["edges"]) / max(1, num_qubits * (num_qubits - 1) / 2)
            ),
            "degree_mean": float(degrees.mean()) if degrees.size else 0.0,
            "degree_max": int(degrees.max()) if degrees.size else 0,
            "J_max": info["J_max"],
            "B_max": info["B_max"],
            "J_components": "".join(info["J_components"]),
            "B_components": "".join(info["B_components"]),
            "coupling_distribution": info["coupling_distribution"],
            "field_distribution": info["field_distribution"],
        }
    )

    evaluator = SubspaceEvaluator(
        H=H,
        exact_energy=ground_energy,
        ground_space=ground_space,
        dense_max_dim=dense_subspace_max_dim,
    )

    label_mi = "inf" if max_interactions is None else str(max_interactions)
    ham_id = f"n{num_qubits}_mi{label_mi}_s{seed}"

    return HamiltonianCase(
        ham_id=ham_id,
        num_qubits=num_qubits,
        dim=H.shape[0],
        max_interactions=max_interactions,
        seed=seed,
        H_spo=H_spo,
        H=H,
        info=info,
        ground_energy=ground_energy,
        first_excited_energy=first_excited,
        gap=gap,
        degeneracy=degeneracy,
        ground_space=ground_space,
        ground_weights=weights,
        spectral_range=spectral_range,
        evaluator=evaluator,
        stats=stats,
        build_timings=timings,
    )


# ----------------------------------------------------------------------
# Initial state selection
# ----------------------------------------------------------------------
def resolve_initial_state(case: HamiltonianCase, spec: str) -> InitialState:
    """Pick the basis state realising (as closely as possible) ``spec``.

    ``spec`` is either ``"max"`` (the highest-weight basis state, the usual
    "informed" start) or a number, in which case the basis state whose
    ground-state weight is closest to it *in log space* is chosen. Log space
    matters because the interesting overlaps span many orders of magnitude.
    """
    weights = case.ground_weights
    ranking = np.argsort(weights)[::-1]
    rank_of = {int(idx): position for position, idx in enumerate(ranking)}

    if spec == "max":
        index = int(ranking[0])
    else:
        try:
            target = float(spec)
        except ValueError as exc:  # pragma: no cover - argparse guards this
            raise ValueError(f"Unknown initial-overlap spec {spec!r}") from exc
        if not 0.0 < target <= 1.0:
            raise ValueError("initial overlap targets must lie in (0, 1]")
        usable = np.where(weights > 1e-15)[0]
        if usable.size == 0:
            index = int(ranking[0])
        else:
            distance = np.abs(np.log10(weights[usable]) - math.log10(target))
            index = int(usable[int(np.argmin(distance))])

    return InitialState(
        spec=spec,
        index=index,
        bitstring=format(index, f"0{case.num_qubits}b"),
        overlap=float(weights[index]),
        rank=int(rank_of[index]),
    )
