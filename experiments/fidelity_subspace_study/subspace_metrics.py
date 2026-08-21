"""Shared subspace evaluation and timing helpers for the fidelity study.

Everything that both SKQD and BARK need in order to be compared on an equal
footing lives here:

* :func:`timed` -- a context manager that accumulates wall-clock seconds into a
  plain ``dict``. Every runner uses it so the per-iteration timing columns have
  identical semantics across methods.
* :class:`SubspaceEvaluator` -- projects ``H`` onto a set of computational basis
  indices, diagonalises **once** (never per bitstring), and reports the
  variational energy plus the fidelity with the exact ground space.

Fidelity convention
-------------------
``fidelity = || P_gs |v> ||^2`` where ``|v>`` is the lowest eigenvector of the
projected Hamiltonian (embedded back into the full space) and ``P_gs`` projects
onto the exact ground *space*. Using the eigenspace projector rather than a
single eigenvector keeps the metric well defined when the ground state is
degenerate.

``captured_weight = sum_{i in S} <i|P_gs|i>`` is the basis-overlap upper bound on
that fidelity: no subspace can beat the ground-state weight it contains.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from typing import Iterator, Sequence

import numpy as np
import scipy.sparse as sp
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh


@contextmanager
def timed(store: dict[str, float], key: str) -> Iterator[None]:
    """Accumulate the wall-clock duration of the block into ``store[key]``."""
    start = time.perf_counter()
    try:
        yield
    finally:
        store[key] = store.get(key, 0.0) + (time.perf_counter() - start)


@dataclass
class SubspaceEvaluation:
    """Result of diagonalising ``H`` inside one subspace."""

    subspace_dim: int
    energy: float
    energy_error: float
    fidelity: float
    captured_weight: float
    t_project: float
    t_diagonalize: float
    t_fidelity: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


class SubspaceEvaluator:
    """Diagonalise ``H`` restricted to a growing set of basis indices.

    Parameters
    ----------
    H:
        Full Hamiltonian as a CSR matrix (dimension ``2**num_qubits``).
    exact_energy:
        Exact ground energy of ``H``.
    ground_space:
        ``(dim, degeneracy)`` matrix whose columns are an orthonormal basis of
        the exact ground eigenspace.
    dense_max_dim:
        Subspaces up to this dimension are diagonalised densely (LAPACK), larger
        ones with ``eigsh``. Dense is both faster and far more robust for the
        small subspaces this study cares about.
    """

    def __init__(
        self,
        H: sp.csr_matrix,
        exact_energy: float,
        ground_space: np.ndarray,
        dense_max_dim: int = 600,
    ) -> None:
        self.H = H.tocsr()
        self.dim = H.shape[0]
        self.exact_energy = float(exact_energy)
        self.ground_space = np.asarray(ground_space)
        self.dense_max_dim = int(dense_max_dim)
        # Diagonal of the ground-space projector: <i|P_gs|i>. Summing it over a
        # subspace gives the captured ground-state weight without ever forming
        # the (dim x dim) projector.
        self._projector_diagonal = np.sum(np.abs(self.ground_space) ** 2, axis=1)

    # ------------------------------------------------------------------
    def captured_weight(self, indices: Sequence[int]) -> float:
        return float(self._projector_diagonal[np.asarray(indices, dtype=int)].sum())

    # ------------------------------------------------------------------
    def evaluate(self, indices: Sequence[int]) -> SubspaceEvaluation:
        """Project, diagonalise once, and score the resulting subspace."""
        idx = np.asarray(indices, dtype=int)
        timings: dict[str, float] = {}

        with timed(timings, "t_project"):
            H_sub = self.H[idx, :][:, idx]

        with timed(timings, "t_diagonalize"):
            energy, vector = self._lowest_eigenpair(H_sub)

        with timed(timings, "t_fidelity"):
            # Embed the subspace eigenvector and project onto the ground space.
            overlaps = self.ground_space[idx, :].conj().T @ vector
            fidelity = float(np.sum(np.abs(overlaps) ** 2))
            captured = self.captured_weight(idx)

        return SubspaceEvaluation(
            subspace_dim=int(idx.size),
            energy=float(energy),
            energy_error=float(energy - self.exact_energy),
            fidelity=min(1.0, max(0.0, fidelity)),
            captured_weight=min(1.0, max(0.0, captured)),
            t_project=timings.get("t_project", 0.0),
            t_diagonalize=timings.get("t_diagonalize", 0.0),
            t_fidelity=timings.get("t_fidelity", 0.0),
        )

    # ------------------------------------------------------------------
    def _lowest_eigenpair(self, H_sub: sp.spmatrix) -> tuple[float, np.ndarray]:
        dim = H_sub.shape[0]
        if dim == 1:
            value = complex(H_sub.toarray()[0, 0]).real
            return float(value), np.array([1.0 + 0.0j])

        if dim <= self.dense_max_dim:
            dense = np.asarray(H_sub.todense())
            dense = 0.5 * (dense + dense.conj().T)
            vals, vecs = eigh(dense, subset_by_index=(0, 0))
            return float(vals[0]), vecs[:, 0]

        try:
            vals, vecs = eigsh(H_sub.tocsc(), k=1, which="SA", maxiter=20_000)
            return float(vals[0]), vecs[:, 0]
        except Exception:  # pragma: no cover - ARPACK convergence fallback
            dense = np.asarray(H_sub.todense())
            dense = 0.5 * (dense + dense.conj().T)
            vals, vecs = eigh(dense, subset_by_index=(0, 0))
            return float(vals[0]), vecs[:, 0]


def first_index_reaching(
    dims: Sequence[int],
    fidelities: Sequence[float],
    target: float,
) -> int | None:
    """Smallest subspace dimension whose fidelity is >= ``target``.

    ``dims``/``fidelities`` are the per-iteration trajectory of a single run.
    Returns ``None`` when the target was never reached.
    """
    for dim, fidelity in zip(dims, fidelities):
        if fidelity >= target:
            return int(dim)
    return None
