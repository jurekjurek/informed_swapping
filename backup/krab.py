from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
from typing import Dict, List, Tuple, Union, Optional, Literal

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh


SparseState = Dict[int, complex]


@dataclass
class IterationLog:
    iteration: int
    energy_before_prune: float
    energy_after_prune: float
    n_active_before_prune: int
    n_active_after_prune: int
    n_raw_generated: int
    n_external_candidates_above_delta: int
    n_selected_new: int
    residual_norm_external: float


@dataclass
class SelectedKrylovResult:
    energy: float
    active_indices: List[int]
    active_bitstrings: List[str]
    coefficients: np.ndarray
    history: List[IterationLog]


def bitstring_to_index(bitstring: str) -> int:
    """
    Convert computational-basis bitstring to integer index.

    Convention:
        '000' -> 0
        '001' -> 1
        '010' -> 2
        ...
        '111' -> 7

    This is the standard big-endian binary convention.
    Make sure this matches the convention used by your Hamiltonian matrix.
    """
    return int(bitstring, 2)


def index_to_bitstring(index: int, n_qubits: int) -> str:
    return format(index, f"0{n_qubits}b")


def infer_n_qubits(dim: int) -> int:
    n_qubits = int(round(np.log2(dim)))
    if 2**n_qubits != dim:
        raise ValueError(f"Hamiltonian dimension {dim} is not a power of 2.")
    return n_qubits


def as_csr_hamiltonian(H) -> sp.csr_matrix:
    """
    Accept either a scipy sparse matrix or a Qiskit SparsePauliOp-like object.
    """
    if sp.issparse(H):
        return H.tocsr()

    # For qiskit.quantum_info.SparsePauliOp
    if hasattr(H, "to_matrix"):
        return sp.csr_matrix(H.to_matrix(sparse=True))

    raise TypeError(
        "H must be a scipy sparse matrix or an object with .to_matrix(sparse=True), "
        "such as qiskit.quantum_info.SparsePauliOp."
    )


def apply_h_to_sparse_state(
    H_csc: sp.csc_matrix,
    state: SparseState,
    drop_tol: float = 0.0,
) -> SparseState:
    """
    Apply H to a sparse computational-basis state.

    Input:
        state = {basis_index: amplitude}

    Output:
        H|state> = {basis_index: amplitude}

    This uses CSC column access. Since H|b> is column b of H,
    this is efficient when the state has sparse support.
    """
    out = defaultdict(complex)

    for col_index, coeff in state.items():
        if abs(coeff) <= drop_tol:
            continue

        start = H_csc.indptr[col_index]
        end = H_csc.indptr[col_index + 1]

        rows = H_csc.indices[start:end]
        vals = H_csc.data[start:end]

        for row_index, h_val in zip(rows, vals):
            out[row_index] += h_val * coeff

    if drop_tol > 0.0:
        return {i: amp for i, amp in out.items() if abs(amp) > drop_tol}

    return dict(out)


def project_hamiltonian_to_subspace(
    H_csc: sp.csc_matrix,
    active: List[int],
    hermitize: bool = True,
) -> sp.csr_matrix:
    """
    Build H_S = P_S H P_S, where S is the active bitstring set.

    The basis order is exactly the order of `active`.
    """
    m = len(active)
    position = {basis_index: i for i, basis_index in enumerate(active)}

    rows = []
    cols = []
    data = []

    for col_pos, basis_col in enumerate(active):
        start = H_csc.indptr[basis_col]
        end = H_csc.indptr[basis_col + 1]

        full_rows = H_csc.indices[start:end]
        full_vals = H_csc.data[start:end]

        for full_row, h_val in zip(full_rows, full_vals):
            row_pos = position.get(full_row)
            if row_pos is not None:
                rows.append(row_pos)
                cols.append(col_pos)
                data.append(h_val)

    H_sub = sp.coo_matrix((data, (rows, cols)), shape=(m, m)).tocsr()

    # Numerically enforce Hermiticity inside the selected subspace.
    if hermitize:
        H_sub = 0.5 * (H_sub + H_sub.getH())

    return H_sub


def lowest_eigenpair(
    H_sub: sp.csr_matrix,
    dense_cutoff: int = 512,
    eigsh_tol: float = 1e-10,
) -> Tuple[float, np.ndarray]:
    """
    Return lowest eigenvalue and corresponding normalized eigenvector
    of the projected Hamiltonian.
    """
    m = H_sub.shape[0]

    if m == 0:
        raise ValueError("Cannot diagonalize an empty subspace.")

    if m == 1:
        return float(np.real(H_sub[0, 0])), np.array([1.0 + 0.0j])

    if m <= dense_cutoff:
        H_dense = H_sub.toarray()
        evals, evecs = np.linalg.eigh(H_dense)
        idx = np.argmin(evals)
        return float(np.real(evals[idx])), evecs[:, idx]

    # Sparse lowest-eigenpair solve for larger active spaces.
    evals, evecs = eigsh(H_sub, k=1, which="SA", tol=eigsh_tol)
    return float(np.real(evals[0])), evecs[:, 0]


def normalize_sparse_state(state: SparseState) -> SparseState:
    norm = np.sqrt(sum(abs(v) ** 2 for v in state.values()))
    if norm == 0.0:
        raise ValueError("Cannot normalize zero state.")
    return {k: v / norm for k, v in state.items()}


def sparse_state_rayleigh_energy(
    Hpsi: SparseState,
    psi: SparseState,
) -> float:
    """
    Compute <psi|H|psi>, assuming Hpsi = H|psi>.
    """
    energy = 0.0 + 0.0j
    for idx, coeff in psi.items():
        energy += np.conjugate(coeff) * Hpsi.get(idx, 0.0)
    return float(np.real(energy))


def prune_by_coefficients(
    active: List[int],
    coeffs: np.ndarray,
    epsilon: float,
    min_keep: int = 1,
    use_probability: bool = True,
) -> Tuple[List[int], np.ndarray]:
    """
    Post-diagonalization pruning.

    If use_probability=True:
        keep b if |c_b|^2 >= epsilon.

    If use_probability=False:
        keep b if |c_b| >= epsilon.
    """
    if len(active) != len(coeffs):
        raise ValueError("active and coeffs must have same length.")

    if use_probability:
        weights = np.abs(coeffs) ** 2
    else:
        weights = np.abs(coeffs)

    keep_positions = [i for i, w in enumerate(weights) if w >= epsilon]

    # Avoid accidentally deleting the entire variational space.
    if len(keep_positions) < min_keep:
        keep_positions = list(np.argsort(weights)[-min_keep:])

    # Keep a deterministic order inherited from `active`.
    keep_positions = sorted(set(keep_positions))

    new_active = [active[i] for i in keep_positions]
    new_coeffs = coeffs[keep_positions]

    # Renormalize the pruned approximate eigenstate.
    norm = np.linalg.norm(new_coeffs)
    if norm == 0.0:
        raise ValueError("Pruned coefficient vector has zero norm.")
    new_coeffs = new_coeffs / norm

    return new_active, new_coeffs


def select_top_initial_components(
    h_seed: SparseState,
    Q: int,
    include_seed_index: Optional[int] = None,
) -> List[int]:
    """
    Initial selection after applying H to the starting bitstring.

    Since we do not yet have a variational eigenvector, we select by
    probability weight |amplitude|^2 in H|seed>.
    """
    ranked = sorted(
        h_seed.items(),
        key=lambda item: abs(item[1]) ** 2,
        reverse=True,
    )

    selected = [idx for idx, amp in ranked[:Q]]

    if include_seed_index is not None and include_seed_index not in selected:
        selected = [include_seed_index] + selected

    # Remove duplicates while preserving order.
    return list(dict.fromkeys(selected))


def select_new_candidates(
    Hpsi: SparseState,
    active_set: set[int],
    Q: int,
    delta: float,
    score_mode: Literal["residual", "pt2"] = "residual",
    diagonal: Optional[np.ndarray] = None,
    current_energy: Optional[float] = None,
    denom_floor: float = 1e-12,
) -> Tuple[List[int], Dict[int, complex], Dict[int, float]]:
    """
    Pre-diagonalization candidate pruning.

    Candidates are external bitstrings a not in the current active set.

    r_a = <a|H|psi>

    If score_mode == "residual":
        score(a) = |r_a|

    If score_mode == "pt2":
        score(a) = |r_a|^2 / max(H_aa - E, denom_floor)

    The threshold delta is always applied to |r_a|.
    """
    external_residuals = {
        idx: amp
        for idx, amp in Hpsi.items()
        if idx not in active_set and abs(amp) >= delta
    }

    scores = {}

    for idx, r_a in external_residuals.items():
        if score_mode == "residual":
            scores[idx] = abs(r_a)

        elif score_mode == "pt2":
            if diagonal is None or current_energy is None:
                raise ValueError(
                    "For score_mode='pt2', diagonal and current_energy are required."
                )

            denom = float(np.real(diagonal[idx])) - current_energy
            denom_eff = max(denom, denom_floor)
            scores[idx] = abs(r_a) ** 2 / denom_eff

        else:
            raise ValueError(f"Unknown score_mode: {score_mode}")

    ranked = sorted(scores.keys(), key=lambda idx: scores[idx], reverse=True)
    selected = ranked[:Q]

    return selected, external_residuals, scores


def make_sparse_state_from_active_and_coeffs(
    active: List[int],
    coeffs: np.ndarray,
    drop_tol: float = 0.0,
) -> SparseState:
    state = {
        basis_index: complex(coeff)
        for basis_index, coeff in zip(active, coeffs)
        if abs(coeff) > drop_tol
    }
    return normalize_sparse_state(state)


def selected_krylov_ground_state(
    H,
    initial_bitstring: Union[str, int],
    Q: int,
    delta: float,
    epsilon: float,
    n_iterations: int,
    *,
    include_initial_bitstring: bool = True,
    score_mode: Literal["residual", "pt2"] = "residual",
    min_keep_after_prune: int = 1,
    coefficient_prune_uses_probability: bool = True,
    dense_cutoff: int = 512,
    eigsh_tol: float = 1e-10,
    drop_tol: float = 0.0,
    hermitize_subspace: bool = True,
) -> SelectedKrylovResult:
    """
    Accumulated selected-subspace Krylov-like ground-state search.

    Parameters
    ----------
    H:
        Hamiltonian as scipy sparse matrix, or Qiskit SparsePauliOp-like object.

    initial_bitstring:
        Starting computational basis state. Either a bitstring like "01011"
        or an integer basis index.

    Q:
        Maximum number of newly generated candidate bitstrings to add per iteration.

    delta:
        Pre-diagonalization threshold.
        A generated external bitstring a is considered only if |r_a| >= delta.

    epsilon:
        Post-diagonalization pruning threshold.
        By default, keep active bitstring b only if |c_b|^2 >= epsilon.

    n_iterations:
        Number of grow-prune iterations after the initial H|seed> selection.

    include_initial_bitstring:
        If True, the original seed bitstring is forced into the initial active set.

    score_mode:
        "residual":
            Select candidates by largest |r_a|.

        "pt2":
            Select candidates by approximate second-order energy score
            |r_a|^2 / (H_aa - E).

    Returns
    -------
    SelectedKrylovResult
        Final variational energy, active basis, bitstrings, coefficients, history.
    """
    H_csr = as_csr_hamiltonian(H)
    H_csc = H_csr.tocsc()

    dim = H_csr.shape[0]
    if H_csr.shape[0] != H_csr.shape[1]:
        raise ValueError("Hamiltonian must be square.")

    n_qubits = infer_n_qubits(dim)
    diagonal = H_csr.diagonal()

    if isinstance(initial_bitstring, str):
        if len(initial_bitstring) != n_qubits:
            raise ValueError(
                f"Initial bitstring has length {len(initial_bitstring)}, "
                f"but Hamiltonian has {n_qubits} qubits."
            )
        seed_index = bitstring_to_index(initial_bitstring)
    else:
        seed_index = int(initial_bitstring)

    if seed_index < 0 or seed_index >= dim:
        raise ValueError("Initial basis index is outside the Hilbert-space dimension.")

    # ------------------------------------------------------------
    # Initial step:
    # Apply H once to the seed bitstring and keep the top-Q components
    # by |amplitude|^2.
    # ------------------------------------------------------------
    seed_state = {seed_index: 1.0 + 0.0j}
    h_seed = apply_h_to_sparse_state(H_csc, seed_state, drop_tol=drop_tol)

    forced_seed = seed_index if include_initial_bitstring else None
    active = select_top_initial_components(
        h_seed=h_seed,
        Q=Q,
        include_seed_index=forced_seed,
    )

    history: List[IterationLog] = []

    # ------------------------------------------------------------
    # Main loop.
    #
    # Each iteration:
    #   1. Project H into current accumulated active subspace.
    #   2. Diagonalize.
    #   3. Prune active basis using |c_b|^2 >= epsilon.
    #   4. Apply H to the pruned approximate ground state.
    #   5. Select up to Q new external candidates using r_a.
    #   6. Accumulate: active <- pruned active union selected new.
    # ------------------------------------------------------------
    for it in range(n_iterations):
        n_active_before = len(active)

        H_sub = project_hamiltonian_to_subspace(
            H_csc,
            active,
            hermitize=hermitize_subspace,
        )

        energy, coeffs = lowest_eigenpair(
            H_sub,
            dense_cutoff=dense_cutoff,
            eigsh_tol=eigsh_tol,
        )

        pruned_active, pruned_coeffs = prune_by_coefficients(
            active=active,
            coeffs=coeffs,
            epsilon=epsilon,
            min_keep=min_keep_after_prune,
            use_probability=coefficient_prune_uses_probability,
        )

        psi_pruned = make_sparse_state_from_active_and_coeffs(
            pruned_active,
            pruned_coeffs,
            drop_tol=drop_tol,
        )

        Hpsi = apply_h_to_sparse_state(H_csc, psi_pruned, drop_tol=drop_tol)

        energy_after_prune = sparse_state_rayleigh_energy(Hpsi, psi_pruned)

        pruned_active_set = set(pruned_active)

        # External residual norm before thresholding.
        residual_norm_external = np.sqrt(
            sum(
                abs(amp) ** 2
                for idx, amp in Hpsi.items()
                if idx not in pruned_active_set
            )
        )

        selected_new, external_residuals, scores = select_new_candidates(
            Hpsi=Hpsi,
            active_set=pruned_active_set,
            Q=Q,
            delta=delta,
            score_mode=score_mode,
            diagonal=diagonal,
            current_energy=energy_after_prune,
        )

        # Accumulate old pruned active basis plus newly selected bitstrings.
        active = list(dict.fromkeys(pruned_active + selected_new))

        history.append(
            IterationLog(
                iteration=it,
                energy_before_prune=energy,
                energy_after_prune=energy_after_prune,
                n_active_before_prune=n_active_before,
                n_active_after_prune=len(pruned_active),
                n_raw_generated=len(Hpsi),
                n_external_candidates_above_delta=len(external_residuals),
                n_selected_new=len(selected_new),
                residual_norm_external=float(residual_norm_external),
            )
        )

        if len(selected_new) == 0:
            # No new directions above threshold delta.
            break

    # ------------------------------------------------------------
    # Final diagonalization in the last accumulated active subspace.
    # This is important because the final iteration may have added
    # candidates that have not yet been variationally optimized.
    # ------------------------------------------------------------
    H_final = project_hamiltonian_to_subspace(
        H_csc,
        active,
        hermitize=hermitize_subspace,
    )

    final_energy, final_coeffs = lowest_eigenpair(
        H_final,
        dense_cutoff=dense_cutoff,
        eigsh_tol=eigsh_tol,
    )

    active_bitstrings = [index_to_bitstring(i, n_qubits) for i in active]

    return SelectedKrylovResult(
        energy=final_energy,
        active_indices=active,
        active_bitstrings=active_bitstrings,
        coefficients=final_coeffs,
        history=history,
    )