from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Literal, Union, Callable, Sequence
import heapq

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh


SparseState = Dict[int, complex]
QSchedule = Union[int, Sequence[int], Callable[[int], int]]


# ============================================================
# Diagnostics
# ============================================================

@dataclass
class StateSparsityDiagnostics:
    active_size: int
    nnz: int
    nnz_fraction_of_active: float
    inverse_participation_ratio: float
    participation_ratio: float
    participation_fraction_of_active: float
    shannon_entropy: float
    exp_shannon_entropy: float
    exp_shannon_fraction_of_active: float
    max_weight: float
    min_nonzero_weight: float
    cumulative_weight_top_k: Dict[int, float]
    n_coeffs_above_weight_thresholds: Dict[float, int]


@dataclass
class InitialDiagnostics:
    seed_index: int
    seed_bitstring: str
    Q_initial_application: int
    n_raw_generated: int
    n_selected_from_initial_application: int
    n_pruned_by_initial_Q: int
    n_pruned_by_initial_cap: int
    initial_active_size: int
    selected_initial_bitstrings: List[str]


@dataclass
class IterationDiagnostics:
    iteration: int
    Q_this_application: int

    active_size_start: int
    energy_before_post_prune: float
    max_coeff_weight_before_post_prune: float
    min_coeff_weight_before_post_prune: float
    ground_state_sparsity_before_post_prune: StateSparsityDiagnostics

    n_pruned_after_diagonalization: int
    active_size_after_post_prune: int
    energy_after_post_prune_rayleigh: float
    ground_state_sparsity_applied_to_H: StateSparsityDiagnostics

    n_generated_total: int
    n_generated_internal: int
    n_generated_external: int
    external_residual_norm: float
    max_abs_external_residual: float

    n_pruned_before_diagonalization_by_delta: int
    n_candidates_after_delta: int
    n_pruned_before_diagonalization_by_Q: int
    n_new_bitstrings_kept: int

    active_size_after_addition_before_cap: int
    n_pruned_by_max_subspace_cap: int
    active_size_end: int

    selected_new_bitstrings: List[str]


@dataclass
class FinalDiagnostics:
    active_size_before_final_cleanup: int
    n_final_cleanup_passes: int
    n_pruned_in_final_cleanup: int
    final_active_size: int
    final_energy: float
    final_ground_state_sparsity: StateSparsityDiagnostics


@dataclass
class SelectedKrylovResult:
    final_energy: float
    final_indices: List[int]
    final_bitstrings: List[str]
    final_coefficients: np.ndarray
    initial_diagnostics: InitialDiagnostics
    iteration_diagnostics: List[IterationDiagnostics]
    final_diagnostics: FinalDiagnostics


# ============================================================
# Basic utilities
# ============================================================

def resolve_Q(Q: QSchedule, application: int) -> int:
    """
    Resolve Q for a given Hamiltonian application.

    application = 0:
        initial H|seed> application.

    application >= 1:
        later H|psi_t> applications.

    Supported:
        Q = 200
        Q = [500, 400, 300, 200, 100]
        Q = lambda app: max(50, int(500 * 0.8**app))
    """
    if callable(Q):
        q_value = Q(application)
    elif isinstance(Q, Sequence) and not isinstance(Q, (str, bytes)):
        if len(Q) == 0:
            raise ValueError("Q schedule cannot be empty.")
        q_value = Q[min(application, len(Q) - 1)]
    else:
        q_value = Q

    q_value = int(q_value)

    if q_value < 0:
        raise ValueError(f"Q must be nonnegative, got {q_value}.")

    return q_value


def bitstring_to_index(bitstring: str) -> int:
    return int(bitstring, 2)


def index_to_bitstring(index: int, n_qubits: int) -> str:
    return format(index, f"0{n_qubits}b")


def infer_n_qubits(dim: int) -> int:
    n = int(round(np.log2(dim)))
    if 2**n != dim:
        raise ValueError(f"Hamiltonian dimension {dim} is not a power of 2.")
    return n


def as_csr_hamiltonian(H) -> sp.csr_matrix:
    """
    Accepts either:
        - scipy sparse matrix,
        - Qiskit SparsePauliOp-like object with .to_matrix(sparse=True).
    """
    if sp.issparse(H):
        return H.tocsr()

    if hasattr(H, "to_matrix"):
        return sp.csr_matrix(H.to_matrix(sparse=True))

    raise TypeError(
        "H must be a scipy sparse matrix or a Qiskit SparsePauliOp-like object "
        "with .to_matrix(sparse=True)."
    )


def normalize_sparse_state(state: SparseState) -> SparseState:
    norm = np.sqrt(sum(abs(v) ** 2 for v in state.values()))
    if norm == 0.0:
        raise ValueError("Cannot normalize zero state.")
    return {k: v / norm for k, v in state.items()}


def sort_by_first_acceptance(
    indices: List[int],
    first_acceptance_rank: Dict[int, int],
) -> List[int]:
    return sorted(indices, key=lambda idx: first_acceptance_rank[idx])


def register_accepted_indices(
    indices: List[int],
    first_acceptance_rank: Dict[int, int],
    last_acceptance_iteration: Dict[int, int],
    next_rank: int,
    iteration: int,
) -> int:
    for idx in indices:
        if idx not in first_acceptance_rank:
            first_acceptance_rank[idx] = next_rank
            next_rank += 1
        last_acceptance_iteration[idx] = iteration
    return next_rank


# ============================================================
# Sparse H application and projection
# ============================================================

def apply_h_to_sparse_state(
    H_csc: sp.csc_matrix,
    state: SparseState,
    drop_tol: float = 0.0,
) -> SparseState:
    """
    Apply H to sparse state {basis_index: amplitude}.

    Uses CSC column access because H|b> is column b.
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
    Build H_S = P_S H P_S in the basis ordering given by active.
    """
    m = len(active)
    position = {basis_index: i for i, basis_index in enumerate(active)}

    rows, cols, data = [], [], []

    for col_pos, basis_col in enumerate(active):
        start = H_csc.indptr[basis_col]
        end = H_csc.indptr[basis_col + 1]

        for full_row, h_val in zip(H_csc.indices[start:end], H_csc.data[start:end]):
            row_pos = position.get(full_row)
            if row_pos is not None:
                rows.append(row_pos)
                cols.append(col_pos)
                data.append(h_val)

    H_sub = sp.coo_matrix((data, (rows, cols)), shape=(m, m)).tocsr()

    if hermitize:
        H_sub = 0.5 * (H_sub + H_sub.getH())

    return H_sub


def lowest_eigenpair(
    H_sub: sp.csr_matrix,
    dense_cutoff: int = 512,
    eigsh_tol: float = 1e-10,
) -> Tuple[float, np.ndarray]:
    m = H_sub.shape[0]

    if m == 0:
        raise ValueError("Cannot diagonalize empty subspace.")

    if m == 1:
        return float(np.real(H_sub[0, 0])), np.array([1.0 + 0.0j])

    if m <= dense_cutoff:
        evals, evecs = np.linalg.eigh(H_sub.toarray())
        i = int(np.argmin(evals))
        return float(np.real(evals[i])), evecs[:, i]

    evals, evecs = eigsh(H_sub, k=1, which="SA", tol=eigsh_tol)
    return float(np.real(evals[0])), evecs[:, 0]


def make_sparse_state_from_active_and_coeffs(
    active: List[int],
    coeffs: np.ndarray,
    drop_tol: float = 0.0,
) -> SparseState:
    state = {
        idx: complex(c)
        for idx, c in zip(active, coeffs)
        if abs(c) > drop_tol
    }
    return normalize_sparse_state(state)


def sparse_state_rayleigh_energy(Hpsi: SparseState, psi: SparseState) -> float:
    e = 0.0 + 0.0j
    for idx, coeff in psi.items():
        e += np.conjugate(coeff) * Hpsi.get(idx, 0.0)
    return float(np.real(e))


# ============================================================
# Sparsity diagnostics
# ============================================================

def diagnose_state_sparsity(
    coeffs: np.ndarray,
    *,
    active_size: Optional[int] = None,
    zero_tol: float = 0.0,
    top_ks: Tuple[int, ...] = (1, 5, 10, 50, 100),
    weight_thresholds: Tuple[float, ...] = (1e-2, 1e-4, 1e-6, 1e-8, 1e-10),
) -> StateSparsityDiagnostics:
    if active_size is None:
        active_size = len(coeffs)

    if len(coeffs) == 0:
        return StateSparsityDiagnostics(
            active_size=active_size,
            nnz=0,
            nnz_fraction_of_active=0.0,
            inverse_participation_ratio=0.0,
            participation_ratio=0.0,
            participation_fraction_of_active=0.0,
            shannon_entropy=0.0,
            exp_shannon_entropy=0.0,
            exp_shannon_fraction_of_active=0.0,
            max_weight=0.0,
            min_nonzero_weight=0.0,
            cumulative_weight_top_k={k: 0.0 for k in top_ks},
            n_coeffs_above_weight_thresholds={thr: 0 for thr in weight_thresholds},
        )

    norm = np.linalg.norm(coeffs)
    if norm == 0.0:
        raise ValueError("Cannot diagnose zero vector.")

    coeffs = coeffs / norm
    weights = np.abs(coeffs) ** 2

    nonzero_mask = np.abs(coeffs) > zero_tol
    nnz = int(np.count_nonzero(nonzero_mask))
    nonzero_weights = weights[nonzero_mask]

    ipr = float(np.sum(weights**2))
    pr = float(1.0 / ipr) if ipr > 0.0 else 0.0

    positive = weights[weights > 0.0]
    entropy = float(-np.sum(positive * np.log(positive)))
    exp_entropy = float(np.exp(entropy))

    sorted_weights = np.sort(weights)[::-1]

    return StateSparsityDiagnostics(
        active_size=int(active_size),
        nnz=nnz,
        nnz_fraction_of_active=float(nnz / active_size) if active_size else 0.0,
        inverse_participation_ratio=ipr,
        participation_ratio=pr,
        participation_fraction_of_active=float(pr / active_size) if active_size else 0.0,
        shannon_entropy=entropy,
        exp_shannon_entropy=exp_entropy,
        exp_shannon_fraction_of_active=float(exp_entropy / active_size) if active_size else 0.0,
        max_weight=float(np.max(weights)) if len(weights) else 0.0,
        min_nonzero_weight=float(np.min(nonzero_weights)) if len(nonzero_weights) else 0.0,
        cumulative_weight_top_k={
            k: float(np.sum(sorted_weights[: min(k, len(sorted_weights))]))
            for k in top_ks
        },
        n_coeffs_above_weight_thresholds={
            thr: int(np.count_nonzero(weights >= thr))
            for thr in weight_thresholds
        },
    )


# ============================================================
# Pruning and candidate selection
# ============================================================

def prune_active_by_coefficients(
    active: List[int],
    coeffs: np.ndarray,
    epsilon: float,
    min_keep: int,
    use_probability: bool,
) -> Tuple[List[int], np.ndarray, int, float, float]:
    """
    Epsilon pruning after diagonalization.

    If use_probability=True:
        keep |c_b|^2 >= epsilon.
    else:
        keep |c_b| >= epsilon.
    """
    weights = np.abs(coeffs) ** 2 if use_probability else np.abs(coeffs)

    max_weight = float(np.max(weights)) if len(weights) else 0.0
    min_weight = float(np.min(weights)) if len(weights) else 0.0

    keep = [i for i, w in enumerate(weights) if w >= epsilon]

    if len(keep) < min_keep:
        keep = list(np.argsort(weights)[-min_keep:])

    keep = sorted(set(keep))

    new_active = [active[i] for i in keep]
    new_coeffs = coeffs[keep]

    norm = np.linalg.norm(new_coeffs)
    if norm == 0.0:
        raise ValueError("Coefficient pruning produced zero vector.")

    new_coeffs = new_coeffs / norm

    return new_active, new_coeffs, len(active) - len(new_active), max_weight, min_weight


def select_initial_components(
    h_seed: SparseState,
    Q: int,
    seed_index: int,
    include_seed_index: bool,
) -> Tuple[List[int], Dict[int, float], int]:
    """
    Initial H|seed> selection by |amplitude|^2.
    """
    items = list(h_seed.items())

    if Q == 0:
        selected_items = []
    elif len(items) > Q:
        selected_items = heapq.nlargest(Q, items, key=lambda item: abs(item[1]) ** 2)
    else:
        selected_items = sorted(items, key=lambda item: abs(item[1]) ** 2, reverse=True)

    selected = [idx for idx, _ in selected_items]
    initial_scores = {idx: float(abs(amp) ** 2) for idx, amp in items}

    if include_seed_index and seed_index not in selected:
        selected = [seed_index] + selected
        initial_scores.setdefault(seed_index, max(initial_scores.values(), default=1.0))

    selected = list(dict.fromkeys(selected))
    n_pruned_by_Q = max(0, len(items) - min(Q, len(items)))

    return selected, initial_scores, n_pruned_by_Q


def select_new_candidates(
    Hpsi: SparseState,
    active_set: set[int],
    Q: int,
    delta: float,
    score_mode: Literal["residual", "pt2"],
    diagonal: Optional[np.ndarray],
    current_energy: Optional[float],
    denom_floor: float,
    n_qubits: int,
    within_iteration_order: Literal["score", "discovery"] = "discovery",
) -> Tuple[List[int], Dict[int, float], dict]:
    """
    Candidate pruning before diagonalization.

    r_a = <a|H|psi>

    delta filters by |r_a|.
    Q caps the number kept.
    """
    external_items = [(idx, amp) for idx, amp in Hpsi.items() if idx not in active_set]
    internal_count = len(Hpsi) - len(external_items)

    above_delta = [(idx, amp) for idx, amp in external_items if abs(amp) >= delta]

    scores: Dict[int, float] = {}

    for idx, r_a in above_delta:
        if score_mode == "residual":
            scores[idx] = float(abs(r_a))

        elif score_mode == "pt2":
            if diagonal is None or current_energy is None:
                raise ValueError("pt2 scoring requires diagonal and current_energy.")

            denom = float(np.real(diagonal[idx])) - float(current_energy)
            denom_eff = max(denom, denom_floor)
            scores[idx] = float(abs(r_a) ** 2 / denom_eff)

        else:
            raise ValueError(f"Unknown score_mode: {score_mode}")

    if Q == 0:
        top = []
    elif len(scores) > Q:
        top = heapq.nlargest(Q, scores.keys(), key=lambda idx: scores[idx])
    else:
        top = sorted(scores.keys(), key=lambda idx: scores[idx], reverse=True)

    top_set = set(top)

    if within_iteration_order == "score":
        selected = top
    elif within_iteration_order == "discovery":
        selected = [idx for idx, _ in above_delta if idx in top_set]
    else:
        raise ValueError("within_iteration_order must be 'score' or 'discovery'.")

    external_residual_norm = float(np.sqrt(sum(abs(amp) ** 2 for _, amp in external_items)))
    max_abs_external_residual = float(max((abs(amp) for _, amp in external_items), default=0.0))

    stats = {
        "n_generated_total": len(Hpsi),
        "n_generated_internal": internal_count,
        "n_generated_external": len(external_items),
        "external_residual_norm": external_residual_norm,
        "max_abs_external_residual": max_abs_external_residual,
        "n_pruned_before_diagonalization_by_delta": len(external_items) - len(above_delta),
        "n_candidates_after_delta": len(above_delta),
        "n_pruned_before_diagonalization_by_Q": max(0, len(above_delta) - len(selected)),
        "n_new_bitstrings_kept": len(selected),
        "selected_new_bitstrings": [index_to_bitstring(idx, n_qubits) for idx in selected],
    }

    return selected, scores, stats


def apply_max_subspace_cap(
    active: List[int],
    max_subspace_dim: Optional[int],
    first_acceptance_rank: Dict[int, int],
    last_acceptance_iteration: Dict[int, int],
    current_iteration: int,
    coeff_weight_by_index: Dict[int, float],
    residual_score_by_index: Dict[int, float],
    *,
    cap_coeff_weight: float,
    cap_residual_weight: float,
    cap_recency_weight: float,
    protected_indices: Optional[set[int]] = None,
) -> Tuple[List[int], List[int], Dict[int, float]]:
    """
    Enforce |S| <= max_subspace_dim using a composite importance score.

    score(b) =
        cap_coeff_weight    * normalized |c_b|^2
      + cap_residual_weight * normalized recent residual/candidate score
      + cap_recency_weight  * recency_score
    """
    if max_subspace_dim is None or len(active) <= max_subspace_dim:
        active_sorted = sort_by_first_acceptance(active, first_acceptance_rank)
        return active_sorted, [], {idx: 0.0 for idx in active_sorted}

    if max_subspace_dim <= 0:
        raise ValueError("max_subspace_dim must be positive or None.")

    protected_indices = protected_indices or set()
    protected = [idx for idx in active if idx in protected_indices]

    if len(protected) > max_subspace_dim:
        protected = sorted(
            protected,
            key=lambda idx: first_acceptance_rank[idx],
        )[:max_subspace_dim]

    max_coeff = max((coeff_weight_by_index.get(idx, 0.0) for idx in active), default=0.0)
    max_resid = max((residual_score_by_index.get(idx, 0.0) for idx in active), default=0.0)

    scores: Dict[int, float] = {}

    for idx in active:
        coeff_norm = coeff_weight_by_index.get(idx, 0.0) / max_coeff if max_coeff > 0 else 0.0
        resid_norm = residual_score_by_index.get(idx, 0.0) / max_resid if max_resid > 0 else 0.0

        age = max(0, current_iteration - last_acceptance_iteration.get(idx, current_iteration))
        recency = 1.0 / (1.0 + age)

        scores[idx] = (
            cap_coeff_weight * coeff_norm
            + cap_residual_weight * resid_norm
            + cap_recency_weight * recency
        )

    n_free = max_subspace_dim - len(protected)
    protected_set = set(protected)
    candidates = [idx for idx in active if idx not in protected_set]

    if n_free > 0:
        kept_free = heapq.nlargest(
            n_free,
            candidates,
            key=lambda idx: (scores[idx], -first_acceptance_rank[idx]),
        )
    else:
        kept_free = []

    kept = set(protected + kept_free)
    pruned = [idx for idx in active if idx not in kept]

    kept_ordered = sort_by_first_acceptance(list(kept), first_acceptance_rank)
    pruned_ordered = sort_by_first_acceptance(pruned, first_acceptance_rank)

    return kept_ordered, pruned_ordered, scores


# ============================================================
# Main algorithm
# ============================================================

def selected_krylov_ground_state(
    H,
    initial_bitstring: Union[str, int],
    Q: QSchedule,
    delta: float,
    epsilon: float,
    n_iterations: int,
    *,
    max_subspace_dim: Optional[int] = None,
    include_initial_bitstring: bool = True,
    protect_seed_initially: bool = True,
    score_mode: Literal["residual", "pt2"] = "residual",
    within_iteration_order: Literal["score", "discovery"] = "discovery",
    coefficient_prune_uses_probability: bool = True,
    min_keep_after_prune: int = 1,
    cap_coeff_weight: float = 1.0,
    cap_residual_weight: float = 1.0,
    cap_recency_weight: float = 0.05,
    dense_cutoff: int = 512,
    eigsh_tol: float = 1e-10,
    drop_tol: float = 0.0,
    hermitize_subspace: bool = True,
    final_cleanup_prune: bool = True,
    max_final_cleanup_passes: int = 10,
    denom_floor: float = 1e-12,
) -> SelectedKrylovResult:
    """
    Selected-subspace Krylov-like ground-state search with:

        1. scheduled Q:
               Q can be int, list/tuple, or callable.

        2. candidate pruning before diagonalization:
               keep up to Q candidates with |r_a| >= delta.

        3. coefficient pruning after diagonalization:
               keep states with |c_b|^2 >= epsilon by default.

        4. hard active-space cap:
               keep at most max_subspace_dim states using composite importance.

    Returned final_bitstrings are ordered by first acceptance into the algorithm.
    """
    H_csr = as_csr_hamiltonian(H)
    H_csc = H_csr.tocsc()

    if H_csr.shape[0] != H_csr.shape[1]:
        raise ValueError("Hamiltonian must be square.")

    dim = H_csr.shape[0]
    n_qubits = infer_n_qubits(dim)
    diagonal = H_csr.diagonal()

    if isinstance(initial_bitstring, str):
        if len(initial_bitstring) != n_qubits:
            raise ValueError(
                f"Initial bitstring length {len(initial_bitstring)} does not match "
                f"{n_qubits} qubits."
            )
        seed_index = bitstring_to_index(initial_bitstring)
    else:
        seed_index = int(initial_bitstring)

    if not (0 <= seed_index < dim):
        raise ValueError("Initial basis index is outside Hilbert space.")

    first_acceptance_rank: Dict[int, int] = {}
    last_acceptance_iteration: Dict[int, int] = {}
    recent_residual_score_by_index: Dict[int, float] = {}
    next_rank = 0

    # ------------------------------------------------------------
    # Initial application H|seed>
    # ------------------------------------------------------------
    Q_initial = resolve_Q(Q, application=0)

    seed_state = {seed_index: 1.0 + 0.0j}
    h_seed = apply_h_to_sparse_state(H_csc, seed_state, drop_tol=drop_tol)

    active, initial_scores, n_pruned_initial_Q = select_initial_components(
        h_seed=h_seed,
        Q=Q_initial,
        seed_index=seed_index,
        include_seed_index=include_initial_bitstring,
    )

    recent_residual_score_by_index.update(initial_scores)

    next_rank = register_accepted_indices(
        active,
        first_acceptance_rank,
        last_acceptance_iteration,
        next_rank,
        iteration=-1,
    )

    protected = {seed_index} if protect_seed_initially and include_initial_bitstring else set()

    active, pruned_initial_cap, _ = apply_max_subspace_cap(
        active=active,
        max_subspace_dim=max_subspace_dim,
        first_acceptance_rank=first_acceptance_rank,
        last_acceptance_iteration=last_acceptance_iteration,
        current_iteration=-1,
        coeff_weight_by_index={},
        residual_score_by_index=recent_residual_score_by_index,
        cap_coeff_weight=cap_coeff_weight,
        cap_residual_weight=cap_residual_weight,
        cap_recency_weight=cap_recency_weight,
        protected_indices=protected,
    )

    initial_diagnostics = InitialDiagnostics(
        seed_index=seed_index,
        seed_bitstring=index_to_bitstring(seed_index, n_qubits),
        Q_initial_application=Q_initial,
        n_raw_generated=len(h_seed),
        n_selected_from_initial_application=min(Q_initial, len(h_seed)),
        n_pruned_by_initial_Q=n_pruned_initial_Q,
        n_pruned_by_initial_cap=len(pruned_initial_cap),
        initial_active_size=len(active),
        selected_initial_bitstrings=[index_to_bitstring(idx, n_qubits) for idx in active],
    )

    iteration_diagnostics: List[IterationDiagnostics] = []

    # ------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------
    for it in range(n_iterations):
        active = sort_by_first_acceptance(active, first_acceptance_rank)
        active_size_start = len(active)

        # Project and diagonalize current active subspace.
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

        sparsity_before_prune = diagnose_state_sparsity(
            coeffs,
            active_size=len(active),
            zero_tol=drop_tol,
        )

        # Post-diagonalization epsilon pruning.
        (
            pruned_active,
            pruned_coeffs,
            n_pruned_after_diag,
            max_coeff_weight,
            min_coeff_weight,
        ) = prune_active_by_coefficients(
            active=active,
            coeffs=coeffs,
            epsilon=epsilon,
            min_keep=min_keep_after_prune,
            use_probability=coefficient_prune_uses_probability,
        )

        # Preserve first-acceptance order after pruning.
        coeff_by_idx = {idx: c for idx, c in zip(pruned_active, pruned_coeffs)}
        pruned_active = sort_by_first_acceptance(pruned_active, first_acceptance_rank)
        pruned_coeffs = np.array([coeff_by_idx[idx] for idx in pruned_active], dtype=complex)
        pruned_coeffs = pruned_coeffs / np.linalg.norm(pruned_coeffs)

        sparsity_applied_to_H = diagnose_state_sparsity(
            pruned_coeffs,
            active_size=len(pruned_active),
            zero_tol=drop_tol,
        )

        psi = make_sparse_state_from_active_and_coeffs(
            pruned_active,
            pruned_coeffs,
            drop_tol=drop_tol,
        )

        Hpsi = apply_h_to_sparse_state(H_csc, psi, drop_tol=drop_tol)
        energy_after_prune = sparse_state_rayleigh_energy(Hpsi, psi)

        # Candidate selection using scheduled Q.
        Q_this_application = resolve_Q(Q, application=it + 1)

        selected_new, candidate_scores, cstats = select_new_candidates(
            Hpsi=Hpsi,
            active_set=set(pruned_active),
            Q=Q_this_application,
            delta=delta,
            score_mode=score_mode,
            diagonal=diagonal,
            current_energy=energy_after_prune,
            denom_floor=denom_floor,
            n_qubits=n_qubits,
            within_iteration_order=within_iteration_order,
        )

        for idx in selected_new:
            recent_residual_score_by_index[idx] = candidate_scores.get(idx, 0.0)

        next_rank = register_accepted_indices(
            selected_new,
            first_acceptance_rank,
            last_acceptance_iteration,
            next_rank,
            iteration=it,
        )

        # Accumulate pruned old basis + newly accepted candidates.
        active_before_cap = list(dict.fromkeys(pruned_active + selected_new))
        active_size_after_addition_before_cap = len(active_before_cap)

        coeff_weight_by_index = {
            idx: float(abs(c) ** 2)
            for idx, c in zip(pruned_active, pruned_coeffs)
        }

        # Enforce hard max-subspace cap.
        active, pruned_by_cap, _ = apply_max_subspace_cap(
            active=active_before_cap,
            max_subspace_dim=max_subspace_dim,
            first_acceptance_rank=first_acceptance_rank,
            last_acceptance_iteration=last_acceptance_iteration,
            current_iteration=it,
            coeff_weight_by_index=coeff_weight_by_index,
            residual_score_by_index=recent_residual_score_by_index,
            cap_coeff_weight=cap_coeff_weight,
            cap_residual_weight=cap_residual_weight,
            cap_recency_weight=cap_recency_weight,
            protected_indices=None,
        )

        iteration_diagnostics.append(
            IterationDiagnostics(
                iteration=it,
                Q_this_application=Q_this_application,
                active_size_start=active_size_start,
                energy_before_post_prune=float(energy),
                max_coeff_weight_before_post_prune=float(max_coeff_weight),
                min_coeff_weight_before_post_prune=float(min_coeff_weight),
                ground_state_sparsity_before_post_prune=sparsity_before_prune,
                n_pruned_after_diagonalization=int(n_pruned_after_diag),
                active_size_after_post_prune=len(pruned_active),
                energy_after_post_prune_rayleigh=float(energy_after_prune),
                ground_state_sparsity_applied_to_H=sparsity_applied_to_H,
                n_generated_total=cstats["n_generated_total"],
                n_generated_internal=cstats["n_generated_internal"],
                n_generated_external=cstats["n_generated_external"],
                external_residual_norm=cstats["external_residual_norm"],
                max_abs_external_residual=cstats["max_abs_external_residual"],
                n_pruned_before_diagonalization_by_delta=cstats[
                    "n_pruned_before_diagonalization_by_delta"
                ],
                n_candidates_after_delta=cstats["n_candidates_after_delta"],
                n_pruned_before_diagonalization_by_Q=cstats[
                    "n_pruned_before_diagonalization_by_Q"
                ],
                n_new_bitstrings_kept=cstats["n_new_bitstrings_kept"],
                active_size_after_addition_before_cap=active_size_after_addition_before_cap,
                n_pruned_by_max_subspace_cap=len(pruned_by_cap),
                active_size_end=len(active),
                selected_new_bitstrings=cstats["selected_new_bitstrings"],
            )
        )

        if len(selected_new) == 0:
            break

    # ------------------------------------------------------------
    # Final cleanup and final diagonalization
    # ------------------------------------------------------------
    active = sort_by_first_acceptance(active, first_acceptance_rank)
    active_size_before_final_cleanup = len(active)
    n_cleanup_passes = 0
    n_pruned_cleanup = 0

    if final_cleanup_prune:
        for _ in range(max_final_cleanup_passes):
            H_final = project_hamiltonian_to_subspace(
                H_csc,
                active,
                hermitize=hermitize_subspace,
            )

            _, final_coeffs = lowest_eigenpair(
                H_final,
                dense_cutoff=dense_cutoff,
                eigsh_tol=eigsh_tol,
            )

            cleaned_active, _, n_pruned, _, _ = prune_active_by_coefficients(
                active=active,
                coeffs=final_coeffs,
                epsilon=epsilon,
                min_keep=min_keep_after_prune,
                use_probability=coefficient_prune_uses_probability,
            )

            if n_pruned == 0:
                break

            active = sort_by_first_acceptance(cleaned_active, first_acceptance_rank)
            n_cleanup_passes += 1
            n_pruned_cleanup += n_pruned

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

    final_sparsity = diagnose_state_sparsity(
        final_coeffs,
        active_size=len(active),
        zero_tol=drop_tol,
    )

    final_bitstrings = [index_to_bitstring(idx, n_qubits) for idx in active]

    return SelectedKrylovResult(
        final_energy=float(final_energy),
        final_indices=active,
        final_bitstrings=final_bitstrings,
        final_coefficients=final_coeffs,
        initial_diagnostics=initial_diagnostics,
        iteration_diagnostics=iteration_diagnostics,
        final_diagnostics=FinalDiagnostics(
            active_size_before_final_cleanup=active_size_before_final_cleanup,
            n_final_cleanup_passes=n_cleanup_passes,
            n_pruned_in_final_cleanup=n_pruned_cleanup,
            final_active_size=len(active),
            final_energy=float(final_energy),
            final_ground_state_sparsity=final_sparsity,
        ),
    )


# ============================================================
# Plotting
# ============================================================

def plot_selected_krylov_diagnostics(result):
    import matplotlib.pyplot as plt

    iterations = [d.iteration for d in result.iteration_diagnostics]

    active_start = [d.active_size_start for d in result.iteration_diagnostics]
    active_after_epsilon = [d.active_size_after_post_prune for d in result.iteration_diagnostics]
    active_after_add = [
        d.active_size_after_addition_before_cap
        for d in result.iteration_diagnostics
    ]
    active_after_cap = [d.active_size_end for d in result.iteration_diagnostics]

    fig1, ax1 = plt.subplots()
    ax1.plot(iterations, active_start, marker="o", label="Diagonalized size")
    ax1.plot(iterations, active_after_epsilon, marker="x", label="After epsilon pruning")
    ax1.plot(iterations, active_after_add, marker="s", label="After adding candidates")
    # ax1.plot(iterations, active_after_cap, marker="d", label="After max-dim cap")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Number of active bitstrings")
    ax1.set_title("Active subspace dimension")
    ax1.legend()
    ax1.grid(True)
    fig1.tight_layout()

    app_ids = [0] + [d.iteration + 1 for d in result.iteration_diagnostics]

    n_added = [result.initial_diagnostics.n_selected_from_initial_application] + [
        d.n_new_bitstrings_kept for d in result.iteration_diagnostics
    ]

    Q_used = [result.initial_diagnostics.Q_initial_application] + [
        d.Q_this_application for d in result.iteration_diagnostics
    ]

    fig2, ax2 = plt.subplots()
    ax2.plot(app_ids, n_added, marker="o", label="New bitstrings accepted")
    ax2.plot(app_ids, Q_used, marker="x", label="Q budget")
    ax2.set_xlabel("Hamiltonian application")
    ax2.set_ylabel("Number of bitstrings")
    ax2.set_title("New bitstrings added vs Q schedule")
    ax2.legend()
    ax2.grid(True)
    fig2.tight_layout()

    app_ids_s = [d.iteration + 1 for d in result.iteration_diagnostics]

    nnz = [
        d.ground_state_sparsity_applied_to_H.nnz
        for d in result.iteration_diagnostics
    ]

    pr = [
        d.ground_state_sparsity_applied_to_H.participation_ratio
        for d in result.iteration_diagnostics
    ]

    fig3, ax3 = plt.subplots()
    ax3.plot(app_ids_s, nnz, marker="o", label="Nonzero coefficients")
    ax3.plot(app_ids_s, pr, marker="x", label="Participation ratio")
    ax3.set_xlabel("Hamiltonian application")
    ax3.set_ylabel("Effective number of bitstrings")
    ax3.set_title("Sparsity of approximate ground state applied to H")
    ax3.legend()
    ax3.grid(True)
    fig3.tight_layout()

    discarded_delta = [
        d.n_pruned_before_diagonalization_by_delta
        for d in result.iteration_diagnostics
    ]

    discarded_Q = [
        d.n_pruned_before_diagonalization_by_Q
        for d in result.iteration_diagnostics
    ]

    discarded_epsilon = [
        d.n_pruned_after_diagonalization
        for d in result.iteration_diagnostics
    ]

    discarded_cap = [
        d.n_pruned_by_max_subspace_cap
        for d in result.iteration_diagnostics
    ]

    fig4, ax4 = plt.subplots()
    ax4.plot(app_ids_s, discarded_delta, marker="o", label="Rejected by delta")
    ax4.plot(app_ids_s, discarded_Q, marker="x", label="Rejected by Q cap")
    ax4.plot(app_ids_s, discarded_epsilon, marker="s", label="Rejected by epsilon")
    ax4.plot(app_ids_s, discarded_cap, marker="d", label="Rejected by max-dim cap")
    ax4.set_xlabel("Hamiltonian application")
    ax4.set_ylabel("Number of discarded bitstrings")
    ax4.set_title("Discarded bitstrings by pruning mechanism")
    ax4.legend()
    ax4.grid(True)
    fig4.tight_layout()

    return fig1, fig2, fig3, fig4
