from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Literal
import numpy as np

import matplotlib.pyplot as plt

from new_approach.krab import   infer_n_qubits, \
                                index_to_bitstring, \
                                bitstring_to_index, \
                                as_csr_hamiltonian, \
                                apply_h_to_sparse_state, \
                                make_sparse_state_from_active_and_coeffs, \
                                sparse_state_rayleigh_energy, \
                                project_hamiltonian_to_subspace, \
                                lowest_eigenpair


@dataclass
class StateSparsityDiagnostics:
    active_size: int

    # Literal number of coefficients above numerical zero threshold.
    nnz: int
    nnz_fraction_of_active: float

    # Effective support measures.
    inverse_participation_ratio: float
    participation_ratio: float
    participation_fraction_of_active: float

    # Entropic effective support.
    shannon_entropy: float
    exp_shannon_entropy: float
    exp_shannon_fraction_of_active: float

    # Weight concentration.
    max_weight: float
    min_nonzero_weight: float
    cumulative_weight_top_k: Dict[int, float]

    # Useful for seeing how many coefficients survive common cutoffs.
    n_coeffs_above_weight_thresholds: Dict[float, int]



@dataclass
class InitialDiagnostics:
    seed_index: int
    seed_bitstring: str
    n_raw_generated: int
    n_selected_from_initial_application: int
    n_pruned_by_initial_Q: int
    initial_active_size: int
    selected_initial_bitstrings: List[str]


@dataclass
class IterationDiagnostics:
    iteration: int

    active_size_start: int

    energy_before_post_prune: float
    max_coeff_weight_before_post_prune: float
    min_coeff_weight_before_post_prune: float

    # New:
    ground_state_sparsity_before_post_prune: StateSparsityDiagnostics
    ground_state_sparsity_applied_to_H: StateSparsityDiagnostics

    n_pruned_after_diagonalization: int
    active_size_after_post_prune: int
    energy_after_post_prune_rayleigh: float

    n_generated_total: int
    n_generated_internal: int
    n_generated_external: int
    external_residual_norm: float
    max_abs_external_residual: float

    n_pruned_before_diagonalization_by_delta: int
    n_candidates_after_delta: int
    n_pruned_before_diagonalization_by_Q: int
    n_new_bitstrings_kept: int

    active_size_end: int
    selected_new_bitstrings: list[str]


@dataclass
class FinalDiagnostics:
    active_size_before_final_cleanup: int
    n_final_cleanup_passes: int
    n_pruned_in_final_cleanup: int
    final_active_size: int
    final_energy: float


@dataclass
class SelectedKrylovResult:
    final_energy: float

    # Final subspace, ordered by first acceptance into the algorithm.
    final_indices: List[int]
    final_bitstrings: List[str]

    # Coefficients of the final approximate ground state.
    # Same order as final_bitstrings/final_indices.
    final_coefficients: np.ndarray

    initial_diagnostics: InitialDiagnostics
    iteration_diagnostics: List[IterationDiagnostics]
    final_diagnostics: FinalDiagnostics



import matplotlib.pyplot as plt


def plot_selected_krylov_diagnostics(result):
    """
    Plot diagnostics for the selected Krylov / selected-subspace method.

    Produces four figures:

        1. Active subspace dimension through the algorithm.
           This is the most important plot for seeing:
               - dimension before diagonalization,
               - dimension after epsilon-pruning,
               - dimension after adding new bitstrings.

        2. Number of newly added bitstrings per Hamiltonian application.

        3. Sparsity of the approximate ground state applied to H.

        4. Number of discarded bitstrings in the two pruning steps.

    Convention:
        application 0:
            initial application H|seed>

        iteration k:
            diagonalize current active subspace,
            prune by epsilon,
            apply H,
            prune candidates by delta and Q,
            add accepted candidates.
    """

    # ============================================================
    # 1. Active subspace dimension
    # ============================================================
    #
    # This shows the sawtooth behavior you care about:
    #
    #   active_size_start:
    #       dimension of the subspace actually diagonalized
    #
    #   active_size_after_post_prune:
    #       dimension after pruning away small-|c_b|^2 states
    #
    #   active_size_end:
    #       dimension after adding newly selected candidates
    #
    # If Q ~ 200 and pruning is aggressive, you should see:
    #
    #   after prune -> after add rises by roughly 200
    #   next iteration after diagonalization/prune drops again
    #

    iterations = [d.iteration for d in result.iteration_diagnostics]

    active_start = [
        d.active_size_start
        for d in result.iteration_diagnostics
    ]

    active_after_epsilon_prune = [
        d.active_size_after_post_prune
        for d in result.iteration_diagnostics
    ]

    active_after_adding_new = [
        d.active_size_end
        for d in result.iteration_diagnostics
    ]

    fig1, ax1 = plt.subplots()

    ax1.plot(
        iterations,
        active_start,
        marker="o",
        label="Diagonalized subspace size",
    )

    ax1.plot(
        iterations,
        active_after_epsilon_prune,
        marker="x",
        label="After epsilon pruning",
    )

    ax1.plot(
        iterations,
        active_after_adding_new,
        marker="s",
        label="After adding new bitstrings",
    )

    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Number of active bitstrings")
    ax1.set_title("Active subspace dimension through the algorithm")
    ax1.legend()
    ax1.grid(True)
    fig1.tight_layout()

    # ============================================================
    # 2. Number of bitstrings added per Hamiltonian application
    # ============================================================

    application_ids_added = [0]
    n_added = [
        result.initial_diagnostics.n_selected_from_initial_application
    ]

    for d in result.iteration_diagnostics:
        application_ids_added.append(d.iteration + 1)
        n_added.append(d.n_new_bitstrings_kept)

    fig2, ax2 = plt.subplots()

    ax2.plot(
        application_ids_added,
        n_added,
        marker="o",
    )

    ax2.set_xlabel("Hamiltonian application")
    ax2.set_ylabel("New bitstrings accepted")
    ax2.set_title("New bitstrings added per Hamiltonian application")
    ax2.grid(True)
    fig2.tight_layout()

    # ============================================================
    # 3. Sparsity of approximate ground state applied to H
    # ============================================================
    #
    # nnz:
    #     Number of coefficients counted as nonzero according to the
    #     zero_tol used in diagnose_state_sparsity.
    #
    # participation_ratio:
    #     Effective number of important basis states:
    #
    #         PR = 1 / sum_b |c_b|^4
    #
    #     This is often more meaningful than literal nnz.
    #

    application_ids_sparsity = []
    ground_state_nnz = []
    ground_state_participation_ratio = []

    for d in result.iteration_diagnostics:
        application_ids_sparsity.append(d.iteration + 1)

        sparsity = d.ground_state_sparsity_applied_to_H

        ground_state_nnz.append(sparsity.nnz)
        ground_state_participation_ratio.append(
            sparsity.participation_ratio
        )

    fig3, ax3 = plt.subplots()

    ax3.plot(
        application_ids_sparsity,
        ground_state_nnz,
        marker="o",
        label="Nonzero coefficients",
    )

    ax3.plot(
        application_ids_sparsity,
        ground_state_participation_ratio,
        marker="x",
        label="Participation ratio",
    )

    ax3.set_xlabel("Hamiltonian application")
    ax3.set_ylabel("Effective number of bitstrings")
    ax3.set_title("Sparsity of approximate ground state applied to H")
    ax3.legend()
    ax3.grid(True)
    fig3.tight_layout()

    # ============================================================
    # 4. Discarded bitstrings in the two pruning steps
    # ============================================================
    #
    # First pruning step:
    #     Happens before adding candidates to the subspace.
    #     This includes:
    #         - rejected by delta threshold,
    #         - rejected by top-Q cap.
    #
    # Second pruning step:
    #     Happens after diagonalization.
    #     Removes active basis states with small |c_b|^2.
    #

    application_ids_pruning = []

    discarded_by_delta = []
    discarded_by_Q = []
    discarded_in_first_pruning_total = []
    discarded_in_second_pruning = []

    for d in result.iteration_diagnostics:
        application_ids_pruning.append(d.iteration + 1)

        n_delta = d.n_pruned_before_diagonalization_by_delta
        n_Q = d.n_pruned_before_diagonalization_by_Q
        n_epsilon = d.n_pruned_after_diagonalization

        discarded_by_delta.append(n_delta)
        discarded_by_Q.append(n_Q)
        discarded_in_first_pruning_total.append(n_delta + n_Q)
        discarded_in_second_pruning.append(n_epsilon)

    fig4, ax4 = plt.subplots()

    ax4.plot(
        application_ids_pruning,
        discarded_by_delta,
        marker="o",
        label="Rejected by delta",
    )

    ax4.plot(
        application_ids_pruning,
        discarded_by_Q,
        marker="x",
        label="Rejected by Q cap",
    )

    ax4.plot(
        application_ids_pruning,
        discarded_in_second_pruning,
        marker="s",
        label="Rejected by epsilon",
    )

    ax4.set_xlabel("Hamiltonian application")
    ax4.set_ylabel("Number of discarded bitstrings")
    ax4.set_title("Discarded bitstrings by pruning mechanism")
    ax4.legend()
    ax4.grid(True)
    fig4.tight_layout()

    return fig1, fig2, fig3, fig4



def diagnose_state_sparsity(
    coeffs: np.ndarray,
    *,
    active_size: int | None = None,
    zero_tol: float = 0.0,
    top_ks: Tuple[int, ...] = (1, 5, 10, 50, 100),
    weight_thresholds: Tuple[float, ...] = (1e-2, 1e-4, 1e-6, 1e-8, 1e-10),
) -> StateSparsityDiagnostics:
    """
    Diagnose sparsity of a normalized state vector represented in the
    current selected bitstring basis.

    coeffs:
        Coefficients c_b of the approximate ground state.

    active_size:
        Dimension of the active basis. If None, len(coeffs) is used.

    zero_tol:
        Numerical threshold for treating a coefficient as nonzero.

    Main quantities:
        nnz:
            Number of coefficients with |c_b| > zero_tol.

        participation_ratio:
            PR = 1 / sum_b |c_b|^4.
            This is the effective number of basis states occupied.

        exp_shannon_entropy:
            exp(-sum_b p_b log p_b), where p_b = |c_b|^2.
            Another effective support size.

        cumulative_weight_top_k:
            Probability mass carried by the largest k coefficients.
    """
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
        raise ValueError("Cannot diagnose sparsity of a zero vector.")

    coeffs = coeffs / norm
    weights = np.abs(coeffs) ** 2

    nonzero_mask = np.abs(coeffs) > zero_tol
    nnz = int(np.count_nonzero(nonzero_mask))

    nonzero_weights = weights[nonzero_mask]

    ipr = float(np.sum(weights**2))
    participation_ratio = float(1.0 / ipr) if ipr > 0.0 else 0.0

    positive_weights = weights[weights > 0.0]
    shannon_entropy = float(-np.sum(positive_weights * np.log(positive_weights)))
    exp_shannon_entropy = float(np.exp(shannon_entropy))

    sorted_weights = np.sort(weights)[::-1]

    cumulative_weight_top_k = {
        k: float(np.sum(sorted_weights[: min(k, len(sorted_weights))]))
        for k in top_ks
    }

    n_coeffs_above_weight_thresholds = {
        thr: int(np.count_nonzero(weights >= thr))
        for thr in weight_thresholds
    }

    max_weight = float(np.max(weights)) if len(weights) > 0 else 0.0
    min_nonzero_weight = (
        float(np.min(nonzero_weights)) if len(nonzero_weights) > 0 else 0.0
    )

    return StateSparsityDiagnostics(
        active_size=int(active_size),
        nnz=nnz,
        nnz_fraction_of_active=float(nnz / active_size) if active_size > 0 else 0.0,
        inverse_participation_ratio=ipr,
        participation_ratio=participation_ratio,
        participation_fraction_of_active=(
            float(participation_ratio / active_size) if active_size > 0 else 0.0
        ),
        shannon_entropy=shannon_entropy,
        exp_shannon_entropy=exp_shannon_entropy,
        exp_shannon_fraction_of_active=(
            float(exp_shannon_entropy / active_size) if active_size > 0 else 0.0
        ),
        max_weight=max_weight,
        min_nonzero_weight=min_nonzero_weight,
        cumulative_weight_top_k=cumulative_weight_top_k,
        n_coeffs_above_weight_thresholds=n_coeffs_above_weight_thresholds,
    )



def sort_by_first_acceptance(
    indices: List[int],
    first_acceptance_rank: Dict[int, int],
) -> List[int]:
    """
    Sort basis indices by the order in which they were first accepted
    into the algorithm.

    This is the order used for the returned final bitstring list.
    """
    return sorted(indices, key=lambda idx: first_acceptance_rank[idx])


def register_accepted_indices(
    indices: List[int],
    first_acceptance_rank: Dict[int, int],
    next_rank: int,
) -> int:
    """
    Register newly accepted bitstrings in first-acceptance order.
    """
    for idx in indices:
        if idx not in first_acceptance_rank:
            first_acceptance_rank[idx] = next_rank
            next_rank += 1
    return next_rank


def select_initial_components_with_diagnostics(
    h_seed: Dict[int, complex],
    Q: int,
    seed_index: int,
    include_seed_index: bool,
    n_qubits: int,
) -> Tuple[List[int], InitialDiagnostics]:
    """
    Initial step.

    We apply H once to the initial bitstring. Since there is no previously
    optimized variational eigenvector yet, we select the largest components
    of H|seed> by |amplitude|^2.

    If include_seed_index=True, the original seed bitstring is also kept,
    possibly making the initial active size Q + 1.
    """
    ranked = sorted(
        h_seed.items(),
        key=lambda item: abs(item[1]) ** 2,
        reverse=True,
    )

    selected_from_h_seed = [idx for idx, amp in ranked[:Q]]

    selected = []
    if include_seed_index:
        selected.append(seed_index)

    for idx in selected_from_h_seed:
        if idx not in selected:
            selected.append(idx)

    n_raw = len(h_seed)
    n_selected_from_application = len(selected_from_h_seed)
    n_pruned_by_Q = max(0, n_raw - n_selected_from_application)

    diagnostics = InitialDiagnostics(
        seed_index=seed_index,
        seed_bitstring=index_to_bitstring(seed_index, n_qubits),
        n_raw_generated=n_raw,
        n_selected_from_initial_application=n_selected_from_application,
        n_pruned_by_initial_Q=n_pruned_by_Q,
        initial_active_size=len(selected),
        selected_initial_bitstrings=[
            index_to_bitstring(idx, n_qubits) for idx in selected
        ],
    )

    return selected, diagnostics


def prune_active_by_coefficients_with_stats(
    active: List[int],
    coeffs: np.ndarray,
    epsilon: float,
    min_keep: int,
    use_probability: bool,
) -> Tuple[List[int], np.ndarray, int, float, float]:
    """
    Post-diagonalization pruning.

    If use_probability=True:
        keep b if |c_b|^2 >= epsilon.

    If use_probability=False:
        keep b if |c_b| >= epsilon.

    Returns:
        pruned_active,
        pruned_coeffs,
        n_pruned,
        max_weight_before_prune,
        min_weight_before_prune
    """
    if use_probability:
        weights = np.abs(coeffs) ** 2
    else:
        weights = np.abs(coeffs)

    max_weight = float(np.max(weights)) if len(weights) > 0 else 0.0
    min_weight = float(np.min(weights)) if len(weights) > 0 else 0.0

    keep_positions = [i for i, w in enumerate(weights) if w >= epsilon]

    # Safety: never delete the entire active space.
    if len(keep_positions) < min_keep:
        keep_positions = list(np.argsort(weights)[-min_keep:])

    keep_positions = sorted(set(keep_positions))

    pruned_active = [active[i] for i in keep_positions]
    pruned_coeffs = coeffs[keep_positions]

    norm = np.linalg.norm(pruned_coeffs)
    if norm == 0.0:
        raise ValueError("Coefficient pruning produced a zero vector.")

    pruned_coeffs = pruned_coeffs / norm

    n_pruned = len(active) - len(pruned_active)

    return pruned_active, pruned_coeffs, n_pruned, max_weight, min_weight


def select_new_candidates_with_diagnostics(
    Hpsi: Dict[int, complex],
    active_set: set[int],
    Q: int,
    delta: float,
    score_mode: Literal["residual", "pt2"],
    diagonal: Optional[np.ndarray],
    current_energy: Optional[float],
    denom_floor: float,
    n_qubits: int,
    within_iteration_order: Literal["score", "discovery"] = "discovery",
) -> Tuple[List[int], IterationDiagnostics]:
    """
    Select external candidate bitstrings from H|psi>.

    Candidates are individual bitstrings a not already in the active set.

    r_a = <a|H|psi>

    delta:
        absolute residual threshold. Candidate survives only if |r_a| >= delta.

    Q:
        maximum number of surviving candidates to keep.

    within_iteration_order:
        "score":
            selected candidates are returned from largest score to smallest.

        "discovery":
            selected candidates are returned in the order in which they appeared
            during the sparse H|psi> construction. This is better if you care
            about discovery/order-of-occurrence bookkeeping.
    """
    external_items = [
        (idx, amp)
        for idx, amp in Hpsi.items()
        if idx not in active_set
    ]

    internal_count = len(Hpsi) - len(external_items)

    external_residual_norm = float(
        np.sqrt(sum(abs(amp) ** 2 for idx, amp in external_items))
    )

    max_abs_external_residual = float(
        max((abs(amp) for idx, amp in external_items), default=0.0)
    )

    above_delta_items = [
        (idx, amp)
        for idx, amp in external_items
        if abs(amp) >= delta
    ]

    n_pruned_by_delta = len(external_items) - len(above_delta_items)

    scores = {}

    for idx, r_a in above_delta_items:
        if score_mode == "residual":
            scores[idx] = abs(r_a)

        elif score_mode == "pt2":
            if diagonal is None or current_energy is None:
                raise ValueError(
                    "For score_mode='pt2', diagonal and current_energy are required."
                )

            denom = float(np.real(diagonal[idx])) - float(current_energy)
            denom_eff = max(denom, denom_floor)
            scores[idx] = abs(r_a) ** 2 / denom_eff

        else:
            raise ValueError(f"Unknown score_mode: {score_mode}")

    ranked_by_score = sorted(
        scores.keys(),
        key=lambda idx: scores[idx],
        reverse=True,
    )

    selected_set = set(ranked_by_score[:Q])

    if within_iteration_order == "score":
        selected = ranked_by_score[:Q]
    elif within_iteration_order == "discovery":
        selected = [
            idx
            for idx, amp in above_delta_items
            if idx in selected_set
        ]
    else:
        raise ValueError("within_iteration_order must be 'score' or 'discovery'.")

    n_pruned_by_Q = max(0, len(above_delta_items) - len(selected))

    # The IterationDiagnostics object will be completed in the main loop.
    partial_diag = {
        "n_generated_total": len(Hpsi),
        "n_generated_internal": internal_count,
        "n_generated_external": len(external_items),
        "external_residual_norm": external_residual_norm,
        "max_abs_external_residual": max_abs_external_residual,
        "n_pruned_before_diagonalization_by_delta": n_pruned_by_delta,
        "n_candidates_after_delta": len(above_delta_items),
        "n_pruned_before_diagonalization_by_Q": n_pruned_by_Q,
        "n_new_bitstrings_kept": len(selected),
        "selected_new_bitstrings": [
            index_to_bitstring(idx, n_qubits) for idx in selected
        ],
    }

    return selected, partial_diag

def selected_krylov_ground_state(
    H,
    initial_bitstring,
    Q: int,
    delta: float,
    epsilon: float,
    n_iterations: int,
    *,
    include_initial_bitstring: bool = True,
    score_mode: Literal["residual", "pt2"] = "residual",
    within_iteration_order: Literal["score", "discovery"] = "discovery",
    min_keep_after_prune: int = 1,
    coefficient_prune_uses_probability: bool = True,
    dense_cutoff: int = 512,
    eigsh_tol: float = 1e-10,
    drop_tol: float = 0.0,
    hermitize_subspace: bool = True,
    final_cleanup_prune: bool = True,
    max_final_cleanup_passes: int = 10,
    denom_floor: float = 1e-12,
) -> SelectedKrylovResult:
    """
    Accumulated selected-subspace Krylov-like ground-state search.

    The algorithm keeps an accumulated active set S.

    Each iteration:
        1. Project H into current active subspace S.
        2. Diagonalize H_S.
        3. Prune active bitstrings with small |c_b|^2 or |c_b|.
        4. Apply H to the pruned approximate ground state.
        5. Keep up to Q external candidates satisfying |r_a| >= delta.
        6. Accumulate the surviving old basis and the newly accepted candidates.

    The returned final bitstrings are the final active subspace, ordered by
    first acceptance into the algorithm.
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
                f"Initial bitstring length {len(initial_bitstring)} does not match "
                f"Hamiltonian size of {n_qubits} qubits."
            )
        seed_index = bitstring_to_index(initial_bitstring)
    else:
        seed_index = int(initial_bitstring)

    if seed_index < 0 or seed_index >= dim:
        raise ValueError("Initial basis index is outside the Hilbert-space dimension.")

    # ------------------------------------------------------------------
    # Bookkeeping for first-acceptance order.
    # ------------------------------------------------------------------
    first_acceptance_rank: Dict[int, int] = {}
    next_rank = 0

    # ------------------------------------------------------------------
    # Initial application: H|seed>.
    # ------------------------------------------------------------------
    seed_state = {seed_index: 1.0 + 0.0j}
    h_seed = apply_h_to_sparse_state(
        H_csc,
        seed_state,
        drop_tol=drop_tol,
    )

    active, initial_diagnostics = select_initial_components_with_diagnostics(
        h_seed=h_seed,
        Q=Q,
        seed_index=seed_index,
        include_seed_index=include_initial_bitstring,
        n_qubits=n_qubits,
    )

    next_rank = register_accepted_indices(
        active,
        first_acceptance_rank,
        next_rank,
    )

    active = sort_by_first_acceptance(active, first_acceptance_rank)

    iteration_diagnostics: List[IterationDiagnostics] = []

    # ------------------------------------------------------------------
    # Main loop.
    # ------------------------------------------------------------------
    for it in range(n_iterations):
        active = sort_by_first_acceptance(active, first_acceptance_rank)
        active_size_start = len(active)

        # 1. Project and diagonalize.
        H_sub = project_hamiltonian_to_subspace(
            H_csc,
            active,
            hermitize=hermitize_subspace,
        )

        energy_before_prune, coeffs = lowest_eigenpair(
            H_sub,
            dense_cutoff=dense_cutoff,
            eigsh_tol=eigsh_tol,
        )

        ground_state_sparsity_before_post_prune = diagnose_state_sparsity(
            coeffs,
            active_size=len(active),
            zero_tol=drop_tol,
        )

        # 2. Post-diagonalization pruning by coefficient weight.
        (
            pruned_active,
            pruned_coeffs,
            n_pruned_after_diag,
            max_coeff_weight,
            min_coeff_weight,
        ) = prune_active_by_coefficients_with_stats(
            active=active,
            coeffs=coeffs,
            epsilon=epsilon,
            min_keep=min_keep_after_prune,
            use_probability=coefficient_prune_uses_probability,
        )

        pruned_active = sort_by_first_acceptance(
            pruned_active,
            first_acceptance_rank,
        )

        # The coefficient array must match pruned_active. Since sorting may
        # have changed order, reconstruct coeffs by dictionary.
        coeff_by_index = {
            idx: coeff
            for idx, coeff in zip(
                [active[i] for i in range(len(active))],
                coeffs,
            )
            if idx in set(pruned_active)
        }

        pruned_coeffs = np.array(
            [coeff_by_index[idx] for idx in pruned_active],
            dtype=complex,
        )

        pruned_coeffs = pruned_coeffs / np.linalg.norm(pruned_coeffs)

        # ...
        ground_state_sparsity_applied_to_H = diagnose_state_sparsity(
            pruned_coeffs,
            active_size=len(pruned_active),
            zero_tol=drop_tol,
        )


        psi_pruned = make_sparse_state_from_active_and_coeffs(
            pruned_active,
            pruned_coeffs,
            drop_tol=drop_tol,
        )

        # 3. Apply H to the pruned approximate ground state.
        Hpsi = apply_h_to_sparse_state(
            H_csc,
            psi_pruned,
            drop_tol=drop_tol,
        )

        energy_after_prune = sparse_state_rayleigh_energy(Hpsi, psi_pruned)

        # 4. Select new external candidates using r_a.
        pruned_active_set = set(pruned_active)

        selected_new, candidate_stats = select_new_candidates_with_diagnostics(
            Hpsi=Hpsi,
            active_set=pruned_active_set,
            Q=Q,
            delta=delta,
            score_mode=score_mode,
            diagonal=diagonal,
            current_energy=energy_after_prune,
            denom_floor=denom_floor,
            n_qubits=n_qubits,
            within_iteration_order=within_iteration_order,
        )

        # 5. Register newly accepted states in first-acceptance order.
        next_rank = register_accepted_indices(
            selected_new,
            first_acceptance_rank,
            next_rank,
        )

        # 6. Accumulate: pruned old active set plus newly selected candidates.
        active = list(dict.fromkeys(pruned_active + selected_new))
        active = sort_by_first_acceptance(active, first_acceptance_rank)

        diag = IterationDiagnostics(
            iteration=it,
            active_size_start=active_size_start,
            energy_before_post_prune=float(energy_before_prune),
            max_coeff_weight_before_post_prune=float(max_coeff_weight),
            min_coeff_weight_before_post_prune=float(min_coeff_weight),

            ground_state_sparsity_before_post_prune=ground_state_sparsity_before_post_prune,
            ground_state_sparsity_applied_to_H=ground_state_sparsity_applied_to_H,

            n_pruned_after_diagonalization=int(n_pruned_after_diag),
            active_size_after_post_prune=len(pruned_active),
            energy_after_post_prune_rayleigh=float(energy_after_prune),

            n_generated_total=candidate_stats["n_generated_total"],
            n_generated_internal=candidate_stats["n_generated_internal"],
            n_generated_external=candidate_stats["n_generated_external"],
            external_residual_norm=candidate_stats["external_residual_norm"],
            max_abs_external_residual=candidate_stats["max_abs_external_residual"],
            n_pruned_before_diagonalization_by_delta=(
                candidate_stats["n_pruned_before_diagonalization_by_delta"]
            ),
            n_candidates_after_delta=candidate_stats["n_candidates_after_delta"],
            n_pruned_before_diagonalization_by_Q=(
                candidate_stats["n_pruned_before_diagonalization_by_Q"]
            ),
            n_new_bitstrings_kept=candidate_stats["n_new_bitstrings_kept"],
            active_size_end=len(active),
            selected_new_bitstrings=candidate_stats["selected_new_bitstrings"],
        )

        iteration_diagnostics.append(diag)

        # If no new bitstrings survive pre-diagonalization pruning, stop.
        if len(selected_new) == 0:
            break

    # ------------------------------------------------------------------
    # Final diagonalization and optional final cleanup.
    #
    # If final_cleanup_prune=True, we prune by epsilon after the final
    # diagonalization and re-diagonalize until the final active set is stable.
    # This ensures that the returned bitstrings are exactly those kept in the
    # final subspace used to compute final_energy.
    # ------------------------------------------------------------------
    active = sort_by_first_acceptance(active, first_acceptance_rank)

    active_size_before_final_cleanup = len(active)
    n_final_cleanup_passes = 0
    n_pruned_in_final_cleanup = 0

    for cleanup_pass in range(max_final_cleanup_passes + 1):
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

        if not final_cleanup_prune:
            break

        (
            cleaned_active,
            cleaned_coeffs,
            n_pruned_this_pass,
            _,
            _,
        ) = prune_active_by_coefficients_with_stats(
            active=active,
            coeffs=final_coeffs,
            epsilon=epsilon,
            min_keep=min_keep_after_prune,
            use_probability=coefficient_prune_uses_probability,
        )

        cleaned_active = sort_by_first_acceptance(
            cleaned_active,
            first_acceptance_rank,
        )

        if n_pruned_this_pass == 0:
            break

        active = cleaned_active
        n_final_cleanup_passes += 1
        n_pruned_in_final_cleanup += n_pruned_this_pass

    # One final diagonalization in the actually returned final subspace.
    active = sort_by_first_acceptance(active, first_acceptance_rank)

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

    final_bitstrings = [
        index_to_bitstring(idx, n_qubits)
        for idx in active
    ]

    final_diagnostics = FinalDiagnostics(
        active_size_before_final_cleanup=active_size_before_final_cleanup,
        n_final_cleanup_passes=n_final_cleanup_passes,
        n_pruned_in_final_cleanup=n_pruned_in_final_cleanup,
        final_active_size=len(active),
        final_energy=float(final_energy),
    )

    return SelectedKrylovResult(
        final_energy=float(final_energy),
        final_indices=active,
        final_bitstrings=final_bitstrings,
        final_coefficients=final_coeffs,
        initial_diagnostics=initial_diagnostics,
        iteration_diagnostics=iteration_diagnostics,
        final_diagnostics=final_diagnostics,
    )