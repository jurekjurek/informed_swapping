"""
Single-job experiment script for the KRAB-vs-SKQD systematic cluster study.

For one combination of hyperparameters
(n_qubits, gs_sparsity, ham_sparsity, overlap, Q, epsilon) the script:

  1. Builds a random Hamiltonian with a controlled sparse ground state.
  2. Computes the true ground energy.
  3. Runs **KRAB** (selected_krylov_ground_state) and builds its
     energy-vs-#basis-states convergence path (states added in the order KRAB
     discovered them), plus its native per-iteration trajectory.
  4. Runs **SKQD** for a sweep of time steps, each giving an ordering of basis
     states and an energy-vs-#states path.
  5. Computes a set of **random** orderings as a baseline.
  6. Records, for every method, the number of sampled basis states needed to
     reach relative error < 1e-3 — the common currency for "who is cheaper".
  7. Saves ONE combined figure:
        - top: the convergence-path comparison (KRAB vs SKQD vs random),
          mirroring the BARK-vs-SKQD plots;
        - bottom: the KRAB internal diagnostics (energy / rel. error,
          subspace size, bitstrings added per iteration).
  8. Appends one comparative CSV row to --data_dir/comparison.csv.

Usage
-----
python run_experiment.py \
    --n_qubits 8 --gs_sparsity 0.1 --ham_sparsity 0.5 \
    --overlap 0.3 --Q_frac 0.03 --epsilon 1e-6 \
    --n_random_paths 20 --seed 42 \
    --figure_dir figures --data_dir data

Q is given as a *fraction* of the full Hilbert space 2ⁿ (e.g. 0.03 = 3%); the
number of new basis states KRAB keeps per iteration is round(Q_frac · 2ⁿ).
"""

import argparse
import csv
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from new_approach.Hamiltonian_controlled_sparsity import (
    make_controlled_sparse_ground_state_hamiltonian_fast,
)
from new_approach.krab_4 import selected_krylov_ground_state
from Permutations.SKQD import do_skqd
from Permutations.Helpers import get_one_path, get_all_paths

# ---------------------------------------------------------------------------
# Fixed settings — not part of the scan
# ---------------------------------------------------------------------------
DELTA = 1e-8
N_ITERATIONS_MIN = 30          # floor on KRAB iterations
GROUND_ENERGY = -5.0
GAP = 1.0
RELATIVE_ERROR_TARGET = 1e-3

# --- Decaying-Q study (Q_mode="decay") -------------------------------------
# KRAB's per-iteration budget Q decays geometrically with the Hamiltonian
# application index `app`, starting from DECAY_Q_START_FRAC·2ⁿ and shrinking by
# DECAY_Q_FACTOR each application, with a floor at DECAY_Q_FLOOR_FRAC·2ⁿ:
#     Q(app) = max(DECAY_Q_FLOOR_FRAC·2ⁿ, int(DECAY_Q_START_FRAC·2ⁿ · f**app))
DECAY_Q_START_FRAC = 0.20      # initial budget, fraction of 2ⁿ
DECAY_Q_FLOOR_FRAC = 0.01      # floor budget, fraction of 2ⁿ
DECAY_Q_FACTOR = 0.75          # geometric decay per application


def resolve_n_iterations(Q_frac: float) -> int:
    """KRAB iteration cap: at least N_ITERATIONS_MIN, up to 3/Q_frac.

    Small Q adds few states per round, so it needs more rounds to build up a
    useful subspace; the cap scales inversely with Q to give it enough room.
    (1% -> 300, 3% -> 100, 5% -> 60, 10% -> 30.)
    """
    return max(N_ITERATIONS_MIN, int(round(3.0 / Q_frac)))


def make_decay_schedule(dim: int):
    """Geometric decaying-Q schedule for the decay study.

    Returns (Q_callable, Q_start_int, Q_floor_int). The callable maps the
    Hamiltonian-application index to an integer budget; KRAB resolves it per
    application via resolve_Q.
    """
    q_start = max(1, int(round(DECAY_Q_START_FRAC * dim)))
    q_floor = max(1, int(round(DECAY_Q_FLOOR_FRAC * dim)))

    def schedule(app: int, _s=q_start, _f=q_floor) -> int:
        return max(_f, int(_s * DECAY_Q_FACTOR ** app))

    return schedule, q_start, q_floor


def resolve_n_iterations_decay() -> int:
    """KRAB iteration cap for the decay study.

    Q starts large and decays to the 1% floor in
    log(FLOOR/START)/log(FACTOR) ≈ 10–11 applications; give enough rounds to
    reach the floor and then run a stretch of floor-sized iterations on top.
    """
    import math
    apps_to_floor = math.ceil(
        math.log(DECAY_Q_FLOOR_FRAC / DECAY_Q_START_FRAC) / math.log(DECAY_Q_FACTOR)
    )
    return max(N_ITERATIONS_MIN, apps_to_floor + N_ITERATIONS_MIN)

# SKQD time steps swept (same sweep as the BARK-vs-SKQD study)
SKQD_TIME_STEPS = [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

CSV_FIELDS = [
    # --- setup ---
    "n_qubits", "gs_sparsity", "ham_sparsity", "overlap", "actual_overlap",
    "Q_frac", "Q_int", "epsilon", "seed",
    "dim", "gs_support_size", "ham_nnz", "gap", "true_ground_energy",
    # --- KRAB ---
    "krab_final_energy", "krab_rel_error_final",
    "krab_final_subspace", "krab_iterations_run", "krab_max_iterations",
    "krab_native_converged",            # KRAB's own per-iteration energy hit target
    "krab_iter_to_target",              # first KRAB iteration reaching target (else NaN)
    "krab_subspace_at_target",          # diagonalized subspace size at that iteration
    "krab_n_states_to_target",          # #states on the discovery-order path to target
    "krab_walltime_s",
    # --- SKQD ---
    "skqd_best_t",                      # time step with fewest states to target
    "skqd_best_n_states_to_target",
    "skqd_any_converged",
    "skqd_best_final_energy",           # min final-path energy over all t
    "skqd_walltime_s",
    # --- random baseline ---
    "rand_median_n_states_to_target",
    "rand_min_n_states_to_target",
    "n_random_paths",
    # --- head-to-head ---
    "krab_beats_skqd",                  # KRAB reaches target with fewer states
    "advantage_ratio",                  # skqd_best_n_states / krab_n_states (>1 ⇒ KRAB cheaper)
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def choose_initial_state(psi: np.ndarray, target_overlap: float) -> tuple[int, float]:
    """Return the basis index whose |psi_i|^2 is closest to target_overlap."""
    probs = np.abs(psi) ** 2
    idx = int(np.argmin(np.abs(probs - target_overlap)))
    return idx, float(probs[idx])


def rel_error(energy: float, true_energy: float) -> float:
    if abs(true_energy) > 0:
        return abs(energy - true_energy) / abs(true_energy)
    return abs(energy - true_energy)


def n_states_to_target(path, true_energy: float, target: float = RELATIVE_ERROR_TARGET):
    """First number of basis states (1-based) at which the path reaches target.

    `path[i]` is the ground-energy estimate using the first (i+1) sampled
    states. Returns None if the target is never reached.
    """
    for i, e in enumerate(path):
        if rel_error(float(e), true_energy) < target:
            return i + 1
    return None


def skqd_ordering_to_indices(arr: np.ndarray) -> list[int]:
    """Strip the -1 sentinels do_skqd leaves in its pre-allocated array."""
    return [int(x) for x in arr if x >= 0]


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def save_figure(
    *,
    figure_dir, fname_stub,
    true_energy, target_band,
    n_qubits, gs_sparsity, ham_sparsity, overlap, actual_overlap,
    Q_frac, Q_int, epsilon, seed, q_label,
    random_paths, skqd_paths, krab_path,
    krab_native_sizes, krab_native_energies,
    krab_n_states, skqd_best_n_states, skqd_best_t,
    krab_diag,
) -> str:
    """Combined figure: convergence-path comparison (top) + KRAB diagnostics (bottom)."""
    fig = plt.figure(figsize=(15, 11))
    gs = fig.add_gridspec(3, 3, height_ratios=[2.1, 1.0, 1.0], hspace=0.42, wspace=0.28)

    # ===================== TOP: convergence-path comparison =====================
    ax = fig.add_subplot(gs[0, :])

    # Random paths in the background
    if random_paths is not None and len(random_paths):
        for path in random_paths:
            ax.plot(np.arange(1, len(path) + 1), path,
                    alpha=0.07, color="steelblue", linewidth=0.6)
        ax.plot([], [], color="steelblue", alpha=0.4, linewidth=1.5,
                label=f"Random ({len(random_paths)})")

    # SKQD sweep
    cmap = plt.cm.plasma
    colors = cmap(np.linspace(0.05, 0.9, len(skqd_paths)))
    for (t, path), color in zip(skqd_paths.items(), colors):
        ax.plot(np.arange(1, len(path) + 1), path, color=color, linewidth=1.1,
                label=f"SKQD t={t}")

    # KRAB discovery-order path (apples-to-apples with SKQD)
    ax.plot(np.arange(1, len(krab_path) + 1), krab_path,
            color="red", linewidth=2.8, linestyle="--",
            label=f"KRAB path ({q_label})")

    # KRAB native trajectory: where KRAB actually diagonalized a subspace
    ax.plot(krab_native_sizes, krab_native_energies, color="black",
            marker="o", markersize=5, linewidth=0.0, zorder=5,
            label="KRAB diagonalized")

    # Reference lines
    ax.axhline(true_energy, color="black", linestyle=":", linewidth=1.3,
               label=f"True E₀={true_energy:.2f}")
    ax.axhspan(true_energy - target_band, true_energy + target_band,
               color="green", alpha=0.12, label=f"±{RELATIVE_ERROR_TARGET:.0e} band")

    # Vertical markers at #states-to-target
    if krab_n_states is not None:
        ax.axvline(krab_n_states, color="red", linestyle="-", linewidth=1.0, alpha=0.6)
    if skqd_best_n_states is not None:
        ax.axvline(skqd_best_n_states, color="purple", linestyle="-", linewidth=1.0, alpha=0.6)

    ax.set_xlabel("Number of sampled basis states (in order)", fontsize=12)
    ax.set_ylabel("Ground-state energy estimate", fontsize=12)

    krab_txt = f"{krab_n_states}" if krab_n_states is not None else "—"
    skqd_txt = (f"{skqd_best_n_states} (t={skqd_best_t})"
                if skqd_best_n_states is not None else "—")
    ax.set_title(
        f"KRAB vs SKQD  |  n={n_qubits}q, gs={gs_sparsity:.2f}, ham={ham_sparsity:.2f}, "
        f"overlap target={overlap:.2f} (actual {actual_overlap:.3f}),  "
        f"{q_label}, ε={epsilon:.0e}\n"
        f"states to reach rel.err<{RELATIVE_ERROR_TARGET:.0e}:  "
        f"KRAB={krab_txt}    best SKQD={skqd_txt}",
        fontsize=11,
    )
    ax.legend(loc="center left", bbox_to_anchor=(1.005, 0.5), fontsize=7.5)
    ax.grid(True, alpha=0.3)

    # ===================== BOTTOM: KRAB internal diagnostics =====================
    iters = krab_diag["iters"]

    # (1) energy + relative error
    ax1 = fig.add_subplot(gs[1, 0])
    ax1.plot(iters, krab_diag["energies"], marker="o", color="steelblue", linewidth=1.4)
    ax1.axhline(true_energy, color="red", linestyle="--", linewidth=1.0)
    ax1.set_xlabel("KRAB iteration"); ax1.set_ylabel("Energy")
    ax1.set_title("KRAB energy vs iteration", fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[1, 1])
    ax2.semilogy(iters, krab_diag["rel_errors"], marker="o", color="darkorange", linewidth=1.4)
    ax2.axhline(RELATIVE_ERROR_TARGET, color="red", linestyle="--", linewidth=1.0)
    ax2.set_xlabel("KRAB iteration"); ax2.set_ylabel("Relative error")
    ax2.set_title("KRAB relative error (log)", fontsize=9)
    ax2.grid(True, alpha=0.3, which="both")

    # (2) subspace size
    ax3 = fig.add_subplot(gs[1, 2])
    ax3.plot(iters, krab_diag["active_sizes"], marker="s", color="seagreen", linewidth=1.4)
    ax3.set_xlabel("KRAB iteration"); ax3.set_ylabel("Active subspace size")
    ax3.set_title("KRAB subspace dimension", fontsize=9)
    ax3.grid(True, alpha=0.3)

    # (3) new bitstrings added vs Q
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.bar(iters, krab_diag["n_new"], color="mediumpurple", alpha=0.85, label="new kept")
    ax4.plot(iters, krab_diag["Q_used"], color="black", marker="x", linewidth=1.0, label="Q budget")
    ax4.set_xlabel("KRAB iteration"); ax4.set_ylabel("Bitstrings")
    ax4.set_title("KRAB bitstrings added / Q", fontsize=9)
    ax4.legend(fontsize=7); ax4.grid(True, alpha=0.3, axis="y")

    # (4) external residual norm (how much "reach" is left outside the subspace)
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.semilogy(iters, np.clip(krab_diag["resid_norm"], 1e-16, None),
                 marker="o", color="firebrick", linewidth=1.4)
    ax5.set_xlabel("KRAB iteration"); ax5.set_ylabel("External residual norm")
    ax5.set_title("KRAB residual reach (log)", fontsize=9)
    ax5.grid(True, alpha=0.3, which="both")

    # (5) participation ratio of the approximate ground state
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.plot(iters, krab_diag["pr"], marker="o", color="teal", linewidth=1.4, label="participation")
    ax6.plot(iters, krab_diag["nnz"], marker="x", color="gray", linewidth=1.0, label="nnz")
    ax6.set_xlabel("KRAB iteration"); ax6.set_ylabel("Effective #states")
    ax6.set_title("KRAB state sparsity", fontsize=9)
    ax6.legend(fontsize=7); ax6.grid(True, alpha=0.3)

    os.makedirs(figure_dir, exist_ok=True)
    out_path = os.path.join(figure_dir, fname_stub + ".png")
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


def append_csv_row(data_dir: str, row: dict, csv_name: str = "comparison.csv") -> None:
    os.makedirs(data_dir, exist_ok=True)
    csv_path = os.path.join(data_dir, csv_name)
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in CSV_FIELDS})


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run(
    n_qubits: int,
    gs_sparsity: float,
    ham_sparsity: float,
    overlap: float,
    Q_frac: float,
    epsilon: float,
    seed: int,
    figure_dir: str,
    data_dir: str,
    n_random_paths: int = 20,
    Q_mode: str = "fixed",
) -> dict:
    dim = 2 ** n_qubits
    decay = Q_mode == "decay"
    if decay:
        # Q decays geometrically with the application index; the integer budget
        # KRAB keeps per iteration is given by the schedule callable. Q_int is
        # the initial (largest) budget, used for reporting/labels.
        Q_arg, Q_int, Q_floor_int = make_decay_schedule(dim)
        n_iterations = resolve_n_iterations_decay()
    else:
        # Q is a fraction of the full Hilbert space; resolve to an integer
        # number of new basis states KRAB keeps per iteration (at least 1).
        Q_int = max(1, int(round(Q_frac * dim)))
        Q_floor_int = Q_int
        Q_arg = Q_int
        n_iterations = resolve_n_iterations(Q_frac)
    target_band = abs(GROUND_ENERGY) * RELATIVE_ERROR_TARGET
    # SKQD steps: keep num_steps a divisor of dim (matches BARK study convention).
    n_skqd_steps = max(8, min(dim, dim // 16))

    if decay:
        Q_desc = f"Q decay {DECAY_Q_START_FRAC:.0%}→{DECAY_Q_FLOOR_FRAC:.0%} (×{DECAY_Q_FACTOR})={Q_int}→{Q_floor_int}"
    else:
        Q_desc = f"Q={Q_frac:.0%}={Q_int}"

    t0 = time.perf_counter()
    print(f"[{n_qubits}q | gs={gs_sparsity} | ham={ham_sparsity} | "
          f"ov={overlap} | {Q_desc} | eps={epsilon:.0e}]", flush=True)

    # ------------------------------------------------------------------
    # 1. Hamiltonian + true ground energy
    # ------------------------------------------------------------------
    print("  Building Hamiltonian ...", flush=True)
    H, _, psi, support, info = make_controlled_sparse_ground_state_hamiltonian_fast(
        n_qubits=n_qubits,
        ground_state_sparsity=gs_sparsity,
        hamiltonian_sparsity=ham_sparsity,
        seed=seed,
        ground_energy=GROUND_ENERGY,
        gap=GAP,
        return_info=True,
        make_pauli_op=False,
    )
    gap_val = info.get("actual_gap", info.get("requested_gap", float("nan")))

    if dim <= 256:
        true_energy = float(np.linalg.eigvalsh(H.toarray())[0])
    else:
        true_energy = float(eigsh(H, k=1, which="SA", return_eigenvectors=False)[0])

    initial_idx, actual_overlap = choose_initial_state(psi, overlap)
    initial_bitstring = format(initial_idx, f"0{n_qubits}b")
    print(f"    dim={dim}, E₀={true_energy:.6f}, gap≈{gap_val:.3f}, "
          f"init idx={initial_idx}, |<ψ|init>|²={actual_overlap:.4f}", flush=True)

    # ------------------------------------------------------------------
    # 2. KRAB
    # ------------------------------------------------------------------
    print(f"  Running KRAB ({Q_desc}, ε={epsilon:.0e}, "
          f"max_iter={n_iterations}) ...", flush=True)
    tK = time.perf_counter()
    result = selected_krylov_ground_state(
        H=H,
        initial_bitstring=initial_bitstring,
        Q=Q_arg,
        delta=DELTA,
        epsilon=epsilon,
        n_iterations=n_iterations,
        max_subspace_dim=None,
        score_mode="residual",
    )
    krab_walltime = time.perf_counter() - tK

    diags = result.iteration_diagnostics
    krab_iters = [d.iteration for d in diags]
    krab_energies = [d.energy_before_post_prune for d in diags]
    krab_active_start = [d.active_size_start for d in diags]
    krab_rel_errors = [rel_error(e, true_energy) for e in krab_energies]

    # native convergence (KRAB's own per-iteration diagonalization)
    krab_iter_to_target = next(
        (it for it, re in zip(krab_iters, krab_rel_errors) if re < RELATIVE_ERROR_TARGET),
        None,
    )
    if krab_iter_to_target is not None:
        krab_subspace_at_target = krab_active_start[krab_iters.index(krab_iter_to_target)]
    else:
        krab_subspace_at_target = None

    # discovery-order path (apples-to-apples with SKQD)
    krab_path = get_one_path(H, result.final_indices)
    krab_n_states = n_states_to_target(krab_path, true_energy)

    krab_final_energy = float(result.final_energy)
    krab_rel_error_final = rel_error(krab_final_energy, true_energy)
    krab_final_subspace = result.final_diagnostics.final_active_size
    print(f"    KRAB: {krab_walltime:.2f}s, final E={krab_final_energy:.6f}, "
          f"final subspace={krab_final_subspace}, "
          f"native iter_to_target={krab_iter_to_target}, "
          f"path states_to_target={krab_n_states}", flush=True)

    krab_diag = dict(
        iters=krab_iters,
        energies=krab_energies,
        rel_errors=np.clip(krab_rel_errors, 1e-16, None),
        active_sizes=[d.active_size_end for d in diags],
        n_new=[d.n_new_bitstrings_kept for d in diags],
        Q_used=[d.Q_this_application for d in diags],
        resid_norm=[d.external_residual_norm for d in diags],
        pr=[d.ground_state_sparsity_applied_to_H.participation_ratio for d in diags],
        nnz=[d.ground_state_sparsity_applied_to_H.nnz for d in diags],
    )

    # ------------------------------------------------------------------
    # 3. SKQD sweep
    # ------------------------------------------------------------------
    print(f"  Running SKQD ({len(SKQD_TIME_STEPS)} time steps × {n_skqd_steps} steps) ...",
          flush=True)
    tS = time.perf_counter()
    skqd_paths: dict[float, np.ndarray] = {}
    skqd_n_states: dict[float, int | None] = {}
    for t in SKQD_TIME_STEPS:
        ordering = do_skqd(H, n_skqd_steps, t=t, initial=initial_idx)
        indices = skqd_ordering_to_indices(ordering)
        path = get_one_path(H, indices)
        skqd_paths[t] = path
        skqd_n_states[t] = n_states_to_target(path, true_energy)
    skqd_walltime = time.perf_counter() - tS

    reached = {t: n for t, n in skqd_n_states.items() if n is not None}
    if reached:
        skqd_best_t = min(reached, key=reached.get)
        skqd_best_n_states = reached[skqd_best_t]
    else:
        skqd_best_t, skqd_best_n_states = None, None
    skqd_best_final_energy = float(min(p[-1] for p in skqd_paths.values()))
    print(f"    SKQD: {skqd_walltime:.2f}s, best t={skqd_best_t}, "
          f"best states_to_target={skqd_best_n_states}", flush=True)

    # ------------------------------------------------------------------
    # 4. Random baseline
    # ------------------------------------------------------------------
    rand_median = rand_min = None
    random_paths = None
    if n_random_paths > 0:
        print(f"  Computing {n_random_paths} random paths ...", flush=True)
        random_paths = get_all_paths(H, n_random_paths, start=initial_idx)
        rand_ns = [n_states_to_target(p, true_energy) for p in random_paths]
        rand_ns_ok = [n for n in rand_ns if n is not None]
        if rand_ns_ok:
            rand_median = float(np.median(rand_ns_ok))
            rand_min = int(min(rand_ns_ok))

    # ------------------------------------------------------------------
    # 5. Head-to-head
    # ------------------------------------------------------------------
    k = krab_n_states if krab_n_states is not None else np.inf
    s = skqd_best_n_states if skqd_best_n_states is not None else np.inf
    krab_beats_skqd = bool(k < s)
    if np.isfinite(k) and np.isfinite(s) and k > 0:
        advantage_ratio = float(s / k)
    elif np.isfinite(k) and not np.isfinite(s):
        advantage_ratio = float("inf")     # KRAB reaches target, SKQD never does
    elif not np.isfinite(k) and np.isfinite(s):
        advantage_ratio = 0.0              # SKQD reaches, KRAB never does
    else:
        advantage_ratio = float("nan")     # neither reaches target

    # ------------------------------------------------------------------
    # 6. Figure
    # ------------------------------------------------------------------
    Q_tag = "Qdecay" if decay else f"Qf{Q_frac:.3f}"
    fname_stub = (
        f"nq{n_qubits}_gs{gs_sparsity:.2f}_ham{ham_sparsity:.2f}"
        f"_ov{overlap:.2f}_{Q_tag}_eps{epsilon:.0e}_seed{seed}"
    )
    fig_path = save_figure(
        figure_dir=figure_dir, fname_stub=fname_stub,
        true_energy=true_energy, target_band=target_band,
        n_qubits=n_qubits, gs_sparsity=gs_sparsity, ham_sparsity=ham_sparsity,
        overlap=overlap, actual_overlap=actual_overlap,
        Q_frac=Q_frac, Q_int=Q_int, epsilon=epsilon, seed=seed, q_label=Q_desc,
        random_paths=random_paths, skqd_paths=skqd_paths, krab_path=krab_path,
        krab_native_sizes=krab_active_start, krab_native_energies=krab_energies,
        krab_n_states=krab_n_states,
        skqd_best_n_states=skqd_best_n_states, skqd_best_t=skqd_best_t,
        krab_diag=krab_diag,
    )
    print(f"    Figure: {fig_path}", flush=True)

    # ------------------------------------------------------------------
    # 7. CSV row
    # ------------------------------------------------------------------
    # In decay mode Q_frac/Q_int report the *initial* (largest) budget; the
    # per-application schedule is captured in the figure's "Q budget" curve.
    row_Q_frac = DECAY_Q_START_FRAC if decay else Q_frac
    row = {
        "n_qubits": n_qubits, "gs_sparsity": gs_sparsity, "ham_sparsity": ham_sparsity,
        "overlap": overlap, "actual_overlap": actual_overlap,
        "Q_frac": row_Q_frac, "Q_int": Q_int, "epsilon": epsilon, "seed": seed,
        "dim": dim, "gs_support_size": info["ground_state_support_size"],
        "ham_nnz": info["hamiltonian_nnz"], "gap": gap_val,
        "true_ground_energy": true_energy,
        "krab_final_energy": krab_final_energy,
        "krab_rel_error_final": krab_rel_error_final,
        "krab_final_subspace": krab_final_subspace,
        "krab_iterations_run": len(diags),
        "krab_max_iterations": n_iterations,
        "krab_native_converged": krab_iter_to_target is not None,
        "krab_iter_to_target": krab_iter_to_target if krab_iter_to_target is not None else float("nan"),
        "krab_subspace_at_target": krab_subspace_at_target if krab_subspace_at_target is not None else float("nan"),
        "krab_n_states_to_target": krab_n_states if krab_n_states is not None else float("nan"),
        "krab_walltime_s": krab_walltime,
        "skqd_best_t": skqd_best_t if skqd_best_t is not None else float("nan"),
        "skqd_best_n_states_to_target": skqd_best_n_states if skqd_best_n_states is not None else float("nan"),
        "skqd_any_converged": skqd_best_n_states is not None,
        "skqd_best_final_energy": skqd_best_final_energy,
        "skqd_walltime_s": skqd_walltime,
        "rand_median_n_states_to_target": rand_median if rand_median is not None else float("nan"),
        "rand_min_n_states_to_target": rand_min if rand_min is not None else float("nan"),
        "n_random_paths": n_random_paths,
        "krab_beats_skqd": krab_beats_skqd,
        "advantage_ratio": advantage_ratio,
    }
    csv_name = "comparison_decay.csv" if decay else "comparison.csv"
    append_csv_row(data_dir, row, csv_name=csv_name)
    print(f"  Total wall time: {time.perf_counter()-t0:.1f}s", flush=True)
    return row


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run one KRAB-vs-SKQD comparison experiment and save figure + CSV row."
    )
    p.add_argument("--n_qubits",       type=int,   required=True)
    p.add_argument("--gs_sparsity",    type=float, required=True)
    p.add_argument("--ham_sparsity",   type=float, required=True)
    p.add_argument("--overlap",        type=float, required=True)
    p.add_argument("--Q_frac",         type=float, required=True,
                   help="KRAB: new basis states kept per iteration, as a fraction "
                        "of the full Hilbert space 2ⁿ (e.g. 0.03 = 3%%). "
                        "Ignored when --Q_mode decay (the budget comes from the "
                        "decay schedule instead).")
    p.add_argument("--Q_mode",         type=str, default="fixed",
                   choices=["fixed", "decay"],
                   help="fixed: constant Q=round(Q_frac·2ⁿ) per iteration (default). "
                        "decay: geometric Q(app)=max(1%%·2ⁿ, int(20%%·2ⁿ·0.75**app)).")
    p.add_argument("--epsilon",        type=float, required=True,
                   help="KRAB: coefficient pruning threshold after diagonalization.")
    p.add_argument("--n_random_paths", type=int,   default=20,
                   help="Random orderings sampled as a baseline (0 disables).")
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--figure_dir",     type=str,   default="figures")
    p.add_argument("--data_dir",       type=str,   default="data")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        n_qubits=args.n_qubits,
        gs_sparsity=args.gs_sparsity,
        ham_sparsity=args.ham_sparsity,
        overlap=args.overlap,
        Q_frac=args.Q_frac,
        epsilon=args.epsilon,
        seed=args.seed,
        figure_dir=args.figure_dir,
        data_dir=args.data_dir,
        n_random_paths=args.n_random_paths,
        Q_mode=args.Q_mode,
    )
