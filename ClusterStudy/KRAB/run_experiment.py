"""
Single-job experiment script for the KRAB systematic cluster study.

For one combination of hyperparameters (n_qubits, gs_sparsity, ham_sparsity,
overlap, Q, epsilon) the script:
  1. Builds a random Hamiltonian with a controlled sparse ground state.
  2. Computes the true ground energy via eigsh.
  3. Runs KRAB (selected_krylov_ground_state) with per-iteration timing.
  4. Records the first iteration at which relative error < 1e-3.
  5. Saves a per-experiment diagnostic figure to --figure_dir.
  6. Appends one CSV row of results to --data_dir/results.csv.

Usage
-----
python run_experiment.py \
    --n_qubits 8 --gs_sparsity 0.1 --ham_sparsity 0.5 \
    --overlap 0.3 --Q 20 --epsilon 1e-6 \
    --seed 42 --figure_dir results --data_dir data
"""

import argparse
import csv
import os
import sys
import time
import traceback

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.linalg import eigsh

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from new_approach.Hamiltonian_controlled_sparsity import (
    make_controlled_sparse_ground_state_hamiltonian_fast,
)
from new_approach.krab_4 import selected_krylov_ground_state

# Fixed KRAB settings — not part of the scan
DELTA = 1e-8
N_ITERATIONS = 30
GROUND_ENERGY = -5.0
GAP = 1.0
RELATIVE_ERROR_TARGET = 1e-3

CSV_FIELDS = [
    "n_qubits", "gs_sparsity", "ham_sparsity", "overlap", "Q", "epsilon", "seed",
    "true_ground_energy",
    "final_energy", "relative_error_final",
    "converged",
    "iter_to_convergence",       # first iteration where rel_error < 1e-3 (0-based); NaN if never
    "subspace_at_convergence",   # active_size_end at that iteration; NaN if never
    "total_iterations_run",
    "final_subspace_size",
    "wall_time_total_s",
    "est_wall_time_to_convergence_s",  # wall_time * (iter_to_conv / total_iters); NaN if never
    "gs_support_size",
    "ham_nnz",
    "dim",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def choose_initial_state(psi: np.ndarray, target_overlap: float) -> tuple[int, float]:
    probs = np.abs(psi) ** 2
    idx = int(np.argmin(np.abs(probs - target_overlap)))
    return idx, float(probs[idx])


def find_convergence(energies: list[float], true_energy: float) -> int | None:
    """Return the first index where relative error < RELATIVE_ERROR_TARGET, else None."""
    for i, e in enumerate(energies):
        if abs(true_energy) > 0 and abs(e - true_energy) / abs(true_energy) < RELATIVE_ERROR_TARGET:
            return i
        # handle near-zero true energy gracefully
        if abs(true_energy) == 0 and abs(e - true_energy) < RELATIVE_ERROR_TARGET:
            return i
    return None


def save_figure(
    result,
    true_energy: float,
    n_qubits: int,
    gs_sparsity: float,
    ham_sparsity: float,
    overlap: float,
    Q: int,
    epsilon: float,
    seed: int,
    figure_dir: str,
) -> str:
    diags = result.iteration_diagnostics
    iterations = [d.iteration for d in diags]
    energies = [d.energy_before_post_prune for d in diags]
    active_sizes = [d.active_size_end for d in diags]
    n_new = [d.n_new_bitstrings_kept for d in diags]

    rel_errors = [
        abs(e - true_energy) / abs(true_energy) if abs(true_energy) > 0 else abs(e - true_energy)
        for e in energies
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    # Top-left: energy convergence
    ax = axes[0, 0]
    ax.plot(iterations, energies, marker="o", color="steelblue", linewidth=1.5)
    ax.axhline(true_energy, color="red", linestyle="--", linewidth=1.2, label=f"True E₀={true_energy}")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Energy estimate")
    ax.set_title("Energy vs iteration")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.4)

    # Top-right: relative error (log scale)
    ax = axes[0, 1]
    ax.semilogy(iterations, rel_errors, marker="o", color="darkorange", linewidth=1.5)
    ax.axhline(RELATIVE_ERROR_TARGET, color="red", linestyle="--", linewidth=1.2,
               label=f"Target {RELATIVE_ERROR_TARGET:.0e}")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Relative error")
    ax.set_title("Relative error (log scale)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.4, which="both")

    # Bottom-left: active subspace size
    ax = axes[1, 0]
    ax.plot(iterations, active_sizes, marker="s", color="seagreen", linewidth=1.5)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Active subspace size")
    ax.set_title("Active subspace dimension")
    ax.grid(True, alpha=0.4)

    # Bottom-right: new bitstrings added per iteration
    ax = axes[1, 1]
    ax.bar(iterations, n_new, color="mediumpurple", alpha=0.8)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("New bitstrings")
    ax.set_title("New bitstrings added per iteration")
    ax.grid(True, alpha=0.4, axis="y")

    conv_iter = find_convergence(energies, true_energy)
    conv_str = f"converges at iter {conv_iter}" if conv_iter is not None else "does not converge"
    fig.suptitle(
        f"KRAB  |  n={n_qubits}q, gs={gs_sparsity:.2f}, ham={ham_sparsity:.2f}, "
        f"ov={overlap:.2f}\n"
        f"Q={Q}, ε={epsilon:.0e}, seed={seed}  |  {conv_str}",
        fontsize=11,
    )
    fig.tight_layout()

    os.makedirs(figure_dir, exist_ok=True)
    fname = (
        f"nq{n_qubits}_gs{gs_sparsity:.2f}_ham{ham_sparsity:.2f}"
        f"_ov{overlap:.2f}_Q{Q:03d}_eps{epsilon:.0e}_seed{seed}.png"
    )
    out_path = os.path.join(figure_dir, fname)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


def append_csv_row(data_dir: str, row: dict) -> None:
    os.makedirs(data_dir, exist_ok=True)
    csv_path = os.path.join(data_dir, "results.csv")
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
    Q: int,
    epsilon: float,
    seed: int,
    figure_dir: str,
    data_dir: str,
) -> dict:
    dim = 2 ** n_qubits
    print(
        f"[{n_qubits}q | gs={gs_sparsity} | ham={ham_sparsity} | "
        f"ov={overlap} | Q={Q} | eps={epsilon:.0e}]",
        flush=True,
    )

    # ------------------------------------------------------------------
    # 1. Build Hamiltonian
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
        make_pauli_op=False,   # not needed for KRAB (works on CSR directly)
    )

    # ------------------------------------------------------------------
    # 2. True ground energy
    # ------------------------------------------------------------------
    if dim <= 256:
        true_energy = float(np.linalg.eigvalsh(H.toarray())[0])
    else:
        true_energy = float(eigsh(H, k=1, which="SA", return_eigenvectors=False)[0])
    print(f"    True E₀ = {true_energy:.8f}", flush=True)

    # ------------------------------------------------------------------
    # 3. Choose initial state
    # ------------------------------------------------------------------
    initial_idx, real_overlap = choose_initial_state(psi, overlap)
    initial_bitstring = format(initial_idx, f"0{n_qubits}b")
    print(f"    Initial state idx={initial_idx}, |<ψ|init>|²={real_overlap:.4f}", flush=True)

    # ------------------------------------------------------------------
    # 4. Run KRAB with timing
    # ------------------------------------------------------------------
    print(f"  Running KRAB (Q={Q}, ε={epsilon:.0e}, δ={DELTA:.0e}, "
          f"max_iter={N_ITERATIONS}) ...", flush=True)
    t0 = time.perf_counter()

    result = selected_krylov_ground_state(
        H=H,
        initial_bitstring=initial_bitstring,
        Q=Q,
        delta=DELTA,
        epsilon=epsilon,
        n_iterations=N_ITERATIONS,
        max_subspace_dim=None,
        score_mode="residual",
    )

    wall_time = time.perf_counter() - t0
    print(f"    Done in {wall_time:.2f}s", flush=True)

    # ------------------------------------------------------------------
    # 5. Extract per-iteration energies and find convergence
    # ------------------------------------------------------------------
    diags = result.iteration_diagnostics
    energies = [d.energy_before_post_prune for d in diags]
    active_sizes_end = [d.active_size_end for d in diags]
    total_iters = len(diags)

    conv_iter = find_convergence(energies, true_energy)
    if conv_iter is not None:
        subspace_at_conv = active_sizes_end[conv_iter]
        est_time_to_conv = wall_time * (conv_iter + 1) / max(total_iters, 1)
        print(f"    Converged at iteration {conv_iter}, "
              f"subspace size {subspace_at_conv}, "
              f"estimated wall time {est_time_to_conv:.2f}s", flush=True)
    else:
        subspace_at_conv = float("nan")
        est_time_to_conv = float("nan")
        print(f"    Did NOT converge within {N_ITERATIONS} iterations.", flush=True)

    final_energy = result.final_energy
    rel_error_final = (
        abs(final_energy - true_energy) / abs(true_energy)
        if abs(true_energy) > 0 else abs(final_energy - true_energy)
    )
    print(f"    Final energy {final_energy:.6f}, rel_error {rel_error_final:.2e}", flush=True)

    # ------------------------------------------------------------------
    # 6. Save figure
    # ------------------------------------------------------------------
    fig_path = save_figure(
        result, true_energy, n_qubits, gs_sparsity, ham_sparsity,
        overlap, Q, epsilon, seed, figure_dir,
    )
    print(f"    Figure: {fig_path}", flush=True)

    # ------------------------------------------------------------------
    # 7. Save CSV row
    # ------------------------------------------------------------------
    row = {
        "n_qubits":                   n_qubits,
        "gs_sparsity":                gs_sparsity,
        "ham_sparsity":               ham_sparsity,
        "overlap":                    overlap,
        "Q":                          Q,
        "epsilon":                    epsilon,
        "seed":                       seed,
        "true_ground_energy":         true_energy,
        "final_energy":               final_energy,
        "relative_error_final":       rel_error_final,
        "converged":                  conv_iter is not None,
        "iter_to_convergence":        conv_iter if conv_iter is not None else float("nan"),
        "subspace_at_convergence":    subspace_at_conv,
        "total_iterations_run":       total_iters,
        "final_subspace_size":        result.final_diagnostics.final_active_size,
        "wall_time_total_s":          wall_time,
        "est_wall_time_to_convergence_s": est_time_to_conv,
        "gs_support_size":            info["ground_state_support_size"],
        "ham_nnz":                    info["hamiltonian_nnz"],
        "dim":                        dim,
    }
    append_csv_row(data_dir, row)
    return row


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run one KRAB experiment and save figure + CSV row."
    )
    p.add_argument("--n_qubits",    type=int,   required=True)
    p.add_argument("--gs_sparsity", type=float, required=True)
    p.add_argument("--ham_sparsity",type=float, required=True)
    p.add_argument("--overlap",     type=float, required=True)
    p.add_argument("--Q",           type=int,   required=True,
                   help="Number of new candidates to keep per iteration.")
    p.add_argument("--epsilon",     type=float, required=True,
                   help="Coefficient pruning threshold after diagonalization.")
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--figure_dir",  type=str,   default="results")
    p.add_argument("--data_dir",    type=str,   default="data")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        n_qubits=args.n_qubits,
        gs_sparsity=args.gs_sparsity,
        ham_sparsity=args.ham_sparsity,
        overlap=args.overlap,
        Q=args.Q,
        epsilon=args.epsilon,
        seed=args.seed,
        figure_dir=args.figure_dir,
        data_dir=args.data_dir,
    )
