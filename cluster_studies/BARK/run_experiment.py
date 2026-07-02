"""
Single-job experiment script for the BARK vs SKQD cluster study.

Runs one combination of hyperparameters:
  n_qubits, gs_sparsity, ham_sparsity, overlap, keep_states

and saves a comparison figure to --output_dir.

Usage
-----
python run_experiment.py \
    --n_qubits 8 \
    --gs_sparsity 0.1 \
    --ham_sparsity 0.5 \
    --overlap 0.3 \
    --keep_states 10 \
    --n_random_paths 100 \
    --seed 42 \
    --output_dir results
"""

import argparse
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for cluster nodes
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh

# ---------------------------------------------------------------------------
# Core routines come from the installed `subspace_search` package
# (`pip install -e ../../subspace_search`), so no sys.path juggling is needed.
# ---------------------------------------------------------------------------
from subspace_search.hamiltonians import (
    make_controlled_sparse_ground_state_hamiltonian_fast,
)
from subspace_search.paths import get_all_paths, get_one_path, project_down
from subspace_search.skqd import do_skqd
from subspace_search.algorithms import BarkBarkBark

# ---------------------------------------------------------------------------
# Time steps swept for SKQD  (hardcoded — we always show the full sweep)
# ---------------------------------------------------------------------------
SKQD_TIME_STEPS = [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def choose_initial_state(psi: np.ndarray, target_overlap: float) -> tuple[int, float]:
    """Return the basis index whose |ψ_i|² is closest to target_overlap."""
    probs = np.abs(psi) ** 2
    idx = int(np.argmin(np.abs(probs - target_overlap)))
    return idx, float(probs[idx])


def skqd_ordering_to_indices(arr: np.ndarray) -> list[int]:
    """Convert the raw do_skqd output to a clean list of integer indices.

    do_skqd fills a pre-allocated array with -1 sentinels; filter those out
    and cast to int.
    """
    return [int(x) for x in arr if x >= 0]


def get_bark_path(H, H_spo, initial_idx: int, n_qubits: int, keep_states: int,
                  seed: int) -> tuple[list[str], np.ndarray]:
    """Run BarkBarkBark and compute the energy-vs-subspace-size path."""
    initial_bitstring = format(initial_idx, f"0{n_qubits}b")
    dim = 2 ** n_qubits

    # max_applications: enough rounds for BARK to encounter all dim states.
    # With keep_states per round, we need at least dim / keep_states rounds.
    # We add a generous buffer and cap at dim.
    max_applications = min(dim, max(n_qubits * 4, 2 * dim // max(1, keep_states)))

    explorer = BarkBarkBark(
        H=H_spo,
        initial_state=initial_bitstring,
        keep_states=keep_states,
        max_applications=max_applications,
        mode="top_m",
        sampling_score="amplitude",
        return_only_applied_bitstrings=True,
        random_seed=seed,
    )
    bitstrings = explorer.run()

    # Convert bitstrings to integer indices and compute energy path
    indices = [int(b, 2) for b in bitstrings]
    energies = get_one_path(H, indices)
    return bitstrings, energies


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run(
    n_qubits: int,
    gs_sparsity: float,
    ham_sparsity: float,
    overlap: float,
    keep_states: int,
    n_random_paths: int,
    seed: int,
    output_dir: str,
) -> None:
    dim = 2 ** n_qubits
    # SKQD uses dim // 16 steps (matching the notebook convention for n_qubits=8).
    # Clamped to [8, dim].
    n_skqd_steps = max(8, min(dim, dim // 16))

    t0 = time.time()
    print(f"[{n_qubits}q | gs={gs_sparsity} | ham={ham_sparsity} | ov={overlap} | ks={keep_states}]")

    # ------------------------------------------------------------------
    # 1. Build Hamiltonian
    # ------------------------------------------------------------------
    print("  Building Hamiltonian ...", flush=True)
    H, H_spo, psi, support, info = make_controlled_sparse_ground_state_hamiltonian_fast(
        n_qubits=n_qubits,
        ground_state_sparsity=gs_sparsity,
        hamiltonian_sparsity=ham_sparsity,
        seed=seed,
        ground_energy=0.0,
        gap=1.0,
        return_info=True,
        make_pauli_op=True,
    )
    gap_val = info.get("actual_gap", info.get("requested_gap", float("nan")))
    print(f"    dim={dim}, gs_support={info['ground_state_support_size']}, "
          f"ham_nnz={info['hamiltonian_nnz']}, gap≈{gap_val:.4f}")

    # ------------------------------------------------------------------
    # 2. Choose initial state closest to the requested overlap
    # ------------------------------------------------------------------
    initial_idx, real_overlap = choose_initial_state(psi, overlap)
    print(f"  Initial state: idx={initial_idx}, |<ψ|init>|²={real_overlap:.6f} "
          f"(requested {overlap})")

    # ------------------------------------------------------------------
    # 3. SKQD — sweep over time steps
    # ------------------------------------------------------------------
    print(f"  Running SKQD for {len(SKQD_TIME_STEPS)} time steps "
          f"({n_skqd_steps} SKQD steps each) ...", flush=True)
    skqd_paths: dict[float, np.ndarray] = {}
    for t in SKQD_TIME_STEPS:
        ordering = do_skqd(H, n_skqd_steps, t=t, initial=initial_idx)
        indices = skqd_ordering_to_indices(ordering)
        skqd_paths[t] = get_one_path(H, indices)
    print(f"    Done. ({time.time()-t0:.1f}s elapsed)", flush=True)

    # ------------------------------------------------------------------
    # 4. Random paths
    # ------------------------------------------------------------------
    print(f"  Computing {n_random_paths} random paths ...", flush=True)
    t1 = time.time()
    random_paths = get_all_paths(H, n_random_paths, start=initial_idx)
    print(f"    Done. ({time.time()-t1:.1f}s, {time.time()-t0:.1f}s total)", flush=True)

    # ------------------------------------------------------------------
    # 5. BARK
    # ------------------------------------------------------------------
    print(f"  Running BARK (keep_states={keep_states}) ...", flush=True)
    t2 = time.time()
    _, bark_path = get_bark_path(H, H_spo, initial_idx, n_qubits, keep_states, seed)
    print(f"    Done. ({time.time()-t2:.1f}s, {time.time()-t0:.1f}s total)", flush=True)

    # ------------------------------------------------------------------
    # 6. Plot
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(13, 6))

    # Random paths in the background
    for path in random_paths:
        ax.plot(path, alpha=0.08, color="steelblue", linewidth=0.6)
    # Dummy artist for legend
    ax.plot([], [], color="steelblue", alpha=0.4, linewidth=1.5, label=f"Random ({n_random_paths})")

    # SKQD sweep
    cmap = plt.cm.plasma
    colors = cmap(np.linspace(0.1, 0.9, len(SKQD_TIME_STEPS)))
    for (t, path), color in zip(skqd_paths.items(), colors):
        ax.plot(path, color=color, linewidth=1.2, label=f"SKQD t={t}")

    # BARK
    ax.plot(bark_path, color="red", linewidth=2.5, linestyle="--",
            label=f"BARK (keep={keep_states})")

    # True ground energy reference line (E₀ = 0 by construction)
    ax.axhline(0.0, color="black", linestyle=":", linewidth=1.2, label="True E₀ = 0")

    ax.set_xlabel("Sampled Basis States (in order)", fontsize=12)
    ax.set_ylabel("Ground-State Energy Estimate", fontsize=12)
    ax.set_title(
        f"BARK vs SKQD  |  n_qubits={n_qubits},  gs_sparsity={gs_sparsity:.2f},  "
        f"ham_sparsity={ham_sparsity:.2f}\n"
        f"target overlap={overlap:.2f}  (actual |⟨ψ|init⟩|²={real_overlap:.4f}),  "
        f"keep_states={keep_states},  seed={seed}",
        fontsize=11,
    )
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)

    os.makedirs(output_dir, exist_ok=True)
    fname = (
        f"nq{n_qubits}_gs{gs_sparsity:.2f}_ham{ham_sparsity:.2f}"
        f"_ov{overlap:.2f}_ks{keep_states:04d}_seed{seed}.png"
    )
    out_path = os.path.join(output_dir, fname)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"  Saved: {out_path}")
    print(f"  Total wall time: {time.time()-t0:.1f}s")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run one BARK vs SKQD comparison experiment and save a figure."
    )
    p.add_argument("--n_qubits",       type=int,   required=True,
                   help="Number of qubits (Hilbert-space dimension = 2^n_qubits).")
    p.add_argument("--gs_sparsity",    type=float, required=True,
                   help="Ground-state sparsity in [0,1]: fraction of basis states with nonzero amplitude.")
    p.add_argument("--ham_sparsity",   type=float, required=True,
                   help="Hamiltonian sparsity in [0,1]: fraction of possible off-diagonal pairs filled.")
    p.add_argument("--overlap",        type=float, required=True,
                   help="Target |<ψ|init>|² overlap in [0,1]; the closest available basis state is chosen.")
    p.add_argument("--keep_states",    type=int,   required=True,
                   help="Number of states BARK keeps per Hamiltonian application.")
    p.add_argument("--n_random_paths", type=int,   default=100,
                   help="Number of random orderings to sample as a baseline (default: 100).")
    p.add_argument("--seed",           type=int,   default=42,
                   help="Random seed for Hamiltonian generation and BARK (default: 42).")
    p.add_argument("--output_dir",     type=str,   default="results",
                   help="Directory where the figure PNG is saved (default: results/).")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        n_qubits=args.n_qubits,
        gs_sparsity=args.gs_sparsity,
        ham_sparsity=args.ham_sparsity,
        overlap=args.overlap,
        keep_states=args.keep_states,
        n_random_paths=args.n_random_paths,
        seed=args.seed,
        output_dir=args.output_dir,
    )
