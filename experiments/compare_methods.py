"""End-to-end example: KRAB vs SKQD vs a random baseline on one Hamiltonian.

Builds a controlled-sparsity Hamiltonian, gets each method's ordering of basis
states, turns those orderings into energy-vs-#states convergence paths, and
plots them on one axis. Copy this as a starting point for new experiments.

Run:
    python compare_methods.py
(after `pip install -e ../subspace_search` inside the .SKQD venv)
"""

import os

import matplotlib
matplotlib.use("Agg")   # headless-safe; drop this line to view interactively
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix

from subspace_search.hamiltonians import (
    diagnostics,
    make_controlled_sparse_ground_state_hamiltonian_fast,
    make_planted_hamiltonian,
)
from subspace_search.skqd import do_skqd
from subspace_search.algorithms import selected_krylov_ground_state
from subspace_search.paths import get_one_path, get_all_paths
from subspace_search.plotting import plot_convergence_paths


def main() -> None:
    n_qubits = 6
    seed = 42

    # 1. Hamiltonian with a controlled-sparsity ground state.
    H, _, psi, support, info = make_controlled_sparse_ground_state_hamiltonian_fast(
        n_qubits=n_qubits,
        ground_state_sparsity=0.1,
        hamiltonian_sparsity=0.3,
        seed=seed,
        ground_energy=-5.0,
        gap=1.0,
        make_pauli_op=False,
        max_amplitude=0.3,      # dominant basis-state probability ~ initial overlap
    )
    true_energy = float(np.linalg.eigvalsh(H.toarray())[0])
    initial = int(np.argmax(np.abs(psi) ** 2))
    print(f"dim={H.shape[0]}, support={info['ground_state_support_size']}, "
          f"E0={true_energy:.4f}, initial index={initial}")

    # Optional example: the newer planted Pauli Hamiltonian has the form
    # H = -Delta |g><g| + lambda R. It returns a SparsePauliOp, so convert it
    # to CSR before passing it to the same SKQD / path utilities used below.
    H_pauli, planted_info = make_planted_hamiltonian(
        num_qubits=n_qubits, ground_support_size=6, pauli_density=0.1,
        Delta=10.0, lam=0.05, seed=seed,
    )
    H_planted = csr_matrix(H_pauli.to_matrix(sparse=True))
    planted_diag = diagnostics(H_pauli, planted_info["planted_ground_state"])
    print(f"planted demo: dim={H_planted.shape[0]}, "
          f"E0={planted_diag['E0']:.4f}, "
          f"initial={planted_info['suggested_initial_bitstring']}")

    paths = {}

    # 2. SKQD reference (sweep a couple of time steps, keep the best).
    best_t, best_path = None, None
    for t in (0.1, 0.5, 1.0):
        order = do_skqd(H, num_steps=8, t=t, initial=initial)
        path = get_one_path(H, [int(i) for i in order if i >= 0])
        if best_path is None or path[-1] < best_path[-1]:
            best_t, best_path = t, path
    paths[f"SKQD (t={best_t})"] = best_path

    # 3. KRAB.
    krab = selected_krylov_ground_state(
        H=H, initial_bitstring=initial, Q=8, delta=1e-8, epsilon=1e-6,
        n_iterations=30,
    )
    krab_order = list(krab.final_indices)
    if initial in krab_order:
        krab_order.remove(initial)
    krab_order.insert(0, initial)
    paths["KRAB"] = get_one_path(H, krab_order)

    # 4. Random baseline (a bundle of orderings).
    paths["Random"] = get_all_paths(H, number_of_paths=20, start=initial)

    # 5. Plot.
    ax = plot_convergence_paths(paths, true_energy=true_energy)
    out = os.path.join(os.path.dirname(__file__), "compare_methods.png")
    ax.get_figure().savefig(out, dpi=130, bbox_inches="tight")
    print(f"KRAB final energy = {krab.final_energy:.6f}")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
