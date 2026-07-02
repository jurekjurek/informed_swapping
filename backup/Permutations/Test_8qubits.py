import matplotlib
matplotlib.use("Agg")

from MakeHam import *
from Helpers import *
import matplotlib.pyplot as plt
from math import factorial
from scipy.sparse.linalg import eigsh
from SKQD import *


def main():
    # Number of qubits and Hamiltonian parameters
    n_qubits = 8
    density = 0.5
    t_values = [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    initial_index = 2 ** (n_qubits - 1)  # analogous to 4 for 4 qubits (highest basis state)

    # Build random Hermitian Hamiltonian
    H = make_hermitian_sparse_random_from_qubits(n_qubits, density)
    print(f"Hamiltonian shape: {H.shape}")

    # Exact ground state energy (for reference)
    correct_energy, correct_state = eigsh(H, k=1, which="SA")
    print(f"Ground state energy (exact): {correct_energy[0]}")

    # Run SKQD for different t values
    skqd_lists = {}
    for t in t_values:
        print(f"Running SKQD for t={t}...")
        skqd_lists[t] = do_skqd(H, num_steps=H.shape[0], t=t, initial=initial_index)

    # Convert SKQD index sequences to energy paths
    skqd_paths = {}
    for t, indices in skqd_lists.items():
        print(f"Computing energy path for t={t}...")
        skqd_paths[t] = get_one_path(H, indices)

    # Sample many random paths for comparison
    print("Sampling random paths...")
    Paths = get_all_paths(H, number_of_paths=10000, start=initial_index)
    print(f"Paths array shape: {Paths.shape}")

    # Plot
    plt.figure(figsize=(10, 6))

    # Plot many random paths with low alpha
    for path in Paths:
        plt.plot(path, color="gray", alpha=0.02)

    # Overlay SKQD paths with clearer colors
    color_map = {
        0.001: "magenta",
        0.01: "cyan",
        0.1: "black",
        0.2: "blue",
        0.3: "green",
        0.4: "red",
        0.5: "purple",
        0.6: "orange",
    }

    for t, path in skqd_paths.items():
        plt.plot(path, label=f"t={t}", color=color_map.get(t, None))

    plt.xlabel("Step")
    plt.ylabel("Ground state energy of projected H")
    plt.legend()
    plt.tight_layout()

    output_png = "skqd_paths_8qubits.pdf"
    plt.savefig(output_png, dpi=300)
    print(f"Saved plot to {output_png}")


if __name__ == "__main__":
    main()
