"""Compare SKQD and two BARK selection rules on the 10-qubit counterexample.

Run from the repository root with the ``bark-counterexample`` environment:

    conda run -n bark-counterexample python couterexample/counterexample.py

The script writes fidelity and infidelity plots, ``fidelity.csv`` and
``summary.json`` next to this file. BARK's amplitude rule is the largest matrix element
``max_{y already expanded} |H_xy|`` on the current global frontier.  Ties are
resolved by the existing stable frontier order. Energy-BARK uses the package's
existing greedy two-dimensional energy-improvement score.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullLocator
import numpy as np
import pandas as pd
import scipy.sparse as sp
from qiskit.quantum_info import SparsePauliOp

# Running this file by path puts ``couterexample`` rather than the repository
# root on sys.path; add the root so the repo's top-level algorithms are importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from BARK import BARK
from SKQD import SKQD
from PaperFigures import (apply_style as apply_paperfigures_style,
                          draw_line as draw_paperfigures_line,
                          style as style_paperfigures_axis)


STUDY_ALGORITHMS = {
    "SKQD": ("#eb6834", "s", "--"),
    "Amplitude-BARK": ("#3aa66f", "^", "-."),
    "Energy-BARK": ("#2a78d6", "o", "-"),
}
STUDY_GRID_KWARGS = dict(color="#c9c8c3", linewidth=0.6, alpha=0.7)


def generate_compare(
    output_dir: str | Path | None = None,
    seed: int = 17,
    shots: int = 128,
    skqd_dt: float = 0.5,
    max_skqd_iterations: int = 12,
    max_bark_applications: int = 1023,
    target_fidelity: float = 0.995,
    interaction_penalty: float = 81.0,
) -> dict[str, object]:
    """Run repository SKQD, energy-BARK, and amplitude-BARK on the counterexample."""
    destination = Path(output_dir) if output_dir is not None else Path(__file__).resolve().parent
    destination.mkdir(parents=True, exist_ok=True)

    # Construct H = -.5 X_0 + sum_i (U/2 I - U/2 Z_i - X_i),
    # which is precisely -.5 X_0 + sum_i (U |1><1| - X_i).
    n_qubits = 10

    def pauli_on(qubit: int, letter: str) -> str:
        label = ["I"] * n_qubits
        label[n_qubits - 1 - qubit] = letter
        return "".join(label)

    terms: list[tuple[str, float]] = [(pauli_on(0, "X"), -0.5)]
    for qubit in range(1, n_qubits):
        terms.extend(
            [
                ("I" * n_qubits, interaction_penalty / 2.0),
                (pauli_on(qubit, "Z"), -interaction_penalty / 2.0),
                (pauli_on(qubit, "X"), -1.0),
            ]
        )
    hamiltonian_pauli = SparsePauliOp.from_list(terms).simplify(atol=1e-12)
    hamiltonian = sp.csr_matrix(hamiltonian_pauli.to_matrix(sparse=True).real)

    # The Hamiltonian is a sum of independent one-qubit terms, so build its
    # exact ground vector from the one-qubit eigenvectors without a full-space
    # diagonalization.
    local_energy = (
        interaction_penalty
        - math.sqrt(interaction_penalty**2 + 4.0)
    ) / 2.0
    local_excited_amplitude = -local_energy
    local_ground = np.array([1.0, local_excited_amplitude], dtype=np.complex128)
    local_ground /= np.linalg.norm(local_ground)
    important_ground = np.array([1.0, 1.0], dtype=np.complex128) / math.sqrt(2.0)
    exact_ground = np.array([1.0 + 0.0j])
    # Qiskit computational-basis indices place qubit 0 in the least
    # significant bit, so tensor factors are assembled from qubit 9 down to 0.
    for qubit in range(n_qubits - 1, -1, -1):
        factor = important_ground if qubit == 0 else local_ground
        exact_ground = np.kron(exact_ground, factor)
    exact_energy = -0.5 + (n_qubits - 1) * local_energy

    initial_fidelity = float(abs(exact_ground[0]) ** 2)
    all_rows: list[dict[str, object]] = []

    def append_curve(method: str, pool_sizes: np.ndarray,
                     fidelities: np.ndarray) -> None:
        all_rows.append({
            "method": method,
            "iteration": 0,
            "subspace_dim": 1,
            "fidelity": initial_fidelity,
        })
        all_rows.extend(
            {
                "method": method,
                "iteration": index,
                "subspace_dim": int(size),
                "fidelity": float(fidelity),
            }
            for index, (size, fidelity) in enumerate(zip(pool_sizes, fidelities), start=1)
        )

    # Use the repository's primary implementations directly. The Hamiltonian
    # is sparse, so SKQD uses its sparse time-evolution path, while BARK grows
    # the projected block incrementally.
    np.random.seed(seed)
    skqd = SKQD(hamiltonian)
    skqd_sizes, skqd_fidelities = skqd.full_skqd_run(
        initial_state_index=0,
        t=skqd_dt,
        n_shots=shots,
        correct_state=exact_ground,
        max_pool_size=2**n_qubits,
        max_iterations=max_skqd_iterations,
    )
    append_curve("SKQD", skqd_sizes, skqd_fidelities)

    energy_bark = BARK(hamiltonian)
    energy_sizes, energy_fidelities = energy_bark.full_bark_run(
        correct_state=exact_ground,
        initial_state_index=0,
        max_pool_size=min(2**n_qubits, max_bark_applications + 1),
    )
    append_curve("Energy-BARK", energy_sizes, energy_fidelities)

    amplitude_bark = BARK(hamiltonian)
    amplitude_sizes, amplitude_fidelities = amplitude_bark.simplified_bark_run(
        correct_state=exact_ground,
        initial_state_index=0,
        max_pool_size=min(2**n_qubits, max_bark_applications + 1),
    )
    append_curve("Amplitude-BARK", amplitude_sizes, amplitude_fidelities)

    def first_flip_dimension(pool_sizes: np.ndarray, method: str) -> int | None:
        # This Hamiltonian's qubit-0=0 sector captures at most half of the
        # exact ground-state weight. The first recorded point over 1/2 must
        # therefore contain at least one q0=1 basis state.
        rows = [row for row in all_rows if row["method"] == method]
        return next(
            (int(row["subspace_dim"]) for row in rows if row["fidelity"] > 0.5 + 1e-10),
            None,
        )

    amplitude_flip_dim = first_flip_dimension(amplitude_sizes, "Amplitude-BARK")
    energy_flip_dim = first_flip_dimension(energy_sizes, "Energy-BARK")
    skqd_flip_dim = first_flip_dimension(skqd_sizes, "SKQD")

    csv_path = destination / "fidelity.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["method", "iteration", "subspace_dim", "fidelity"],
        )
        writer.writeheader()
        writer.writerows(all_rows)

    plot_path = destination / "comparison.png"
    fig, axis = plt.subplots(figsize=(9.2, 5.6), constrained_layout=True)
    colors = {"SKQD": "#0072B2", "Amplitude-BARK": "#D55E00", "Energy-BARK": "#009E73"}
    for method in ("SKQD", "Amplitude-BARK", "Energy-BARK"):
        rows = [row for row in all_rows if row["method"] == method]
        axis.plot(
            [row["subspace_dim"] for row in rows],
            [row["fidelity"] for row in rows],
            marker="o" if len(rows) < 30 else None,
            markersize=3,
            linewidth=2,
            label=method,
            color=colors[method],
        )
    axis.axvline(512, color="#777777", linestyle="--", linewidth=1)
    axis.annotate("512 states with qubit 0 = 0", xy=(512, 0.5), xytext=(24, 0.35),
                  arrowprops={"arrowstyle": "->", "color": "#555555"}, fontsize=9)
    axis.set_xscale("log", base=2)
    axis.set(
        xlabel="Selected subspace dimension",
        ylabel="Ground-state fidelity",
        title="",
        xlim=(1, 1024),
        ylim=(0, 1.02),
    )
    axis.grid(True, alpha=0.25)
    axis.legend(frameon=False)
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    infidelity_plot_path = destination / "infidelity_vs_bitstrings.png"
    apply_paperfigures_style()
    fig, axis = plt.subplots(figsize=(7.05, 2.3), layout="constrained")
    for method in ("Amplitude-BARK", "SKQD"):
        rows = [row for row in all_rows if row["method"] == method]
        infidelities = np.maximum(
            1.0 - np.asarray([row["fidelity"] for row in rows], dtype=float),
            1e-16,
        )
        color, marker, linestyle = STUDY_ALGORITHMS[method]
        plot_rows = pd.DataFrame({
            "Budget_Fraction": [
                row["subspace_dim"] / (2**n_qubits) for row in rows
            ]
        })
        if method == "SKQD":
            axis.plot(
                plot_rows["Budget_Fraction"], infidelities,
                color=color, linestyle=linestyle, linewidth=3.0,
                marker=marker, markersize=5, markevery=1,
                markerfacecolor="none", markeredgecolor=color,
                markeredgewidth=1.2, label=method, zorder=4,
            )
        else:
            draw_paperfigures_line(
                axis, plot_rows, method, color, marker, linestyle, infidelities
            )
    style_paperfigures_axis(axis)
    axis.set_yscale("log")
    axis.yaxis.set_major_locator(LogLocator(numticks=5))
    axis.yaxis.set_minor_locator(NullLocator())
    axis.set(
        xlabel=r"Bitstrings $K/2^N$",
        ylabel=r"Infidelity $1-F$",
        xlim=(1 / (2**n_qubits) / 1.3, 0.8),
        ylim=(1e-10, 1.05),
    )
    axis.legend(frameon=False)
    fig.savefig(infidelity_plot_path, bbox_inches="tight", dpi=300)
    plt.close(fig)

    def first_dimension(rows: list[dict[str, object]]) -> int | None:
        return next(
            (int(row["subspace_dim"]) for row in rows if row["fidelity"] >= target_fidelity),
            None,
        )

    skqd_target_dimension = first_dimension(
        [row for row in all_rows if row["method"] == "SKQD"]
    )
    amplitude_target_dimension = first_dimension(
        [row for row in all_rows if row["method"] == "Amplitude-BARK"]
    )
    energy_target_dimension = first_dimension(
        [row for row in all_rows if row["method"] == "Energy-BARK"]
    )
    summary = {
        "hamiltonian": (
            f"-0.5 X_0 + sum_{{i=1}}^9 "
            f"({interaction_penalty:g} |1><1|_i - X_i)"
        ),
        "qubits": n_qubits,
        "initial_state": "|0>^10",
        "exact_ground_energy": exact_energy,
        "target_fidelity": target_fidelity,
        "skqd_shots_per_iteration": shots,
        "skqd_dt": skqd_dt,
        "skqd_unitary_applications": int(skqd.iterations),
        "skqd_termination": skqd.termination,
        "skqd_pool_growth_steps": int(len(skqd_sizes)),
        "skqd_final_pool_size": int(skqd.projection.size),
        "first_dimension_at_target": {
            "SKQD": skqd_target_dimension,
            "Amplitude-BARK": amplitude_target_dimension,
            "Energy-BARK": energy_target_dimension,
        },
        "first_dimension_with_qubit_0_flipped": {
            # The primary runners expose the projected subspace size rather
            # than every selected index. Fidelity above 1/2 proves q0=1 is
            # present because the q0=0 sector contains at most half the ground
            # state weight.
            "SKQD": skqd_flip_dim,
            "Amplitude-BARK": amplitude_flip_dim,
            "Energy-BARK": energy_flip_dim,
        },
        "data_csv": str(csv_path),
        "plot": str(plot_path),
        "infidelity_plot": str(infidelity_plot_path),
    }
    summary_path = destination / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return {
        "plot": plot_path,
        "infidelity_plot": infidelity_plot_path,
        "data": csv_path,
        "summary": summary_path,
        "results": summary,
    }


if __name__ == "__main__":
    results = generate_compare()
    print(json.dumps(results["results"], indent=2))
