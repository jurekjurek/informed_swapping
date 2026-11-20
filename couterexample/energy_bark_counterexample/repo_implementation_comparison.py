"""Run the repo's SKQD and BARK implementations on this sparse Hamiltonian."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import numpy as np
from scipy.linalg import eigh

REPO_ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from BARK import BARK
from SKQD import SKQD
from hamiltonian_10qubit import make_counterexample


def exact_ground_state(m: int, dimension: int, ids: dict[str, int]) -> tuple[float, np.ndarray]:
    """Ground vector in the invariant five-state target sector."""
    root_coupling = 2.0
    gateway_coupling = 1.8
    target_diagonal = -2.2
    target_block = np.array(
        [
            [root_coupling, np.sqrt(2.0) * gateway_coupling, 0.0, 0.0],
            [np.sqrt(2.0) * gateway_coupling, 0.0, -1.0, 0.0],
            [0.0, -1.0, 0.0, -1.0],
            [0.0, 0.0, -1.0, target_diagonal],
        ],
        dtype=float,
    )
    eigenvalues, eigenvectors = eigh(target_block)
    coefficients = eigenvectors[:, 0]
    state = np.zeros(dimension, dtype=float)
    state[ids["s"]] = coefficients[0] / np.sqrt(2.0)
    state[ids["a"]] = -coefficients[0] / np.sqrt(2.0)
    state[ids["y"]] = coefficients[1]
    state[ids["d"]] = coefficients[2]
    state[ids["c"]] = coefficients[3]
    return float(eigenvalues[0]), state


def write_curve(path: Path, pool_sizes: np.ndarray, fidelities: np.ndarray,
                initial_fidelity: float) -> None:
    rows = [(1, initial_fidelity)]
    rows.extend((int(size), float(fidelity))
                for size, fidelity in zip(pool_sizes, fidelities))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["D", "fidelity"])
        writer.writerows(rows)


def main() -> dict[str, object]:
    m = 9
    shots = 200
    seed = 13
    evolution_time = 1.5
    output = HERE / "results_m9"
    output.mkdir(parents=True, exist_ok=True)

    hamiltonian, ids = make_counterexample(m=m, U=20 * m**2, embed_qubits=True)
    ground_energy, ground_state = exact_ground_state(m, hamiltonian.shape[0], ids)
    initial_fidelity = float(abs(ground_state[ids["s"]]) ** 2)
    target_pool_size = (1 << m) + 4

    np.random.seed(seed)
    skqd = SKQD(hamiltonian)
    skqd_sizes, skqd_fidelities = skqd.full_skqd_run(
        initial_state_index=ids["s"],
        t=evolution_time,
        n_shots=shots,
        correct_state=ground_state,
        max_pool_size=target_pool_size,
        max_iterations=10_000,
    )

    energy_bark = BARK(hamiltonian)
    energy_sizes, energy_fidelities = energy_bark.full_bark_run(
        correct_state=ground_state,
        initial_state_index=ids["s"],
        max_pool_size=target_pool_size,
    )

    amplitude_bark = BARK(hamiltonian)
    amplitude_sizes, amplitude_fidelities = amplitude_bark.simplified_bark_run(
        correct_state=ground_state,
        initial_state_index=ids["s"],
        max_pool_size=target_pool_size,
    )

    write_curve(output / "skqd_fidelity.csv", skqd_sizes, skqd_fidelities,
                initial_fidelity)
    write_curve(output / "bark_fidelity.csv", energy_sizes, energy_fidelities,
                initial_fidelity)
    write_curve(output / "amplitude_bark_fidelity.csv", amplitude_sizes,
                amplitude_fidelities, initial_fidelity)

    def first_above_half(sizes: np.ndarray, fidelities: np.ndarray) -> int | None:
        found = np.flatnonzero(np.asarray(fidelities) > 0.5 + 1e-10)
        return int(sizes[found[0]]) if found.size else None

    result = {
        "hamiltonian_source": "hamiltonian_10qubit.make_counterexample",
        "algorithm_implementations": {
            "SKQD": "SKQD.full_skqd_run",
            "Energy-BARK": "BARK.full_bark_run",
            "Amplitude-BARK": "BARK.simplified_bark_run",
        },
        "m": m,
        "qubits": 10,
        "hamiltonian_dimension": int(hamiltonian.shape[0]),
        "connected_component_dimension": target_pool_size,
        "ground_energy": ground_energy,
        "initial_fidelity": initial_fidelity,
        "skqd_time": evolution_time,
        "skqd_shots_per_application": shots,
        "skqd_applications": int(skqd.iterations),
        "skqd_termination": skqd.termination,
        "final_subspace_dimensions": {
            "SKQD": int(skqd_sizes[-1]) if skqd_sizes.size else 1,
            "Energy-BARK": int(energy_sizes[-1]) if energy_sizes.size else 1,
            "Amplitude-BARK": int(amplitude_sizes[-1]) if amplitude_sizes.size else 1,
        },
        "first_dimension_with_fidelity_above_one_half": {
            "SKQD": first_above_half(skqd_sizes, skqd_fidelities),
            "Energy-BARK": first_above_half(energy_sizes, energy_fidelities),
            "Amplitude-BARK": first_above_half(amplitude_sizes, amplitude_fidelities),
        },
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n",
                                         encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
