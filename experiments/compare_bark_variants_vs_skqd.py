"""Compare BARK variants against SKQD on random spin Hamiltonians.

This script uses the new BARK implementation from ``bark_best_first_baab.py``
exclusively. It compares BARK settings formed from:

1. baseline: pool strategy, amplitude score
2. best_first_only: best-first strategy, amplitude score
3. two_dimensional_only: pool strategy, two-dimensional score
4. best_first_two_dimensional: best-first strategy, two-dimensional score
5. perturbative_only: pool strategy, perturbative score
6. best_first_perturbative: best-first strategy, perturbative score

For each Hamiltonian setting, it plots every requested SKQD timestep, a random
ordering baseline, the selected BARK curve, and the exact ground energy.

Run, for example:

    python experiments/compare_bark_variants_vs_skqd.py

or:

    python experiments/compare_bark_variants_vs_skqd.py --num-sites 6 \
        --complexities low medium high --bark-settings all \
        --skqd-timesteps 0.001 0.01 0.1 0.3 0.5 1.0
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from qiskit.quantum_info import SparsePauliOp
from scipy.sparse.linalg import eigsh

from subspace_search.algorithms import BarkBestFirstPerturbative
from subspace_search.hamiltonians import make_random_spin_hamiltonian
from subspace_search.paths import get_all_paths, get_one_path
from subspace_search.skqd import do_skqd


BARK_SETTING_KWARGS: dict[str, dict[str, Any]] = {
    "baseline": {
        "score_mode": "amplitude",
    },
    "best_first_only": {
        "selection_strategy": "best_first",
        "score_mode": "amplitude",
    },
    "two_dimensional_only": {
        "score_mode": "two_dimensional",
    },
    "best_first_two_dimensional": {
        "selection_strategy": "best_first",
        "score_mode": "two_dimensional",
    },
    "perturbative_only": {
        "score_mode": "perturbative",
    },
    "best_first_perturbative": {
        "selection_strategy": "best_first",
        "score_mode": "perturbative",
    },
}


COMPLEXITY_PRESETS: dict[str, dict[str, Any]] = {
    "low": {
        "max_interactions": 1,
        "J_max": 0.5,
        "B_max": 0.2,
        "J_components": ("z",),
        "B_components": ("x", "z"),
        "coupling_distribution": "uniform",
        "field_distribution": "uniform",
    },
    "medium": {
        "max_interactions": 2,
        "J_max": 1.0,
        "B_max": 0.5,
        "J_components": ("x", "z"),
        "B_components": ("x", "z"),
        "coupling_distribution": "uniform",
        "field_distribution": "uniform",
    },
    "high": {
        "max_interactions": None,
        "J_max": 1.0,
        "B_max": 1.0,
        "J_components": ("x", "y", "z"),
        "B_components": ("x", "y", "z"),
        "coupling_distribution": "normal",
        "field_distribution": "normal",
    },
}


@dataclass
class HamiltonianCase:
    label: str
    H_spo: SparsePauliOp
    H: sp.csr_matrix
    info: dict[str, Any]
    true_energy: float
    ground_state: np.ndarray
    initial_index: int
    initial_bitstring: str
    ground_state_density: float
    hamiltonian_density: float


@dataclass
class CurveResult:
    label: str
    path: np.ndarray
    runtime_seconds: float
    convergence_dim: int | None
    final_subspace_dim: int
    final_energy: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare four BARK variants against multi-timestep SKQD.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--num-sites", type=int, default=6)
    parser.add_argument(
        "--complexities",
        nargs="+",
        default=["low", "medium", "high"],
        choices=sorted(COMPLEXITY_PRESETS),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument(
        "--bark-settings",
        nargs="+",
        default=["all"],
        choices=["all", *BARK_SETTING_KWARGS],
    )
    parser.add_argument(
        "--skqd-timesteps",
        nargs="+",
        type=float,
        default=[0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    )
    parser.add_argument("--skqd-num-steps", type=int, default=8)
    parser.add_argument("--random-paths", type=int, default=8)
    parser.add_argument("--keep-states", type=int, default=4)
    parser.add_argument("--max-applications", type=int, default=12)
    parser.add_argument("--bark-mode", choices=["top_m", "importance_sample"], default="top_m")
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--perturbative-epsilon", type=float, default=1e-12)
    parser.add_argument("--ground-update-interval", type=int, default=1)
    parser.add_argument("--no-aggregate-frontier-amplitudes", action="store_true")
    parser.add_argument("--restrict-equal-ones-zeros", action="store_true")
    parser.add_argument("--convergence-tol", type=float, default=1e-6)
    parser.add_argument(
        "--plot-bark-settings-together",
        action="store_true",
        help="Also plot all selected BARK settings against each other for each Hamiltonian.",
    )
    parser.add_argument("--output-dir", default="experiments/bark_variant_results")
    return parser.parse_args()


def selected_bark_settings(names: list[str]) -> list[str]:
    if "all" in names:
        return list(BARK_SETTING_KWARGS)
    return names


def validate_args(args: argparse.Namespace) -> None:
    if args.num_sites > 8:
        raise ValueError("--num-sites must be <= 8 for this experiment.")
    if args.num_sites <= 0:
        raise ValueError("--num-sites must be positive.")
    if args.skqd_num_steps <= 0:
        raise ValueError("--skqd-num-steps must be positive.")
    if args.convergence_tol <= 0:
        raise ValueError("--convergence-tol must be positive.")


def compute_ground_state(H: sp.csr_matrix) -> tuple[float, np.ndarray]:
    dim = H.shape[0]
    if dim <= 2:
        evals, evecs = np.linalg.eigh(H.toarray())
        return float(evals[0]), evecs[:, 0]
    evals, evecs = eigsh(H, k=1, which="SA")
    return float(evals[0]), evecs[:, 0]


def convergence_dimension(
    path: np.ndarray,
    true_energy: float,
    tolerance: float,
) -> int | None:
    gaps = path - true_energy
    hits = np.where(gaps <= tolerance)[0]
    if len(hits) == 0:
        return None
    return int(hits[0] + 1)


def path_from_order(H: sp.csr_matrix, order: list[int]) -> np.ndarray:
    if not order:
        return np.array([], dtype=float)
    return get_one_path(H, order)


def make_case(
    num_sites: int,
    complexity: str,
    seed: int,
) -> HamiltonianCase:
    params = COMPLEXITY_PRESETS[complexity]
    H_spo, info = make_random_spin_hamiltonian(
        num_sites=num_sites,
        seed=seed,
        **params,
    )
    H = sp.csr_matrix(H_spo.to_matrix(sparse=True))
    true_energy, ground_state = compute_ground_state(H)

    probabilities = np.abs(ground_state) ** 2
    initial_index = int(np.argmax(probabilities))
    initial_bitstring = format(initial_index, f"0{num_sites}b")
    dim = H.shape[0]
    ground_state_density = float(np.count_nonzero(probabilities > 1e-12) / dim)
    hamiltonian_density = float(H.nnz / (dim * dim))

    label = f"{complexity}_seed{seed}_n{num_sites}"
    return HamiltonianCase(
        label=label,
        H_spo=H_spo,
        H=H,
        info=info,
        true_energy=true_energy,
        ground_state=ground_state,
        initial_index=initial_index,
        initial_bitstring=initial_bitstring,
        ground_state_density=ground_state_density,
        hamiltonian_density=hamiltonian_density,
    )


def run_skqd_sweep(
    case: HamiltonianCase,
    timesteps: list[float],
    num_steps: int,
    tolerance: float,
) -> dict[float, CurveResult]:
    results = {}
    for t in timesteps:
        start = time.perf_counter()
        order = do_skqd(case.H, num_steps=num_steps, t=t, initial=case.initial_index)
        order = [int(i) for i in order if i >= 0]
        path = path_from_order(case.H, order)
        runtime = time.perf_counter() - start
        results[t] = CurveResult(
            label=f"SKQD t={t:g}",
            path=path,
            runtime_seconds=runtime,
            convergence_dim=convergence_dimension(path, case.true_energy, tolerance),
            final_subspace_dim=len(path),
            final_energy=float(path[-1]),
        )
    return results


def run_bark(
    case: HamiltonianCase,
    setting_name: str,
    args: argparse.Namespace,
) -> CurveResult:
    setting_kwargs = dict(BARK_SETTING_KWARGS[setting_name])
    start = time.perf_counter()
    bark = BarkBestFirstPerturbative(
        H=case.H_spo,
        initial_state=case.initial_bitstring,
        keep_states=args.keep_states,
        max_applications=args.max_applications,
        mode=args.bark_mode,
        sample_size=args.sample_size,
        restrict_equal_ones_zeros=args.restrict_equal_ones_zeros,
        return_only_applied_bitstrings=True,
        perturbative_epsilon=args.perturbative_epsilon,
        ground_update_interval=args.ground_update_interval,
        aggregate_frontier_amplitudes=not args.no_aggregate_frontier_amplitudes,
        random_seed=0,
        **setting_kwargs,
    )
    bitstrings = bark.run()
    order = [int(bitstring, 2) for bitstring in bitstrings]
    path = path_from_order(case.H, order)
    runtime = time.perf_counter() - start
    return CurveResult(
        label=f"BARK {setting_name}",
        path=path,
        runtime_seconds=runtime,
        convergence_dim=convergence_dimension(path, case.true_energy, args.convergence_tol),
        final_subspace_dim=len(path),
        final_energy=float(path[-1]),
    )


def best_skqd_result(skqd_results: dict[float, CurveResult]) -> tuple[float, CurveResult]:
    def key(item: tuple[float, CurveResult]) -> tuple[int, int, float]:
        _, result = item
        converged = result.convergence_dim is not None
        convergence_rank = result.convergence_dim if converged else math.inf
        return (0 if converged else 1, int(convergence_rank), result.final_energy)

    return min(skqd_results.items(), key=key)


def plot_setting_comparison(
    case: HamiltonianCase,
    bark_setting: str,
    bark_result: CurveResult,
    skqd_results: dict[float, CurveResult],
    random_paths: np.ndarray,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))

    x_random = np.arange(1, random_paths.shape[1] + 1)
    for i, random_path in enumerate(random_paths):
        ax.plot(
            x_random,
            random_path,
            color="gray",
            alpha=0.18,
            linewidth=0.8,
            label="random paths" if i == 0 else None,
        )
    ax.plot(x_random, random_paths.mean(axis=0), color="black", linestyle=":", label="random mean")

    for t, result in skqd_results.items():
        x = np.arange(1, len(result.path) + 1)
        ax.plot(x, result.path, linewidth=1.2, label=f"SKQD t={t:g}")

    x_bark = np.arange(1, len(bark_result.path) + 1)
    ax.plot(x_bark, bark_result.path, "o-", linewidth=2.0, label=bark_result.label)
    ax.axhline(case.true_energy, color="red", linestyle="--", linewidth=1.2, label="exact ground energy")

    ax.set_xlabel("projected subspace dimension")
    ax.set_ylabel("lowest projected energy")
    ax.set_title(
        f"{case.label}: {bark_setting}\n"
        f"gs density={case.ground_state_density:.3f}, H density={case.hamiltonian_density:.3f}"
    )
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def plot_bark_settings_together(
    case: HamiltonianCase,
    bark_results: dict[str, CurveResult],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for setting, result in bark_results.items():
        x = np.arange(1, len(result.path) + 1)
        ax.plot(x, result.path, "o-", label=setting)
    ax.axhline(case.true_energy, color="red", linestyle="--", linewidth=1.2, label="exact ground energy")
    ax.set_xlabel("projected subspace dimension")
    ax.set_ylabel("lowest projected energy")
    ax.set_title(f"{case.label}: BARK settings comparison")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def write_summary(rows: list[dict[str, Any]], output_path: Path) -> None:
    if not rows:
        return
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    validate_args(args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bark_settings = selected_bark_settings(args.bark_settings)
    summary_rows: list[dict[str, Any]] = []

    for complexity in args.complexities:
        for seed in args.seeds:
            case = make_case(args.num_sites, complexity, seed)
            print(
                f"\n{case.label}: E0={case.true_energy:.8f}, "
                f"initial={case.initial_bitstring}, "
                f"ground_state_density={case.ground_state_density:.4f}, "
                f"hamiltonian_density={case.hamiltonian_density:.4f}"
            )

            skqd_results = run_skqd_sweep(
                case,
                timesteps=args.skqd_timesteps,
                num_steps=args.skqd_num_steps,
                tolerance=args.convergence_tol,
            )
            best_t, best_skqd = best_skqd_result(skqd_results)
            print(
                f"  best SKQD: t={best_t:g}, convergence_dim={best_skqd.convergence_dim}, "
                f"final_dim={best_skqd.final_subspace_dim}, runtime={best_skqd.runtime_seconds:.3f}s"
            )

            random_paths = get_all_paths(case.H, number_of_paths=args.random_paths, start=case.initial_index)
            bark_results: dict[str, CurveResult] = {}

            for setting in bark_settings:
                bark_result = run_bark(case, setting, args)
                bark_results[setting] = bark_result
                print(
                    f"  {setting}: convergence_dim={bark_result.convergence_dim}, "
                    f"final_dim={bark_result.final_subspace_dim}, runtime={bark_result.runtime_seconds:.3f}s"
                )

                plot_path = output_dir / f"{case.label}__{setting}__vs_skqd.png"
                plot_setting_comparison(
                    case=case,
                    bark_setting=setting,
                    bark_result=bark_result,
                    skqd_results=skqd_results,
                    random_paths=random_paths,
                    output_path=plot_path,
                )

                summary_rows.append(
                    {
                        "hamiltonian": case.label,
                        "num_sites": args.num_sites,
                        "complexity": complexity,
                        "seed": seed,
                        "ground_energy": case.true_energy,
                        "ground_state_density": case.ground_state_density,
                        "hamiltonian_density": case.hamiltonian_density,
                        "initial_index": case.initial_index,
                        "initial_bitstring": case.initial_bitstring,
                        "bark_setting": setting,
                        "bark_convergence_dim": bark_result.convergence_dim,
                        "bark_final_subspace_dim": bark_result.final_subspace_dim,
                        "bark_final_energy": bark_result.final_energy,
                        "bark_runtime_seconds": bark_result.runtime_seconds,
                        "best_skqd_timestep": best_t,
                        "best_skqd_convergence_dim": best_skqd.convergence_dim,
                        "best_skqd_final_subspace_dim": best_skqd.final_subspace_dim,
                        "best_skqd_final_energy": best_skqd.final_energy,
                        "best_skqd_runtime_seconds": best_skqd.runtime_seconds,
                        "convergence_tolerance": args.convergence_tol,
                    }
                )

            if args.plot_bark_settings_together and len(bark_results) > 1:
                plot_bark_settings_together(
                    case,
                    bark_results,
                    output_dir / f"{case.label}__bark_settings_together.png",
                )

    summary_path = output_dir / "summary.csv"
    write_summary(summary_rows, summary_path)
    print(f"\nWrote plots and summary to {output_dir}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
