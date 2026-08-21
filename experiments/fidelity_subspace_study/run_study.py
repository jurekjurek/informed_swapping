"""Systematic BARK vs SKQD scan: subspace dimension needed to reach a fidelity.

For every sampled random spin Hamiltonian, every scanned initial overlap and
every target fidelity, this script answers

    what is the smallest *normalised* subspace dimension (dim / 2**n) at which
    the method reaches the target fidelity, and what parameters achieve it?

SKQD is optimised over ``(dt, shots)``, BARK over
``(score_mode, selection_strategy, keep_states)``. Both methods start from the
same computational basis state, chosen to realise the requested initial overlap
with the exact ground space.

Every iteration of every run is logged with a full wall-clock breakdown
(propagation, sampling, scoring, expansion, projection, diagonalisation,
fidelity evaluation) and with the running subspace statistics, so the resulting
tables support runtime analyses that were never planned in advance.

Usage
-----
    python experiments/fidelity_subspace_study/run_study.py --preset smoke
    python experiments/fidelity_subspace_study/run_study.py --preset default \
        --output-dir experiments/fidelity_subspace_study/results
    python experiments/fidelity_subspace_study/run_study.py --preset full --dry-run

Long scans can be split across processes/machines with ``--chunk``/``--num-chunks``
and re-run safely with ``--resume`` (per-Hamiltonian parquet shards).
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import zlib
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from tqdm import tqdm

from bark_runner import BarkConfig, run_bark
from hamiltonian_cases import HamiltonianCase, InitialState, build_case
from optimization import method_comparison, optimal_configurations
from skqd_runner import SkqdConfig, make_propagator_cache, run_skqd
from study_config import PRESETS, StudyConfig, build_config


SHARD_TABLES = ("hamiltonians", "runs", "iterations", "convergence")


# ----------------------------------------------------------------------
def stable_seed(*parts: Any) -> int:
    """Deterministic 32-bit seed from arbitrary run identifiers."""
    payload = "|".join(str(p) for p in parts).encode("utf-8")
    return int(zlib.crc32(payload))


def case_context(case: HamiltonianCase, initial: InitialState) -> dict[str, Any]:
    """Columns identifying the (Hamiltonian, initial state) a run belongs to."""
    return {
        "ham_id": case.ham_id,
        "num_qubits": case.num_qubits,
        "hilbert_dim": case.dim,
        "max_interactions": (
            -1 if case.max_interactions is None else int(case.max_interactions)
        ),
        "max_interactions_label": (
            "all-to-all" if case.max_interactions is None else str(case.max_interactions)
        ),
        "seed": case.seed,
        "ground_energy": case.ground_energy,
        "gap": case.gap,
        "degeneracy": case.degeneracy,
        "gs_participation_fraction": case.stats["gs_participation_fraction"],
        "gs_support_fraction": case.stats["gs_support_fraction"],
        "gs_entropy_normalized": case.stats["gs_entropy_normalized"],
        "ham_density": case.stats["ham_density"],
        "ham_num_pauli_terms": case.stats["ham_num_pauli_terms"],
        "initial_spec": initial.spec,
        "initial_index": initial.index,
        "initial_bitstring": initial.bitstring,
        "initial_overlap": initial.overlap,
        "initial_rank": initial.rank,
    }


# ----------------------------------------------------------------------
def summarise_run(
    records: list[dict[str, Any]],
    context: dict[str, Any],
    run_id: str,
    config_label: str,
    params: dict[str, Any],
    target_fidelities: Iterable[float],
) -> tuple[pd.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    """Turn one run's per-iteration records into the three table fragments."""
    frame = pd.DataFrame(records)
    method = str(records[0]["method"])
    frame.insert(0, "run_id", run_id)
    for key, value in reversed(list({**context, **params}.items())):
        frame.insert(1, key, value)
    frame["config_label"] = config_label

    frame["cum_t_algorithm"] = frame["t_algorithm"].cumsum()
    frame["cum_t_solve"] = frame["t_solve"].cumsum()
    frame["cum_t_fidelity"] = frame["t_fidelity"].cumsum()
    frame["cum_t_total"] = frame["cum_t_algorithm"] + frame["cum_t_solve"]
    frame["best_fidelity_so_far"] = frame["fidelity"].cummax()

    final = frame.iloc[-1]
    run_row: dict[str, Any] = {
        "run_id": run_id,
        "method": method,
        **context,
        **params,
        "config_label": config_label,
        "num_iterations": int(final["iteration"]),
        "num_diagonalizations": int(frame["diagonalized"].sum()),
        "final_subspace_dim": int(final["subspace_dim"]),
        "final_subspace_fraction": float(final["subspace_fraction"]),
        "final_fidelity": float(final["fidelity"]),
        "best_fidelity": float(frame["fidelity"].max()),
        "final_energy_error": float(final["energy_error"]),
        "final_captured_weight": float(final["captured_weight"]),
        "total_t_algorithm": float(frame["t_algorithm"].sum()),
        "total_t_solve": float(frame["t_solve"].sum()),
        "total_t_fidelity": float(frame["t_fidelity"].sum()),
        "total_t_propagate": float(frame["t_propagate"].sum()),
        "total_t_sample": float(frame["t_sample"].sum()),
        "total_t_subspace_update": float(frame["t_subspace_update"].sum()),
        "total_t_select": float(frame["t_select"].sum()),
        "total_t_expand": float(frame["t_expand"].sum()),
        "total_t_ground_update": float(frame["t_ground_update"].sum()),
        "total_t_project": float(frame["t_project"].sum()),
        "total_t_diagonalize": float(frame["t_diagonalize"].sum()),
        "total_runtime": float(frame["t_iteration"].sum()),
    }

    convergence_rows: list[dict[str, Any]] = []
    for target in target_fidelities:
        hits = frame.index[frame["fidelity"] >= target]
        reached = len(hits) > 0
        row: dict[str, Any] = {
            "run_id": run_id,
            "method": method,
            **context,
            **params,
            "config_label": config_label,
            "target_fidelity": float(target),
            "reached": bool(reached),
            "final_fidelity": run_row["final_fidelity"],
            "final_subspace_fraction": run_row["final_subspace_fraction"],
        }
        if reached:
            hit = frame.loc[hits[0]]
            row.update(
                {
                    "iteration_at_target": int(hit["iteration"]),
                    "subspace_dim_at_target": int(hit["subspace_dim"]),
                    "subspace_fraction_at_target": float(hit["subspace_fraction"]),
                    "shots_at_target": int(hit["shots_cumulative"]),
                    "fidelity_at_target": float(hit["fidelity"]),
                    "energy_error_at_target": float(hit["energy_error"]),
                    "t_algorithm_at_target": float(hit["cum_t_algorithm"]),
                    "t_solve_at_target": float(hit["cum_t_solve"]),
                    "t_total_at_target": float(hit["cum_t_total"]),
                }
            )
        else:
            row.update(
                {
                    "iteration_at_target": np.nan,
                    "subspace_dim_at_target": np.nan,
                    "subspace_fraction_at_target": np.nan,
                    "shots_at_target": np.nan,
                    "fidelity_at_target": np.nan,
                    "energy_error_at_target": np.nan,
                    "t_algorithm_at_target": np.nan,
                    "t_solve_at_target": np.nan,
                    "t_total_at_target": np.nan,
                }
            )
        convergence_rows.append(row)

    return frame, run_row, convergence_rows


# ----------------------------------------------------------------------
def run_case(
    case: HamiltonianCase,
    config: StudyConfig,
    progress: bool = True,
) -> dict[str, pd.DataFrame]:
    """Run the complete SKQD and BARK grids for one Hamiltonian."""
    iteration_frames: list[pd.DataFrame] = []
    run_rows: list[dict[str, Any]] = []
    convergence_rows: list[dict[str, Any]] = []

    skqd_grid = config.skqd_grid()
    bark_grid = config.bark_grid()
    total = len(config.initial_overlaps) * (len(skqd_grid) + len(bark_grid))
    bar = tqdm(total=total, desc=case.ham_id, leave=False, disable=not progress)

    for spec in config.initial_overlaps:
        initial = case.initial_state(spec)
        context = case_context(case, initial)
        propagator_cache = make_propagator_cache()

        for dt, shots, repeat in skqd_grid:
            run_config = SkqdConfig(
                dt=dt,
                shots=shots,
                max_iterations=config.skqd_max_iterations,
                repeat=repeat,
                seed=stable_seed(case.ham_id, spec, "skqd", dt, shots, repeat),
            )
            records = run_skqd(
                case=case,
                initial=initial,
                config=run_config,
                stop_fidelity=config.stop_fidelity,
                max_subspace_fraction=config.max_subspace_fraction,
                dense_propagator_max_dim=config.skqd_dense_propagator_max_dim,
                propagator_cache=propagator_cache,
            )
            label = f"dt={dt:g},shots={shots}"
            run_id = f"{case.ham_id}|{spec}|SKQD|{label}|r{repeat}"
            frame, run_row, rows = summarise_run(
                records, context, run_id, label, run_config.as_dict(),
                config.target_fidelities,
            )
            iteration_frames.append(frame)
            run_rows.append(run_row)
            convergence_rows.extend(rows)
            bar.update(1)

        for score_mode, strategy, keep_states, repeat in bark_grid:
            run_config = BarkConfig(
                score_mode=score_mode,
                selection_strategy=strategy,
                keep_states=keep_states,
                max_iterations=config.bark_max_iterations,
                mode=config.bark_mode,
                ground_update_interval=config.bark_ground_update_interval,
                subspace=config.bark_subspace,
                restrict_equal_ones_zeros=config.bark_restrict_equal_ones_zeros,
                repeat=repeat,
                seed=stable_seed(
                    case.ham_id, spec, "bark", score_mode, strategy, keep_states, repeat
                ),
            )
            records = run_bark(
                case=case,
                initial=initial,
                config=run_config,
                stop_fidelity=config.stop_fidelity,
                max_subspace_fraction=config.max_subspace_fraction,
            )
            label = f"{score_mode}/{strategy}/keep={keep_states}"
            run_id = f"{case.ham_id}|{spec}|BARK|{label}|r{repeat}"
            frame, run_row, rows = summarise_run(
                records, context, run_id, label, run_config.as_dict(),
                config.target_fidelities,
            )
            iteration_frames.append(frame)
            run_rows.append(run_row)
            convergence_rows.extend(rows)
            bar.update(1)

    bar.close()

    return {
        "hamiltonians": pd.DataFrame([case.row()]),
        "runs": pd.DataFrame(run_rows),
        "iterations": pd.concat(iteration_frames, ignore_index=True),
        "convergence": pd.DataFrame(convergence_rows),
    }


# ----------------------------------------------------------------------
def shard_paths(shard_dir: Path, ham_id: str) -> dict[str, Path]:
    return {name: shard_dir / f"{ham_id}__{name}.parquet" for name in SHARD_TABLES}


def write_shards(shard_dir: Path, ham_id: str, tables: dict[str, pd.DataFrame]) -> None:
    shard_dir.mkdir(parents=True, exist_ok=True)
    for name, path in shard_paths(shard_dir, ham_id).items():
        tables[name].to_parquet(path, index=False)


def merge_shards(shard_dir: Path) -> dict[str, pd.DataFrame]:
    merged: dict[str, pd.DataFrame] = {}
    for name in SHARD_TABLES:
        files = sorted(shard_dir.glob(f"*__{name}.parquet"))
        if not files:
            merged[name] = pd.DataFrame()
            continue
        merged[name] = pd.concat(
            (pd.read_parquet(f) for f in files), ignore_index=True
        )
    return merged


# ----------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--preset", default="default", choices=sorted(PRESETS))
    parser.add_argument(
        "--output-dir",
        default="experiments/fidelity_subspace_study/results",
        help="Directory for parquet tables and shards.",
    )

    scan = parser.add_argument_group("scan overrides")
    scan.add_argument("--num-qubits", nargs="+", type=int)
    scan.add_argument(
        "--max-interactions",
        nargs="+",
        type=int,
        help="Bonds per site; use -1 for all-to-all coupling.",
    )
    scan.add_argument("--num-hamiltonians", type=int)
    scan.add_argument("--seed-offset", type=int)
    scan.add_argument(
        "--initial-overlaps",
        nargs="+",
        help="'max' for the highest-weight basis state, or target overlaps like 0.05.",
    )
    scan.add_argument("--target-fidelities", nargs="+", type=float)
    scan.add_argument("--max-subspace-fraction", type=float)

    model = parser.add_argument_group("random spin model")
    model.add_argument("--J-max", type=float, dest="J_max")
    model.add_argument("--B-max", type=float, dest="B_max")
    model.add_argument("--J-components", nargs="+", dest="J_components")
    model.add_argument("--B-components", nargs="+", dest="B_components")
    model.add_argument(
        "--coupling-distribution", choices=["uniform", "normal", "bimodal"]
    )
    model.add_argument("--field-distribution", choices=["uniform", "normal", "bimodal"])

    skqd = parser.add_argument_group("SKQD grid")
    skqd.add_argument("--skqd-dt", nargs="+", type=float)
    skqd.add_argument("--skqd-shots", nargs="+", type=int)
    skqd.add_argument("--skqd-repeats", type=int)
    skqd.add_argument("--skqd-max-iterations", type=int)

    bark = parser.add_argument_group("BARK grid")
    bark.add_argument(
        "--bark-score-modes",
        nargs="+",
        choices=[
            "amplitude",
            "probability",
            "coupling",
            "perturbative",
            "two_dimensional",
        ],
    )
    bark.add_argument(
        "--bark-selection-strategies", nargs="+", choices=["pool", "best_first"]
    )
    bark.add_argument("--bark-keep-states", nargs="+", type=int)
    bark.add_argument("--bark-max-iterations", type=int)
    bark.add_argument("--bark-subspace", choices=["applied", "encountered"])

    control = parser.add_argument_group("execution")
    control.add_argument("--chunk", type=int, default=0)
    control.add_argument("--num-chunks", type=int, default=1)
    control.add_argument("--resume", action="store_true")
    control.add_argument("--dry-run", action="store_true")
    control.add_argument("--no-progress", action="store_true")
    control.add_argument("--min-reach-fraction", type=float, default=0.5)
    control.add_argument(
        "--merge-only",
        action="store_true",
        help="Skip simulation; just merge existing shards and redo the optimisation.",
    )
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> StudyConfig:
    overrides = {
        key: getattr(args, key)
        for key in (
            "num_qubits",
            "max_interactions",
            "num_hamiltonians",
            "seed_offset",
            "initial_overlaps",
            "target_fidelities",
            "max_subspace_fraction",
            "J_max",
            "B_max",
            "coupling_distribution",
            "field_distribution",
            "skqd_dt",
            "skqd_shots",
            "skqd_repeats",
            "skqd_max_iterations",
            "bark_score_modes",
            "bark_selection_strategies",
            "bark_keep_states",
            "bark_max_iterations",
            "bark_subspace",
        )
        if getattr(args, key, None) is not None
    }
    if args.J_components is not None:
        overrides["J_components"] = tuple(args.J_components)
    if args.B_components is not None:
        overrides["B_components"] = tuple(args.B_components)
    return build_config(args.preset, **overrides)


def finalise(output_dir: Path, min_reach_fraction: float) -> dict[str, pd.DataFrame]:
    """Merge shards, run the parameter optimisation, and write the top-level tables."""
    tables = merge_shards(output_dir / "shards")
    for name, frame in tables.items():
        frame.to_parquet(output_dir / f"{name}.parquet", index=False)

    convergence = tables["convergence"]
    if convergence.empty:
        return tables

    optimal = optimal_configurations(convergence, min_reach_fraction)
    optimal.to_parquet(output_dir / "optimal.parquet", index=False)
    comparison = method_comparison(optimal)
    comparison.to_parquet(output_dir / "method_comparison.parquet", index=False)
    tables["optimal"] = optimal
    tables["method_comparison"] = comparison
    return tables


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = config_from_args(args)
    output_dir = Path(args.output_dir)
    shard_dir = output_dir / "shards"

    counts = config.run_count()
    print("Scan size:")
    for key, value in counts.items():
        print(f"  {key:>20}: {value}")

    if args.dry_run:
        print("\n--dry-run: nothing executed.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(config.to_json(), encoding="utf-8")
    (output_dir / "environment.json").write_text(
        json.dumps(
            {
                "python": sys.version,
                "platform": platform.platform(),
                "processor": platform.processor(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "started": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    if not args.merge_only:
        specs = config.case_specs()
        if args.num_chunks > 1:
            specs = specs[args.chunk :: args.num_chunks]
            print(f"\nChunk {args.chunk}/{args.num_chunks}: {len(specs)} Hamiltonians")

        for num_qubits, max_interactions, seed in tqdm(
            specs, desc="Hamiltonians", disable=args.no_progress
        ):
            label_mi = "inf" if max_interactions is None else str(max_interactions)
            ham_id = f"n{num_qubits}_mi{label_mi}_s{seed}"
            if args.resume and all(
                p.exists() for p in shard_paths(shard_dir, ham_id).values()
            ):
                continue

            case = build_case(
                num_qubits=num_qubits,
                max_interactions=max_interactions,
                seed=seed,
                model_kwargs=config.model_kwargs,
                degeneracy_tol=config.degeneracy_tol,
                num_eigenvalues=config.num_reference_eigenvalues,
                dense_subspace_max_dim=config.dense_subspace_max_dim,
            )
            tables = run_case(case, config, progress=not args.no_progress)
            write_shards(shard_dir, case.ham_id, tables)

    tables = finalise(output_dir, args.min_reach_fraction)

    print(f"\nWrote tables to {output_dir}")
    for name, frame in tables.items():
        print(f"  {name:>18}.parquet : {len(frame):>8} rows")


if __name__ == "__main__":
    main()
