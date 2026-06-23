"""
Generate the KRAB parameter grid and write it to params/param_grid.txt.

Each line: n_qubits  gs_sparsity  ham_sparsity  overlap  Q_frac  epsilon

Q_frac is the fraction of the full Hilbert space 2ⁿ that KRAB keeps per
iteration (in [0, 1]); the integer budget round(Q_frac · 2ⁿ) is resolved per
system at run time.

Usage
-----
python generate_jobs.py [--preset full|small] [--dry_run]
"""

import argparse
import itertools
import os

FULL_GRID = dict(
    n_qubits      = [6, 8, 10],
    gs_sparsity   = [0.05, 0.1, 0.25],
    ham_sparsity  = [0.1, 0.25, 0.5],
    overlap       = [0.1, 0.3, 0.5, 0.8],
    Q             = [0.01, 0.03, 0.05, 0.10],   # fraction of 2ⁿ kept per iter
    epsilon       = [1e-7, 1e-6, 1e-5, 1e-4],
)

SMALL_GRID = dict(
    n_qubits      = [6, 8],
    gs_sparsity   = [0.1, 0.5],
    ham_sparsity  = [0.25, 0.5],
    overlap       = [0.1, 0.5],
    Q             = [0.01, 0.10],               # fraction of 2ⁿ kept per iter
    epsilon       = [1e-6, 1e-4],
)

PRESETS = {"full": FULL_GRID, "small": SMALL_GRID}


def build_param_lines(grid: dict) -> list[str]:
    lines = []
    for (n_q, gs, ham, ov, Q, eps) in itertools.product(
        grid["n_qubits"], grid["gs_sparsity"], grid["ham_sparsity"],
        grid["overlap"], grid["Q"], grid["epsilon"],
    ):
        lines.append(f"{n_q} {gs} {ham} {ov} {Q:.2f} {eps:.2e}")
    return lines


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--preset",  choices=list(PRESETS), default="full")
    p.add_argument("--output",  default="params/param_grid.txt")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    grid = PRESETS[args.preset]
    lines = build_param_lines(grid)

    print(f"Preset      : {args.preset}")
    print(f"Total jobs  : {len(lines)}")
    print(f"Array range : 0–{len(lines)-1}")
    print()
    print("First 5 lines (n_qubits  gs_sparsity  ham_sparsity  overlap  Q_frac  epsilon):")
    for line in lines[:5]:
        print(f"  {line}")
    if len(lines) > 5:
        print(f"  ... ({len(lines)-5} more)")

    if args.dry_run:
        print("\n[dry_run] File not written.")
        return

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nWritten to: {args.output}")
    print(f"Submit with: sbatch run_array.slurm   (array=0-29 hardcoded)")


if __name__ == "__main__":
    main()
