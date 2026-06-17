"""
Generate the parameter grid and write it to params/param_grid.txt.

Each line in the file is one job:
    n_qubits  gs_sparsity  ham_sparsity  overlap  keep_states

Run this script once before submitting the Slurm array job:
    python generate_jobs.py

To preview the grid without writing:
    python generate_jobs.py --dry_run

To use a smaller grid for quick testing:
    python generate_jobs.py --preset small
"""

import argparse
import itertools
import os

# ---------------------------------------------------------------------------
# Parameter grids
# ---------------------------------------------------------------------------

# Full grid — intended for the large cluster run.
FULL_GRID = dict(
    n_qubits       = [6, 7, 8, 9, 10],
    gs_sparsity    = [0.05, 0.1, 0.25, 0.5, 0.75, 1.0],
    ham_sparsity   = [0.0, 0.25, 0.5, 0.75, 1.0],
    overlap        = [0.01, 0.1, 0.3, 0.5, 0.8],
    # keep_states expressed as a fraction of dim = 2^n_qubits;
    # the actual integer is computed per n_qubits below.
    keep_states_frac = [0.01, 0.05, 0.1, 0.5],
)

# Small grid — quick sanity check, runs in minutes on a laptop.
SMALL_GRID = dict(
    n_qubits         = [6, 8],
    gs_sparsity      = [0.1, 0.5],
    ham_sparsity     = [0.25, 0.75],
    overlap          = [0.1, 0.5],
    keep_states_frac = [0.05, 0.25],
)

PRESETS = {"full": FULL_GRID, "small": SMALL_GRID}


# ---------------------------------------------------------------------------
# Grid generation
# ---------------------------------------------------------------------------

def build_param_lines(grid: dict) -> list[str]:
    """Return one line per job: 'n_qubits gs_sparsity ham_sparsity overlap keep_states'."""
    lines = []
    combos = itertools.product(
        grid["n_qubits"],
        grid["gs_sparsity"],
        grid["ham_sparsity"],
        grid["overlap"],
        grid["keep_states_frac"],
    )
    for n_qubits, gs_sp, ham_sp, overlap, ks_frac in combos:
        dim = 2 ** n_qubits
        keep_states = max(1, int(round(ks_frac * dim)))
        # Skip keep_states >= dim (no point in keeping more states than exist)
        if keep_states >= dim:
            keep_states = dim // 2
        lines.append(f"{n_qubits} {gs_sp} {ham_sp} {overlap} {keep_states}")

    # Deduplicate while preserving order (different fracs can map to the same int)
    seen = set()
    unique = []
    for line in lines:
        if line not in seen:
            seen.add(line)
            unique.append(line)
    return unique


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Generate parameter grid for Slurm array job.")
    p.add_argument(
        "--preset", choices=list(PRESETS), default="full",
        help="Which parameter grid to use (default: full).",
    )
    p.add_argument(
        "--output", default="params/param_grid.txt",
        help="Path to write the parameter file (default: params/param_grid.txt).",
    )
    p.add_argument(
        "--dry_run", action="store_true",
        help="Print statistics and the first few lines without writing the file.",
    )
    args = p.parse_args()

    grid = PRESETS[args.preset]
    lines = build_param_lines(grid)

    print(f"Preset      : {args.preset}")
    print(f"Total jobs  : {len(lines)}")
    print(f"Array indices: 0–{len(lines)-1}")
    print()
    print("First 5 lines (n_qubits  gs_sparsity  ham_sparsity  overlap  keep_states):")
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
    print(f"Submit with: sbatch --array=0-{len(lines)-1} run_array.slurm")


if __name__ == "__main__":
    main()
