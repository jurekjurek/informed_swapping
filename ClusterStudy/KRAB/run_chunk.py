"""
Chunk runner for the KRAB cluster study.

Reads params/param_grid.txt, slices it into N_CHUNKS parts, and runs
every experiment in this chunk sequentially.

Usage
-----
python run_chunk.py --chunk 0 --n_chunks 30
"""

import argparse
import os
import sys
import time
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from run_experiment import run


def load_param_lines(param_file: str) -> list[tuple]:
    lines = []
    with open(param_file) as f:
        for raw in f:
            raw = raw.strip()
            if not raw or raw.startswith("#"):
                continue
            parts = raw.split()
            if len(parts) != 6:
                raise ValueError(f"Expected 6 fields, got: {raw!r}")
            n_qubits, gs_sp, ham_sp, overlap, Q, epsilon = parts
            lines.append((int(n_qubits), float(gs_sp), float(ham_sp),
                          float(overlap), int(Q), float(epsilon)))
    return lines


def chunk_slice(total: int, n_chunks: int, chunk_idx: int) -> range:
    base, remainder = divmod(total, n_chunks)
    start = chunk_idx * base + min(chunk_idx, remainder)
    stop  = start + base + (1 if chunk_idx < remainder else 0)
    return range(start, stop)


def figure_fname(n_qubits, gs_sp, ham_sp, overlap, Q, epsilon, seed):
    return (
        f"nq{n_qubits}_gs{gs_sp:.2f}_ham{ham_sp:.2f}"
        f"_ov{overlap:.2f}_Q{Q:03d}_eps{epsilon:.0e}_seed{seed}.png"
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--chunk",      type=int, required=True)
    p.add_argument("--n_chunks",   type=int, default=30)
    p.add_argument("--param_file", type=str, default="params/param_grid.txt")
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--figure_dir", type=str, default="results")
    p.add_argument("--data_dir",   type=str, default="data")
    args = p.parse_args()

    if args.chunk < 0 or args.chunk >= args.n_chunks:
        raise ValueError(f"--chunk must be in [0, {args.n_chunks - 1}]")

    all_params = load_param_lines(args.param_file)
    total = len(all_params)
    my_range = chunk_slice(total, args.n_chunks, args.chunk)
    my_params = [all_params[i] for i in my_range]

    print(f"Chunk {args.chunk}/{args.n_chunks - 1}  —  "
          f"jobs {my_range.start}–{my_range.stop - 1} of {total - 1}  "
          f"({len(my_params)} experiments)", flush=True)
    print("=" * 65, flush=True)

    t_chunk = time.time()

    for local_idx, (n_qubits, gs_sp, ham_sp, overlap, Q, epsilon) in enumerate(my_params):
        global_idx = my_range.start + local_idx
        print(f"\n[{local_idx + 1}/{len(my_params)}  global={global_idx}]", flush=True)

        # Skip if figure already exists (safe re-submission)
        fname = figure_fname(n_qubits, gs_sp, ham_sp, overlap, Q, epsilon, args.seed)
        if os.path.exists(os.path.join(args.figure_dir, fname)):
            print(f"  Skipping (figure exists): {fname}", flush=True)
            continue

        try:
            run(
                n_qubits=n_qubits,
                gs_sparsity=gs_sp,
                ham_sparsity=ham_sp,
                overlap=overlap,
                Q=Q,
                epsilon=epsilon,
                seed=args.seed,
                figure_dir=args.figure_dir,
                data_dir=args.data_dir,
            )
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            print("  Continuing to next experiment.", flush=True)

    elapsed = time.time() - t_chunk
    print(f"\nChunk {args.chunk} finished in {elapsed/3600:.2f}h  "
          f"({len(my_params)} experiments).", flush=True)


if __name__ == "__main__":
    main()
