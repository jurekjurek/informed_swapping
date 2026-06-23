"""
Chunk runner for the KRAB-vs-SKQD cluster study.

Reads params/param_grid.txt and distributes the experiments across N_CHUNKS so
that every chunk takes roughly the same wall time (cost-balanced, not
contiguous): the most expensive jobs — large n_qubits and large Q — are spread
one per chunk. Each chunk then runs its assigned experiments sequentially.

Usage
-----
python run_chunk.py --chunk 0 --n_chunks 30
"""

import argparse
import heapq
import os
import sys
import time
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from run_experiment import run, SKQD_TIME_STEPS, resolve_n_iterations


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
            n_qubits, gs_sp, ham_sp, overlap, Q_frac, epsilon = parts
            lines.append((int(n_qubits), float(gs_sp), float(ham_sp),
                          float(overlap), float(Q_frac), float(epsilon)))
    return lines


def estimate_cost(n_qubits: int, Q_frac: float, n_random_paths: int) -> float:
    """Relative wall-time estimate for one experiment (used for load balancing).

    Dominated by the SKQD time-step sweep + random baseline, each reconstructed
    with get_one_path (one eigsh per added state on a growing matrix → ~dim²).
    KRAB adds a smaller, Q-dependent term that grows with the subspace size.
    """
    dim = 2 ** n_qubits
    n_skqd = len(SKQD_TIME_STEPS)
    skqd_random = (n_skqd + n_random_paths) * dim ** 2
    q_int = max(1, round(Q_frac * dim))
    krab = resolve_n_iterations(Q_frac) * q_int * dim
    return float(skqd_random + krab)


def assign_chunks(all_params: list[tuple], n_chunks: int,
                  n_random_paths: int) -> dict[int, list[int]]:
    """Greedy longest-processing-time bin packing → balanced total cost per chunk.

    Deterministic given the param file and n_random_paths, so the assignment is
    identical across all array tasks.
    """
    costs = [estimate_cost(p[0], p[4], n_random_paths) for p in all_params]
    order = sorted(range(len(all_params)), key=lambda i: costs[i], reverse=True)

    heap = [(0.0, c) for c in range(n_chunks)]          # (current load, chunk id)
    heapq.heapify(heap)
    chunk_jobs: dict[int, list[int]] = {c: [] for c in range(n_chunks)}
    for i in order:
        load, c = heapq.heappop(heap)
        chunk_jobs[c].append(i)
        heapq.heappush(heap, (load + costs[i], c))

    # Run cheapest-first within a chunk so quick wins land early in the log.
    for c in chunk_jobs:
        chunk_jobs[c].sort(key=lambda i: costs[i])
    return chunk_jobs


def figure_fname(n_qubits, gs_sp, ham_sp, overlap, Q_frac, epsilon, seed):
    return (
        f"nq{n_qubits}_gs{gs_sp:.2f}_ham{ham_sp:.2f}"
        f"_ov{overlap:.2f}_Qf{Q_frac:.3f}_eps{epsilon:.0e}_seed{seed}.png"
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--chunk",          type=int, required=True)
    p.add_argument("--n_chunks",       type=int, default=30)
    p.add_argument("--param_file",     type=str, default="params/param_grid.txt")
    p.add_argument("--seed",           type=int, default=42)
    p.add_argument("--figure_dir",     type=str, default="figures")
    p.add_argument("--data_dir",       type=str, default="data")
    p.add_argument("--n_random_paths", type=int, default=20,
                   help="Random baseline orderings per experiment (0 disables).")
    args = p.parse_args()

    if args.chunk < 0 or args.chunk >= args.n_chunks:
        raise ValueError(f"--chunk must be in [0, {args.n_chunks - 1}]")

    all_params = load_param_lines(args.param_file)
    total = len(all_params)
    chunk_jobs = assign_chunks(all_params, args.n_chunks, args.n_random_paths)
    my_indices = chunk_jobs[args.chunk]
    est_cost = sum(estimate_cost(all_params[i][0], all_params[i][4], args.n_random_paths)
                   for i in my_indices)

    print(f"Chunk {args.chunk}/{args.n_chunks - 1}  —  "
          f"{len(my_indices)} experiments (cost-balanced) of {total} total  "
          f"(relative cost ≈ {est_cost:.2e})", flush=True)
    print("=" * 65, flush=True)

    t_chunk = time.time()

    for local_idx, global_idx in enumerate(my_indices):
        n_qubits, gs_sp, ham_sp, overlap, Q_frac, epsilon = all_params[global_idx]
        print(f"\n[{local_idx + 1}/{len(my_indices)}  global={global_idx}]", flush=True)

        # Skip if figure already exists (safe re-submission)
        fname = figure_fname(n_qubits, gs_sp, ham_sp, overlap, Q_frac, epsilon, args.seed)
        if os.path.exists(os.path.join(args.figure_dir, fname)):
            print(f"  Skipping (figure exists): {fname}", flush=True)
            continue

        try:
            run(
                n_qubits=n_qubits,
                gs_sparsity=gs_sp,
                ham_sparsity=ham_sp,
                overlap=overlap,
                Q_frac=Q_frac,
                epsilon=epsilon,
                seed=args.seed,
                figure_dir=args.figure_dir,
                data_dir=args.data_dir,
                n_random_paths=args.n_random_paths,
            )
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            print("  Continuing to next experiment.", flush=True)

    elapsed = time.time() - t_chunk
    print(f"\nChunk {args.chunk} finished in {elapsed/3600:.2f}h  "
          f"({len(my_indices)} experiments).", flush=True)


if __name__ == "__main__":
    main()
