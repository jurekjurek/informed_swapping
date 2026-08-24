"""
Distributed version of SystematicStudy.py for a SLURM job array.

The unit of work is a "cell": one ``(hamiltonian_index, num_sites,
max_interactions)`` combination. A cell builds one Hamiltonian, solves for its
ground state, and then runs both protocols for every fidelity and every starting
state.

Load balancing is structural rather than measured. Cost per cell grows steeply
with ``num_sites``, so slicing the cell list contiguously would give some jobs
only cheap 6-site cells and others only expensive 14-site ones. Instead the cells
are grouped by ``num_sites`` and dealt round-robin within each group, so every
job receives the same number of cells of each system size (to within one) and
therefore roughly the same total workload. No profiling required.

    e.g. num_sites = [6, 8, 10, 12, 14], 200 cells, 20 jobs
         -> every job runs 10 cells: 2 x 6, 2 x 8, 2 x 10, 2 x 12, 2 x 14

Each job writes its own shard, so a merge is a concatenation and a re-submission
skips whatever already finished.

Usage
-----
    # inspect the split before submitting anything
    python ClusterStudy.py plan   --num-hamiltonians 20 --num-jobs 20

    # one array task (this is what the submit script calls)
    python ClusterStudy.py run    --num-hamiltonians 20 --num-jobs 20 --job-index 3

    # collect the shards
    python ClusterStudy.py merge  --output systematic_study_results.csv
"""

import argparse
import hashlib
import os
import sys
import time
from collections import defaultdict, namedtuple

import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh

from RandomSpinModel import make_random_spin_hamiltonian
from BARK import BARK
from SKQD import SKQD

DEFAULT_NUM_SITES = [6, 8, 10, 12, 14]
DEFAULT_MAX_INTERACTIONS = [1, 2]
DEFAULT_FIDELITIES = [0.8, 0.85, 0.9]

COLUMNS = ["Hamiltonian_Index", "Number_of_Sites", "Max_Interactions", "Fidelity",
           "Ground_State_Density", "Hamiltonian_Density", "Overlap",
           "BARK_Pool_Size", "SKQD_Pool_Size", "Seed"]

Cell = namedtuple("Cell", ["hamiltonian_index", "num_sites", "max_interactions"])


# --------------------------------------------------------------------------- #
# Work definition and distribution
# --------------------------------------------------------------------------- #

def cell_seed(cell: Cell) -> int:
    """
    Deterministic 32-bit seed derived from the cell's identity.

    SHA-256 rather than ``hash()``: Python salts string hashing per process, so
    ``hash()`` would give different Hamiltonians in different jobs for the same
    cell. With this, any job -- or a re-run after a preemption -- reproduces
    exactly the same Hamiltonian.
    """
    key = f"{cell.hamiltonian_index}|{cell.num_sites}|{cell.max_interactions}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "little")


def enumerate_cells(num_hamiltonians, num_sites, max_interactions):
    """All cells of the study, in a fixed order independent of how they are split."""
    return [Cell(hamiltonian_index, n_sites, max_interaction)
            for hamiltonian_index in range(num_hamiltonians)
            for n_sites in num_sites
            for max_interaction in max_interactions]


def assign_cells(cells, num_jobs, job_index):
    """
    The cells belonging to one job of the array.

    Cells are bucketed by ``num_sites`` and dealt round-robin inside each bucket,
    which spreads every system size evenly over the jobs. Each job therefore gets
    ``len(bucket) // num_jobs`` cells of each size, plus at most one extra when a
    bucket does not divide evenly.
    """
    buckets = defaultdict(list)
    for cell in cells:
        buckets[cell.num_sites].append(cell)

    assigned = []
    for n_sites in sorted(buckets):
        assigned.extend(buckets[n_sites][job_index::num_jobs])
    return assigned


def describe_plan(cells, num_jobs):
    """Per-job composition table, so the balance can be checked before submitting."""
    sizes = sorted({cell.num_sites for cell in cells})
    rows = []
    for job_index in range(num_jobs):
        mine = assign_cells(cells, num_jobs, job_index)
        counts = defaultdict(int)
        for cell in mine:
            counts[cell.num_sites] += 1
        row = {"Job": job_index, "Cells": len(mine)}
        row.update({f"n={n}": counts[n] for n in sizes})
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Running a cell
# --------------------------------------------------------------------------- #

def run_cell(cell: Cell, fidelities, sparse: bool = True) -> list:
    """
    Run one cell and return its rows as a list of dicts.

    ``sparse=True`` keeps the Hamiltonian in CSR form and solves only for the
    ground state, which is what is needed once the Hilbert space is too large to
    diagonalize densely. ``sparse=False`` computes the full dense
    eigendecomposition and hands it to SKQD, which then builds U(t) from it
    instead of going through a matrix exponential -- faster, but only affordable
    for small systems.
    """
    seed = cell_seed(cell)
    # SKQD draws its shots through the global RNG, so seed that as well.
    np.random.seed(seed)

    hamiltonian = make_random_spin_hamiltonian(
        num_sites=cell.num_sites,
        max_interactions=cell.max_interactions,
        seed=seed,
    )[0].to_matrix(sparse=sparse)

    if sparse:
        eigenvalues, eigenvectors = eigsh(hamiltonian, k=1, which="SA")
        all_eigenvalues, all_eigenvectors = None, None
    else:
        eigenvalues, eigenvectors = eigh(hamiltonian)
        all_eigenvalues, all_eigenvectors = eigenvalues, eigenvectors

    ground_state = eigenvectors[:, int(np.argmin(eigenvalues))]
    ground_state_density = np.count_nonzero(np.abs(ground_state) > 1e-3) / ground_state.shape[0]

    stored_nonzeros = hamiltonian.nnz if sparse else np.count_nonzero(hamiltonian)
    hamiltonian_density = stored_nonzeros / hamiltonian.shape[0] ** 2

    probabilities = np.abs(ground_state) ** 2
    largest_indices = np.argsort(probabilities)[-5:]
    smallest_indices = np.argsort(probabilities)[:5]
    indices_to_test = np.concatenate((largest_indices, smallest_indices))
    overlaps = probabilities[indices_to_test]

    rows = []
    for fidelity in fidelities:
        for initial_state_index, overlap in zip(indices_to_test, overlaps):
            bark_protocol = BARK(hamiltonian)
            bark_pool_size = bark_protocol.test_run(
                target_fidelity=fidelity,
                correct_state=ground_state,
                initial_state_index=initial_state_index,
            )

            skqd_protocol = SKQD(hamiltonian, eigenvalues=all_eigenvalues,
                                 eigenvectors=all_eigenvectors)
            t, shots, _ = skqd_protocol.optimize(
                initial_state_index=initial_state_index,
                correct_state=ground_state,
                target_fidelity=fidelity,
            )
            skqd_pool_size = skqd_protocol.test_run(
                initial_state_index=initial_state_index,
                t=t,
                n_shots=shots,
                correct_state=ground_state,
                target_fidelity=fidelity,
            )

            rows.append({
                "Hamiltonian_Index": cell.hamiltonian_index,
                "Number_of_Sites": cell.num_sites,
                "Max_Interactions": cell.max_interactions,
                "Fidelity": fidelity,
                "Ground_State_Density": ground_state_density,
                "Hamiltonian_Density": hamiltonian_density,
                "Overlap": overlap,
                "BARK_Pool_Size": bark_pool_size,
                "SKQD_Pool_Size": skqd_pool_size,
                "Seed": seed,
            })
    return rows


def shard_path(shard_dir, job_index):
    return os.path.join(shard_dir, f"shard_{job_index:05d}.csv")


def run_job(args):
    """Run every cell assigned to this job and write one shard."""
    cells = enumerate_cells(args.num_hamiltonians, args.num_sites, args.max_interactions)
    mine = assign_cells(cells, args.num_jobs, args.job_index)

    os.makedirs(args.shard_dir, exist_ok=True)
    destination = shard_path(args.shard_dir, args.job_index)
    if os.path.exists(destination) and not args.overwrite:
        print(f"[job {args.job_index}] {destination} exists, nothing to do "
              f"(pass --overwrite to force a re-run)", flush=True)
        return

    counts = defaultdict(int)
    for cell in mine:
        counts[cell.num_sites] += 1
    composition = ", ".join(f"{count} x n={n}" for n, count in sorted(counts.items()))
    print(f"[job {args.job_index}/{args.num_jobs}] {len(mine)} cells ({composition})", flush=True)

    rows = []
    started = time.time()
    for position, cell in enumerate(mine, start=1):
        cell_started = time.time()
        rows.extend(run_cell(cell, args.fidelities, sparse=args.sparse))
        print(f"[job {args.job_index}] {position}/{len(mine)} "
              f"n={cell.num_sites} mi={cell.max_interactions} "
              f"ham={cell.hamiltonian_index} took {time.time() - cell_started:.1f}s "
              f"(elapsed {time.time() - started:.1f}s)", flush=True)

        # Write after every cell so a timeout or preemption keeps the finished work.
        pd.DataFrame(rows, columns=COLUMNS).to_csv(destination + ".partial", index=False)

    pd.DataFrame(rows, columns=COLUMNS).to_csv(destination, index=False)
    if os.path.exists(destination + ".partial"):
        os.remove(destination + ".partial")
    print(f"[job {args.job_index}] wrote {len(rows)} rows to {destination} "
          f"in {time.time() - started:.1f}s", flush=True)


def merge_shards(args):
    """Concatenate the shards into one CSV and report anything missing."""
    cells = enumerate_cells(args.num_hamiltonians, args.num_sites, args.max_interactions)

    frames, missing = [], []
    for job_index in range(args.num_jobs):
        path = shard_path(args.shard_dir, job_index)
        if os.path.exists(path):
            frames.append(pd.read_csv(path))
        elif assign_cells(cells, args.num_jobs, job_index):
            missing.append(job_index)

    if not frames:
        sys.exit(f"no shards found in {args.shard_dir}")

    merged = pd.concat(frames, ignore_index=True)
    merged.sort_values(["Number_of_Sites", "Max_Interactions", "Hamiltonian_Index", "Fidelity"],
                       inplace=True, ignore_index=True)
    merged.to_csv(args.output, index=False)

    expected = len(cells) * len(args.fidelities) * 10
    print(f"merged {len(frames)} shards -> {args.output}: {len(merged)} rows "
          f"({expected} expected for a complete run)")
    if missing:
        print(f"WARNING: {len(missing)} shards missing (jobs {missing}); "
              f"re-submit those array indices, then merge again")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def build_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["plan", "run", "merge"])
    parser.add_argument("--num-hamiltonians", type=int, default=20,
                        help="number of random Hamiltonians per (num_sites, max_interactions)")
    parser.add_argument("--num-sites", type=int, nargs="+", default=DEFAULT_NUM_SITES)
    parser.add_argument("--max-interactions", type=int, nargs="+", default=DEFAULT_MAX_INTERACTIONS)
    parser.add_argument("--fidelities", type=float, nargs="+", default=DEFAULT_FIDELITIES)
    parser.add_argument("--num-jobs", type=int, default=20,
                        help="size of the SLURM array")
    parser.add_argument("--job-index", type=int, default=None,
                        help="this task's index; defaults to $SLURM_ARRAY_TASK_ID")
    parser.add_argument("--shard-dir", default="shards")
    parser.add_argument("--output", default="systematic_study_results.csv",
                        help="merge mode: where to write the combined CSV")
    parser.add_argument("--dense", dest="sparse", action="store_false",
                        help="use dense Hamiltonians and pass the full eigendecomposition to SKQD")
    parser.add_argument("--overwrite", action="store_true",
                        help="re-run a job even if its shard already exists")
    parser.set_defaults(sparse=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    if args.mode == "run" and args.job_index is None:
        env = os.environ.get("SLURM_ARRAY_TASK_ID")
        if env is None:
            sys.exit("--job-index is required outside a SLURM array")
        args.job_index = int(env)

    if args.mode == "plan":
        cells = enumerate_cells(args.num_hamiltonians, args.num_sites, args.max_interactions)
        plan = describe_plan(cells, args.num_jobs)
        print(f"{len(cells)} cells over {args.num_jobs} jobs "
              f"({len(cells) * len(args.fidelities) * 10} rows total)\n")
        print(plan.to_string(index=False))
        spread = plan["Cells"].max() - plan["Cells"].min()
        print(f"\ncells per job: min {plan['Cells'].min()}, max {plan['Cells'].max()} "
              f"(spread {spread})")
        for column in [c for c in plan.columns if c.startswith("n=")]:
            values = plan[column]
            if values.max() != values.min():
                print(f"  {column}: uneven by {values.max() - values.min()} "
                      f"(bucket does not divide by {args.num_jobs})")
        print(f"\nsbatch --array=0-{args.num_jobs - 1} submit_study.sh")
    elif args.mode == "run":
        run_job(args)
    else:
        merge_shards(args)


if __name__ == "__main__":
    main()
