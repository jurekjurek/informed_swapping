"""
Distributed version of SystematicStudy.py for a SLURM job array.

The unit of work is a "cell": one ``(hamiltonian_index, num_sites,
max_interactions)`` combination. A cell builds one Hamiltonian, solves for its
ground state, and then runs both protocols for every fidelity and every starting
state.

Load balancing is done by cost model. Cost per cell grows steeply with
``num_sites`` -- and, as the first production run showed, by a further factor of
about 2.6 from ``max_interactions=1`` to ``max_interactions=3`` -- so cells are
weighted by ``estimate_cost`` and dealt out longest-processing-time-first: each
cell in turn goes to whichever job is currently the lightest. That balances the
two axes that actually matter instead of only the system size.

    e.g. num_sites = [6, 8, 10, 12, 14], 300 cells, 40 jobs
         -> every job gets a similar *estimated runtime*, not a similar count

Each job writes its own shard, so a merge is a concatenation and a re-submission
skips whatever already finished. A job that is killed mid-shard leaves a
``.partial`` file behind; re-running picks up from there and only redoes the
cells that were still missing.

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
import heapq
import os
import sys
import time
from collections import defaultdict, namedtuple

import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.sparse import issparse
from scipy.sparse.linalg import eigsh

from RandomSpinModel import make_random_spin_hamiltonian
from BARK import BARK
from SKQD import SKQD

DEFAULT_NUM_SITES = [6, 8, 10, 12, 14]
DEFAULT_MAX_INTERACTIONS = [1, 2]
DEFAULT_FIDELITIES = [0.8, 0.85, 0.9]

# Hilbert-space dimension up to which the full dense eigendecomposition is
# computed once per cell and handed to SKQD. 4096 states (12 sites) needs a
# 268 MB eigenvector matrix and about a minute of LAPACK time -- against which
# every single time evolution in the cell drops from a few hundred sparse
# mat-vecs plus norm estimation to two dense mat-vecs.
DEFAULT_DENSE_LIMIT = 4096

COLUMNS = ["Hamiltonian_Index", "Number_of_Sites", "Max_Interactions", "Fidelity",
           "Ground_State_Density", "Hamiltonian_Density", "Overlap",
           "BARK_Pool_Size", "SKQD_Pool_Size", "Seed"]

CELL_KEY = ["Hamiltonian_Index", "Number_of_Sites", "Max_Interactions"]

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


def estimate_cost(cell: Cell) -> float:
    """
    Relative cost of a cell. Used only for load balancing, so the constant and
    the units are irrelevant -- only the ratios matter.

    Calibrated on job array 27141335, whose per-cell wall times were

        n=6    36 / 41 / 43 s      (max_interactions = 1 / 2 / 3)
        n=8   314 / 448 / 496 s
        n=10 5496 / 13099 / 14259 s

    i.e. roughly ``dimension ** 2.3`` in the system size and a further ~1.8x
    per extra bond per site, saturating once the interaction graph is complete.
    """
    dimension = 2.0 ** cell.num_sites
    bonds = max(1, min(cell.max_interactions, max(cell.num_sites - 1, 1)))
    return dimension ** 2.3 * (1.0 + 0.8 * (bonds - 1))


def assign_all_cells(cells, num_jobs, balance="cost"):
    """
    Split every cell over the jobs, returning one list per job.

    ``balance="cost"`` weights cells by ``estimate_cost`` and hands each in turn
    to the currently lightest job (longest-processing-time-first). Within a job
    the cells are then ordered cheapest first, so a task that runs out of wall
    clock still banks as many finished cells as possible in its ``.partial``.

    ``balance="stratified"`` is the previous behaviour: bucket by ``num_sites``
    and deal round-robin inside each bucket. It equalises the *count* of each
    system size per job but ignores ``max_interactions``, which is worth a
    factor of 2.6 at n=10.

    Both are pure functions of ``(cells, num_jobs)``, so ``plan``, ``run`` and
    ``merge`` all agree on the split without communicating.
    """
    jobs = [[] for _ in range(num_jobs)]

    if balance == "stratified":
        buckets = defaultdict(list)
        for cell in cells:
            buckets[cell.num_sites].append(cell)
        for n_sites in sorted(buckets):
            for job_index in range(num_jobs):
                jobs[job_index].extend(buckets[n_sites][job_index::num_jobs])
        return jobs

    if balance != "cost":
        raise ValueError(f"unknown balance strategy {balance!r}")

    # (load, job_index) so ties break on the lower index and the split is
    # reproducible across processes.
    loads = [(0.0, job_index) for job_index in range(num_jobs)]
    heapq.heapify(loads)
    for cell in sorted(cells, key=lambda c: (-estimate_cost(c), c)):
        load, job_index = heapq.heappop(loads)
        jobs[job_index].append(cell)
        heapq.heappush(loads, (load + estimate_cost(cell), job_index))

    for job in jobs:
        job.sort(key=lambda c: (estimate_cost(c), c))
    return jobs


def assign_cells(cells, num_jobs, job_index, balance="cost"):
    """The cells belonging to one job of the array."""
    return assign_all_cells(cells, num_jobs, balance)[job_index]


def describe_plan(cells, num_jobs, balance="cost"):
    """Per-job composition table, so the balance can be checked before submitting."""
    sizes = sorted({cell.num_sites for cell in cells})
    jobs = assign_all_cells(cells, num_jobs, balance)
    reference = min((estimate_cost(cell) for cell in cells), default=1.0)

    rows = []
    for job_index, mine in enumerate(jobs):
        counts = defaultdict(int)
        for cell in mine:
            counts[cell.num_sites] += 1
        row = {"Job": job_index, "Cells": len(mine)}
        row.update({f"n={n}": counts[n] for n in sizes})
        row["Load"] = round(sum(estimate_cost(cell) for cell in mine) / reference, 1)
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Running a cell
# --------------------------------------------------------------------------- #

def solve_ground_state(hamiltonian, dense_limit: int):
    """
    Ground state of the Hamiltonian, plus the full eigendecomposition when the
    Hilbert space is small enough to afford one.

    Solving only for the ground state with ``eigsh`` looks like the frugal
    choice, but it leaves SKQD without an eigendecomposition, so every time
    evolution has to go through ``expm_multiply`` -- a few hundred sparse
    mat-vecs plus operator-norm estimation, repeated tens of thousands of times
    per cell. One dense ``eigh`` up front costs a fraction of that and turns
    every evolution into two dense mat-vecs.

    Returns ``(ground_state, eigenvalues, eigenvectors)``, the latter two being
    ``None`` when the dense route was not taken.
    """
    if hamiltonian.shape[0] <= dense_limit:
        dense = hamiltonian.toarray() if issparse(hamiltonian) else np.asarray(hamiltonian)
        eigenvalues, eigenvectors = eigh(dense)
        # eigh returns eigenvalues in ascending order.
        return eigenvectors[:, 0], eigenvalues, eigenvectors

    eigenvalues, eigenvectors = eigsh(hamiltonian, k=1, which="SA")
    return eigenvectors[:, int(np.argmin(eigenvalues))], None, None


def run_cell(cell: Cell, fidelities, sparse: bool = True,
             dense_limit: int = DEFAULT_DENSE_LIMIT, n_target: int | None = None,
             penalty_strength: float = 0.0,) -> list:
    """
    Run one cell and return its rows as a list of dicts.

    ``sparse=True`` keeps the Hamiltonian in CSR form, which is what both
    protocols want for reading rows and growing the projected block.
    ``sparse=False`` materializes it densely instead; it no longer changes which
    eigen-route SKQD takes, since that is now governed by ``dense_limit``.

    The Hamiltonian, the eigendecomposition and both protocol objects are built
    once for the whole cell. They used to be rebuilt inside the fidelity/initial
    state loop, which threw away every per-t cache 50 times over.
    """
    seed = cell_seed(cell)
    # SKQD draws its shots through the global RNG, so seed that as well.
    np.random.seed(seed)

    hamiltonian = make_random_spin_hamiltonian(
        num_sites=cell.num_sites,
        max_interactions=cell.max_interactions,
        J_components=("z"),
        B_components=("x"),
        B_max = 10,
        seed=seed,
        N_target=n_target,
        penalty_strength=penalty_strength,
    )[0].to_matrix(sparse=sparse)

    ground_state, all_eigenvalues, all_eigenvectors = solve_ground_state(
        hamiltonian, dense_limit)
    ground_state_density = np.count_nonzero(np.abs(ground_state) > 1e-3) / ground_state.shape[0]

    stored_nonzeros = hamiltonian.nnz if issparse(hamiltonian) else np.count_nonzero(hamiltonian)
    hamiltonian_density = stored_nonzeros / hamiltonian.shape[0] ** 2

    probabilities = np.abs(ground_state) ** 2
    largest_indices = np.argsort(probabilities)[-5:]
    smallest_indices = np.argsort(probabilities)[:5]
    indices_to_test = np.concatenate((largest_indices, smallest_indices))
    overlaps = probabilities[indices_to_test]

    bark_protocol = BARK(hamiltonian)
    skqd_protocol = SKQD(hamiltonian, eigenvalues=all_eigenvalues,
                         eigenvectors=all_eigenvectors)

    rows = []
    for initial_state_index, overlap in zip(indices_to_test, overlaps):
        # One walk of BARK's (deterministic, fidelity-independent) trajectory
        # answers every fidelity at once.
        bark_pool_sizes = bark_protocol.sweep(
            target_fidelities=fidelities,
            correct_state=ground_state,
            initial_state_index=initial_state_index,
        )

        # One grid scan likewise answers every fidelity at once, though each
        # fidelity still gets its own (t, n_shots).
        optima = skqd_protocol.optimize_many(
            initial_state_index=initial_state_index,
            correct_state=ground_state,
            target_fidelities=fidelities,
            correct_probabilities=probabilities,
        )

        # Fidelities that picked the same (t, n_shots) share the confirming run.
        grouped = defaultdict(list)
        for position, (t, shots, _) in enumerate(optima):
            grouped[(t, shots)].append(position)

        skqd_pool_sizes = np.empty(len(fidelities))
        for (t, shots), positions in grouped.items():
            wanted = [fidelities[position] for position in positions]
            sizes = skqd_protocol.sweep(
                initial_state_index=initial_state_index,
                t=t,
                n_shots=shots,
                correct_state=ground_state,
                target_fidelities=wanted,
                correct_probabilities=probabilities,
            )
            for position, size in zip(positions, sizes):
                skqd_pool_sizes[position] = size

        for position, fidelity in enumerate(fidelities):
            rows.append({
                "Hamiltonian_Index": cell.hamiltonian_index,
                "Number_of_Sites": cell.num_sites,
                "Max_Interactions": cell.max_interactions,
                "Fidelity": fidelity,
                "Ground_State_Density": ground_state_density,
                "Hamiltonian_Density": hamiltonian_density,
                "Overlap": overlap,
                "BARK_Pool_Size": bark_pool_sizes[position],
                "SKQD_Pool_Size": skqd_pool_sizes[position],
                "Seed": seed,
            })
    return rows


def shard_path(shard_dir, job_index):
    return os.path.join(shard_dir, f"shard_{job_index:05d}.csv")


def load_partial(path, fidelities):
    """
    Rows already written by an earlier, interrupted run of this job.

    A cell is only treated as finished if the partial holds a full set of rows
    for it, and the partial is discarded outright if it was written for a
    different set of fidelities -- otherwise a re-submission with changed
    settings would silently mix two studies into one shard.

    Returns ``(rows, finished_cell_keys)``.
    """
    if not os.path.exists(path):
        return [], set()

    try:
        frame = pd.read_csv(path)
    except Exception as error:
        print(f"  ignoring unreadable partial {path}: {error}", flush=True)
        return [], set()

    if frame.empty or not set(COLUMNS).issubset(frame.columns):
        return [], set()

    recorded = np.sort(frame["Fidelity"].unique())
    expected = np.sort(np.unique(np.asarray(fidelities, dtype=float)))
    if recorded.shape != expected.shape or not np.allclose(recorded, expected):
        print(f"  ignoring partial {path}: it was written for fidelities "
              f"{list(recorded)}, not {list(expected)}", flush=True)
        return [], set()

    rows_per_cell = len(expected) * 10
    counts = frame.groupby(CELL_KEY).size()
    finished = {key for key, count in counts.items() if count == rows_per_cell}
    if not finished:
        return [], set()

    keep = frame.set_index(CELL_KEY).index.isin(finished)
    return frame[keep][COLUMNS].to_dict("records"), finished


def run_job(args):
    """Run every cell assigned to this job and write one shard."""
    cells = enumerate_cells(args.num_hamiltonians, args.num_sites, args.max_interactions)
    mine = assign_cells(cells, args.num_jobs, args.job_index, args.balance)

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
    print(f"[job {args.job_index}/{args.num_jobs}] {len(mine)} cells ({composition})",
          flush=True)

    rows, finished = [], set()
    if args.resume and not args.overwrite:
        rows, finished = load_partial(destination + ".partial", args.fidelities)
        if finished:
            print(f"[job {args.job_index}] resuming: {len(finished)} of {len(mine)} "
                  f"cells already in {destination}.partial", flush=True)

    started = time.time()
    for position, cell in enumerate(mine, start=1):
        if (cell.hamiltonian_index, cell.num_sites, cell.max_interactions) in finished:
            print(f"[job {args.job_index}] {position}/{len(mine)} "
                  f"n={cell.num_sites} mi={cell.max_interactions} "
                  f"ham={cell.hamiltonian_index} already done, skipping", flush=True)
            continue

        cell_started = time.time()
        rows.extend(run_cell(cell, args.fidelities, sparse=args.sparse,
                             dense_limit=args.dense_limit, n_target=args.n_target, 
                             penalty_strength=args.penalty_strength,))
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
    jobs = assign_all_cells(cells, args.num_jobs, args.balance)

    frames, partials, missing = [], [], []
    for job_index in range(args.num_jobs):
        path = shard_path(args.shard_dir, job_index)
        if os.path.exists(path):
            frames.append(pd.read_csv(path))
            continue
        # A job that ran out of wall clock left its finished cells behind in a
        # .partial; those cells are complete results, so they are worth merging
        # even though the job as a whole never got to write its shard.
        rows, finished = ([], set())
        if args.include_partial:
            rows, finished = load_partial(path + ".partial", args.fidelities)
        if rows:
            frames.append(pd.DataFrame(rows, columns=COLUMNS))
            partials.append(job_index)
        elif jobs[job_index]:
            missing.append(job_index)

    if not frames:
        hint = "" if args.include_partial else " (pass --include-partial to also " \
                                              "merge finished cells from .partial files)"
        sys.exit(f"no shards found in {args.shard_dir}{hint}")

    merged = pd.concat(frames, ignore_index=True)
    merged.sort_values(["Number_of_Sites", "Max_Interactions", "Hamiltonian_Index", "Fidelity"],
                       inplace=True, ignore_index=True)
    merged.to_csv(args.output, index=False)

    expected = len(cells) * len(args.fidelities) * 10
    print(f"merged {len(frames)} shards -> {args.output}: {len(merged)} rows "
          f"({expected} expected for a complete run)")
    if partials:
        print(f"note: {len(partials)} of those came from unfinished jobs' .partial "
              f"files (jobs {partials}); only cells with a full set of fidelities "
              f"were taken")

    # Which parts of the grid are actually usable matters more than the row
    # total when a run was cut short, so spell the coverage out per n.
    have = set(merged.set_index(CELL_KEY).index)
    print("coverage per number of sites:")
    for num_sites in sorted(args.num_sites):
        wanted = [c for c in cells if c.num_sites == num_sites]
        done = sum(1 for c in wanted if c in have)
        print(f"  n={num_sites:>3}: {done:>4}/{len(wanted)} cells"
              f"{'  COMPLETE' if done == len(wanted) else ''}")

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
    parser.add_argument("--n-target", type=int, default=None)
    parser.add_argument("--penalty-strength", type=float, default=0.0)
    parser.add_argument("--fidelities", type=float, nargs="+", default=DEFAULT_FIDELITIES)
    parser.add_argument("--num-jobs", type=int, default=20,
                        help="size of the SLURM array")
    parser.add_argument("--job-index", type=int, default=None,
                        help="this task's index; defaults to $SLURM_ARRAY_TASK_ID")
    parser.add_argument("--shard-dir", default="shards")
    parser.add_argument("--output", default="systematic_study_results.csv",
                        help="merge mode: where to write the combined CSV")
    parser.add_argument("--balance", choices=["cost", "stratified"], default="cost",
                        help="how cells are spread over the array; must match "
                             "between plan, run and merge")
    parser.add_argument("--dense-limit", type=int, default=DEFAULT_DENSE_LIMIT,
                        help="Hilbert-space dimension up to which the full dense "
                             "eigendecomposition is computed and handed to SKQD "
                             "(0 disables it)")
    parser.add_argument("--dense", dest="sparse", action="store_false",
                        help="build the Hamiltonian densely instead of in CSR form")
    parser.add_argument("--overwrite", action="store_true",
                        help="re-run a job even if its shard already exists")
    parser.add_argument("--no-resume", dest="resume", action="store_false",
                        help="ignore any .partial file and re-run every cell")
    parser.add_argument("--include-partial", action="store_true",
                        help="merge: also take the finished cells out of the "
                             ".partial files of jobs that never completed")
    parser.set_defaults(sparse=True, resume=True)
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
        plan = describe_plan(cells, args.num_jobs, args.balance)
        print(f"{len(cells)} cells over {args.num_jobs} jobs "
              f"({len(cells) * len(args.fidelities) * 10} rows total, "
              f"balance={args.balance})\n")
        print(plan.to_string(index=False))
        spread = plan["Cells"].max() - plan["Cells"].min()
        print(f"\ncells per job: min {plan['Cells'].min()}, max {plan['Cells'].max()} "
              f"(spread {spread})")
        imbalance = plan["Load"].max() / max(plan["Load"].min(), 1e-9)
        print(f"estimated load: min {plan['Load'].min()}, max {plan['Load'].max()} "
              f"(heaviest job is {imbalance:.2f}x the lightest)")
        for column in [c for c in plan.columns if c.startswith("n=")]:
            values = plan[column]
            if values.max() != values.min():
                print(f"  {column}: uneven by {values.max() - values.min()} "
                      f"(expected under cost balancing -- Load is what is equalised)")
        print(f"\nsbatch --array=0-{args.num_jobs - 1} submit_study.sh")
    elif args.mode == "run":
        run_job(args)
    else:
        merge_shards(args)


if __name__ == "__main__":
    main()
