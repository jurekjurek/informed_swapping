"""
Distributed fidelity-at-fixed-budget study, for a SLURM job array.

This is the counterpart to ClusterStudy.py. There, the target fidelity was fixed
and the question was how many bitstrings each protocol needs to reach it. Here
nothing is fixed: both protocols run until they finish, and the comparison is
the fidelity reached with a given number of unique bitstrings. The fixed-target
study is one contour of this one, and because the Hamiltonians come from the same
``cell_seed`` the two can be checked against each other directly.

Four curves are compared: BARK, the simplified BARK that ranks by the bare
coupling instead of Johann's method, SKQD, and the ceiling. The ceiling is the
fidelity of the *best possible* pool of K states, i.e. the summed probability of
the K largest components of the true ground state -- no protocol can beat it, so
it turns an A-against-B plot into an absolute scale.

Every full run records its fidelity at the pool sizes it happens to pass
through. BARK adds one state per iteration; SKQD adds a random number, because
its shots routinely resample states that are already pooled. The two are
therefore sampled on different grids and their raw points cannot be averaged
against each other. Instead every run is read as a step function -- the fidelity
at the last pool size not exceeding K, see ``fidelity_at_budget`` -- on a common
log-spaced grid of budgets K. That gives each (Hamiltonian, initial state) run
exactly one value per grid point, so the average over Hamiltonians is a plain
equal-weight mean rather than one weighted by how densely SKQD happened to
sample that region.

The grid is log-spaced and the y-axis is the infidelity 1 - F on a log scale,
because everything that decides the comparison happens at K / 2^N << 1 and in
the last digits of the fidelity. Linear axes spend their resolution on the
regime where all protocols have already won. Curves are not only averaged over
everything: the runs are clustered by system size, lattice dimension,
anisotropy, coupling, field, ground-state density, Hamiltonian density and the
overlap of the initial state, and by every combination of those, one folder each. See
``plot_results``.

The unit of work is a "cell": one ``(hamiltonian_index, num_sites, dimensions,
delta, J, Bx, By, Bz)`` combination. A cell builds one Hamiltonian, solves for its
ground state, and then runs all three protocols from every starting state.
Distribution, resuming and merging all work exactly as in ClusterStudy.py and
reuse its machinery -- only the cost model differs, since these runs grow a pool
proportional to the Hilbert space rather than stopping at a target fidelity.

Usage
-----
    # inspect the split before submitting anything
    python EqualNumberOfBitstrings.py plan  --num-hamiltonians 20 --num-jobs 20

    # one array task (this is what the submit script calls)
    python EqualNumberOfBitstrings.py run   --num-hamiltonians 20 --num-jobs 20 --job-index 3

    # collect the shards and draw the figures
    python EqualNumberOfBitstrings.py merge --output equal_bitstrings_results.csv
    python EqualNumberOfBitstrings.py plot  --output equal_bitstrings_results.csv

    # fewer figures: only up to two clustering axes at a time, four bins each
    python EqualNumberOfBitstrings.py plot  --max-combination 2 --num-bins 4

Figures
-------
``plot`` writes one folder per way of clustering the runs, and one subfolder per
cluster inside it, e.g. ``by_N_overlap/N-10_overlap-high/curves_sem.pdf``. Each
holds the fidelity curves and the ratio against SKQD, once with the spread of
the runs, once with the error of the mean and once as the median with its
interquartile range (``_median``), plus where the protocols stopped
by themselves. Whenever the ceiling drops off a cliff -- which it does as soon as
the pool covers the support of the ground state, taking the y-axis down with it
-- a second copy of each curve stopping in front of that jump is written next to
it, with a ``_zoom`` suffix. Next to those, ``distribution_overlap.pdf`` and
its counterparts for the two densities turn the bin names back into numbers:
the histogram of the quantity over all runs, the quantile cuts as labelled
vertical lines, and the band the folder keeps highlighted in the bars.
``plot_index.csv`` lists every cluster with the number of runs behind it, and
``bin_edges.csv`` every cut as a number.
"""

import argparse
import hashlib
import itertools
import os
import sys
import time
from collections import defaultdict, namedtuple
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse.linalg import eigsh

import ClusterStudy
from RandomSpinModel import make_heisenberg_hamiltonian
from BARK import BARK
from SKQD import SKQD, fidelity_at_budget
from ClusterStudy import (DEFAULT_DENSE_LIMIT, assign_all_cells, assign_cells,
                          describe_plan, shard_path, solve_ground_state)

DEFAULT_NUM_SITES = [6, 8, 10, 12]
DEFAULT_DIMENSIONS = [1, 2]
DEFAULT_DELTAS = [0.0, 0.5, 1.0, 10.0, 100.0]

# Seed of the draw of J and B_z. These are sampled rather than swept, and they
# have to be sampled *once* for the whole study: plan, run and merge each
# enumerate the cells in their own process, so drawing them from the global RNG
# would hand every one of the three a different set of Hamiltonians -- a split
# that is printed but not executed, a resume that never recognises its own
# rows, and a coverage report on cells that were never asked for.
DEFAULT_COUPLING_SEED = 20250921

# Decimals the drawn couplings are rounded to. They are part of the cell key, so
# they have to survive a round trip through a CSV: ``to_csv`` writes floats at
# about 16 significant digits, which drops the last bit of a full double and
# leaves a resumed job -- and merge's coverage report -- unable to recognise the
# rows it wrote itself. Six decimals round-trip exactly and are far finer than
# anything the study resolves.
COUPLING_DECIMALS = 6

# Initial states per Hamiltonian, drawn uniformly at random -- without
# replacement -- from every basis state the ground state has non-zero overlap
# with. Unlike ClusterStudy.run_cell, which takes both ends of the overlap
# distribution, this samples the distribution itself, so an average over the
# initial states is an average over a typical starting point instead of over a
# mixture of two tails.
DEFAULT_NUM_INITIAL_STATES = 6

# What counts as non-zero overlap, as a probability. The zeros that matter are
# the exact ones of the symmetry sectors; an eigensolver leaves those at about
# 1e-16 in the amplitude, i.e. 1e-32 here, so anything well below the smallest
# physically meaningful weight and well above that separates the two.
OVERLAP_FLOOR = 1e-12

# Repeats of the SKQD trajectory per run. BARK is deterministic and needs none.
DEFAULT_N_REPEATS = 3

# Budget grid: this many log-spaced integers from one bitstring up to
# ``max_fraction`` of the Hilbert space. Running truly to the end of the Hilbert
# space costs 2^N growing diagonalizations per BARK run, and the last decade is
# the regime in which a subspace method has lost anyway, so the runs are capped.
DEFAULT_GRID_POINTS = 30
DEFAULT_MAX_FRACTION = 0.5

COLUMNS = ["Hamiltonian_Index", "Number_of_Sites", "Dimensions", "Delta", "J", "Bx", "By", "Bz",
           "Ground_State_Density", "Hamiltonian_Density", "Gap", "Overlap",
           "Initial_State_Index", "Algorithm", "Budget", "Budget_Fraction", "Fidelity",
           "Final_Pool_Size", "Final_Pool_Fraction", "Terminated_Early", "T", "N_Shots",
           "Seed"]

CELL_KEY = ["Hamiltonian_Index", "Number_of_Sites", "Dimensions", "Delta", "J", "Bx", "By", "Bz"]
# What identifies one curve: one protocol on one initial state of one Hamiltonian.
RUN_KEY = CELL_KEY + ["Initial_State_Index"]
PANEL_KEY = ["Number_of_Sites", "Dimensions", "Delta", "J", "Bx", "By", "Bz"]

# Categorical slots of the validated default palette, plus grey for the ceiling.
ALGORITHMS = (
    # name                colour     marker  linestyle
    ("BARK",              "#2a78d6", "o",    "-"),
    ("Simplified BARK",   "#3aa66f", "^",    "-."),
    ("SKQD",              "#eb6834", "s",    "--"),
    ("Ceiling",           "#52514e", None,   ":"),
)
REFERENCE = "SKQD"          # denominator of the ratio plots

GRID_KWARGS = dict(color="#c9c8c3", linewidth=0.6, alpha=0.7)
# Fidelities of exactly 1 do occur once the pool spans the support of the ground
# state, and log(0) is not plottable, so the infidelity is floored.
INFIDELITY_FLOOR = 1e-16

# A ground state whose largest component carries at least this much weight is
# a single basis state, see ``drop_trivial_ground_states``.
TRIVIAL_GROUND_STATE = 1.0 - 1e-10

# Cells whose gap E_1 - E_0 lies below this are dropped from the plots, see
# ``drop_small_gaps``. The gaps of the study do not separate cleanly into two
# groups, so this is a choice: it removes the quasi-degenerate Neel pairs of
# the large anisotropies and little else.
MIN_GAP = 1e-4

OUTPUT_ROOT = Path("equal_bitstrings_plots_heisenberg")


# --------------------------------------------------------------------------- #
# Work definition and distribution
# --------------------------------------------------------------------------- #

# ClusterStudy's Cell -- and with it its cell_seed -- names a cell by its
# maximum number of interactions, which a uniform XXZ Hamiltonian does not have.
# Both are redefined here rather than changed there, so the fixed-target study
# keeps running unchanged; everything else imported from it only ever reads
# ``num_sites`` off a cell, or hands it to the cost model patched in below.
Cell = namedtuple("Cell", ["hamiltonian_index", "num_sites", "dimensions", "delta",
                           "J", "Bx", "By", "Bz"])


def cell_seed(cell: Cell) -> int:
    """
    Deterministic 32-bit seed derived from the cell's identity.

    SHA-256 rather than ``hash()``, as in ClusterStudy.cell_seed: Python salts
    string hashing per process, so ``hash()`` would give different initial
    states in different jobs for the same cell.

    The couplings go into the key through ``repr``, which writes a float
    back exactly. They are drawn from a continuous distribution, so a key that
    rounded them further would hand two different Hamiltonians the same seed.
    """
    key = "|".join(repr(field) for field in cell).encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "little")


def estimate_cost(cell: Cell) -> float:
    """
    Relative cost of a cell, for load balancing only; the units are irrelevant.

    ClusterStudy.py measured ``dimension ** 2.3`` for the fixed-target study, but
    that exponent does not carry over: those runs stopped at F = 0.9 and never
    grew a pool proportional to the Hilbert space. Here every run goes to the
    same *fraction* of it, so a run pays a growing diagonalization at each of
    O(dimension) steps -- between ``dimension ** 3`` if BARK's warm-started
    LOBPCG keeps converging and ``dimension ** 4`` if it falls back to exact
    solves. The middle of that range is the safer guess for a split.

    This is a scaling argument, not a measurement. The first production array
    prints a wall time per cell; recalibrate from those, as was done for the
    other study.
    """
    dimension = 2.0 ** cell.num_sites
    # Nearest neighbours only, so the bond count follows from the geometry: a
    # chain has just under one bond per site, the open rectangle just under two.
    bonds_per_site = 1.0 if cell.dimensions == 1 else 2.0
    return dimension ** 3.5 * (1.0 + 0.8 * (bonds_per_site - 1))


# ClusterStudy's splitter weights cells with ClusterStudy's cost model, and it
# takes it from its own module global. Redirect that at ours here, at import
# time, so plan, run and merge all agree on the split -- patching it in only one
# of them would print one split and execute another. The two studies are never
# in the same process, so nothing else is affected.
ClusterStudy.estimate_cost = estimate_cost

def enumerate_cells(num_hamiltonians, num_sites, dimensions, deltas,
                    coupling_seed=DEFAULT_COUPLING_SEED):
    """All cells of the study, in a fixed order independent of how they are split."""

    # Sample J and Bz uniformly in [-1, 1] for num_hamiltonian times. Through
    # generators of their own rather than the global RNG: the draw has to come
    # out the same in plan, run and merge (see DEFAULT_COUPLING_SEED), and the
    # global one is also what SKQD shoots with, so seeding it here would move
    # every trajectory of the study as well.
    #
    # One generator per Hamiltonian index rather than one vector draw, so that
    # a Hamiltonian's couplings depend on its index alone. Drawing the whole
    # vector at once would give Hamiltonian 0 different couplings as soon as
    # ``num_hamiltonians`` changes, i.e. raising it to extend a study would
    # quietly turn every shard already on disk into a shard of another study.
    couplings = [np.round(np.random.default_rng([coupling_seed, index]).uniform(-1, 1, 2),
                          COUPLING_DECIMALS)
                 for index in range(num_hamiltonians)]

    # Everything is stored as a float, including the two field components that
    # are always zero, so that a cell compares equal to the row it wrote once
    # that row has been through a CSV -- which is what the resume check and the
    # coverage report of merge are.
    return [Cell(hamiltonian_index, n_sites, dim, float(delta),
                 float(J), 0.0, 0.0, float(Bz))
            for hamiltonian_index, (J, Bz) in enumerate(couplings)
            for n_sites in num_sites
            for dim in dimensions
            for delta in deltas
            ]


def study_cells(args) -> list:
    """
    All cells of the study, in a fixed order independent of how they are split.

    ``enumerate_cells`` draws J and B_z itself, from a fixed seed rather than
    from the CLI, so that plan, run and merge all see the same Hamiltonians
    without the couplings having to be passed around as arguments.
    """
    return enumerate_cells(args.num_hamiltonians, args.num_sites,
                           args.dimensions, args.deltas)


# --------------------------------------------------------------------------- #
# Running a cell
# --------------------------------------------------------------------------- #

def budget_grid(dimension: int, grid_points: int, max_fraction: float) -> np.ndarray:
    """Log-spaced integer budgets from one bitstring up to a fraction of the Hilbert space."""
    grid = np.logspace(0.0, np.log10(max_fraction * dimension), grid_points)
    return np.unique(np.round(grid).astype(int))


def with_initial_point(pool_sizes: np.ndarray, fidelities: np.ndarray,
                       overlap: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Prepend the pool of size one to a recorded trajectory.

    Both protocols only record after they have added a state, so neither reports
    anything at a budget of one bitstring -- even though a one-state pool is a
    perfectly well defined subspace whose ground state is that basis state, with
    fidelity equal to its overlap. Without this the first grid point reads as a
    fidelity of zero for every run.
    """
    if pool_sizes.size and pool_sizes[0] <= 1:
        return pool_sizes, fidelities
    return (np.concatenate(([1], pool_sizes)), np.concatenate(([overlap], fidelities)))


def rows_per_initial_state(cell: Cell, args) -> int:
    """
    How many rows one initial state of a cell contributes.

    This, rather than the total, is what a ``.partial`` is checked against: it
    depends on the grid, so a partial written under a different grid or a
    different cap fails the check and is redone rather than mixed in, while a
    cell that simply had fewer initial states to draw still passes. See
    ``load_partial``.
    """
    budgets = budget_grid(2 ** cell.num_sites, args.grid_points, args.max_fraction)
    return len(ALGORITHMS) * budgets.size


def expected_rows(cell: Cell, args) -> int:
    """
    How many rows a finished cell contributes at most.

    A ground state supported on fewer than ``num_initial_states`` basis states
    contributes fewer. The Ising-like anisotropies of this family, and every
    coupling the field dominates, give a ground state of one or two components,
    so this is an upper bound on the row count and not the row count itself; it
    is used to say how large a complete run would be, never to decide whether a
    cell is finished.
    """
    return args.num_initial_states * rows_per_initial_state(cell, args)


def run_cell(cell: Cell, args) -> list:
    """
    Run all three protocols on one Hamiltonian, for every initial state.

    The Hamiltonian, the eigendecomposition and the protocol objects are built
    once for the whole cell, as in ClusterStudy.run_cell.

    Returns the cell's rows as a list of dicts. The termination of a run is
    repeated on each of its rows rather than kept in a second table: it costs a
    few duplicated columns and saves a second shard, a second merge and a second
    resume check.
    """
    seed = cell_seed(cell)
    # SKQD draws its shots through the global RNG, so seed that as well.
    np.random.seed(seed)

    hamiltonian = make_heisenberg_hamiltonian(
        num_sites=cell.num_sites,
        dimension=cell.dimensions,
        delta=cell.delta,
        J=cell.J,
        h = (cell.Bx, cell.By, cell.Bz),
        spin = 1
    )[0].to_matrix(sparse=True)

    ground_state, all_eigenvalues, all_eigenvectors = solve_ground_state(
        hamiltonian, args.dense_limit)
    probabilities = np.abs(ground_state) ** 2

    dimension = hamiltonian.shape[0]
    ground_state_density = np.count_nonzero(np.abs(ground_state) > 1e-3) / dimension
    hamiltonian_density = hamiltonian.nnz / dimension ** 2

    # Gap above the ground state, so that the plots can drop the cells where the
    # ground state is (quasi-)degenerate and its fidelity is ill-defined.
    if all_eigenvalues is not None:
        gap = float(all_eigenvalues[1] - all_eigenvalues[0])
    else:
        lowest = np.sort(eigsh(hamiltonian, k=2, which="SA")[0])
        gap = float(lowest[1] - lowest[0])

    budgets = budget_grid(dimension, args.grid_points, args.max_fraction)
    max_pool_size = int(budgets[-1])

    # The best pool of K states is the K most probable ones, so the ceiling is a
    # cumulative sum of the sorted probabilities. It bounds *any* protocol at
    # pool size K: the fidelity of a vector supported on the pool cannot exceed
    # the probability mass of the ground state that the pool covers.
    ceiling = np.cumsum(np.sort(probabilities)[::-1])

    bark_protocol = BARK(hamiltonian)
    skqd_protocol = SKQD(hamiltonian, eigenvalues=all_eigenvalues,
                         eigenvectors=all_eigenvectors)

    # Initial states: a uniform sample, without replacement, from the basis
    # states the ground state actually has weight on. A state of exactly zero
    # overlap is not a starting point at all -- BARK has nothing to rank from it
    # and SKQD's first shot is uncorrelated with the target -- and the symmetry
    # sectors of these Hamiltonians make a large part of the basis exactly that.
    # Drawing uniformly from the rest makes the overlap column a sample of the
    # true overlap distribution of the Hamiltonian, so the plots can be read as
    # statements about a typical starting point; the earlier extremes-only choice
    # sampled two tails that no run would ever draw by itself.
    # ``rng`` is separate from the global RNG so that changing this selection
    # leaves SKQD's shots, which come from the global one, exactly as they were.
    rng = np.random.default_rng(seed)
    non_zero_indices = np.flatnonzero(probabilities > OVERLAP_FLOOR)
    indices_to_test = rng.choice(non_zero_indices,
                                 size=min(args.num_initial_states, non_zero_indices.size),
                                 replace=False)

    rows = []
    for initial_state_index in indices_to_test:
        overlap = float(probabilities[initial_state_index])
        # name -> (fidelity per budget, mean final pool size)
        results = {}

        for name, run in (("BARK", bark_protocol.full_bark_run),
                          ("Simplified BARK", bark_protocol.simplified_bark_run)):
            pool_sizes, fidelities = run(ground_state, initial_state_index,
                                         max_pool_size=max_pool_size)
            final_pool_size = float(pool_sizes[-1]) if pool_sizes.size else 1.0
            pool_sizes, fidelities = with_initial_point(pool_sizes, fidelities, overlap)
            results[name] = (fidelity_at_budget(pool_sizes, fidelities, budgets),
                             final_pool_size)

        # (t, n_shots) are tuned once per run, for the mean infidelity along the
        # whole budget grid, and then held fixed along the whole curve. Tuning
        # for the largest budget alone chose on a point where every setting has
        # already exhausted the symmetry sector. Re-optimizing at every budget
        # would draw an envelope over hyperparameters that no single SKQD run
        # realizes. The evaluation repeats below draw fresh shots -- the
        # optimizer's own repeats have already advanced the global RNG -- so the
        # reported curve is not the noise the optimum was selected on.
        t, n_shots = skqd_protocol.optimize_general(
            initial_state_index, ground_state, max_pool_size=max_pool_size,
            n_repeats=args.n_repeats, budgets=budgets)
        runs = [skqd_protocol.full_skqd_run(initial_state_index, t, n_shots,
                                            ground_state, max_pool_size=max_pool_size,
                                            budgets=budgets)
                for _ in range(args.n_repeats)]
        results["SKQD"] = (
            np.mean([fidelity_at_budget(*with_initial_point(pool_sizes, fidelities, overlap),
                                        budgets)
                     for pool_sizes, fidelities in runs], axis=0),
            float(np.mean([pool_sizes[-1] if pool_sizes.size else 1
                           for pool_sizes, _ in runs])),
        )

        results["Ceiling"] = (ceiling[budgets - 1], float(max_pool_size))

        for name, (fidelities, final_pool_size) in results.items():
            for budget, fidelity in zip(budgets, fidelities):
                rows.append({
                    "Hamiltonian_Index": cell.hamiltonian_index,
                    "Number_of_Sites": cell.num_sites,
                    "Dimensions": cell.dimensions,
                    "Delta": cell.delta,
                    "J": cell.J,
                    "Bx": cell.Bx,
                    "By": cell.By,
                    "Bz": cell.Bz,
                    "Ground_State_Density": ground_state_density,
                    "Hamiltonian_Density": hamiltonian_density,
                    "Gap": gap,
                    "Overlap": overlap,
                    "Initial_State_Index": int(initial_state_index),
                    "Algorithm": name,
                    "Budget": int(budget),
                    "Budget_Fraction": budget / dimension,
                    "Fidelity": float(fidelity),
                    "Final_Pool_Size": final_pool_size,
                    "Final_Pool_Fraction": final_pool_size / dimension,
                    # A run that used up the budget was cut off by us, not by
                    # itself; only a shorter one ran out of states to add.
                    "Terminated_Early": final_pool_size < max_pool_size,
                    # SKQD's tuned hyperparameters; the other curves have none.
                    "T": t if name == "SKQD" else np.nan,
                    "N_Shots": n_shots if name == "SKQD" else np.nan,
                    "Seed": seed,
                })
    return rows


# --------------------------------------------------------------------------- #
# Shards
# --------------------------------------------------------------------------- #

def load_partial(path, cells, args):
    """
    Rows already written by an earlier, interrupted run of this job.

    A cell counts as finished only if the partial holds its full set of rows
    *and* it is still part of the study, i.e. it appears in ``cells``. Without
    the membership check a re-submission with a changed grid would silently mix
    two studies into one shard; checking membership rather than comparing the
    requested settings is what makes a genuine resume still work.

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

    counts = frame.groupby(CELL_KEY).size()
    wanted = {(cell.hamiltonian_index, cell.num_sites, cell.dimensions, cell.delta,
               cell.J, cell.Bx, cell.By, cell.Bz): rows_per_initial_state(cell, args)
              for cell in cells}

    # A cell is written in one piece -- run_cell hands back all of its rows at
    # once and the partial is rewritten after every cell -- so a cell holding
    # fewer rows than a full set is not an interrupted one, it is a cell whose
    # ground state lived on fewer basis states than ``num_initial_states``.
    # Matching on the block one initial state writes therefore still rejects a
    # partial written under a different grid, which is the point of the check,
    # without re-running every near-product Hamiltonian on every resume.
    finished = set()
    for key, count in counts.items():
        block = wanted.get((int(key[0]), int(key[1]), int(key[2]),
                            float(key[3]), float(key[4]), float(key[5]),
                            float(key[6]), float(key[7])))
        if block and count % block == 0 and count <= block * args.num_initial_states:
            finished.add(key)

    dropped = len(counts) - len(finished)
    if dropped:
        print(f"  ignoring {dropped} cell(s) in {path}: incomplete, or not part of "
              f"this job under the current settings", flush=True)
    if not finished:
        return [], set()

    keep = frame.set_index(CELL_KEY).index.isin(finished)
    return frame[keep][COLUMNS].to_dict("records"), finished


def run_job(args):
    """Run every cell assigned to this job and write one shard."""
    cells = study_cells(args)
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
        rows, finished = load_partial(destination + ".partial", mine, args)
        if finished:
            print(f"[job {args.job_index}] resuming: {len(finished)} of {len(mine)} "
                  f"cells already in {destination}.partial", flush=True)

    started = time.time()
    for position, cell in enumerate(mine, start=1):
        label = (f"n={cell.num_sites} dimensions={cell.dimensions} delta={cell.delta} J={cell.J} "
                 f"Bx={cell.Bx} By={cell.By} Bz={cell.Bz} "
                 f"ham={cell.hamiltonian_index}")
        if (cell.hamiltonian_index, cell.num_sites, cell.dimensions, cell.delta, cell.J, cell.Bx, cell.By, cell.Bz) in finished:
            print(f"[job {args.job_index}] {position}/{len(mine)} {label} "
                  f"already done, skipping", flush=True)
            continue

        cell_started = time.time()
        rows.extend(run_cell(cell, args))
        print(f"[job {args.job_index}] {position}/{len(mine)} {label} "
              f"took {time.time() - cell_started:.1f}s "
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
    cells = study_cells(args)
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
            rows, finished = load_partial(path + ".partial", jobs[job_index], args)
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
    merged.sort_values(RUN_KEY + ["Algorithm", "Budget"], inplace=True, ignore_index=True)
    merged.to_csv(args.output, index=False)

    expected = sum(expected_rows(cell, args) for cell in cells)
    print(f"merged {len(frames)} shards -> {args.output}: {len(merged)} rows "
          f"({expected} expected for a complete run)")
    if partials:
        print(f"note: {len(partials)} of those came from unfinished jobs' .partial "
              f"files (jobs {partials}); only complete cells were taken")

    # Which parts of the grid are usable matters more than the row total when a
    # run was cut short, so spell the coverage out per n.
    have = set(merged.set_index(CELL_KEY).index)
    print("coverage per number of sites:")
    for num_sites in sorted(args.num_sites):
        wanted = [c for c in cells if c.num_sites == num_sites]
        done = sum(1 for c in wanted
                   if (c.hamiltonian_index, c.num_sites, c.dimensions, c.delta, c.J, c.Bx, c.By, c.Bz) in have)
        print(f"  n={num_sites:>3}: {done:>4}/{len(wanted)} cells"
              f"{'  COMPLETE' if done == len(wanted) else ''}")

    if missing:
        print(f"WARNING: {len(missing)} shards missing (jobs {missing}); "
              f"re-submit those array indices, then merge again")


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #

def drop_trivial_ground_states(data: pd.DataFrame) -> pd.DataFrame:
    """
    Remove every cell whose ground state is a single basis state.

    There the only initial state of non-zero overlap is the ground state itself,
    so every protocol has F = 1 with one bitstring and the cell compares nothing.
    These are the ferromagnetic cells with delta >= 1 and the ones whose field
    saturates the magnetization. The ceiling at a budget of one bitstring is the
    weight of the largest component, so it identifies them exactly.
    """
    ceiling = data[(data["Algorithm"] == "Ceiling") & (data["Budget"] == 1)]
    trivial = ceiling.loc[ceiling["Fidelity"] >= TRIVIAL_GROUND_STATE, CELL_KEY].drop_duplicates()
    keep = ~data.set_index(CELL_KEY).index.isin(trivial.set_index(CELL_KEY).index)
    print(f"Dropped {len(trivial)} of {len(data.drop_duplicates(CELL_KEY))} cells "
          f"with a trivial ground state")
    return data[keep].copy()


def drop_small_gaps(data: pd.DataFrame) -> pd.DataFrame:
    """
    Remove every cell whose ground state is (quasi-)degenerate.

    At large anisotropies the two Neel-like states are split only by high-order
    tunnelling. The fidelity with one eigenvector is then ill-defined: a pool
    holding one of the two Neel sectors has an essentially exact energy, but a
    fidelity near 1/2. Results written before the ``Gap`` column existed are
    left untouched.
    """
    if "Gap" not in data.columns:
        print("No Gap column in the data, skipping the gap filter")
        return data
    cells = data.drop_duplicates(CELL_KEY)
    small = cells.loc[cells["Gap"] < MIN_GAP, CELL_KEY]
    keep = ~data.set_index(CELL_KEY).index.isin(small.set_index(CELL_KEY).index)
    print(f"Dropped {len(small)} of {len(cells)} cells with a gap below {MIN_GAP:g}")
    return data[keep].copy()


def add_log_infidelity(curves: pd.DataFrame) -> pd.DataFrame:
    """Add log10(1 - F), which is what the averages are taken in."""
    infidelity = np.clip(1.0 - curves["Fidelity"], INFIDELITY_FLOOR, 1.0)
    curves["Log_Infidelity"] = np.log10(infidelity)
    return curves


def on_common_grid(group: pd.DataFrame) -> pd.DataFrame:
    """
    Read every run of a cluster at every budget fraction the cluster contains.

    Budgets are whole bitstrings and the axis is K / 2^N, so two system sizes
    share a grid point only by accident. Grouping the recorded rows by
    Budget_Fraction therefore averages a *different subset of the runs* at every
    point as soon as a cluster mixes system sizes, which shows up as a zigzag and
    is not a property of the protocols. Here each run is instead read as the step
    function it is -- its value at the last budget it recorded that does not
    exceed the grid point, the same reading ``fidelity_at_budget`` performs when
    the runs are recorded -- on the union of the cluster's budget fractions.

    A run contributes nothing below its own first budget and is not extrapolated
    backwards: a fraction of 10^-3 is less than one bitstring of a 2^6
    dimensional space, and no reading of that run can invent one. Clusters that
    do fix the system size are untouched, since there the union is exactly the
    grid every run was already recorded on.
    """
    wide = group.pivot_table(index="Budget_Fraction", columns=RUN_KEY + ["Algorithm"],
                             values="Log_Infidelity").ffill()
    curves = wide.melt(ignore_index=False, value_name="Log_Infidelity").reset_index()
    return curves.dropna(subset=["Log_Infidelity"])


def aggregate(frame: pd.DataFrame, column: str, error: str) -> pd.DataFrame:
    """
    Mean of ``column`` per (algorithm, budget), with a spread of ``error``.

    Averaging is done in the log of the infidelity: it runs over several decades,
    so the arithmetic mean of the fidelity would be dominated by whichever run
    happens to be furthest from converged, and a symmetric band around it would
    leave the interval (0, 1] that the fidelity lives in.

    ``error="std"`` is the spread of the runs -- how much the protocols differ
    from Hamiltonian to Hamiltonian. ``error="sem"`` is the uncertainty of the
    plotted mean itself, which is the one to read when asking whether two curves
    are actually distinguishable.

    ``error="median"`` replaces the mean by the median and the band by the
    interquartile range. Runs that reach F = 1 to machine precision all sit at
    INFIDELITY_FLOOR, an arbitrary constant that the mean depends on strongly
    and the median does not. The median is still stored in the ``mean`` column,
    so the plotting code reads every kind of average the same way.
    """
    groups = frame.groupby(["Algorithm", "Budget_Fraction"])[column]
    if error == "median":
        stats = groups.agg(["median", "count"]).reset_index().rename(columns={"median": "mean"})
        stats["lower"] = groups.quantile(0.25).to_numpy()
        stats["upper"] = groups.quantile(0.75).to_numpy()
        return stats

    stats = groups.agg(["mean", "std", "count"]).reset_index()
    # A single sample has no spread rather than an undefined one.
    stats["std"] = stats["std"].fillna(0.0)
    spread = stats["std"] if error == "std" else stats["std"] / np.sqrt(stats["count"])
    stats["lower"] = stats["mean"] - spread
    stats["upper"] = stats["mean"] + spread
    return stats


def ratio_frame(curves: pd.DataFrame) -> pd.DataFrame:
    """
    Log infidelity of every algorithm relative to REFERENCE, run by run.

    The ratio is formed within a run before averaging, so the reference and the
    algorithm are compared on the same Hamiltonian and the same initial state.
    Dividing the two ensemble means instead would compare averages taken over
    different mixtures whenever a run is missing from one of them.
    """
    keys = RUN_KEY + ["Budget_Fraction"]
    reference = curves[curves["Algorithm"] == REFERENCE][keys + ["Log_Infidelity"]]
    ratios = curves.merge(reference, on=keys, suffixes=("", "_Reference"))
    ratios["Log_Ratio"] = ratios["Log_Infidelity"] - ratios["Log_Infidelity_Reference"]
    return ratios


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #

def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def style_axes(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(True, which="both", **GRID_KWARGS)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(frameon=False, fontsize=9)


def _band_label(error: str) -> str:
    if error == "median":
        return "interquartile range, line: median"
    return "standard deviation" if error == "std" else "standard error"


def _annotate_counts(ax: plt.Axes, stats: pd.DataFrame) -> None:
    """State how many runs entered each average."""
    counts = stats.groupby("Algorithm")["count"].agg(["min", "max"])
    # Below their own first budget the runs of the smaller systems are absent,
    # so a cluster that mixes system sizes has a range rather than one number.
    text = ", ".join(f"{name}: {int(row['min'])}" if row["min"] == row["max"]
                     else f"{name}: {int(row['min'])}-{int(row['max'])}"
                     for name, row in counts.iterrows())
    ax.text(0.0, -0.22, f"runs per point -- {text}", transform=ax.transAxes,
            fontsize=7, color="#52514e", va="top")


def plot_curves(group: pd.DataFrame, title: str, path: Path, error: str) -> None:
    """Infidelity against budget, one line per algorithm plus the ceiling."""
    stats = aggregate(group, "Log_Infidelity", error)

    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    for name, colour, marker, linestyle in ALGORITHMS:
        rows = stats[stats["Algorithm"] == name].sort_values("Budget_Fraction")
        if rows.empty:
            continue
        mean = rows["mean"].to_numpy()
        ax.plot(rows["Budget_Fraction"], 10.0 ** mean, marker=marker, markersize=5,
                linewidth=2, linestyle=linestyle, color=colour,
                markeredgecolor="white", markeredgewidth=0.8, label=name)
        ax.fill_between(rows["Budget_Fraction"], 10.0 ** rows["lower"].to_numpy(),
                        10.0 ** rows["upper"].to_numpy(), color=colour, alpha=0.12, linewidth=0)

    style_axes(ax, f"{title}\n(band: {_band_label(error)})",
               "Number of bitstrings K / Hilbert space dimension", "Infidelity 1 - F")
    _annotate_counts(ax, stats)
    save_figure(fig, path)


def plot_ratio(group: pd.DataFrame, title: str, path: Path, error: str) -> None:
    """Infidelity relative to REFERENCE on the same runs; the reference is 1 by construction."""
    stats = aggregate(ratio_frame(group), "Log_Ratio", error)

    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    for name, colour, marker, linestyle in ALGORITHMS:
        rows = stats[stats["Algorithm"] == name].sort_values("Budget_Fraction")
        if rows.empty or name == REFERENCE:
            continue
        mean = rows["mean"].to_numpy()
        ax.plot(rows["Budget_Fraction"], 10.0 ** mean, marker=marker, markersize=5,
                linewidth=2, linestyle=linestyle, color=colour,
                markeredgecolor="white", markeredgewidth=0.8, label=name)
        ax.fill_between(rows["Budget_Fraction"], 10.0 ** rows["lower"].to_numpy(),
                        10.0 ** rows["upper"].to_numpy(), color=colour, alpha=0.12, linewidth=0)

    ax.axhline(1.0, color=dict((n, c) for n, c, *_ in ALGORITHMS)[REFERENCE],
               linewidth=2, label=REFERENCE)
    style_axes(ax, f"{title}\n(band: {_band_label(error)})",
               "Number of bitstrings K / Hilbert space dimension",
               f"Infidelity relative to {REFERENCE}")
    _annotate_counts(ax, stats)
    save_figure(fig, path)


def plot_termination(group: pd.DataFrame, title: str, path: Path) -> None:
    """
    Where the protocols stop when they stop by themselves.

    A run that reached the budget was cut off by the study, so only the early
    terminators are averaged here -- BARK when the states reachable from the
    initial state are exhausted, SKQD when an iteration draws nothing new. The
    share of runs that got there at all is printed on top of each bar, because a
    mean over a handful of runs says little on its own.
    """
    # One row per run: the termination is repeated on all of a run's curve rows.
    group = group.drop_duplicates(RUN_KEY + ["Algorithm"])

    names, values, spreads, colours, labels = [], [], [], [], []
    for name, colour, *_ in ALGORITHMS:
        if name == "Ceiling":
            continue   # not a run, it has no termination
        rows = group[group["Algorithm"] == name]
        early = rows[rows["Terminated_Early"]]
        names.append(name)
        colours.append(colour)
        values.append(early["Final_Pool_Fraction"].mean() if len(early) else 0.0)
        spreads.append(early["Final_Pool_Fraction"].std(ddof=1) if len(early) > 1 else 0.0)
        labels.append(f"{len(early)}/{len(rows)}")

    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    positions = np.arange(len(names))
    bars = ax.bar(positions, values, yerr=spreads, color=colours, width=0.6,
                  capsize=4, error_kw=dict(elinewidth=1.2))

    # Clear of the error bar, so the share is readable next to the mean.
    top = max([value + spread for value, spread in zip(values, spreads)] or [1.0]) or 1.0
    for bar, spread, label in zip(bars, spreads, labels):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + spread + 0.02 * top,
                label, ha="center", va="bottom", fontsize=8, color="#52514e")

    ax.set_xticks(positions, names)
    ax.set_ylabel("Pool size at early termination / Hilbert space dimension")
    ax.set_title(f"{title}\n(bars: mean over the runs that terminated early; "
                 f"labels: how many did)", fontsize=10)
    ax.set_ylim(0.0, top * 1.2)
    ax.grid(True, axis="y", **GRID_KWARGS)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    save_figure(fig, path)


# --------------------------------------------------------------------------- #
# Clustering the runs
# --------------------------------------------------------------------------- #

# The axes the runs can be clustered by, in the order the folder names list
# them. ``binned`` says whether the column is continuous and has to be cut into
# bins before it can be grouped on.
FACETS = (
    # short name  column                  binned  name in the titles
    ("N",         "Number_of_Sites",      False,  "N"),
    ("dim",       "Dimensions",           False,  "lattice dimension"),
    ("delta",     "Delta",                False,  "anisotropy delta"),
    ("J",         "J",                    True,   "coupling J"),
    ("Bz",        "Bz",                   True,   "field B_z"),
    ("gsdens",    "Ground_State_Density", True,   "ground-state density"),
    ("hamdens",   "Hamiltonian_Density",  True,   "Hamiltonian density"),
    ("overlap",   "Overlap",              True,   "initial overlap"),
)

DEFAULT_NUM_BINS = 3
# Below this many runs a cluster is skipped instead of drawn -- see plot_results.
DEFAULT_MIN_RUNS = 5
# Decades the ceiling has to lose in one grid step to count as a jump.
DEFAULT_CEILING_JUMP = 2.0


def bin_names(count: int) -> list:
    """Readable names for ``count`` quantile bins."""
    if count == 2:
        return ["low", "high"]
    if count == 3:
        return ["low", "mid", "high"]
    return [f"q{i + 1}" for i in range(count)]


def quantile_edges(values: pd.Series, num_bins: int) -> np.ndarray:
    """
    Equal-frequency bin edges of ``values``.

    Equal frequency rather than equal width: the densities, and the overlaps far
    more so, are spread over decades -- the initial states are sampled from the
    overlap distribution of the ground state, which is itself heavy-tailed --
    and equal-width bins would leave nearly every run in a single one. Repeated
    values share an edge, which is dropped, so a column with few distinct values
    comes out with fewer bins rather than with ties split between them.
    """
    quantiles = np.linspace(0.0, 1.0, num_bins + 1)
    return np.unique(np.quantile(values.to_numpy(dtype=float), quantiles))


def apply_bins(values: pd.Series, edges: np.ndarray) -> pd.Series:
    """Name the bin every value falls in."""
    if edges.size < 3:
        # A single bin: the column does not vary here, so it clusters nothing.
        return pd.Series("all", index=values.index)
    return pd.cut(values, edges, labels=bin_names(edges.size - 1),
                  include_lowest=True).astype(str)


def add_facet_labels(data: pd.DataFrame, runs: pd.DataFrame, num_bins: int,
                     scope: str) -> tuple:
    """
    Add one label column per facet; return their names and the edges cut on.

    The edges come back keyed by ``(short name, number of sites)`` -- with
    ``None`` in place of the number of sites under ``scope="global"``, where one
    cut covers the whole study -- so the figures can say in numbers what a
    folder called ``overlap-low`` actually holds. See plot_bin_histogram.

    The edges come from the runs, not from the rows: a run contributes one row
    per algorithm and budget and the number of budgets grows with the system
    size, so cutting on the rows would let the large systems decide where the
    bins of the small ones lie.

    ``scope="per-N"`` bins within each system size. Both densities and the
    overlaps shrink with N -- a ground state spread over 5% of a 2^12 dimensional
    space is a wide one -- so global bins would largely re-sort the runs by N and
    the density folders would be showing the size dependence again under another
    name. ``scope="global"`` is the literal reading, for when the absolute value
    is what matters.
    """
    labels, edges = [], {}
    for name, column, binned, _ in FACETS:
        label = f"Facet_{name}"
        if not binned:
            # ``%g`` rather than ``astype(int)``: the anisotropies are not all
            # integers, and truncating them would put delta = 0.5 and delta = 0
            # into the same folder while still calling it delta-0.
            data[label] = f"{name}-" + data[column].astype(float).map("{:g}".format)
        elif scope == "per-N":
            data[label] = ""
            for num_sites, block in data.groupby("Number_of_Sites"):
                cut = quantile_edges(runs.loc[runs["Number_of_Sites"] == num_sites, column],
                                     num_bins)
                edges[(name, int(num_sites))] = cut
                data.loc[block.index, label] = f"{name}-" + apply_bins(block[column], cut)
        else:
            cut = quantile_edges(runs[column], num_bins)
            edges[(name, None)] = cut
            data[label] = f"{name}-" + apply_bins(data[column], cut)
        labels.append(label)
    return labels, edges


# --------------------------------------------------------------------------- #
# What the bins mean in numbers
# --------------------------------------------------------------------------- #

# Neutral bars for every run, one categorical slot for the band a cluster keeps
# and the grey ink of the ceiling for the cuts: a histogram carries one series,
# so the accent is free to mark the selection instead of an identity.
HIST_COLOUR = "#d6d5d0"
HIST_SELECTED_COLOUR = "#2a78d6"
CUT_COLOUR = "#52514e"

HIST_NUM_BINS = 24
# Ratio of largest to smallest value above which the axis is drawn logarithmic.
HIST_LOG_SPAN = 50.0
# Decades a logarithmic axis is allowed to span. The overlap of a basis state
# that lies in another symmetry sector is zero up to round-off -- 1e-280 and
# exact zeros both occur -- and an axis reaching down there would squeeze the
# runs that have any overlap at all into its last inch. Everything further down
# than this goes into one bar outside the axis instead, which is the honest
# reading anyway: below the floor the values are numerically indistinguishable
# from zero and only their count means anything.
HIST_MAX_DECADES = 12.0


# One histogram's geometry: the bin edges, whether the axis is logarithmic,
# the value under which everything is drawn off the scale, the limits (fixed
# rather than autoscaled, so the panels of one figure agree and a label can be
# placed in axes coordinates), and whether anything is off the scale at all.
Layout = namedtuple("Layout", "bins log floor xlim underflow")


def format_cut(value: float) -> str:
    """A bin edge as a number that can be carried over into the text."""
    return f"{value:.2e}" if 0.0 < abs(value) < 1e-2 else f"{value:.3g}"


def histogram_layout(values: np.ndarray) -> Layout:
    """
    Bins and axis for a distribution of run properties.

    The overlaps and the densities run over decades, so equal-width bins on a
    linear axis would pile nearly every run into the first one -- the same
    reason quantile_edges cuts on equal frequency. Anything spanning less than
    HIST_LOG_SPAN is left linear, where the eye reads the shape more easily, and
    nothing is off the scale there: a linear axis holds zeros without trouble.

    On a logarithmic axis everything more than HIST_MAX_DECADES below the
    largest value is taken out of the bins and drawn in one bar off the scale
    instead, and the axis then starts at the smallest value that survives rather
    than at that threshold, so the decades between the negligible tail and the
    bulk of the runs do not take up most of the figure.
    """
    positive = values[values > 0.0]
    high = float(values.max()) if values.size else 0.0
    if positive.size and high > 0.0 and high / float(positive.min()) >= HIST_LOG_SPAN:
        kept = positive[positive >= high * 10.0 ** -HIST_MAX_DECADES]
        floor = float(kept.min()) if kept.size else high * 10.0 ** -HIST_MAX_DECADES
        bins = np.geomspace(floor, high, HIST_NUM_BINS + 1)
        underflow = bool((values < floor).any())
        ratio = bins[1] / bins[0]
        left = bins[0] / ratio ** 3 if underflow else bins[0] / np.sqrt(ratio)
        return Layout(bins, True, floor, (left, bins[-1] * np.sqrt(ratio)), underflow)

    low = float(values.min()) if values.size else 0.0
    if not high > low:
        # A single distinct value: a narrow window, so the one bar is visible.
        margin = abs(low) * 0.05 or 1.0
        bins = np.linspace(low - margin, high + margin, 4)
    else:
        bins = np.linspace(low, high, HIST_NUM_BINS + 1)
    pad = 0.04 * (bins[-1] - bins[0])
    return Layout(bins, False, -np.inf, (bins[0] - pad, bins[-1] + pad), False)


def underflow_bar(layout: Layout) -> tuple:
    """Left edge, width and centre of the bar holding what is off the scale."""
    ratio = layout.bins[1] / layout.bins[0]
    # A bin width clear of the first bin, so the bar reads as being off the
    # scale rather than as the start of it.
    left, right = layout.bins[0] / ratio ** 2, layout.bins[0] / ratio
    return left, right - left, float(np.sqrt(left * right))


def layout_fraction(layout: Layout, position: float) -> float:
    """Where ``position`` sits across the axes, as a fraction of their width."""
    low, high = layout.xlim
    fraction = ((np.log10(position / low) / np.log10(high / low)) if layout.log
                else (position - low) / (high - low))
    return float(min(max(fraction, 0.0), 1.0))


def place_label(ax: plt.Axes, fraction: float, height: float, text: str,
                colour: str, size: float) -> None:
    """
    Write ``text`` at ``fraction`` of the axes, kept inside them where it would
    otherwise hang over the edge -- the outermost band of a cut often ends at
    the last bin, and a tie can leave it one bin wide.
    """
    align = "left" if fraction < 0.06 else "right" if fraction > 0.94 else "center"
    ax.text(fraction, height, text, transform=ax.transAxes, ha=align,
            va="bottom" if height > 1.0 else "top", fontsize=size, color=colour)


def draw_bin_panel(ax: plt.Axes, values: pd.Series, held, edges: np.ndarray,
                   selected, layout: Layout) -> None:
    """
    One distribution with the cuts that named its bins written on top.

    ``held`` are the runs of the folder the figure is written into, drawn over
    the others. For a folder cut on this axis alone that is exactly the band
    between two cuts; for one cut on several axes it is the part of the band
    that survived the other cuts, which is what the folder actually averages.
    """
    array = values.to_numpy(dtype=float)
    names = bin_names(edges.size - 1) if edges.size >= 3 else ["all"]
    labels = apply_bins(values, edges)
    chosen = (np.empty(0) if held is None else held.to_numpy(dtype=float))
    inside = array >= layout.floor
    chosen_inside = chosen >= layout.floor

    ax.hist(array[inside], bins=layout.bins, color=HIST_COLOUR, label="all runs")
    if chosen_inside.any():
        ax.hist(chosen[chosen_inside], bins=layout.bins, color=HIST_SELECTED_COLOUR,
                label="runs in this folder")

    # Everything under the floor in one bar left of the axis. A cut down there
    # is drawn inside that bar, since its own place is not on the scale.
    under = None
    if layout.underflow:
        left, width, under = underflow_bar(layout)
        ax.bar(left, int((~inside).sum()), width=width, align="edge", color=HIST_COLOUR)
        if (~chosen_inside).any():
            ax.bar(left, int((~chosen_inside).sum()), width=width, align="edge",
                   color=HIST_SELECTED_COLOUR)

    if layout.log:
        ax.set_xscale("log")
    # Fixed limits, so every panel of the figure shows the same window and the
    # labels below can be placed as a fraction of it.
    ax.set_xlim(*layout.xlim)

    # The cuts, with their value spelled out -- the whole point of the figure.
    # Two cuts under the floor share the one bar, and so share one label.
    drawn = defaultdict(list)
    for edge in edges[1:-1]:
        position = under if under is not None and edge < layout.floor else edge
        drawn[position].append(format_cut(edge))
    for position, texts in drawn.items():
        ax.axvline(position, color=CUT_COLOUR, linestyle="--", linewidth=1.2, zorder=3)
        place_label(ax, layout_fraction(layout, position), 1.02, " / ".join(texts),
                    CUT_COLOUR, 7.0)

    # Which band is which, and how many runs ended up in it. A band reaching
    # under the floor is labelled over the part of it that is on the scale, or
    # over the off-scale bar if none of it is. Ties in a column with few
    # distinct values can leave a band a bin wide, so a label that would land on
    # its neighbour is dropped a line instead.
    counts = labels.value_counts()
    height, previous = 0.95, None
    for name, low, high in zip(names, edges[:-1], edges[1:]):
        if not layout.log:
            position = 0.5 * (low + high)
        elif high <= layout.floor:
            position = under if under is not None else layout.bins[0]
        else:
            position = float(np.sqrt(max(low, layout.bins[0])
                                     * min(high, layout.bins[-1])))
        fraction = layout_fraction(layout, position)
        if previous is not None and abs(fraction - previous) < 0.14:
            height = 0.76 if height == 0.95 else 0.95
        # The band holds this many runs; the folder, whose runs are the bars
        # drawn over them, can hold fewer once the other cuts have had their say.
        text = f"{name}\n{int(counts.get(name, 0))} runs"
        if name == selected and chosen.size and chosen.size != counts.get(name, 0):
            text += f", {chosen.size} here"
        place_label(ax, fraction, height, text,
                    HIST_SELECTED_COLOUR if name == selected else CUT_COLOUR, 7.5)
        previous = fraction

    ax.margins(y=0.35)
    ax.grid(True, axis="y", **GRID_KWARGS)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def name_underflow_bar(ax: plt.Axes, layout: Layout, values: np.ndarray) -> None:
    """Label the off-scale bar where its tick label would be."""
    ticks = [tick for tick in ax.get_xticks()
             if layout.bins[0] <= tick <= layout.bins[-1]]
    zeros = "0 or\n" if (values == 0.0).any() else ""
    ax.set_xticks(ticks + [underflow_bar(layout)[2]],
                  labels=[f"$10^{{{int(round(np.log10(tick)))}}}$" for tick in ticks]
                         + [f"{zeros}$<${format_cut(layout.floor)}"])
    ax.set_xlim(*layout.xlim)


def plot_bin_histogram(runs: pd.DataFrame, cluster_runs: pd.DataFrame, column: str,
                       axis_title: str, edges: dict, sites: list, selected,
                       title: str, path: Path) -> None:
    """
    The distribution a cut was taken from, with the cut drawn as a line.

    ``low``, ``mid`` and ``high`` in a folder name say where a run sits relative
    to the others and nothing about the value itself. This puts the numbers back:
    the histogram of every run the cut was taken over, a vertical line at each
    edge labelled with its value, and the runs this folder holds drawn over the
    rest, so the name, the number and the runs behind it are in one figure.

    Under ``bin_scope="per-N"`` every system size has its own cut, so there is
    one panel per size the folder holds. The bins are taken from all of them at
    once and the axis is shared, so the shift of the whole distribution with N
    is visible next to the cuts that follow it.
    """
    def per_site(frame):
        return dict((site, frame[column] if site is None
                     else frame.loc[frame["Number_of_Sites"] == site, column])
                    for site in sites)

    values, held = per_site(runs), per_site(cluster_runs)
    pooled = pd.concat(values.values()).to_numpy(dtype=float)
    layout = histogram_layout(pooled)
    # A folder holding every run it would highlight -- ``all/``, ``by_N/`` --
    # has nothing to pick out of its own distribution.
    if sum(len(held[site]) for site in sites) == pooled.size:
        held = dict((site, None) for site in sites)

    fig, axes = plt.subplots(len(sites), 1, sharex=True, squeeze=False,
                             figsize=(5.5, 1.1 + 1.9 * len(sites)))
    for ax, site in zip(axes[:, 0], sites):
        draw_bin_panel(ax, values[site], held[site], edges[site], selected, layout)
        ax.set_ylabel("runs" if site is None else f"runs (N = {site})")
    axes[-1, 0].set_xlabel(axis_title)
    if layout.underflow:
        name_underflow_bar(axes[-1, 0], layout, pooled)

    fig.suptitle(f"{axis_title} over all runs, and the cuts behind the bins"
                 f"\n{title}", fontsize=10)
    fig.tight_layout()
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if len(handles) > 1:
        fig.legend(handles, labels, frameon=False, fontsize=8, ncol=len(handles),
                   loc="lower center", bbox_to_anchor=(0.5, -0.02))
    save_figure(fig, path)


def cluster_histograms(group: pd.DataFrame, names: list, key: tuple,
                       runs: pd.DataFrame, edges: dict, title: str,
                       directory: Path, scope: str) -> int:
    """
    The distributions behind one cluster's bins; returns how many were written.

    Drawn for the continuous axes the cluster is actually cut on, so the folder
    holding ``overlap-low`` also holds the histogram that says what low means.
    A cluster cut on none of them -- ``all/``, ``by_N/``, ``by_dim/`` -- gets
    all of them instead: it holds every run at that size, which is exactly the
    distribution the other folders are carved out of.
    """
    selected = dict(zip(names, (value.split("-", 1)[1] for value in key)))
    binned = [(name, column, axis_title) for name, column, is_binned, axis_title
              in FACETS if is_binned]
    targets = [facet for facet in binned if facet[0] in selected] or binned

    sites = ([None] if scope == "global"
             else sorted(int(site) for site in group["Number_of_Sites"].unique()))
    cluster_runs = group.drop_duplicates(RUN_KEY)
    for name, column, axis_title in targets:
        plot_bin_histogram(runs, cluster_runs, column, axis_title,
                           dict((site, edges[(name, site)]) for site in sites),
                           sites, selected.get(name), title,
                           directory / f"distribution_{name}.pdf")
    return len(targets)


def bin_edge_table(runs: pd.DataFrame, edges: dict) -> pd.DataFrame:
    """Every cut as a row, so the bins can also be read off a table."""
    columns = dict((name, column) for name, column, *_ in FACETS)
    rows = []
    for (name, site), cut in sorted(edges.items(), key=lambda item: (item[0][0],
                                                                    item[0][1] or 0)):
        values = (runs[columns[name]] if site is None
                  else runs.loc[runs["Number_of_Sites"] == site, columns[name]])
        labels = apply_bins(values, cut)
        names = bin_names(cut.size - 1) if cut.size >= 3 else ["all"]
        for bin_name, low, high in zip(names, cut[:-1], cut[1:]):
            rows.append({"Axis": name, "Column": columns[name],
                         "Number_of_Sites": "all" if site is None else site,
                         "Bin": bin_name, "Lower": low, "Upper": high,
                         "Runs": int((labels == bin_name).sum())})
    return pd.DataFrame(rows)


def cluster_title(names: list, key: tuple) -> str:
    """Human-readable version of a cluster's label tuple."""
    titles = dict((name, title) for name, _, _, title in FACETS)
    return ", ".join(f"{titles[name]} = {value.split('-', 1)[1]}"
                     for name, value in zip(names, key))


def ceiling_cutoff(group: pd.DataFrame, min_drop: float):
    """
    The budget just before the ceiling falls off a cliff, or None if it does not.

    Once a pool covers the support of the ground state the ceiling's infidelity
    drops to the numerical floor within one grid step. The y-axis then has to
    span every decade down to that floor and the region where the protocols
    actually differ is squeezed into the top of the figure. This finds the first
    step in which the ceiling loses more than ``min_drop`` decades -- a smooth
    stretch loses a fraction of one -- and returns the last budget before it,
    which is where a zoomed figure should stop.
    """
    ceiling = group[group["Algorithm"] == "Ceiling"]
    if ceiling.empty:
        return None
    mean = ceiling.groupby("Budget_Fraction")["Log_Infidelity"].mean().sort_index()
    drops = -mean.diff()
    jumps = drops.index[drops > min_drop]
    if not len(jumps):
        return None
    before = mean.index[mean.index < jumps[0]]
    # Cutting down to a couple of points does not make a figure worth drawing.
    return float(before[-1]) if before.size >= 3 else None


def cluster_plots(group: pd.DataFrame, title: str, directory: Path, min_drop: float) -> int:
    """Every figure of one cluster of runs; returns how many were written."""
    written = 0
    curves = on_common_grid(group)
    cutoff = ceiling_cutoff(curves, min_drop)
    zoomed = curves[curves["Budget_Fraction"] <= cutoff] if cutoff is not None else None
    for error in ("std", "sem", "median"):
        plot_curves(curves, title, directory / f"curves_{error}.pdf", error)
        plot_ratio(curves, title, directory / f"ratio_{error}.pdf", error)
        written += 2
        if zoomed is not None:
            zoom_title = f"{title} -- up to the jump in the ceiling"
            plot_curves(zoomed, zoom_title, directory / f"curves_{error}_zoom.pdf", error)
            plot_ratio(zoomed, zoom_title, directory / f"ratio_{error}_zoom.pdf", error)
            written += 2
    plot_termination(group, title, directory / "termination.pdf")
    return written + 1


def plot_results(data_file: str, output_root: Path = OUTPUT_ROOT,
                 num_bins: int = DEFAULT_NUM_BINS, min_runs: int = DEFAULT_MIN_RUNS,
                 bin_scope: str = "per-N", ceiling_jump: float = DEFAULT_CEILING_JUMP,
                 max_combination=None) -> None:
    """
    Every figure, in one folder per way of clustering the runs.

    The runs are cut by the axes of FACETS -- system size, lattice dimension,
    anisotropy, coupling, field, ground-state density, Hamiltonian density and
    the overlap of the initial state -- and this walks every combination of them,
    from the plain average over everything (``all/``) through the single axes
    (``by_gsdens/``) up to all of them at once. Every
    cluster gets a subfolder holding the same figures, so the path says exactly
    what is held fixed and nothing has to be read off a file name:

        by_N_overlap/N-10_overlap-high/curves_sem.pdf

    Since the x-axis is K / 2^N and not K, clusters that do not fix N are
    averages over system sizes, which is a different statement from the panels
    per (N, lattice dimension) -- those live in ``by_N_dim/``.

    Clusters holding fewer than ``min_runs`` runs are skipped rather than drawn:
    the deep combinations cut the runs into hundreds of cells, and a mean with a
    standard error over two of them is a figure that invites conclusions it
    cannot carry. ``plot_index.csv`` lists everything that was drawn together
    with the run count behind it.

    Since ``overlap-low`` says only where a run sits among the others, every
    cluster also gets ``distribution_<axis>.pdf`` for the continuous axes it is
    cut on: the histogram of that quantity over all runs, with the quantile cuts
    as labelled vertical lines and the band this folder keeps picked out of the
    bars. The same numbers are tabulated once in ``bin_edges.csv``.
    """
    data = add_log_infidelity(pd.read_csv(data_file))
    print(f"Loaded {len(data)} rows from {data_file}")
    data = drop_trivial_ground_states(data)
    data = drop_small_gaps(data)

    # One row per run, which is what the cuts are taken on and what the
    # histograms of them show -- a run contributes many rows to ``data``.
    runs_frame = data.drop_duplicates(RUN_KEY).copy()
    labels, edges = add_facet_labels(data, runs_frame, num_bins, bin_scope)
    names = [name for name, *_ in FACETS]
    depth = len(FACETS) if max_combination is None else max_combination

    index, written, skipped = [], 0, 0
    for size in range(depth + 1):
        for combination in itertools.combinations(range(len(FACETS)), size):
            folder = ("by_" + "_".join(names[i] for i in combination)) if combination else "all"
            groups = (data.groupby([labels[i] for i in combination]) if combination
                      else [((), data)])
            for key, group in groups:
                key = key if isinstance(key, tuple) else (key,)
                runs = len(group.drop_duplicates(RUN_KEY))
                if runs < min_runs:
                    skipped += 1
                    continue
                directory = output_root / folder / "_".join(key) if key else output_root / folder
                title = cluster_title([names[i] for i in combination], key) or "all runs"
                written += cluster_plots(group, title, directory, ceiling_jump)
                written += cluster_histograms(group, [names[i] for i in combination],
                                              key, runs_frame, edges, title,
                                              directory, bin_scope)
                index.append({"Folder": folder, "Cluster": "_".join(key) or "all",
                              "Runs": runs, "Path": str(directory)})
            print(f"{folder}: {len(index)} clusters so far, {written} figures")

    output_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(index).to_csv(output_root / "plot_index.csv", index=False)
    bin_edge_table(runs_frame, edges).to_csv(output_root / "bin_edges.csv", index=False)
    print(f"Wrote {written} PDF files in {len(index)} clusters under {output_root}/ "
          f"({skipped} clusters skipped for holding fewer than {min_runs} runs)")
    print(f"index: {output_root / 'plot_index.csv'}")
    print(f"bins:  {output_root / 'bin_edges.csv'}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def build_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["plan", "run", "merge", "plot"])
    parser.add_argument("--num-hamiltonians", type=int, default=20,
                        help="number of (J, B_z) draws; each one is run at every "
                             "(num_sites, dimension, delta)")
    parser.add_argument("--num-sites", type=int, nargs="+", default=DEFAULT_NUM_SITES)
    parser.add_argument("--dimensions", type=int, nargs="+", default=DEFAULT_DIMENSIONS,
                        help="lattice dimensions: 1 for a chain, 2 for the "
                             "most-square open rectangle")
    parser.add_argument("--deltas", type=float, nargs="+", default=DEFAULT_DELTAS,
                        help="anisotropies of the XXZ Hamiltonians")
    parser.add_argument("--num-initial-states", type=int, default=DEFAULT_NUM_INITIAL_STATES,
                        help="initial states per Hamiltonian, drawn uniformly at "
                             "random from the basis states of non-zero overlap")
    parser.add_argument("--n-repeats", type=int, default=DEFAULT_N_REPEATS,
                        help="repeats of the SKQD trajectory per run")
    parser.add_argument("--grid-points", type=int, default=DEFAULT_GRID_POINTS,
                        help="log-spaced budgets at which the curves are read")
    parser.add_argument("--max-fraction", type=float, default=DEFAULT_MAX_FRACTION,
                        help="largest budget, as a fraction of the Hilbert space")
    parser.add_argument("--num-jobs", type=int, default=20,
                        help="size of the SLURM array")
    parser.add_argument("--job-index", type=int, default=None,
                        help="this task's index; defaults to $SLURM_ARRAY_TASK_ID")
    parser.add_argument("--shard-dir", default="bitstring_shards_heisenberg")
    parser.add_argument("--output", default="equal_bitstrings_results.csv",
                        help="merge mode: where to write the combined CSV; "
                             "plot mode: which CSV to read")
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT),
                        help="plot mode: where to write the figures")
    parser.add_argument("--num-bins", type=int, default=DEFAULT_NUM_BINS,
                        help="plot mode: quantile bins the continuous axes "
                             "(the densities and the overlap) are cut into")
    parser.add_argument("--bin-scope", choices=["per-N", "global"], default="per-N",
                        help="plot mode: whether those bins are taken within each "
                             "system size or over the whole study")
    parser.add_argument("--min-runs", type=int, default=DEFAULT_MIN_RUNS,
                        help="plot mode: clusters holding fewer runs than this are "
                             "not drawn")
    parser.add_argument("--ceiling-jump", type=float, default=DEFAULT_CEILING_JUMP,
                        help="plot mode: decades the ceiling has to lose in one "
                             "budget step for the extra figures that stop in front "
                             "of it to be drawn")
    parser.add_argument("--max-combination", type=int, default=None,
                        help="plot mode: cluster by at most this many axes at once "
                             "(default: all of them)")
    parser.add_argument("--balance", choices=["cost", "stratified"], default="cost",
                        help="how cells are spread over the array; must match "
                             "between plan, run and merge")
    parser.add_argument("--dense-limit", type=int, default=DEFAULT_DENSE_LIMIT,
                        help="Hilbert-space dimension up to which the full dense "
                             "eigendecomposition is computed and handed to SKQD "
                             "(0 disables it)")
    parser.add_argument("--overwrite", action="store_true",
                        help="re-run a job even if its shard already exists")
    parser.add_argument("--no-resume", dest="resume", action="store_false",
                        help="ignore any .partial file and re-run every cell")
    parser.add_argument("--include-partial", action="store_true",
                        help="merge: also take the finished cells out of the "
                             ".partial files of jobs that never completed")
    parser.set_defaults(resume=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    if args.mode == "run" and args.job_index is None:
        env = os.environ.get("SLURM_ARRAY_TASK_ID")
        if env is None:
            sys.exit("--job-index is required outside a SLURM array")
        args.job_index = int(env)

    if args.mode == "plan":
        cells = study_cells(args)
        plan = describe_plan(cells, args.num_jobs, args.balance)
        rows = sum(expected_rows(cell, args) for cell in cells)
        print(f"{len(cells)} cells over {args.num_jobs} jobs "
              f"({rows} rows total, balance={args.balance})\n")
        print(plan.to_string(index=False))
        spread = plan["Cells"].max() - plan["Cells"].min()
        print(f"\ncells per job: min {plan['Cells'].min()}, max {plan['Cells'].max()} "
              f"(spread {spread})")
        imbalance = plan["Load"].max() / max(plan["Load"].min(), 1e-9)
        print(f"estimated load: min {plan['Load'].min()}, max {plan['Load'].max()} "
              f"(heaviest job is {imbalance:.2f}x the lightest)")
        print(f"\nsbatch --array=0-{args.num_jobs - 1} submit_bitstrings.sh")
    elif args.mode == "run":
        run_job(args)
    elif args.mode == "merge":
        merge_shards(args)
    else:
        plot_results(args.output, Path(args.output_root), num_bins=args.num_bins,
                     min_runs=args.min_runs, bin_scope=args.bin_scope,
                     ceiling_jump=args.ceiling_jump,
                     max_combination=args.max_combination)


if __name__ == "__main__":
    main()
