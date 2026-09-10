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
regime where all protocols have already won. Curves are never averaged across
the number of sites: the difficulty grows with the system size, so every
(Number_of_Sites, Max_Interactions) gets its own figure.

The unit of work is a "cell": one ``(hamiltonian_index, num_sites,
max_interactions)`` combination. A cell builds one Hamiltonian, solves for its
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
"""

import argparse
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import ClusterStudy
from RandomSpinModel import make_random_spin_hamiltonian
from BARK import BARK
from SKQD import SKQD, fidelity_at_budget
from ClusterStudy import (DEFAULT_DENSE_LIMIT, Cell, assign_all_cells, assign_cells,
                          cell_seed, describe_plan, enumerate_cells, shard_path,
                          solve_ground_state)

DEFAULT_NUM_SITES = [6, 8, 10, 12]
DEFAULT_MAX_INTERACTIONS = [1, 2]

# Field strength of the Hamiltonians. A single value rather than the sweep of
# ClusterStudy.py: this study spends its budget on resolving the whole
# fidelity-against-K curve instead of on a fourth axis.
DEFAULT_B_MAX = 1.0

# Initial states per Hamiltonian, taken from both ends of the overlap
# distribution as in ClusterStudy.run_cell -- so this many largest and this many
# smallest. They are kept apart in the plots rather than averaged over: picking
# the extremes makes the sample bimodal by construction, and the mean of a
# bimodal mixture describes neither mode.
DEFAULT_NUM_INITIAL_STATES = 3

# Repeats of the SKQD trajectory per run. BARK is deterministic and needs none.
DEFAULT_N_REPEATS = 3

# Budget grid: this many log-spaced integers from one bitstring up to
# ``max_fraction`` of the Hilbert space. Running truly to the end of the Hilbert
# space costs 2^N growing diagonalizations per BARK run, and the last decade is
# the regime in which a subspace method has lost anyway, so the runs are capped.
DEFAULT_GRID_POINTS = 30
DEFAULT_MAX_FRACTION = 0.5

COLUMNS = ["Hamiltonian_Index", "Number_of_Sites", "Max_Interactions",
           "Ground_State_Density", "Hamiltonian_Density", "Overlap",
           "Initial_State_Index", "Algorithm", "Budget", "Budget_Fraction", "Fidelity",
           "Final_Pool_Size", "Final_Pool_Fraction", "Terminated_Early", "Seed"]

CELL_KEY = ["Hamiltonian_Index", "Number_of_Sites", "Max_Interactions"]
# What identifies one curve: one protocol on one initial state of one Hamiltonian.
RUN_KEY = CELL_KEY + ["Initial_State_Index"]
PANEL_KEY = ["Number_of_Sites", "Max_Interactions"]

# Categorical slots of the validated default palette, plus grey for the ceiling.
ALGORITHMS = (
    # name                colour     marker  linestyle
    ("BARK",              "#2a78d6", "o",    "-"),
    ("Simplified BARK",   "#3aa66f", "^",    "-."),
    ("SKQD",              "#eb6834", "s",    "--"),
    ("Ceiling",           "#52514e", None,   ":"),
)
REFERENCE = "BARK"          # denominator of the ratio plots

GRID_KWARGS = dict(color="#c9c8c3", linewidth=0.6, alpha=0.7)
# Fidelities of exactly 1 do occur once the pool spans the support of the ground
# state, and log(0) is not plottable, so the infidelity is floored.
INFIDELITY_FLOOR = 1e-16

OUTPUT_ROOT = Path("equal_bitstrings_plots")


# --------------------------------------------------------------------------- #
# Work definition and distribution
# --------------------------------------------------------------------------- #

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
    bonds = max(1, min(cell.max_interactions, max(cell.num_sites - 1, 1)))
    return dimension ** 3.5 * (1.0 + 0.8 * (bonds - 1))


# ClusterStudy's splitter weights cells with ClusterStudy's cost model, and it
# takes it from its own module global. Redirect that at ours here, at import
# time, so plan, run and merge all agree on the split -- patching it in only one
# of them would print one split and execute another. The two studies are never
# in the same process, so nothing else is affected.
ClusterStudy.estimate_cost = estimate_cost


def study_cells(args) -> list:
    """
    All cells of the study, in a fixed order independent of how they are split.

    ``enumerate_cells`` carries a penalty axis; passing the single field strength
    as that axis keeps ``Cell`` -- and therefore ``cell_seed`` -- identical to
    the fixed-target study, so both studies see the same disorder realisations.
    """
    return enumerate_cells(args.num_hamiltonians, args.num_sites,
                           args.max_interactions, [args.b_max])


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


def expected_rows(cell: Cell, args) -> int:
    """
    How many rows a finished cell contributes.

    Used to tell a complete cell in a ``.partial`` from one that was interrupted
    half way. It depends on the grid, so a partial written under a different
    grid, a different number of initial states or a different cap fails the check
    and is redone rather than mixed in.
    """
    budgets = budget_grid(2 ** cell.num_sites, args.grid_points, args.max_fraction)
    return 2 * args.num_initial_states * len(ALGORITHMS) * budgets.size


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

    hamiltonian = make_random_spin_hamiltonian(
        num_sites=cell.num_sites,
        max_interactions=cell.max_interactions,
        J_components=("x", "y"),
        B_components=("z"),
        B_max=cell.penalty_strength,
        seed=seed,
        N_target=cell.num_sites // 2,
        penalty_strength=0,
    )[0].to_matrix(sparse=True)

    ground_state, all_eigenvalues, all_eigenvectors = solve_ground_state(
        hamiltonian, args.dense_limit)
    probabilities = np.abs(ground_state) ** 2

    dimension = hamiltonian.shape[0]
    ground_state_density = np.count_nonzero(np.abs(ground_state) > 1e-3) / dimension
    hamiltonian_density = hamiltonian.nnz / dimension ** 2

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

    largest_indices = np.argsort(probabilities)[-args.num_initial_states:]
    smallest_indices = np.argsort(probabilities)[:args.num_initial_states]
    indices_to_test = np.concatenate((largest_indices, smallest_indices))

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

        # (t, n_shots) are tuned for the largest budget of the grid, once per
        # run, and then held fixed along the whole curve. Re-optimizing at every
        # budget would draw an envelope over hyperparameters that no single SKQD
        # run realizes. The evaluation repeats below draw fresh shots -- the
        # optimizer's own repeats have already advanced the global RNG -- so the
        # reported curve is not the noise the optimum was selected on.
        t, n_shots = skqd_protocol.optimize_general(
            initial_state_index, ground_state, max_pool_size=max_pool_size,
            n_repeats=args.n_repeats)
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
                    "Max_Interactions": cell.max_interactions,
                    "Ground_State_Density": ground_state_density,
                    "Hamiltonian_Density": hamiltonian_density,
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
    wanted = {(cell.hamiltonian_index, cell.num_sites, cell.max_interactions):
              expected_rows(cell, args) for cell in cells}
    finished = {key for key, count in counts.items()
                if count == wanted.get((int(key[0]), int(key[1]), int(key[2])))}

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
        label = (f"n={cell.num_sites} mi={cell.max_interactions} "
                 f"ham={cell.hamiltonian_index}")
        if (cell.hamiltonian_index, cell.num_sites, cell.max_interactions) in finished:
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
                   if (c.hamiltonian_index, c.num_sites, c.max_interactions) in have)
        print(f"  n={num_sites:>3}: {done:>4}/{len(wanted)} cells"
              f"{'  COMPLETE' if done == len(wanted) else ''}")

    if missing:
        print(f"WARNING: {len(missing)} shards missing (jobs {missing}); "
              f"re-submit those array indices, then merge again")


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #

def add_log_infidelity(curves: pd.DataFrame) -> pd.DataFrame:
    """Add log10(1 - F), which is what the averages are taken in."""
    infidelity = np.clip(1.0 - curves["Fidelity"], INFIDELITY_FLOOR, 1.0)
    curves["Log_Infidelity"] = np.log10(infidelity)
    return curves


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
    """
    stats = frame.groupby(["Algorithm", "Budget_Fraction"])[column].agg(
        ["mean", "std", "count"]).reset_index()
    # A single sample has no spread rather than an undefined one.
    stats["std"] = stats["std"].fillna(0.0)
    stats["spread"] = stats["std"] if error == "std" else stats["std"] / np.sqrt(stats["count"])
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
    return "standard deviation" if error == "std" else "standard error"


def _annotate_counts(ax: plt.Axes, stats: pd.DataFrame) -> None:
    """State how many runs entered each average."""
    counts = stats.groupby("Algorithm")["count"].max()
    text = ", ".join(f"{name}: {int(count)}" for name, count in counts.items())
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
        spread = rows["spread"].to_numpy()
        ax.plot(rows["Budget_Fraction"], 10.0 ** mean, marker=marker, markersize=5,
                linewidth=2, linestyle=linestyle, color=colour,
                markeredgecolor="white", markeredgewidth=0.8, label=name)
        ax.fill_between(rows["Budget_Fraction"], 10.0 ** (mean - spread),
                        10.0 ** (mean + spread), color=colour, alpha=0.12, linewidth=0)

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
        spread = rows["spread"].to_numpy()
        ax.plot(rows["Budget_Fraction"], 10.0 ** mean, marker=marker, markersize=5,
                linewidth=2, linestyle=linestyle, color=colour,
                markeredgecolor="white", markeredgewidth=0.8, label=name)
        ax.fill_between(rows["Budget_Fraction"], 10.0 ** (mean - spread),
                        10.0 ** (mean + spread), color=colour, alpha=0.12, linewidth=0)

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


def plot_results(data_file: str, output_root: Path = OUTPUT_ROOT) -> None:
    """Produce every figure from the merged results."""
    data = add_log_infidelity(pd.read_csv(data_file))
    print(f"Loaded {len(data)} rows from {data_file}")

    written = 0
    for (num_sites, max_interaction), group in data.groupby(PANEL_KEY):
        title = f"N = {num_sites}, max. interactions = {max_interaction}"
        stem = f"N{num_sites}_maxint{max_interaction}"
        for error in ("std", "sem"):
            plot_curves(group, title, output_root / f"curves_{stem}_{error}.pdf", error)
            plot_ratio(group, title, output_root / f"ratio_{stem}_{error}.pdf", error)
            written += 2
        plot_termination(group, title, output_root / f"termination_{stem}.pdf")
        written += 1

    print(f"Wrote {written} PDF files under {output_root}/")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def build_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["plan", "run", "merge", "plot"])
    parser.add_argument("--num-hamiltonians", type=int, default=20,
                        help="number of random Hamiltonians per (num_sites, max_interactions)")
    parser.add_argument("--num-sites", type=int, nargs="+", default=DEFAULT_NUM_SITES)
    parser.add_argument("--max-interactions", type=int, nargs="+",
                        default=DEFAULT_MAX_INTERACTIONS)
    parser.add_argument("--b-max", type=float, default=DEFAULT_B_MAX,
                        help="field strength of the Hamiltonians")
    parser.add_argument("--num-initial-states", type=int, default=DEFAULT_NUM_INITIAL_STATES,
                        help="initial states per Hamiltonian from each end of the "
                             "overlap distribution")
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
    parser.add_argument("--shard-dir", default="bitstring_shards")
    parser.add_argument("--output", default="equal_bitstrings_results.csv",
                        help="merge mode: where to write the combined CSV; "
                             "plot mode: which CSV to read")
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT),
                        help="plot mode: where to write the figures")
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
        plot_results(args.output, Path(args.output_root))


if __name__ == "__main__":
    main()
