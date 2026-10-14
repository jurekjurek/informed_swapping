"""
Draw whole studies the way EqualNumberOfBitstrings.enumerate_cells does, for a
few candidate sampling schemes, and record what the fixed-budget study would
see: which cells survive the gap and trivial filters, their ground-state
densities, and the overlaps of the initial states drawn from them.

    python simulate_schemes.py     # writes data/schemes_cells.csv, data/schemes_overlaps.csv

A scheme is a function ``draw(rng, delta) -> (J, Bx, Bz)`` plus a list of deltas. The
same number of Hamiltonians per (N, dimension, delta) as the default study is
drawn (20), with the default sizes and geometries.

Initial states are drawn in two ways from every kept cell:

``uniform``     uniformly from the support, as EqualNumberOfBitstrings.run_cell
                does now;
``stratified``  one state per equal-width bin of log10(overlap) between the
                smallest and the largest overlap of the support.
"""

import os

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(variable, "1")

import itertools
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

from common import HERE, OVERLAP_FLOOR, measure

NUM_SITES = [6, 8, 10, 12]
DIMENSIONS = [1, 2]
NUM_HAMILTONIANS = 20
NUM_INITIAL_STATES = 6
SEED = 20251014

CURRENT_DELTAS = [0.0, 0.5, 1.0, 10.0, 100.0]
PROPOSED_DELTAS = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]


def current_longitudinal(rng, delta):
    J, Bz = rng.uniform(-1, 1, 2)
    return J, 0.0, Bz


def current_transversal(rng, delta):
    J, Bx = rng.uniform(-1, 1, 2)
    return J, Bx, 0.0


def tilted(ratio_range, angle_sampler, scale_with_delta=False):
    """
    J uniform in [-1, 1]; |B| = b |J| with b log-uniform; theta from the sampler.

    ``scale_with_delta`` measures the field against the largest coupling instead,
    |B| = b |J| max(1, delta), so that b means the same at every anisotropy.
    """
    low, high = np.log10(ratio_range)

    def draw(rng, delta):
        J = rng.uniform(-1, 1)
        b = 10 ** rng.uniform(low, high)
        theta = angle_sampler(rng)
        B = b * abs(J) * (max(1.0, delta) if scale_with_delta else 1.0)
        return J, B * np.sin(theta), B * np.cos(theta)
    return draw


def uniform_angle(rng):
    return rng.uniform(0, np.pi / 2)


def uniform_cos_angle(rng):
    # Field direction uniform on the sphere, restricted to the x-z quarter.
    return np.arccos(rng.uniform(0, 1))


SCHEMES = {
    "longitudinal (current)": (current_longitudinal, CURRENT_DELTAS),
    "transversal (current)": (current_transversal, CURRENT_DELTAS),
    "tilted, b in [0.1, 10]": (tilted((0.1, 10.0), uniform_angle), PROPOSED_DELTAS),
    "tilted, b in [0.03, 30]": (tilted((0.03, 30.0), uniform_angle), PROPOSED_DELTAS),
    "tilted, b in [0.1, 10], cos-uniform": (tilted((0.1, 10.0), uniform_cos_angle),
                                            PROPOSED_DELTAS),
    "tilted, b in [0.1, 10], scaled by max(1, delta)": (
        tilted((0.1, 10.0), uniform_angle, scale_with_delta=True), PROPOSED_DELTAS),
}


def stratified(probabilities, rng, count):
    """One support state per equal-width bin of log10(p); empty bins are skipped."""
    support = np.flatnonzero(probabilities > OVERLAP_FLOOR)
    logs = np.log10(probabilities[support])
    edges = np.linspace(logs.min(), logs.max(), count + 1)
    edges[-1] += 1e-12
    chosen = []
    for low, high in zip(edges[:-1], edges[1:]):
        members = support[(logs >= low) & (logs < high)]
        if members.size:
            chosen.append(rng.choice(members))
    return np.array(chosen, dtype=int)


def cell(task):
    scheme, index, num_sites, dimension, delta = task
    draw, _ = SCHEMES[scheme]
    # Couplings depend on the scheme and the Hamiltonian index only, as in the study.
    rng = np.random.default_rng([SEED, list(SCHEMES).index(scheme), index])
    J, Bx, Bz = draw(rng, delta)
    row, probabilities = measure(num_sites, dimension, delta, J, (Bx, 0.0, Bz))
    row.update(scheme=scheme, hamiltonian_index=index, num_sites=num_sites,
               dimension=dimension, delta=delta, J=J, Bx=Bx, Bz=Bz)
    overlaps = []
    if row["kept"]:
        pick = np.random.default_rng([SEED, index, num_sites, dimension, int(delta * 10)])
        support = np.flatnonzero(probabilities > OVERLAP_FLOOR)
        uniform = pick.choice(support, size=min(NUM_INITIAL_STATES, support.size), replace=False)
        for method, states in (("uniform", uniform),
                               ("stratified", stratified(probabilities, pick, NUM_INITIAL_STATES))):
            overlaps += [dict(scheme=scheme, method=method, num_sites=num_sites,
                              dimension=dimension, delta=delta, density=row["density"],
                              overlap=float(probabilities[s])) for s in states]
    return row, overlaps


def main():
    tasks = [(scheme, index, n, dim, delta)
             for scheme, (_, deltas) in SCHEMES.items()
             for index, n, dim, delta in itertools.product(range(NUM_HAMILTONIANS), NUM_SITES,
                                                           DIMENSIONS, deltas)]
    start = time.time()
    with Pool() as pool:
        results = pool.map(cell, tasks, chunksize=8)
    cells = pd.DataFrame([row for row, _ in results])
    overlaps = pd.DataFrame([o for _, rows in results for o in rows])
    os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
    cells.to_csv(os.path.join(HERE, "data", "schemes_cells.csv"), index=False)
    overlaps.to_csv(os.path.join(HERE, "data", "schemes_overlaps.csv"), index=False)
    print(f"{len(cells)} cells, {len(overlaps)} initial states in {time.time() - start:.0f} s")


if __name__ == "__main__":
    main()
