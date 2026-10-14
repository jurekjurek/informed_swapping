"""
Density-balanced selection of Hamiltonians for EqualNumberOfBitstrings.py.

Plain random sampling of the tilted-field scheme already spans densities from
below 0.1 to above 0.9, but the distribution is U-shaped and shifts with N: at
N = 12 most draws are either nearly product states or nearly flat. Since one
ground state costs milliseconds, the cure is to draw candidates and keep a
fixed number per density bin, for every (N, dimension, delta) separately.

The result is a table of couplings, one row per cell, that the study can read
instead of drawing J and B itself:

    python select_hamiltonians.py      # writes data/selected_hamiltonians.csv

Columns: num_sites, dimension, delta, hamiltonian_index, J, Bx, By, Bz, plus the
density, gap and top weight that were measured, so the table documents itself.
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

from common import HERE, measure

NUM_SITES = [6, 8, 10, 12]
DIMENSIONS = [1, 2]
DELTAS = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]
DENSITY_BINS = np.linspace(0.0, 1.0, 11)   # ten bins of width 0.1
PER_BIN = 2                                # 2 x 10 bins = 20 Hamiltonians per cell group
MAX_CANDIDATES = 10000
RATIO_RANGE = (0.1, 10.0)                  # b = |B| / (|J| max(1, delta)), log-uniform
SEED = 20251014
COUPLING_DECIMALS = 6                      # as in EqualNumberOfBitstrings.py

OUTPUT = os.path.join(HERE, "data", "selected_hamiltonians.csv")


def draw(rng, delta):
    """
    The tilted-field scheme: J ~ U(-1, 1), theta ~ U(0, 90 deg), and the field
    measured against the largest coupling, |B| = b |J| max(1, delta).
    """
    J = rng.uniform(-1, 1)
    b = 10 ** rng.uniform(*np.log10(RATIO_RANGE))
    theta = rng.uniform(0, np.pi / 2)
    B = b * abs(J) * max(1.0, delta)
    Bx, Bz = B * np.sin(theta), B * np.cos(theta)
    return tuple(float(np.round(x, COUPLING_DECIMALS)) for x in (J, Bx, Bz))


def group(task):
    num_sites, dimension, delta = task
    rng = np.random.default_rng([SEED, num_sites, dimension, int(delta * 10)])
    filled = [[] for _ in range(len(DENSITY_BINS) - 1)]
    candidates = 0
    while candidates < MAX_CANDIDATES and min(map(len, filled)) < PER_BIN:
        candidates += 1
        J, Bx, Bz = draw(rng, delta)
        row, _ = measure(num_sites, dimension, delta, J, (Bx, 0.0, Bz))
        if not row["kept"]:
            continue
        index = min(np.searchsorted(DENSITY_BINS, row["density"], side="right") - 1,
                    len(filled) - 1)
        if len(filled[index]) < PER_BIN:
            filled[index].append(dict(J=J, Bx=Bx, By=0.0, Bz=Bz, density=row["density"],
                                      gap=row["gap"], top_weight=row["top_weight"]))
    rows = [dict(num_sites=num_sites, dimension=dimension, delta=delta, **r)
            for members in filled for r in members]
    for index, r in enumerate(rows):
        r["hamiltonian_index"] = index
    return rows, candidates, [len(m) for m in filled]


def main():
    tasks = list(itertools.product(NUM_SITES, DIMENSIONS, DELTAS))
    start = time.time()
    with Pool() as pool:
        results = pool.map(group, tasks)
    rows, total = [], 0
    for task, (group_rows, candidates, fill) in zip(tasks, results):
        rows += group_rows
        total += candidates
        missing = "" if min(fill) == PER_BIN else f"  bins short: {fill}"
        print(f"N={task[0]:2d} dim={task[1]} delta={task[2]:4g}: "
              f"{len(group_rows)} kept from {candidates} candidates{missing}")
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    pd.DataFrame(rows).to_csv(OUTPUT, index=False)
    print(f"{len(rows)} Hamiltonians from {total} candidates in {time.time() - start:.0f} s "
          f"-> {OUTPUT}")


if __name__ == "__main__":
    main()
