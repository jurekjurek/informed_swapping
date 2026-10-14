"""
Grid scan of the XXZ Hamiltonian of EqualNumberOfBitstrings.py.

The Hamiltonian

    H = J sum_<ij> (X_i X_j + Y_i Y_j + delta Z_i Z_j) - sum_i (Bx X_i + Bz Z_i)

has, up to an overall scale, four parameters: the sign of J, the anisotropy
delta, the field strength relative to the coupling, b = |B| / |J|, and the angle
theta of the field away from the z axis (Bz = |B| cos theta, Bx = |B| sin theta).
theta = 0 is the longitudinal study, theta = 90 deg the transversal one.

Every point of the grid is diagonalised once; the output is one row per point
with the density, gap, trivial flag and quantiles of the overlap distribution.

    python scan_grid.py            # writes data/grid_scan.csv
"""

import itertools
import os
import time

# One BLAS thread per worker: with the default, eight workers each start eight
# threads and the scan runs slower than a single process.
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(variable, "1")

from multiprocessing import Pool

import numpy as np
import pandas as pd

from common import HERE, measure

NUM_SITES = [6, 8, 10, 12]
DIMENSIONS = [1, 2]
DELTAS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 100.0]
SIGNS = [-1.0, 1.0]
RATIOS = np.round(np.logspace(-1.5, 1.5, 10), 4)      # |B| / |J| from 0.03 to 30
ANGLES = [0.0, 5.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0]  # degrees from the z axis

OUTPUT = os.path.join(HERE, "data", "grid_scan.csv")


def point(args):
    num_sites, dimension, delta, sign, ratio, angle = args
    theta = np.deg2rad(angle)
    field = (ratio * np.sin(theta), 0.0, ratio * np.cos(theta))
    row, _ = measure(num_sites, dimension, delta, sign, field)
    row.update(num_sites=num_sites, dimension=dimension, delta=delta, J=sign,
               ratio=ratio, angle=angle)
    return row


def main():
    grid = list(itertools.product(NUM_SITES, DIMENSIONS, DELTAS, SIGNS, RATIOS, ANGLES))
    # Largest systems first, so the pool is not left waiting on them at the end.
    grid.sort(key=lambda g: -g[0])
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    start = time.time()
    rows = []
    with Pool() as pool:
        for row in pool.imap_unordered(point, grid, chunksize=8):
            rows.append(row)
            if len(rows) % 1000 == 0:
                print(f"{len(rows)}/{len(grid)} after {time.time() - start:.0f} s", flush=True)
    pd.DataFrame(rows).to_csv(OUTPUT, index=False)
    print(f"{len(rows)} points in {time.time() - start:.0f} s -> {OUTPUT}")


if __name__ == "__main__":
    main()
