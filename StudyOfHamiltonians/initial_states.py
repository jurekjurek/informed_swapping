"""
Initial overlaps of the log-balanced study, and how to raise them without
touching the Hamiltonians.

The Hamiltonians are the ones the production run selected, read from the shard
directory; nothing in the main folder is written. Every ground state is solved
once, and from it the initial states are drawn by several schemes, six per
Hamiltonian as in the study:

``current``           uniformly from every basis state with p > OVERLAP_FLOOR,
                      exactly as EqualNumberOfBitstrings.run_cell did before
                      this study (same generator, so it reproduces the
                      production overlaps);
``dense``             uniformly from the states the density counts,
                      p > 0.1 / 2^N;
``dense_filled``      the same, and where fewer than six such states exist,
                      the rest uniformly from the remaining support;
``born``              with probability p, i.e. as a measurement of the ground
                      state would return them (without replacement);
``stratified``        one state per equal-width bin of log10(p) between the
                      smallest and the largest p of the support;
``stratified_dense``  the same between 0.1 / 2^N and the largest p;
``top``               the six most probable states, as a reference for the
                      largest overlaps a Hamiltonian can offer at all.

    python initial_states.py      # writes data/initial_overlaps.csv, data/ground_state_summary.csv
"""

import os

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(variable, "1")

import sys
import time
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.sparse.linalg import eigsh

HERE = os.path.dirname(os.path.abspath(__file__))
MAIN = os.path.dirname(HERE)
sys.path.insert(0, MAIN)
warnings.filterwarnings("ignore")

# The study's own definitions, imported so that nothing can drift; the module
# is only read.
from EqualNumberOfBitstrings import (Cell, build_hamiltonian, cell_seed,  # noqa: E402
                                     ground_state_density)

# The floor run_cell drew the initial states above before this study.
OVERLAP_FLOOR = 1e-12

TABLE = os.path.join(MAIN, "bitstring_shards_heisenberg_balanced_log",
                     "balanced_hamiltonians_log.csv")
SHARDS = os.path.join(MAIN, "bitstring_shards_heisenberg_balanced_log")
DATA = os.path.join(HERE, "data")
NUM_INITIAL_STATES = 6
DENSE_FRACTION = 0.1          # the density's threshold, as a fraction of 1 / 2^N
SCHEMES = ["current", "dense", "dense_filled", "born", "stratified", "stratified_dense", "top"]


def cells():
    table = pd.read_csv(TABLE, dtype={"J_Per_Distance": str})
    return [Cell(int(r.Hamiltonian_Index), int(r.Number_of_Sites), int(r.Dimensions),
                 float(r.Delta), tuple(float(J) for J in r.J_Per_Distance.split(";")),
                 float(r.Bx), float(r.By), float(r.Bz), int(r.Max_Distance))
            for r in table.itertuples()]


def ground_state(cell):
    hamiltonian = build_hamiltonian(cell)
    if hamiltonian.shape[0] <= 256:
        values, vectors = np.linalg.eigh(hamiltonian.toarray())
        return vectors[:, 0]
    values, vectors = eigsh(hamiltonian, k=2, which="SA", tol=1e-12,
                            v0=np.random.default_rng(0).uniform(-1, 1, hamiltonian.shape[0]))
    return vectors[:, np.argmin(values)]


def stratified(probabilities, candidates, rng, count):
    """One of ``candidates`` per equal-width bin of log10(p); empty bins are skipped."""
    logs = np.log10(probabilities[candidates])
    edges = np.linspace(logs.min(), logs.max(), count + 1)
    edges[-1] += 1e-9
    chosen = [rng.choice(candidates[(logs >= low) & (logs < high)])
              for low, high in zip(edges[:-1], edges[1:])
              if np.any((logs >= low) & (logs < high))]
    return np.array(chosen, dtype=int)


def draw(scheme, probabilities, cell):
    size = probabilities.size
    support = np.flatnonzero(probabilities > OVERLAP_FLOOR)
    dense = np.flatnonzero(probabilities > DENSE_FRACTION / size)
    if scheme == "current":
        # Same generator and call as run_cell.
        rng = np.random.default_rng(cell_seed(cell))
        return rng.choice(support, size=min(NUM_INITIAL_STATES, support.size), replace=False)
    rng = np.random.default_rng([cell_seed(cell), SCHEMES.index(scheme)])
    if scheme == "dense":
        return rng.choice(dense, size=min(NUM_INITIAL_STATES, dense.size), replace=False)
    if scheme == "dense_filled":
        chosen = rng.choice(dense, size=min(NUM_INITIAL_STATES, dense.size), replace=False)
        rest = np.setdiff1d(support, dense)
        missing = min(NUM_INITIAL_STATES - chosen.size, rest.size)
        return np.concatenate([chosen, rng.choice(rest, size=missing, replace=False)])
    if scheme == "born":
        weights = probabilities[support] / probabilities[support].sum()
        return rng.choice(support, size=min(NUM_INITIAL_STATES, support.size), replace=False,
                          p=weights)
    if scheme == "stratified":
        return stratified(probabilities, support, rng, NUM_INITIAL_STATES)
    if scheme == "stratified_dense":
        return stratified(probabilities, dense, rng, NUM_INITIAL_STATES)
    if scheme == "top":
        return np.argsort(probabilities)[::-1][:NUM_INITIAL_STATES]
    raise ValueError(scheme)


def analyse(cell):
    psi = ground_state(cell)
    probabilities = np.abs(psi) ** 2
    size = probabilities.size
    key = dict(hamiltonian_index=cell.hamiltonian_index, num_sites=cell.num_sites,
               dimension=cell.dimensions, delta=cell.delta, max_distance=cell.max_distance)
    summary = dict(key, density=ground_state_density(psi), top=float(probabilities.max()),
                   support=int(np.count_nonzero(probabilities > OVERLAP_FLOOR)),
                   dense_support=int(np.count_nonzero(probabilities > DENSE_FRACTION / size)),
                   # Born-weighted mean overlap: the participation sum.
                   ipr=float(np.sum(probabilities ** 2)))
    rows = [dict(key, scheme=scheme, state=int(state), overlap=float(probabilities[state]),
                 density=summary["density"])
            for scheme in SCHEMES for state in draw(scheme, probabilities, cell)]
    return summary, rows


def production_overlaps():
    """Overlaps of the production run, read from the shards, one row per run."""
    columns = ["Hamiltonian_Index", "Number_of_Sites", "Dimensions", "Delta", "Max_Distance",
               "Initial_State_Index", "Overlap"]
    frames = [pd.read_csv(os.path.join(SHARDS, name), usecols=columns)
              for name in sorted(os.listdir(SHARDS)) if name.startswith("shard_")]
    return pd.concat(frames).drop_duplicates()


def main():
    start = time.time()
    with Pool() as pool:
        results = pool.map(analyse, cells(), chunksize=4)
    summary = pd.DataFrame([s for s, _ in results])
    overlaps = pd.DataFrame([r for _, rows in results for r in rows])
    os.makedirs(DATA, exist_ok=True)
    summary.to_csv(os.path.join(DATA, "ground_state_summary.csv"), index=False)
    overlaps.to_csv(os.path.join(DATA, "initial_overlaps.csv"), index=False)
    print(f"{len(summary)} Hamiltonians in {time.time() - start:.0f}s")

    # The 'current' scheme has to reproduce what the production run drew.
    production = production_overlaps().rename(columns={
        "Hamiltonian_Index": "hamiltonian_index", "Number_of_Sites": "num_sites",
        "Dimensions": "dimension", "Delta": "delta", "Max_Distance": "max_distance",
        "Initial_State_Index": "state", "Overlap": "production_overlap"})
    mine = overlaps[overlaps.scheme == "current"]
    merged = mine.merge(production, how="outer", indicator=True,
                        on=["hamiltonian_index", "num_sites", "dimension", "delta",
                            "max_distance", "state"])
    both = merged[merged._merge == "both"]
    print(f"current vs production: {len(both)} of {len(production)} runs drawn identically, "
          f"max |log10 ratio| {np.abs(np.log10(both.overlap / both.production_overlap)).max():.2e}")


if __name__ == "__main__":
    main()
