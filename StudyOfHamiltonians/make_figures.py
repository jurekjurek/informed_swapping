"""
Figures of the Hamiltonian study, written to figures/ for report.tex.

Needs data/grid_scan.csv, data/schemes_cells.csv, data/schemes_overlaps.csv and
data/selected_hamiltonians.csv (scan_grid.py, simulate_schemes.py and
select_hamiltonians.py). The initial states of the selected Hamiltonians are
drawn here and cached in data/selected_overlaps.csv.

    python make_figures.py
"""

import os

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(variable, "1")

from multiprocessing import Pool

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from common import HERE, OVERLAP_FLOOR, measure
from simulate_schemes import NUM_INITIAL_STATES, stratified

DATA = os.path.join(HERE, "data")
FIGURES = os.path.join(HERE, "figures")

BLUE = "#2a78d6"
ORANGE = "#eb6834"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#c9c8c3"
DROPPED = "#e4e3de"
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "blue", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])

DENSITY_BINS = np.linspace(0.0, 1.0, 11)

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.edgecolor": INK_SECONDARY, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK_SECONDARY, "ytick.color": INK_SECONDARY,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "grid.alpha": 0.7,
    "axes.axisbelow": True, "savefig.bbox": "tight", "pdf.fonttype": 42,
})


def selected_overlaps():
    """Initial states of the selected Hamiltonians, uniform and stratified."""
    path = os.path.join(DATA, "selected_overlaps.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    selected = pd.read_csv(os.path.join(DATA, "selected_hamiltonians.csv"))
    with Pool() as pool:
        rows = pool.map(_overlaps_of, selected.to_dict("records"), chunksize=8)
    frame = pd.DataFrame([r for group in rows for r in group])
    frame.to_csv(path, index=False)
    return frame


def _overlaps_of(h):
    _, probabilities = measure(int(h["num_sites"]), int(h["dimension"]), h["delta"], h["J"],
                               (h["Bx"], h["By"], h["Bz"]))
    rng = np.random.default_rng([int(h["num_sites"]), int(h["dimension"]),
                                 int(h["delta"] * 10), int(h["hamiltonian_index"])])
    support = np.flatnonzero(probabilities > OVERLAP_FLOOR)
    uniform = rng.choice(support, size=min(NUM_INITIAL_STATES, support.size), replace=False)
    return [dict(method=method, num_sites=h["num_sites"], density=h["density"],
                 overlap=float(probabilities[s]))
            for method, states in (("uniform", uniform),
                                   ("stratified", stratified(probabilities, rng,
                                                             NUM_INITIAL_STATES)))
            for s in states]


def density_comparison(cells, selected):
    """Rows: sampling scheme. Columns: N. Bars: kept cells per density bin."""
    rows = [
        ("Current, longitudinal ($B_x=0$)", cells[cells.scheme == "longitudinal (current)"]),
        ("Current, transversal ($B_z=0$)", cells[cells.scheme == "transversal (current)"]),
        ("Proposed, tilted field, random",
         cells[cells.scheme == "tilted, b in [0.1, 10], scaled by max(1, delta)"]),
        ("Proposed, tilted field, density-balanced",
         selected.assign(kept=True)),
    ]
    sizes = sorted(selected.num_sites.unique())
    fig, axes = plt.subplots(len(rows), len(sizes), figsize=(7.0, 6.4), sharex=True)
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.08, top=0.93, wspace=0.14, hspace=0.55)
    centres = 0.5 * (DENSITY_BINS[1:] + DENSITY_BINS[:-1])
    for r, (title, frame) in enumerate(rows):
        for c, n in enumerate(sizes):
            ax = axes[r, c]
            block = frame[frame.num_sites == n]
            kept = block[block.kept]
            counts = np.histogram(kept.density, DENSITY_BINS)[0]
            share = counts / max(len(block), 1)
            ax.bar(centres, share, width=0.1 - 0.012, color=BLUE, edgecolor="none")
            ax.set_xlim(0, 1)
            ax.set_xticks([0, 0.5, 1], ["0", "0.5", "1"])
            ax.set_ylim(0, 0.75)
            ax.set_yticks([0, 0.25, 0.5, 0.75])
            dropped = 1 - len(kept) / max(len(block), 1)
            # The balanced selection applies the filters while it selects, so
            # nothing it returns is dropped and the label would say nothing.
            if r < len(rows) - 1:
                ax.text(0.97, 0.93, f"dropped {dropped:.0%}", transform=ax.transAxes,
                        ha="right", va="top", fontsize=6.5, color=INK_SECONDARY)
            if r == 0:
                ax.set_title(f"N = {n}")
            if c == 0:
                ax.set_ylabel("share of cells")
            else:
                ax.set_yticklabels([])
            if r == len(rows) - 1:
                ax.set_xlabel("ground-state density")
        top = axes[r, 0].get_position().y1
        fig.text(0.08, top + (0.035 if r == 0 else 0.012), title, ha="left", va="bottom",
                 fontsize=8, fontweight="bold")
    fig.savefig(os.path.join(FIGURES, "density_comparison.pdf"))
    plt.close(fig)


def mechanism(grid):
    """Density over (field angle, field strength) at N = 12 in 1D."""
    panels = [(1.0, 0.0), (1.0, 2.0), (1.0, 10.0), (-1.0, 0.0), (-1.0, 2.0), (-1.0, 10.0)]
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.2), sharex=True, sharey=True)
    block = grid[(grid.num_sites == 12) & (grid.dimension == 1)]
    ratios = np.sort(block.ratio.unique())
    angles = np.sort(block.angle.unique())
    for ax, (sign, delta) in zip(axes.flat, panels):
        s = block[(block.J == sign) & (block.delta == delta)]
        density = s.pivot(index="angle", columns="ratio", values="density").loc[angles, ratios]
        kept = s.pivot(index="angle", columns="ratio", values="kept").loc[angles, ratios]
        values = np.ma.masked_where(~kept.to_numpy(dtype=bool), density.to_numpy())
        ax.set_facecolor(DROPPED)
        mesh = ax.pcolormesh(np.arange(len(ratios) + 1), np.arange(len(angles) + 1), values,
                             cmap=SEQUENTIAL, vmin=0, vmax=1, edgecolors="white",
                             linewidth=0.6)
        ax.grid(False)
        name = "AFM, $J>0$" if sign > 0 else "FM, $J<0$"
        ax.set_title(f"{name}, $\\Delta = {delta:g}$")
        ax.set_xticks(np.arange(len(ratios)) + 0.5)
        ax.set_xticklabels([f"{r:.2g}" for r in ratios], rotation=90)
        ax.set_yticks(np.arange(len(angles)) + 0.5)
        ax.set_yticklabels([f"{a:g}" for a in angles])
        for spine in ax.spines.values():
            spine.set_visible(False)
    for ax in axes[1]:
        ax.set_xlabel("$b = |B| / |J|$")
    for ax in axes[:, 0]:
        ax.set_ylabel("field angle $\\theta$ from $z$ [deg]")
    fig.tight_layout()
    bar = fig.colorbar(mesh, ax=axes, shrink=0.8, pad=0.015)
    bar.set_label("ground-state density")
    bar.outline.set_visible(False)
    fig.savefig(os.path.join(FIGURES, "mechanism.pdf"))
    plt.close(fig)


def overlap_comparison(overlaps, chosen):
    """Distribution of log10(initial overlap), and its joint with the density."""
    decades = np.arange(-12, 0.5, 0.5)
    rows = [
        ("Current\nlongitudinal", overlaps[(overlaps.scheme == "longitudinal (current)")
                                           & (overlaps.method == "uniform")]),
        ("Current\ntransversal", overlaps[(overlaps.scheme == "transversal (current)")
                                          & (overlaps.method == "uniform")]),
        ("Balanced selection\nuniform from support", chosen[chosen.method == "uniform"]),
        ("Balanced selection\nstratified in log overlap", chosen[chosen.method == "stratified"]),
    ]
    fig = plt.figure(figsize=(7.0, 4.4))
    grid = fig.add_gridspec(2, 4, height_ratios=[1, 1.25], hspace=0.55, wspace=0.12)
    for c, (title, frame) in enumerate(rows):
        ax = fig.add_subplot(grid[0, c])
        logs = np.log10(frame.overlap)
        ax.hist(logs, bins=decades, weights=np.full(len(logs), 1 / len(logs)),
                color=BLUE, rwidth=0.85)
        ax.set_title(title, fontsize=7.5)
        ax.set_xlim(-12.2, 0.2)
        ax.set_ylim(0, 0.2)
        ax.set_xlabel("$\\log_{10}$ overlap")
        if c == 0:
            ax.set_ylabel("share of initial states")
        else:
            ax.set_yticklabels([])
        above = (frame.overlap > 1e-2).mean()
        ax.text(0.03, 0.95, f"$>10^{{-2}}$: {above:.0%}", transform=ax.transAxes,
                va="top", fontsize=6.5, color=INK_SECONDARY)
    for c, method in enumerate(["uniform", "stratified"]):
        ax = fig.add_subplot(grid[1, 2 * c:2 * c + 2])
        frame = chosen[chosen.method == method]
        ax.scatter(frame.density, np.log10(frame.overlap), s=4, color=BLUE, alpha=0.35,
                   linewidths=0)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-12.5, 0.3)
        ax.set_xlabel("ground-state density")
        ax.set_title(f"Balanced selection, initial states {method}", fontsize=7.5)
        if c == 0:
            ax.set_ylabel("$\\log_{10}$ overlap")
        else:
            ax.set_yticklabels([])
    fig.savefig(os.path.join(FIGURES, "overlap_comparison.pdf"))
    plt.close(fig)


def main():
    os.makedirs(FIGURES, exist_ok=True)
    grid = pd.read_csv(os.path.join(DATA, "grid_scan.csv"))
    cells = pd.read_csv(os.path.join(DATA, "schemes_cells.csv"))
    overlaps = pd.read_csv(os.path.join(DATA, "schemes_overlaps.csv"))
    selected = pd.read_csv(os.path.join(DATA, "selected_hamiltonians.csv"))
    density_comparison(cells, selected)
    mechanism(grid)
    overlap_comparison(overlaps, selected_overlaps())
    print(f"figures written to {FIGURES}")


if __name__ == "__main__":
    main()
