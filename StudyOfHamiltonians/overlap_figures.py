"""
Figures of the initial-overlap study, written to figures/ for overlap_report.tex.

Needs data/initial_overlaps.csv and data/ground_state_summary.csv from
initial_states.py.

    python overlap_figures.py
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

# Same look as the figures of report.pdf.
from make_figures import BLUE, DATA, FIGURES, GRID, INK_SECONDARY

SIZES = [6, 8, 10, 12]
ROWS = [
    ("current", "Now: uniform from the support, $p > 10^{-12}$"),
    ("dense", "Proposed: uniform from the states the density counts, $p > 0.1/2^N$"),
    ("dense_filled", "The same, topped up to six from the rest of the support"),
    ("stratified", "Stratified in $\\log_{10} p$ over the whole support"),
    ("born", "Drawn with probability $p$ (as a measurement would)"),
]


def overlap_histograms(overlaps):
    """Rows: scheme. Columns: N. Bars: share of runs per half decade of overlap."""
    edges = np.arange(-12, 0.5, 0.5)
    fig, axes = plt.subplots(len(ROWS), len(SIZES), figsize=(7.0, 7.6), sharex=True,
                             sharey=True)
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.06, top=0.95, wspace=0.12, hspace=0.62)
    for r, (scheme, title) in enumerate(ROWS):
        for c, n in enumerate(SIZES):
            ax = axes[r, c]
            block = overlaps[(overlaps.scheme == scheme) & (overlaps.num_sites == n)]
            logs = np.log10(block.overlap)
            ax.hist(logs.clip(lower=-11.99), bins=edges, weights=np.full(len(logs), 1 / len(logs)),
                    color=BLUE, rwidth=0.85)
            ax.axvline(np.log10(0.1 / 2 ** n), color=INK_SECONDARY, lw=0.8, ls="--")
            ax.text(0.03, 0.93, f"median $10^{{{logs.median():.1f}}}$\n{len(block)} runs",
                    transform=ax.transAxes, va="top", fontsize=6.3, color=INK_SECONDARY)
            ax.set_xlim(-12.2, 0.2)
            ax.set_xticks([-12, -9, -6, -3, 0])
            ax.set_ylim(0, 0.42)
            if r == 0:
                ax.set_title(f"N = {n}")
            if c == 0:
                ax.set_ylabel("share of runs")
            if r == len(ROWS) - 1:
                ax.set_xlabel("$\\log_{10}$ overlap")
        top = axes[r, 0].get_position().y1
        fig.text(0.08, top + (0.03 if r == 0 else 0.008), title, ha="left", va="bottom",
                 fontsize=8, fontweight="bold")
    fig.savefig(os.path.join(FIGURES, "overlap_schemes.pdf"))
    plt.close(fig)


def overlap_against_density(overlaps, summary):
    """Rows: now and proposed. Columns: N. Grey: the largest overlap a Hamiltonian offers."""
    rows = ROWS[:2]
    fig, axes = plt.subplots(len(rows), len(SIZES), figsize=(7.0, 3.9), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.12, top=0.9, wspace=0.12, hspace=0.5)
    for r, (scheme, title) in enumerate(rows):
        for c, n in enumerate(SIZES):
            ax = axes[r, c]
            tops = summary[summary.num_sites == n]
            ax.scatter(np.log10(tops.density), np.log10(tops.top), s=3, color=GRID, linewidths=0)
            block = overlaps[(overlaps.scheme == scheme) & (overlaps.num_sites == n)]
            ax.scatter(np.log10(block.density), np.log10(block.overlap).clip(lower=-12), s=3,
                       color=BLUE, alpha=0.3, linewidths=0)
            ax.axhline(np.log10(0.1 / 2 ** n), color=INK_SECONDARY, lw=0.8, ls="--")
            ax.set_xlim(-3.7, 0.1)
            ax.set_ylim(-12.5, 0.4)
            ax.set_yticks([-12, -9, -6, -3, 0])
            if r == 0:
                ax.set_title(f"N = {n}")
            if c == 0:
                ax.set_ylabel("$\\log_{10}$ overlap")
            if r == len(rows) - 1:
                ax.set_xlabel("$\\log_{10}$ density")
        top = axes[r, 0].get_position().y1
        fig.text(0.08, top + (0.055 if r == 0 else 0.012), title, ha="left", va="bottom",
                 fontsize=8, fontweight="bold")
    # Stand-ins rather than the scatters themselves, which are too faint to read.
    handles = [Line2D([], [], ls="", marker="o", ms=4, color=colour, label=label)
               for colour, label in ((GRID, "largest $p$ of the Hamiltonian"),
                                     (BLUE, "initial states"))]
    fig.legend(handles=handles, loc="upper right", ncol=2, frameon=False, fontsize=7,
               bbox_to_anchor=(0.99, 1.0))
    fig.savefig(os.path.join(FIGURES, "overlap_vs_density.pdf"))
    plt.close(fig)


def main():
    overlaps = pd.read_csv(os.path.join(DATA, "initial_overlaps.csv"))
    summary = pd.read_csv(os.path.join(DATA, "ground_state_summary.csv"))
    overlap_histograms(overlaps)
    overlap_against_density(overlaps, summary)
    print(f"figures written to {FIGURES}")


if __name__ == "__main__":
    main()
