"""
Figures of the paper: BARK against SKQD at a fixed number of bitstrings, Heisenberg model.

Reads the merged results of EqualNumberOfBitstrings.py and writes every figure of
the paper to ``PaperFigures/``, numbered in the order they appear:

    fig1_system_size                     all runs, one column per system size N
    fig2_ground_state_density            N = 12, clustered by ground-state density
    fig3_initial_overlap                 N = 12, clustered by initial overlap
    figA1_lattice_dimension              N = 12, 1D against 2D lattices
    figA2_ground_state_density_other_N   Fig. 2 for N = 6, 8 and 10
    figA3_initial_overlap_other_N        Fig. 3 for N = 6, 8 and 10
    figA4_choice_of_statistic            median against mean, and the mean's floor dependence

Every figure except A4 pairs two rows of panels per cluster of runs: the median
infidelity 1 - F against the number of bitstrings K / 2^N with the interquartile
range as band, and below it the fraction of runs that have converged, i.e. reached
1 - F < CONVERGED.

Why the median. A run that has converged has 1 - F = 0 to machine precision, and
is floored at INFIDELITY_FLOOR to be drawn on a log axis. Wherever some runs have
converged and others have not -- which is most of the budget range here -- the
mean of log(1 - F) is pulled down by those runs by an amount that depends on
that arbitrary floor (figA4 shows it moving with the floor), and it reports a
gradual gain that no typical run shows. The median does not depend on it. What
the median hides in exchange, the minority of runs that already converged, is
exactly what the converged fraction shows.

The density and overlap bins are equal-frequency quantile bins taken within each
system size, the same cuts as ``EqualNumberOfBitstrings.py plot``, so the folders
there and the panels here hold the same runs. Next to the figures,
``bin_edges.csv`` lists every cut as a number and ``figure_data.csv`` every
plotted value with the number of runs behind it, for the captions and the text.

Usage
-----
    python PaperFigures.py
    python PaperFigures.py --data equal_bitstrings_heisenberg_results.csv --format png
"""

import argparse
import string
from collections import namedtuple
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import NullLocator

import EqualNumberOfBitstrings as study

DEFAULT_DATA = "equal_bitstrings_heisenberg_results.csv"
DEFAULT_OUTPUT = Path("PaperFigures")

# The system size of the main-text figures; the others go to the appendix.
MAIN_N = 12

# A run counts as converged below this infidelity. It sits six decades above the
# floor, so the floor's exact value does not decide who has converged.
CONVERGED = 1e-10
INFIDELITY_FLOOR = study.INFIDELITY_FLOOR
# The second floor of figA4, to show how much the mean depends on the first.
ALTERNATIVE_FLOOR = 1e-12

NUM_BINS = study.DEFAULT_NUM_BINS

# Notation of the binned quantities, as it appears in the panel titles. Change
# these to whatever symbols the paper defines.
GSDENS_SYMBOL = r"\rho_0"
OVERLAP_SYMBOL = r"|\langle b_0|\psi_0\rangle|^2"

# Widths of a two-column layout (revtex), in inches.
TEXT_WIDTH = 7.05

# Heights of the two rows of a pair, in inches: the infidelity carries the
# figure, the converged fraction only has to resolve 0 to 1.
INFIDELITY_HEIGHT = 1.55
CONVERGED_HEIGHT = 0.95

# Interquartile bands only for the two protocols the paper compares. Simplified
# BARK lies on top of BARK and the ceiling's band would cover both.
BANDED = ("BARK", "SKQD")
BAND_ALPHA = 0.15
# Every grid point carries a marker otherwise, and at N = 12 that is thirty of
# them in a panel two inches wide.
MARK_EVERY = 2

INK = "#2b2a28"
MUTED_INK = "#6b6a66"

Panel = namedtuple("Panel", ["title", "group"])     # group None: no runs in this bin


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

def load(path: str) -> pd.DataFrame:
    """The study's rows, with the same cells dropped as in the study's own plots."""
    data = pd.read_csv(path)
    data = study.drop_trivial_ground_states(data)
    return study.drop_small_gaps(data)


def curves(group: pd.DataFrame, floor: float = INFIDELITY_FLOOR) -> pd.DataFrame:
    """Every run of ``group`` as log10(1 - F) on one common budget grid."""
    group = group.assign(Log_Infidelity=np.log10(np.clip(1.0 - group["Fidelity"], floor, 1.0)))
    return study.on_common_grid(group)


def summarize(group: pd.DataFrame, floor: float = INFIDELITY_FLOOR) -> pd.DataFrame:
    """
    Per algorithm and budget: median and quartiles, mean and its standard error,
    and the fraction of converged runs -- all of log10(1 - F).
    """
    if floor >= CONVERGED:
        raise ValueError(f"a floor of {floor:g} leaves nothing below the "
                         f"convergence threshold {CONVERGED:g}")
    frame = curves(group, floor)
    frame["Converged"] = frame["Log_Infidelity"] < np.log10(CONVERGED)
    groups = frame.groupby(["Algorithm", "Budget_Fraction"])
    values = groups["Log_Infidelity"]
    stats = values.agg(median="median", mean="mean", std="std", runs="count")
    stats["q1"] = values.quantile(0.25)
    stats["q3"] = values.quantile(0.75)
    stats["sem"] = stats["std"].fillna(0.0) / np.sqrt(stats["runs"])
    stats["converged"] = groups["Converged"].mean()
    return stats.drop(columns="std").reset_index()


def runs_of(group: pd.DataFrame) -> pd.DataFrame:
    """One row per run, which is what quantile cuts are taken on."""
    return group.drop_duplicates(study.RUN_KEY)


def binned(data: pd.DataFrame, column: str, num_sites: int) -> tuple:
    """
    The runs of one system size split into quantile bins of ``column``.

    Returns ``({bin name: rows}, edges)``. The cuts are taken on the runs of that
    system size only, as in ``EqualNumberOfBitstrings.add_facet_labels`` with
    ``scope="per-N"``: both densities and the overlaps shrink with N, and global
    cuts would mostly re-sort the runs by system size.
    """
    block = data[data["Number_of_Sites"] == num_sites]
    edges = study.quantile_edges(runs_of(block)[column], NUM_BINS)
    labels = study.apply_bins(block[column], edges)
    return {name: rows for name, rows in block.groupby(labels)}, edges


# --------------------------------------------------------------------------- #
# Labels
# --------------------------------------------------------------------------- #

def format_number(value: float) -> str:
    """A bin edge as mathtext: plain near unity, in powers of ten otherwise."""
    if value == 0:
        return "0"
    exponent = int(np.floor(np.log10(abs(value))))
    if -2 <= exponent <= 1:
        return f"{value:.2g}"
    return rf"{value / 10.0 ** exponent:.1f}\times 10^{{{exponent}}}"


def bin_title(name: str, edges: np.ndarray, symbol: str) -> str:
    """
    What a bin holds, e.g. ``low: rho_0 <= 0.11``.

    ``pd.cut`` closes the bins on the right, so a run on an edge belongs to the
    bin below it -- hence <= and >.
    """
    names = study.bin_names(edges.size - 1)
    index = names.index(name)
    lower, upper = format_number(edges[index]), format_number(edges[index + 1])
    if index == 0:
        condition = rf"{symbol} \leq {upper}"
    elif index == len(names) - 1:
        condition = rf"{symbol} > {lower}"
    else:
        condition = rf"{lower} < {symbol} \leq {upper}"
    return f"{name}: ${condition}$"


# --------------------------------------------------------------------------- #
# Drawing
# --------------------------------------------------------------------------- #

def apply_style() -> None:
    plt.rcParams.update({
        "font.family": "STIXGeneral",       # matches Times in revtex
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 8,
        "axes.edgecolor": MUTED_INK,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": MUTED_INK,
        "ytick.color": MUTED_INK,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.width": 0.4,
        "pdf.fonttype": 42,                 # editable text in the PDF
        "savefig.dpi": 300,
    })


def style(ax: plt.Axes) -> None:
    # ax.set_xscale("log")
    ax.grid(True, which="major", **study.GRID_KWARGS)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def draw_line(ax: plt.Axes, rows: pd.DataFrame, name: str, colour: str, marker,
              linestyle: str, values: np.ndarray) -> None:
    ax.plot(rows["Budget_Fraction"], values, color=colour, linestyle=linestyle,
            linewidth=1.2, marker=marker, markersize=2.5, markevery=MARK_EVERY,
            label=name, zorder=1 if name == "Ceiling" else 2)


def draw_infidelity(ax: plt.Axes, stats: pd.DataFrame, statistic: str = "median") -> None:
    """Median (or mean) infidelity with the interquartile range (or SEM) as band."""
    for name, colour, marker, linestyle in study.ALGORITHMS:
        rows = stats[stats["Algorithm"] == name].sort_values("Budget_Fraction")
        if rows.empty:
            continue
        if statistic == "median":
            centre, lower, upper = rows["median"], rows["q1"], rows["q3"]
        else:
            centre = rows["mean"]
            lower, upper = centre - rows["sem"], centre + rows["sem"]
        draw_line(ax, rows, name, colour, marker, linestyle, 10.0 ** centre.to_numpy())
        if name in BANDED:
            ax.fill_between(rows["Budget_Fraction"], 10.0 ** lower.to_numpy(),
                            10.0 ** upper.to_numpy(), color=colour, alpha=BAND_ALPHA,
                            linewidth=0)
    style(ax)
    ax.set_yscale("log")
    ax.set_ylim(10.0 ** -16.8, 3.0)
    ax.set_yticks([1e0, 1e-4, 1e-8, 1e-12, 1e-16])
    ax.yaxis.set_minor_locator(NullLocator())


def draw_converged(ax: plt.Axes, stats: pd.DataFrame) -> None:
    """Fraction of runs with 1 - F below CONVERGED."""
    for name, colour, marker, linestyle in study.ALGORITHMS:
        rows = stats[stats["Algorithm"] == name].sort_values("Budget_Fraction")
        if not rows.empty:
            draw_line(ax, rows, name, colour, marker, linestyle, rows["converged"].to_numpy())
    style(ax)
    ax.set_ylim(-0.04, 1.04)
    ax.set_yticks([0.0, 0.5, 1.0])


def budget_limits(group: pd.DataFrame) -> tuple:
    """x-range of a panel: from one bitstring to the largest budget recorded."""
    fractions = group["Budget_Fraction"]
    return fractions.min() / 1.3, fractions.max() * 1.3


def figure_legend(fig: plt.Figure, ax: plt.Axes) -> None:
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=len(labels), frameon=False)


class Recorder:
    """Collects every plotted number, so the text can quote them."""

    def __init__(self):
        self.frames = []

    def add(self, figure: str, panel: str, title: str, stats: pd.DataFrame, **extra) -> None:
        self.frames.append(stats.assign(Figure=figure, Panel=panel,
                                        Title=title.replace("\n", " "), **extra))

    def save(self, path: Path) -> None:
        frame = pd.concat(self.frames, ignore_index=True)
        first = ["Figure", "Panel", "Title"]
        frame[first + [c for c in frame.columns if c not in first]].to_csv(path, index=False)


def paired_figure(name: str, rows: list, recorder: Recorder) -> plt.Figure:
    """
    One pair of rows per entry of ``rows`` -- infidelity above, converged
    fraction below -- and one column per panel.

    The panels of a row share their budget range, those of different rows need
    not (fig1 and the appendix figures change N from one to the next), so every
    pair gets its own x-limits and the infidelity row borrows the tick labels of
    the converged row underneath it.
    """
    num_cols = max(len(row) for row in rows)
    heights = [INFIDELITY_HEIGHT, CONVERGED_HEIGHT] * len(rows)
    fig, axes = plt.subplots(2 * len(rows), num_cols, layout="constrained",
                             figsize=(TEXT_WIDTH, sum(heights) + 0.35),
                             gridspec_kw={"height_ratios": heights}, squeeze=False)
    letters = iter(string.ascii_lowercase)
    legend_axis = None

    for r, row in enumerate(rows):
        top, bottom = axes[2 * r], axes[2 * r + 1]
        stats = [summarize(panel.group) if panel.group is not None else None for panel in row]
        for draw, line in ((draw_infidelity, top), (draw_converged, bottom)):
            for c, (panel, panel_stats) in enumerate(zip(row, stats)):
                ax = line[c]
                if panel_stats is None:
                    ax.axis("off")
                    if draw is draw_infidelity:
                        ax.text(0.5, 0.5, f"{panel.title}\n(no runs in this bin)",
                                transform=ax.transAxes, ha="center", va="center",
                                color=MUTED_INK)
                    continue
                letter = next(letters)
                draw(ax, panel_stats)
                ax.set_xlim(*budget_limits(panel.group))
                ax.set_title(f"({letter})", loc="left", fontweight="bold")
                if draw is draw_infidelity:
                    ax.set_title(panel.title)
                    ax.tick_params(labelbottom=False)
                    recorder.add(name, letter, panel.title, panel_stats)
                    legend_axis = legend_axis or ax
                if c > 0:
                    ax.tick_params(labelleft=False)
        top[0].set_ylabel(r"Infidelity $1-F$")
        bottom[0].set_ylabel("Fraction\nconverged")
        for c in range(len(row), num_cols):
            top[c].axis("off")
            bottom[c].axis("off")

    for ax in axes[-1]:
        ax.set_xlabel(r"Bitstrings $K/2^N$")
    figure_legend(fig, legend_axis)
    return fig


# --------------------------------------------------------------------------- #
# The figures
# --------------------------------------------------------------------------- #

def by_size(data: pd.DataFrame) -> list:
    sizes = sorted(data["Number_of_Sites"].unique())
    return [[Panel(f"$N = {n}$", data[data["Number_of_Sites"] == n]) for n in sizes]]


def by_bin(data: pd.DataFrame, column: str, symbol: str, sizes: list,
           edge_rows: list, facet: str, with_size: bool) -> list:
    """One row of panels per system size, one panel per quantile bin of ``column``."""
    rows = []
    for n in sizes:
        groups, edges = binned(data, column, n)
        names = study.bin_names(edges.size - 1)
        row = []
        # Columns stay low / mid / high even where ties leave fewer bins.
        for name in study.bin_names(NUM_BINS):
            if name in groups:
                title = bin_title(name, edges, symbol)
                index = names.index(name)
                edge_rows.append({"Facet": facet, "N": n, "Bin": name,
                                  "Lower": edges[index], "Upper": edges[index + 1],
                                  "Runs": len(runs_of(groups[name]))})
            else:
                title = f"{name}: merged by ties"
            if with_size:
                title = f"$N = {n}$, {title}"
            row.append(Panel(title, groups.get(name)))
        rows.append(row)
    return rows


def by_dimension(data: pd.DataFrame) -> list:
    block = data[data["Number_of_Sites"] == MAIN_N]
    return [[Panel(f"{d}D lattice, $N = {MAIN_N}$", rows)
             for d, rows in block.groupby("Dimensions")]]


def statistic_figure(data: pd.DataFrame, recorder: Recorder) -> plt.Figure:
    """
    figA4: the same runs (N = MAIN_N) summarised three ways.

    The median, the mean with its standard error, and the mean again with the
    floor raised from INFIDELITY_FLOOR to ALTERNATIVE_FLOOR. The median of the
    first and third would be identical; the means are not, which is the reason
    the paper uses the median.
    """
    group = data[data["Number_of_Sites"] == MAIN_N]
    variants = (
        ("median", INFIDELITY_FLOOR, "median, interquartile range"),
        ("mean", INFIDELITY_FLOOR, rf"mean $\pm$ SEM, floor $10^{{{np.log10(INFIDELITY_FLOOR):.0f}}}$"),
        ("mean", ALTERNATIVE_FLOOR, rf"mean $\pm$ SEM, floor $10^{{{np.log10(ALTERNATIVE_FLOOR):.0f}}}$"),
    )
    fig, axes = plt.subplots(1, len(variants), layout="constrained", sharey=True,
                             figsize=(TEXT_WIDTH, INFIDELITY_HEIGHT + 0.75))
    for ax, letter, (statistic, floor, title) in zip(axes, string.ascii_lowercase, variants):
        stats = summarize(group, floor)
        draw_infidelity(ax, stats, statistic)
        ax.set_xlim(*budget_limits(group))
        ax.set_title(f"({letter})", loc="left", fontweight="bold")
        ax.set_title(title)
        ax.set_xlabel(r"Bitstrings $K/2^N$")
        recorder.add("figA4_choice_of_statistic", letter, title, stats, Floor=floor)
    axes[0].set_ylabel(r"Infidelity $1-F$")
    figure_legend(fig, axes[0])
    return fig


def make_figures(data_file: str, output: Path, fmt: str) -> None:
    apply_style()
    data = load(data_file)
    output.mkdir(parents=True, exist_ok=True)
    recorder, edge_rows = Recorder(), []
    other_sizes = sorted(n for n in data["Number_of_Sites"].unique() if n != MAIN_N)

    figures = [
        ("fig1_system_size", lambda: by_size(data)),
        ("fig2_ground_state_density",
         lambda: by_bin(data, "Ground_State_Density", GSDENS_SYMBOL, [MAIN_N],
                        edge_rows, "gsdens", with_size=False)),
        ("fig3_initial_overlap",
         lambda: by_bin(data, "Overlap", OVERLAP_SYMBOL, [MAIN_N],
                        edge_rows, "overlap", with_size=False)),
        ("figA1_lattice_dimension", lambda: by_dimension(data)),
        ("figA2_ground_state_density_other_N",
         lambda: by_bin(data, "Ground_State_Density", GSDENS_SYMBOL, other_sizes,
                        edge_rows, "gsdens", with_size=True)),
        ("figA3_initial_overlap_other_N",
         lambda: by_bin(data, "Overlap", OVERLAP_SYMBOL, other_sizes,
                        edge_rows, "overlap", with_size=True)),
    ]
    for name, rows in figures:
        save(paired_figure(name, rows(), recorder), output / f"{name}.{fmt}")
    save(statistic_figure(data, recorder), output / f"figA4_choice_of_statistic.{fmt}")

    recorder.save(output / "figure_data.csv")
    pd.DataFrame(edge_rows).to_csv(output / "bin_edges.csv", index=False)
    print(f"Wrote {output / 'figure_data.csv'} and {output / 'bin_edges.csv'}")


def save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", default=DEFAULT_DATA,
                        help="merged results of EqualNumberOfBitstrings.py")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="folder the figures are written to")
    parser.add_argument("--format", default="pdf", choices=("pdf", "png", "svg"),
                        help="file format of the figures")
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    make_figures(args.data, args.output, args.format)


if __name__ == "__main__":
    main()
