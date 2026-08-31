"""
This script uses the result from SystematicStudy.py to plot the results of the systematic study
comparing SKQD and BARK protocols for random spin models.

The main plot is always the pool size against the fidelity, with one line per protocol. The pool
size is always normalised to the Hilbert space dimension 2**Number_of_Sites, so that systems of
different size can be compared. Every figure is written as a PDF into a structured output folder:

    plots/
      01_individual/            one plot per (Hamiltonian, initial state) -- no clustering at all
      02_density_bins/          clustered by Ground_State_Density / Hamiltonian_Density (3 bins each)
      03_max_interactions/      clustered by Max_Interactions
      04_number_of_sites/       clustered by Number_of_Sites

A pool size of ``inf`` means the protocol never reached the target fidelity. Those runs are dropped
from the averages and reported separately in the figure annotation, so that a missing point is never
silently read as a small pool.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Categorical slots 1 and 2 of the validated default palette (blue / orange).
PROTOCOLS = (
    # name    raw column          normalised column  colour     marker  linestyle
    ("BARK", "BARK_Pool_Size", "BARK_Norm", "#2a78d6", "o", "-"),
    ("SKQD", "SKQD_Pool_Size", "SKQD_Norm", "#eb6834", "s", "--"),
)

GRID_KWARGS = dict(color="#c9c8c3", linewidth=0.6, alpha=0.7)
BIN_LABELS = ("low", "mid", "high")
OUTPUT_ROOT = Path("ising_plots")


# --------------------------------------------------------------------------------------
# data handling
# --------------------------------------------------------------------------------------

def load_data(data_file: str) -> pd.DataFrame:
    """Read the study results and add the Hilbert-space-normalised pool sizes."""
    data = pd.read_csv(data_file)
    data["Hilbert_Space_Size"] = 2 ** data["Number_of_Sites"]
    for _, raw_column, norm_column, *_ in PROTOCOLS:
        # inf marks a run that never reached the target fidelity -> not a number, not a big number
        finite = data[raw_column].replace([np.inf, -np.inf], np.nan)
        data[norm_column] = finite / data["Hilbert_Space_Size"]
    return data


def add_density_bins(data: pd.DataFrame, n_bins: int = 3) -> dict:
    """Bin both density columns into ``n_bins`` equal-width bins.

    Returns a mapping column -> readable label for every bin, so the bin edges can be put
    into the figure titles instead of a bare 'low'/'mid'/'high'.
    """
    edge_labels = {}
    for column in ("Ground_State_Density", "Hamiltonian_Density"):
        labels = list(BIN_LABELS[:n_bins])
        binned, edges = pd.cut(data[column], bins=n_bins, labels=labels, retbins=True)
        data[f"{column}_Bin"] = binned
        edge_labels[column] = {
            label: f"{label} [{edges[i]:.3f}, {edges[i + 1]:.3f}]"
            for i, label in enumerate(labels)
        }
    return edge_labels


def aggregate_by_fidelity(group: pd.DataFrame) -> pd.DataFrame:
    """Mean, standard deviation, sample count and failure count per fidelity."""
    rows = []
    for fidelity, sub in group.groupby("Fidelity"):
        row = {"Fidelity": fidelity}
        for name, raw_column, norm_column, *_ in PROTOCOLS:
            values = sub[norm_column].dropna()
            row[f"{name}_mean"] = values.mean() if len(values) else np.nan
            # a single sample has no spread rather than an undefined one
            row[f"{name}_std"] = values.std(ddof=1) if len(values) > 1 else 0.0
            row[f"{name}_n"] = len(values)
            row[f"{name}_failed"] = int(sub[norm_column].isna().sum())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("Fidelity")


# --------------------------------------------------------------------------------------
# plotting
# --------------------------------------------------------------------------------------

def make_one_plot(group: pd.DataFrame, ax: plt.Axes, title: str, show_errors: bool = True) -> None:
    """Plot normalised pool size against fidelity for both protocols on ``ax``.

    Args:
        group: the rows of the study that belong to this cluster.
        ax: axes to draw on.
        title: axes title.
        show_errors: draw the standard deviation as error bars (switched off for single runs).
    """
    stats = aggregate_by_fidelity(group)

    for name, _, _, colour, marker, linestyle in PROTOCOLS:
        mean = stats[f"{name}_mean"].to_numpy(dtype=float)
        std = stats[f"{name}_std"].to_numpy(dtype=float) if show_errors else None
        ax.errorbar(
            stats["Fidelity"], mean, yerr=std,
            marker=marker, markersize=7, linewidth=2, linestyle=linestyle,
            color=colour, markeredgecolor="white", markeredgewidth=0.8,
            capsize=3, elinewidth=1.2, label=name,
        )
        if show_errors and std is not None:
            # the normalised pool size lives in [0, 1] by construction, so clip the band there
            ax.fill_between(stats["Fidelity"], np.clip(mean - std, 0.0, 1.0),
                            np.clip(mean + std, 0.0, 1.0), color=colour, alpha=0.12,
                            linewidth=0)

    ax.set_xlabel("Target fidelity")
    ax.set_ylabel("Pool size / Hilbert space dimension")
    ax.set_title(title, fontsize=10)
    ax.set_xticks(sorted(group["Fidelity"].unique()))
    ax.set_ylim(0.0, min(ax.get_ylim()[1], 1.02))
    ax.grid(True, axis="y", **GRID_KWARGS)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(frameon=False, fontsize=9)

    _annotate_counts(ax, stats)


def _annotate_counts(ax: plt.Axes, stats: pd.DataFrame) -> None:
    """State how many runs entered each average and how many never converged."""
    counts = ", ".join(
        f"{f:.2f}: " + "/".join(str(int(stats.loc[stats['Fidelity'] == f, f'{name}_n'].iloc[0]))
                                for name, *_ in PROTOCOLS)
        for f in stats["Fidelity"]
    )
    lines = [f"n (BARK/SKQD) per fidelity -- {counts}"]

    failures = []
    for name, *_ in PROTOCOLS:
        total_failed = int(stats[f"{name}_failed"].sum())
        if total_failed:
            failures.append(f"{name}: {total_failed}")
    if failures:
        lines.append("runs without convergence -- " + ", ".join(failures))

    ax.text(0.0, -0.22, "\n".join(lines), transform=ax.transAxes, fontsize=7,
            color="#52514e", va="top")


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def single_figure(group: pd.DataFrame, title: str, path: Path, show_errors: bool = True) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    make_one_plot(group, ax, title, show_errors=show_errors)
    save_figure(fig, path)


def overview_figure(groups: list, title: str, path: Path, n_columns: int = 3) -> None:
    """Put every cluster of one family on a single grid page for quick comparison."""
    n_rows = int(np.ceil(len(groups) / n_columns))
    fig, axes = plt.subplots(n_rows, n_columns, figsize=(5.0 * n_columns, 4.0 * n_rows),
                             squeeze=False)
    flat_axes = axes.flatten()
    for ax, (subtitle, group) in zip(flat_axes, groups):
        make_one_plot(group, ax, subtitle)
    for ax in flat_axes[len(groups):]:
        ax.axis("off")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save_figure(fig, path)


def _slug(value: float) -> str:
    return f"{value:.4f}".replace(".", "p")


# --------------------------------------------------------------------------------------
# the four plot families
# --------------------------------------------------------------------------------------

def plot_individual(data: pd.DataFrame, root: Path) -> int:
    """One plot per (Hamiltonian, initial state): no clustering, no averaging."""
    hamiltonian_keys = ["Number_of_Sites", "Max_Interactions",
                        "Ground_State_Density", "Hamiltonian_Density"]
    written = 0
    for (n_sites, max_int, gs_density, h_density), ham_group in data.groupby(hamiltonian_keys):
        folder = (root / f"N{n_sites}_maxint{max_int}"
                         f"_gsd{_slug(gs_density)}_hd{_slug(h_density)}")
        # order the initial states by overlap so the file names are readable
        overlaps = sorted(ham_group["Overlap"].unique())
        for rank, overlap in enumerate(overlaps, start=1):
            group = ham_group[ham_group["Overlap"] == overlap]
            title = (f"N = {n_sites}, max. interactions = {max_int}\n"
                     f"GS density = {gs_density:.3f}, H density = {h_density:.3f}, "
                     f"overlap = {overlap:.2e}")
            single_figure(group, title, folder / f"overlap{rank:02d}_{overlap:.2e}.pdf",
                          show_errors=False)
            written += 1
    return written


def plot_density_bins(data: pd.DataFrame, edge_labels: dict, root: Path) -> int:
    """Clustered by ground state density and Hamiltonian density, three bins each."""
    written = 0

    # marginal clustering: one plot per bin of each density
    for column in ("Ground_State_Density", "Hamiltonian_Density"):
        folder = root / column.lower()
        overview = []
        for label in BIN_LABELS:
            group = data[data[f"{column}_Bin"] == label]
            if group.empty:
                print(f"  skipping empty bin {column} = {label}")
                continue
            title = f"{column.replace('_', ' ')}: {edge_labels[column][label]}"
            single_figure(group, title, folder / f"{label}.pdf")
            overview.append((title, group))
            written += 1
        if overview:
            overview_figure(overview, f"Clustered by {column.replace('_', ' ')}",
                            folder / "_overview.pdf", n_columns=len(overview))
            written += 1

    # joint clustering: 3 x 3 bins
    folder = root / "joint"
    overview = []
    for gs_label in BIN_LABELS:
        for h_label in BIN_LABELS:
            group = data[(data["Ground_State_Density_Bin"] == gs_label)
                         & (data["Hamiltonian_Density_Bin"] == h_label)]
            if group.empty:
                print(f"  skipping empty bin GS = {gs_label}, H = {h_label}")
                continue
            title = (f"GS density {edge_labels['Ground_State_Density'][gs_label]}\n"
                     f"H density {edge_labels['Hamiltonian_Density'][h_label]}")
            single_figure(group, title, folder / f"gsd_{gs_label}__hd_{h_label}.pdf")
            overview.append((title, group))
            written += 1
    if overview:
        overview_figure(overview, "Clustered by ground state and Hamiltonian density",
                        folder / "_overview.pdf")
        written += 1
    return written


def plot_by_column(data: pd.DataFrame, column: str, root: Path, label: str) -> int:
    """Clustered by a single discrete parameter (max interactions, number of sites)."""
    written = 0
    overview = []
    for value, group in data.groupby(column):
        title = f"{label} = {value}"
        single_figure(group, title, root / f"{column.lower()}_{value}.pdf")
        overview.append((title, group))
        written += 1
    if overview:
        overview_figure(overview, f"Clustered by {label.lower()}", root / "_overview.pdf",
                        n_columns=len(overview))
        written += 1
    return written


def plot_results(data_file: str, output_root: Path = OUTPUT_ROOT) -> None:
    """Produce every figure family from the study results."""
    data = load_data(data_file)
    edge_labels = add_density_bins(data)

    n_failed = {name: int(data[norm].isna().sum()) for name, _, norm, *_ in PROTOCOLS}
    print(f"Loaded {len(data)} runs from {data_file}; "
          f"non-converged runs: {n_failed}")

    total = 0
    print("01_individual (no clustering) ...")
    total += plot_individual(data, output_root / "01_individual")
    print("02_density_bins ...")
    total += plot_density_bins(data, edge_labels, output_root / "02_density_bins")
    print("03_max_interactions ...")
    total += plot_by_column(data, "Max_Interactions", output_root / "03_max_interactions",
                            "Max. interactions")
    print("04_number_of_sites ...")
    total += plot_by_column(data, "Number_of_Sites", output_root / "04_number_of_sites",
                            "Number of sites")
    print(f"Wrote {total} PDF files under {output_root}/")


if __name__ == "__main__":
    plot_results("/home/erosanow_hpc/informed_swapping/ising_systematic_study_results.csv")
