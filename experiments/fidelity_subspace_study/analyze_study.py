"""Turn the study tables into the comparison figures.

The headline figure is ``fidelity_vs_subspace``: **one** axis carrying every
Hamiltonian, every qubit count and every parameter setting of the scan, with
fidelity on y and the subspace dimension normalised by the Hilbert space
dimension on x. Each thin line is the *best-parameter envelope* of one
(Hamiltonian, initial overlap) pair -- the highest fidelity any scanned
parameter setting reached at each subspace size -- so reading the plot at a
fidelity gives exactly the quantity the study optimises for. Two bold lines give
the per-method median across the whole scan.

Colour carries method identity only (SKQD vs BARK); qubit count and interaction
cap are read off the supporting panels, which keeps the headline plot legible
even with hundreds of curves on it.

Usage
-----
    python experiments/fidelity_subspace_study/analyze_study.py \
        --results-dir experiments/fidelity_subspace_study/results
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from optimization import CASE_COLUMNS, method_comparison, optimal_configurations


# ----------------------------------------------------------------------
# Palette (validated categorical slots; see README for the provenance)
# ----------------------------------------------------------------------
THEMES: dict[str, dict[str, Any]] = {
    "light": {
        "surface": "#fcfcfb",
        "text_primary": "#0b0b0b",
        "text_secondary": "#52514e",
        "grid": "#dedcd6",
        "series": {"SKQD": "#2a78d6", "BARK": "#eb6834"},
        "stages": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"],
    },
    "dark": {
        "surface": "#1a1a19",
        "text_primary": "#ffffff",
        "text_secondary": "#c3c2b7",
        "grid": "#3a3a37",
        "series": {"SKQD": "#3987e5", "BARK": "#d95926"},
        "stages": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300"],
    },
}

METHOD_ORDER = ["SKQD", "BARK"]

# Uniform runtime taxonomy so both methods stack the same six slots.
STAGE_COLUMNS = {
    "state generation": ["t_propagate", "t_sample"],
    "candidate scoring": ["t_select"],
    "H application": ["t_expand"],
    "internal ground update": ["t_ground_update"],
    "subspace solve": ["t_project", "t_diagonalize"],
    "bookkeeping": ["t_subspace_update"],
}


def apply_theme(theme: dict[str, Any]) -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": theme["surface"],
            "axes.facecolor": theme["surface"],
            "savefig.facecolor": theme["surface"],
            "text.color": theme["text_primary"],
            "axes.labelcolor": theme["text_secondary"],
            "axes.edgecolor": theme["grid"],
            "xtick.color": theme["text_secondary"],
            "ytick.color": theme["text_secondary"],
            "grid.color": theme["grid"],
            "grid.linewidth": 0.6,
            "axes.grid": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 9,
            "legend.frameon": False,
            "figure.dpi": 150,
        }
    )


# ----------------------------------------------------------------------
# Best-parameter envelopes
# ----------------------------------------------------------------------
def envelope_for_group(group: pd.DataFrame, hilbert_dim: int) -> np.ndarray:
    """Highest fidelity reachable at each subspace dimension 1..hilbert_dim.

    The envelope is the pointwise maximum over every scanned parameter setting,
    made monotone with a running maximum: a subspace of dimension d can always
    be padded to d' > d without losing fidelity, so the reachable fidelity is
    non-decreasing in d by construction.
    """
    best = group.groupby("subspace_dim")["fidelity"].max()
    curve = np.full(hilbert_dim, -np.inf)
    curve[best.index.to_numpy() - 1] = best.to_numpy()

    # Running maximum forward-fills and enforces monotonicity in one pass;
    # dimensions before the first observation stay NaN.
    curve = np.maximum.accumulate(curve)
    curve[np.isneginf(curve)] = np.nan
    # Nothing is known past the largest subspace the runs actually reached
    # (runs stop at the hardest target fidelity or at --max-subspace-fraction),
    # so the envelope ends there rather than extending flat to dimension 1.
    curve[int(best.index.max()) :] = np.nan
    return curve


def build_envelopes(iterations: pd.DataFrame) -> pd.DataFrame:
    """One envelope per (Hamiltonian, initial overlap, method, repeat).

    Repeats are kept separate so the stochastic spread of SKQD survives into the
    plot instead of being hidden by an optimistic best-of-all-repeats curve.
    """
    rows: list[dict[str, Any]] = []
    keys = ["ham_id", "num_qubits", "hilbert_dim", "max_interactions_label",
            "initial_spec", "initial_overlap", "method", "repeat"]
    for key, group in iterations.groupby(keys, dropna=False, observed=True):
        record = dict(zip(keys, key))
        hilbert_dim = int(record["hilbert_dim"])
        curve = envelope_for_group(group, hilbert_dim)
        dims = np.arange(1, hilbert_dim + 1)
        record["fractions"] = dims / hilbert_dim
        record["fidelities"] = curve
        rows.append(record)
    return pd.DataFrame(rows)


def step_interpolate(
    fractions: np.ndarray, fidelities: np.ndarray, grid: np.ndarray
) -> np.ndarray:
    """Evaluate a monotone step curve on ``grid`` (previous-value lookup)."""
    positions = np.searchsorted(fractions, grid, side="right") - 1
    out = np.full(grid.shape, np.nan)
    valid = positions >= 0
    out[valid] = fidelities[positions[valid]]
    return out


def median_envelope(
    envelopes: pd.DataFrame, grid: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Median and interquartile band of a set of envelopes on a common grid."""
    stacked = np.vstack(
        [
            step_interpolate(row.fractions, row.fidelities, grid)
            for row in envelopes.itertuples()
        ]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        median = np.nanmedian(stacked, axis=0)
        low = np.nanpercentile(stacked, 25, axis=0)
        high = np.nanpercentile(stacked, 75, axis=0)
    return median, low, high


# ----------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------
def plot_fidelity_vs_subspace(
    envelopes: pd.DataFrame,
    target_fidelities: Iterable[float],
    theme: dict[str, Any],
    output_path: Path,
) -> pd.DataFrame:
    """The headline figure: every run of the scan on one axis."""
    fig, ax = plt.subplots(figsize=(8.2, 5.2))

    smallest = float(min(1.0 / envelopes["hilbert_dim"]))
    grid = np.logspace(np.log10(smallest), 0.0, 400)

    table_rows: list[dict[str, Any]] = []
    medians: dict[str, np.ndarray] = {}
    for method in METHOD_ORDER:
        subset = envelopes[envelopes["method"] == method]
        if subset.empty:
            continue
        color = theme["series"][method]

        for row in subset.itertuples():
            ax.plot(
                row.fractions,
                row.fidelities,
                color=color,
                alpha=0.13,
                linewidth=0.8,
                solid_capstyle="round",
                zorder=2,
            )

        median, low, high = median_envelope(subset, grid)
        medians[method] = median
        ax.fill_between(grid, low, high, color=color, alpha=0.15, linewidth=0, zorder=3)
        ax.plot(grid, median, color=color, linewidth=2.2, zorder=4,
                solid_capstyle="round")

        table_rows.extend(
            {
                "method": method,
                "subspace_fraction": float(f),
                "median_fidelity": float(m),
                "q25_fidelity": float(lo),
                "q75_fidelity": float(hi),
            }
            for f, m, lo, hi in zip(grid, median, low, high)
        )

    # Direct-label the medians where they are furthest apart, so the two bands
    # are identified without the reader tracking colour back to the legend.
    if len(medians) == 2:
        a, b = (medians[m] for m in METHOD_ORDER if m in medians)
        separation = np.abs(a - b)
        if np.isfinite(separation).any():
            at = int(np.nanargmax(separation))
            for method in METHOD_ORDER:
                if method not in medians:
                    continue
                curve = medians[method]
                above = curve[at] >= max(a[at], b[at])
                ax.annotate(
                    method,
                    xy=(grid[at], curve[at]),
                    xytext=(0, 9 if above else -9),
                    textcoords="offset points",
                    color=theme["series"][method],
                    fontsize=10,
                    fontweight="bold",
                    ha="center",
                    va="bottom" if above else "top",
                    zorder=5,
                )

    for target in sorted(target_fidelities):
        ax.axhline(target, color=theme["grid"], linewidth=0.8, linestyle=(0, (4, 3)),
                   zorder=1)
        ax.annotate(
            f"F = {target:g}",
            xy=(smallest, target),
            xytext=(2, -3),
            textcoords="offset points",
            color=theme["text_secondary"],
            fontsize=7.5,
            va="top",
        )

    ax.set_xscale("log")
    ax.set_xlim(smallest, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("subspace dimension / Hilbert space dimension")
    ax.set_ylabel("fidelity with exact ground space")
    ax.set_title(
        "Fidelity reached per unit of subspace, best parameters per run",
        color=theme["text_primary"],
        fontsize=11,
        loc="left",
        pad=12,
    )
    n_curves = len(envelopes)
    ax.annotate(
        f"{n_curves} envelopes · thin = one Hamiltonian/initial-overlap run, "
        f"bold = median, band = IQR",
        xy=(0, 1.005),
        xycoords="axes fraction",
        color=theme["text_secondary"],
        fontsize=8,
    )
    ax.grid(True, which="both", alpha=0.5)
    ax.set_axisbelow(True)

    handles = [
        Line2D([], [], color=theme["series"][m], linewidth=2.2, label=m)
        for m in METHOD_ORDER
        if (envelopes["method"] == m).any()
    ]
    # Outside the axes: the curves fill the plot area and the direct labels move
    # with the data, so an in-axes legend has nowhere collision-free to sit.
    ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        fontsize=9,
    )

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(table_rows)


def add_method_legend(fig, theme: dict[str, Any], marker: str | None = None) -> None:
    """One figure-level legend, placed clear of the data."""
    handles = [
        Line2D(
            [], [],
            color=theme["series"][m],
            linewidth=0 if marker else 2.0,
            marker=marker or "o",
            linestyle="none" if marker else "-",
            markersize=6,
            label=m,
        )
        for m in METHOD_ORDER
    ]
    fig.legend(
        handles=handles,
        loc="upper right",
        bbox_to_anchor=(1.0, 1.0),
        ncol=len(handles),
        fontsize=9,
    )


def plot_subspace_scaling(
    optimal: pd.DataFrame, theme: dict[str, Any], output_path: Path
) -> pd.DataFrame:
    """Normalised subspace dimension needed per qubit count, per target fidelity."""
    targets = sorted(optimal["target_fidelity"].unique())
    fig, axes = plt.subplots(
        1, len(targets), figsize=(3.1 * len(targets) + 1.0, 3.8), sharey=True
    )
    axes = np.atleast_1d(axes)

    summary: list[dict[str, Any]] = []
    for ax, target in zip(axes, targets):
        panel = optimal[optimal["target_fidelity"] == target]
        for offset, method in enumerate(METHOD_ORDER):
            subset = panel[(panel["method"] == method) & panel["feasible"]]
            if subset.empty:
                continue
            color = theme["series"][method]
            jitter = (offset - 0.5) * 0.16
            ax.scatter(
                subset["num_qubits"] + jitter,
                subset["best_subspace_fraction"],
                s=26,
                color=color,
                alpha=0.55,
                edgecolors=theme["surface"],
                linewidths=0.8,
                zorder=3,
            )
            medians = (
                subset.groupby("num_qubits")["best_subspace_fraction"].median().sort_index()
            )
            ax.plot(
                medians.index + jitter,
                medians.to_numpy(),
                color=color,
                linewidth=2.0,
                marker="o",
                markersize=5,
                markeredgecolor=theme["surface"],
                markeredgewidth=0.8,
                zorder=4,
            )
            summary.extend(
                {
                    "target_fidelity": float(target),
                    "method": method,
                    "num_qubits": int(n),
                    "median_subspace_fraction": float(v),
                    "n_cases": int((subset["num_qubits"] == n).sum()),
                }
                for n, v in medians.items()
            )

        ax.set_yscale("log")
        ax.set_xlabel("qubits")
        ax.set_title(f"target F = {target:g}", fontsize=9.5,
                     color=theme["text_primary"], loc="left")
        ax.grid(True, which="both", alpha=0.5)
        ax.set_axisbelow(True)
        ax.set_xticks(sorted(optimal["num_qubits"].unique()))

    axes[0].set_ylabel("subspace fraction needed")
    add_method_legend(fig, theme)
    fig.suptitle(
        "Subspace fraction needed to reach a target fidelity",
        color=theme["text_primary"],
        fontsize=11,
        x=0.01,
        ha="left",
    )
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(summary)


def plot_initial_overlap_effect(
    optimal: pd.DataFrame, theme: dict[str, Any], output_path: Path
) -> None:
    targets = sorted(optimal["target_fidelity"].unique())
    fig, axes = plt.subplots(
        1, len(targets), figsize=(3.1 * len(targets) + 1.0, 3.8), sharey=True
    )
    axes = np.atleast_1d(axes)

    for ax, target in zip(axes, targets):
        panel = optimal[(optimal["target_fidelity"] == target) & optimal["feasible"]]
        for method in METHOD_ORDER:
            subset = panel[panel["method"] == method]
            if subset.empty:
                continue
            ax.scatter(
                subset["initial_overlap"],
                subset["best_subspace_fraction"],
                s=26,
                color=theme["series"][method],
                alpha=0.6,
                edgecolors=theme["surface"],
                linewidths=0.8,
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("initial overlap with ground space")
        ax.set_title(f"target F = {target:g}", fontsize=9.5,
                     color=theme["text_primary"], loc="left")
        ax.grid(True, which="both", alpha=0.5)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("subspace fraction needed")
    add_method_legend(fig, theme, marker="o")
    fig.suptitle(
        "Effect of the shared initial overlap",
        color=theme["text_primary"], fontsize=11, x=0.01, ha="left",
    )
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_sparsity_effect(
    optimal: pd.DataFrame,
    hamiltonians: pd.DataFrame,
    theme: dict[str, Any],
    output_path: Path,
) -> None:
    merged = optimal.merge(
        hamiltonians[["ham_id", "gs_participation_fraction", "ham_density"]],
        on="ham_id",
        how="left",
    )
    targets = sorted(merged["target_fidelity"].unique())
    fig, axes = plt.subplots(
        1, len(targets), figsize=(3.1 * len(targets) + 1.0, 3.8), sharey=True
    )
    axes = np.atleast_1d(axes)

    for ax, target in zip(axes, targets):
        panel = merged[(merged["target_fidelity"] == target) & merged["feasible"]]
        for method in METHOD_ORDER:
            subset = panel[panel["method"] == method]
            if subset.empty:
                continue
            ax.scatter(
                subset["gs_participation_fraction"],
                subset["best_subspace_fraction"],
                s=26,
                color=theme["series"][method],
                alpha=0.6,
                edgecolors=theme["surface"],
                linewidths=0.8,
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("ground-state participation fraction")
        ax.set_title(f"target F = {target:g}", fontsize=9.5,
                     color=theme["text_primary"], loc="left")
        ax.grid(True, which="both", alpha=0.5)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("subspace fraction needed")
    add_method_legend(fig, theme, marker="o")
    fig.suptitle(
        "Ground-state sparsity vs the subspace each method needs",
        color=theme["text_primary"], fontsize=11, x=0.01, ha="left",
    )
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def runtime_breakdown_table(runs: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for stage, columns in STAGE_COLUMNS.items():
        present = [f"total_{c}" for c in columns if f"total_{c}" in runs.columns]
        if not present:
            continue
        totals = runs.groupby(["method", "num_qubits"])[present].sum().sum(axis=1)
        frames.append(totals.rename(stage))
    table = pd.concat(frames, axis=1).fillna(0.0)
    return table.reset_index()


def plot_runtime_breakdown(
    runs: pd.DataFrame, theme: dict[str, Any], output_path: Path
) -> pd.DataFrame:
    table = runtime_breakdown_table(runs)
    stages = [c for c in STAGE_COLUMNS if c in table.columns]
    methods = [m for m in METHOD_ORDER if (table["method"] == m).any()]

    fig, axes = plt.subplots(
        1, len(methods), figsize=(3.6 * len(methods) + 1.0, 3.8), sharey=True
    )
    axes = np.atleast_1d(axes)

    for ax, method in zip(axes, methods):
        panel = table[table["method"] == method].sort_values("num_qubits")
        totals = panel[stages].sum(axis=1).replace(0.0, np.nan)
        bottom = np.zeros(len(panel))
        x = np.arange(len(panel))
        for color, stage in zip(theme["stages"], stages):
            share = (panel[stage] / totals).fillna(0.0).to_numpy()
            ax.bar(
                x,
                share,
                bottom=bottom,
                width=0.62,
                color=color,
                edgecolor=theme["surface"],
                linewidth=1.4,          # the 2px surface gap between segments
                label=stage,
            )
            bottom = bottom + share
        ax.set_xticks(x)
        ax.set_xticklabels(panel["num_qubits"].astype(int))
        ax.set_xlabel("qubits")
        ax.set_ylim(0, 1)
        ax.set_title(method, fontsize=10, color=theme["text_primary"], loc="left")
        ax.grid(True, axis="y", alpha=0.4)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("share of total runtime")
    axes[-1].legend(
        loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8.5, title="stage"
    )
    fig.suptitle(
        "Where the wall-clock time goes",
        color=theme["text_primary"], fontsize=11, x=0.01, ha="left",
    )
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return table


# ----------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir", default="experiments/fidelity_subspace_study/results"
    )
    parser.add_argument("--figure-dir", default=None)
    parser.add_argument("--theme", default="light", choices=sorted(THEMES))
    parser.add_argument("--min-reach-fraction", type=float, default=0.5)
    parser.add_argument(
        "--recompute-optimal",
        action="store_true",
        help="Re-run the parameter optimisation instead of reading optimal.parquet.",
    )
    args = parser.parse_args(argv)

    results_dir = Path(args.results_dir)
    figure_dir = Path(args.figure_dir) if args.figure_dir else results_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    theme = THEMES[args.theme]
    apply_theme(theme)

    iterations = pd.read_parquet(results_dir / "iterations.parquet")
    runs = pd.read_parquet(results_dir / "runs.parquet")
    convergence = pd.read_parquet(results_dir / "convergence.parquet")
    hamiltonians = pd.read_parquet(results_dir / "hamiltonians.parquet")

    optimal_path = results_dir / "optimal.parquet"
    if args.recompute_optimal or not optimal_path.exists():
        optimal = optimal_configurations(convergence, args.min_reach_fraction)
        optimal.to_parquet(optimal_path, index=False)
        method_comparison(optimal).to_parquet(
            results_dir / "method_comparison.parquet", index=False
        )
    else:
        optimal = pd.read_parquet(optimal_path)

    targets = sorted(convergence["target_fidelity"].unique())

    envelopes = build_envelopes(iterations)
    headline = plot_fidelity_vs_subspace(
        envelopes, targets, theme, figure_dir / "fidelity_vs_subspace.png"
    )
    headline.to_csv(figure_dir / "fidelity_vs_subspace.csv", index=False)

    scaling = plot_subspace_scaling(
        optimal, theme, figure_dir / "subspace_scaling.png"
    )
    scaling.to_csv(figure_dir / "subspace_scaling.csv", index=False)

    plot_initial_overlap_effect(
        optimal, theme, figure_dir / "initial_overlap_effect.png"
    )
    plot_sparsity_effect(
        optimal, hamiltonians, theme, figure_dir / "sparsity_effect.png"
    )
    breakdown = plot_runtime_breakdown(
        runs, theme, figure_dir / "runtime_breakdown.png"
    )
    breakdown.to_csv(figure_dir / "runtime_breakdown.csv", index=False)

    comparison = method_comparison(optimal)
    comparison.to_csv(figure_dir / "method_comparison.csv", index=False)

    advantage = comparison.get("subspace_advantage_bark")
    print(f"Figures and table views written to {figure_dir}")
    if advantage is not None and advantage.notna().any():
        print(
            "BARK subspace advantage over SKQD (>1 favours BARK): "
            f"median {advantage.median():.2f}, "
            f"range {advantage.min():.2f}-{advantage.max():.2f}"
        )


if __name__ == "__main__":
    main()
