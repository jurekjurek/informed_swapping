"""Plot the sparse counterexample trajectories in the PaperFigures_v2 style.

Run after ``energy_bark_counterexample.py`` has generated the fidelity CSVs:

    conda run -n bark-counterexample python couterexample/energy_bark_counterexample/plot_infidelity.py
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullLocator
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from PaperFigures import apply_style, draw_line, style


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results_m9"
N_QUBITS = 10
DIMENSION = 2**N_QUBITS
CURVES = {
    "Energy-BARK": ("bark_fidelity.csv", "#2a78d6", "o", "-"),
    "Amplitude-BARK": ("amplitude_bark_fidelity.csv", "#3aa66f", "^", "-."),
    "SKQD": ("skqd_fidelity.csv", "#eb6834", "s", "--"),
}


def main() -> Path:
    apply_style()
    fig, ax = plt.subplots(figsize=(7.05, 2.3), layout="constrained")

    # Draw SKQD last so its markers remain visible at overlapping dimensions,
    # matching the ordering used for the earlier counterexample.
    for name, (filename, color, marker, linestyle) in CURVES.items():
        data = pd.read_csv(RESULTS / filename)
        dimension_column = "D"
        fidelity_column = "fidelity"
        x = data[dimension_column].to_numpy(dtype=float) / DIMENSION
        infidelity = np.maximum(1.0 - data[fidelity_column].to_numpy(dtype=float), 1e-16)
        if name == "SKQD":
            ax.plot(
                x, infidelity, color=color, linestyle=linestyle, linewidth=3.0,
                marker=marker, markersize=5, markevery=1,
                markerfacecolor="none", markeredgecolor=color,
                markeredgewidth=1.2, label=name, zorder=4,
            )
        else:
            rows = pd.DataFrame({"Budget_Fraction": x})
            draw_line(ax, rows, name, color, marker, linestyle, infidelity)

    style(ax)
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(numticks=5))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set(
        xlabel=r"Bitstrings $K/2^N$",
        ylabel=r"Infidelity $1-F$",
        xlim=(1 / DIMENSION / 1.3, 0.8),
        ylim=(1e-10, 1.05),
    )
    ax.legend(frameon=False)

    output = RESULTS / "infidelity_vs_bitstrings.png"
    fig.savefig(output, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return output


if __name__ == "__main__":
    print(main())
