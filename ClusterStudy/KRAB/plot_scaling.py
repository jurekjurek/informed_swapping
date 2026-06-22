"""
Post-processing script: load data/results.csv and produce scaling plots.

Run after all cluster jobs have finished:
    python plot_scaling.py [--data_dir data] [--out_dir scaling_plots]

Plots produced
--------------
1.  iter_vs_nqubits.png   — iterations to convergence vs n_qubits,
                            one panel per (Q, epsilon) combination.
2.  iter_vs_Q.png         — iterations to convergence vs Q,
                            one curve per n_qubits, averaged over
                            gs_sparsity / ham_sparsity / overlap.
3.  iter_vs_epsilon.png   — iterations to convergence vs epsilon,
                            one curve per n_qubits.
4.  subspace_vs_nqubits.png — subspace size at convergence vs n_qubits
                              (shows how many basis states are actually needed).
5.  walltime_vs_nqubits.png — estimated wall time to convergence vs n_qubits.
6.  convergence_rate.png  — fraction of experiments that converge vs
                            each hyperparameter (4 subplots).
7.  heatmap_Q_eps.png     — convergence rate as a 2-D heat map over Q × epsilon,
                            one panel per n_qubits.
"""

import argparse
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TARGET = 1e-3   # relative-error threshold used in the study


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load(data_dir: str) -> pd.DataFrame:
    path = os.path.join(data_dir, "results.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Results file not found: {path}")
    df = pd.read_csv(path)
    df["converged"] = df["converged"].astype(bool)
    return df


def savefig(fig: plt.Figure, out_dir: str, name: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def mean_ci(series: pd.Series) -> tuple[float, float]:
    """Return (mean, half 95%-CI using normal approximation)."""
    n = series.notna().sum()
    if n == 0:
        return float("nan"), float("nan")
    m = series.mean()
    s = series.std(ddof=1) if n > 1 else 0.0
    ci = 1.96 * s / np.sqrt(n) if n > 0 else 0.0
    return float(m), float(ci)


# ---------------------------------------------------------------------------
# Individual plot functions
# ---------------------------------------------------------------------------

def plot_iter_vs_nqubits(df: pd.DataFrame, out_dir: str) -> None:
    """Panel grid: rows = epsilon, cols = Q."""
    conv = df[df["converged"]].copy()
    if conv.empty:
        print("  [iter_vs_nqubits] No converged experiments — skipping.")
        return

    epsilons = sorted(df["epsilon"].unique())
    Qs       = sorted(df["Q"].unique())
    nqs      = sorted(df["n_qubits"].unique())

    fig, axes = plt.subplots(
        len(epsilons), len(Qs),
        figsize=(4 * len(Qs), 3.5 * len(epsilons)),
        sharex=True, sharey=False,
    )
    if len(epsilons) == 1:
        axes = axes[np.newaxis, :]
    if len(Qs) == 1:
        axes = axes[:, np.newaxis]

    for ri, eps in enumerate(epsilons):
        for ci, Q in enumerate(Qs):
            ax = axes[ri, ci]
            sub = conv[(conv["epsilon"] == eps) & (conv["Q"] == Q)]

            if not sub.empty:
                grp = sub.groupby("n_qubits")["iter_to_convergence"]
                means = grp.mean()
                stds  = grp.std(ddof=1).fillna(0)
                ax.errorbar(means.index, means.values, yerr=stds.values,
                            marker="o", linewidth=1.5, capsize=4)
            ax.set_title(f"Q={Q}, ε={eps:.0e}", fontsize=9)
            ax.set_xlabel("n_qubits")
            ax.set_ylabel("Iterations to convergence")
            ax.set_xticks(nqs)
            ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"KRAB: iterations to reach relative error < {TARGET:.0e}\n"
        f"(averaged over gs_sparsity, ham_sparsity, overlap, seed; error bars = 1σ)",
        fontsize=11,
    )
    fig.tight_layout()
    savefig(fig, out_dir, "iter_vs_nqubits.png")


def plot_iter_vs_Q(df: pd.DataFrame, out_dir: str) -> None:
    conv = df[df["converged"]].copy()
    if conv.empty:
        print("  [iter_vs_Q] No converged experiments — skipping.")
        return

    nqs = sorted(df["n_qubits"].unique())
    Qs  = sorted(df["Q"].unique())
    epsilons = sorted(df["epsilon"].unique())

    fig, axes = plt.subplots(1, len(epsilons), figsize=(5 * len(epsilons), 4.5),
                             sharey=False)
    if len(epsilons) == 1:
        axes = [axes]

    cmap = plt.cm.viridis
    colors = {nq: cmap(i / max(len(nqs) - 1, 1)) for i, nq in enumerate(nqs)}

    for ax, eps in zip(axes, epsilons):
        sub_eps = conv[conv["epsilon"] == eps]
        for nq in nqs:
            sub = sub_eps[sub_eps["n_qubits"] == nq].groupby("Q")["iter_to_convergence"]
            if sub.ngroups == 0:
                continue
            means = sub.mean()
            stds  = sub.std(ddof=1).fillna(0)
            ax.errorbar(means.index, means.values, yerr=stds.values,
                        marker="o", linewidth=1.5, capsize=4,
                        color=colors[nq], label=f"{nq}q")
        ax.set_title(f"ε = {eps:.0e}", fontsize=10)
        ax.set_xlabel("Q")
        ax.set_ylabel("Iterations to convergence")
        ax.set_xticks(Qs)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"KRAB: iterations to convergence vs Q\n"
        f"(averaged over gs_sparsity, ham_sparsity, overlap, seed)",
        fontsize=11,
    )
    fig.tight_layout()
    savefig(fig, out_dir, "iter_vs_Q.png")


def plot_iter_vs_epsilon(df: pd.DataFrame, out_dir: str) -> None:
    conv = df[df["converged"]].copy()
    if conv.empty:
        print("  [iter_vs_epsilon] No converged experiments — skipping.")
        return

    nqs      = sorted(df["n_qubits"].unique())
    Qs       = sorted(df["Q"].unique())
    epsilons = sorted(df["epsilon"].unique())

    cmap = plt.cm.plasma
    colors_nq = {nq: cmap(i / max(len(nqs) - 1, 1)) for i, nq in enumerate(nqs)}
    ls_map = {Q: ls for Q, ls in zip(Qs, ["-", "--", "-.", ":"])}

    fig, ax = plt.subplots(figsize=(7, 5))

    for nq in nqs:
        for Q in Qs:
            sub = conv[(conv["n_qubits"] == nq) & (conv["Q"] == Q)].groupby("epsilon")[
                "iter_to_convergence"
            ]
            if sub.ngroups == 0:
                continue
            means = sub.mean().sort_index()
            stds  = sub.std(ddof=1).fillna(0).reindex(means.index)
            ax.errorbar(
                means.index, means.values, yerr=stds.values,
                marker="o", linewidth=1.3, capsize=3,
                color=colors_nq[nq], linestyle=ls_map.get(Q, "-"),
                label=f"{nq}q Q={Q}",
            )

    ax.set_xscale("log")
    ax.set_xlabel("epsilon (log scale)")
    ax.set_ylabel("Iterations to convergence")
    ax.set_title(
        f"KRAB: iterations to convergence vs ε\n"
        f"(averaged over gs_sparsity, ham_sparsity, overlap)"
    )
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    savefig(fig, out_dir, "iter_vs_epsilon.png")


def plot_subspace_vs_nqubits(df: pd.DataFrame, out_dir: str) -> None:
    conv = df[df["converged"]].copy()
    if conv.empty:
        print("  [subspace_vs_nqubits] No converged experiments — skipping.")
        return

    Qs       = sorted(df["Q"].unique())
    epsilons = sorted(df["epsilon"].unique())
    nqs      = sorted(df["n_qubits"].unique())

    fig, axes = plt.subplots(1, len(Qs), figsize=(5 * len(Qs), 4.5), sharey=False)
    if len(Qs) == 1:
        axes = [axes]

    cmap = plt.cm.cool
    colors_eps = {eps: cmap(i / max(len(epsilons) - 1, 1)) for i, eps in enumerate(epsilons)}

    for ax, Q in zip(axes, Qs):
        sub_Q = conv[conv["Q"] == Q]
        for eps in epsilons:
            sub = sub_Q[sub_Q["epsilon"] == eps].groupby("n_qubits")["subspace_at_convergence"]
            if sub.ngroups == 0:
                continue
            means = sub.mean()
            stds  = sub.std(ddof=1).fillna(0)
            # also plot 2^n for reference
            ax.errorbar(means.index, means.values, yerr=stds.values,
                        marker="o", linewidth=1.5, capsize=4,
                        color=colors_eps[eps], label=f"ε={eps:.0e}")

        # Reference line: full Hilbert space
        ref_x = np.array(nqs)
        ax.plot(ref_x, 2**ref_x, "k--", linewidth=1, label="Full dim = 2ⁿ", alpha=0.6)
        ax.set_title(f"Q={Q}")
        ax.set_xlabel("n_qubits")
        ax.set_ylabel("Subspace size at convergence")
        ax.set_xticks(nqs)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")

    fig.suptitle(
        "KRAB: subspace size at convergence vs n_qubits\n"
        "(dashed: full Hilbert-space dimension 2ⁿ)",
        fontsize=11,
    )
    fig.tight_layout()
    savefig(fig, out_dir, "subspace_vs_nqubits.png")


def plot_walltime_vs_nqubits(df: pd.DataFrame, out_dir: str) -> None:
    conv = df[df["converged"]].copy()
    if conv.empty:
        print("  [walltime_vs_nqubits] No converged experiments — skipping.")
        return

    Qs   = sorted(df["Q"].unique())
    nqs  = sorted(df["n_qubits"].unique())

    cmap = plt.cm.tab10
    colors = {Q: cmap(i / max(len(Qs) - 1, 1)) for i, Q in enumerate(Qs)}

    fig, ax = plt.subplots(figsize=(7, 5))

    for Q in Qs:
        sub = conv[conv["Q"] == Q].groupby("n_qubits")["est_wall_time_to_convergence_s"]
        if sub.ngroups == 0:
            continue
        means = sub.mean()
        stds  = sub.std(ddof=1).fillna(0)
        ax.errorbar(means.index, means.values, yerr=stds.values,
                    marker="o", linewidth=1.5, capsize=4,
                    color=colors[Q], label=f"Q={Q}")

    ax.set_xlabel("n_qubits")
    ax.set_ylabel("Estimated wall time to convergence (s)")
    ax.set_yscale("log")
    ax.set_xticks(nqs)
    ax.set_title(
        "KRAB: estimated wall time to convergence vs n_qubits\n"
        "(averaged over ε, gs_sparsity, ham_sparsity, overlap)"
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    savefig(fig, out_dir, "walltime_vs_nqubits.png")


def plot_convergence_rate(df: pd.DataFrame, out_dir: str) -> None:
    """4-panel bar chart: convergence rate vs each hyperparameter."""
    params = ["n_qubits", "Q", "epsilon", "gs_sparsity"]
    labels = ["n_qubits", "Q", "ε (epsilon)", "GS sparsity"]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))

    for ax, param, label in zip(axes, params, labels):
        grp = df.groupby(param)["converged"].mean() * 100
        vals = grp.sort_index()
        ax.bar(range(len(vals)), vals.values, color="steelblue", alpha=0.85)
        tick_labels = [f"{v:.0e}" if isinstance(v, float) and v < 1e-2 else str(v)
                       for v in vals.index]
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(tick_labels, rotation=30, ha="right", fontsize=9)
        ax.set_xlabel(label)
        ax.set_ylabel("Convergence rate (%)")
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3, axis="y")
        ax.axhline(100, color="red", linestyle="--", linewidth=0.8)

    fig.suptitle(
        f"KRAB: fraction of experiments reaching relative error < {TARGET:.0e}\n"
        f"by hyperparameter value (marginalized over all others)",
        fontsize=11,
    )
    fig.tight_layout()
    savefig(fig, out_dir, "convergence_rate.png")


def plot_heatmap_Q_eps(df: pd.DataFrame, out_dir: str) -> None:
    """2-D heat map of convergence rate over Q × epsilon for each n_qubits."""
    nqs      = sorted(df["n_qubits"].unique())
    Qs       = sorted(df["Q"].unique())
    epsilons = sorted(df["epsilon"].unique(), reverse=True)   # large ε at top

    fig, axes = plt.subplots(1, len(nqs), figsize=(4.5 * len(nqs), 4.5))
    if len(nqs) == 1:
        axes = [axes]

    for ax, nq in zip(axes, nqs):
        sub = df[df["n_qubits"] == nq]
        matrix = np.zeros((len(epsilons), len(Qs)))
        for ri, eps in enumerate(epsilons):
            for ci, Q in enumerate(Qs):
                cell = sub[(sub["epsilon"] == eps) & (sub["Q"] == Q)]
                matrix[ri, ci] = cell["converged"].mean() * 100 if len(cell) else float("nan")

        im = ax.imshow(matrix, vmin=0, vmax=100, cmap="RdYlGn", aspect="auto")
        ax.set_xticks(range(len(Qs)))
        ax.set_xticklabels([str(Q) for Q in Qs])
        ax.set_yticks(range(len(epsilons)))
        ax.set_yticklabels([f"{e:.0e}" for e in epsilons])
        ax.set_xlabel("Q")
        ax.set_ylabel("epsilon")
        ax.set_title(f"{nq} qubits")

        for ri in range(len(epsilons)):
            for ci in range(len(Qs)):
                v = matrix[ri, ci]
                txt = f"{v:.0f}%" if not np.isnan(v) else "—"
                ax.text(ci, ri, txt, ha="center", va="center", fontsize=9,
                        color="black" if 20 < v < 80 else "white")

        plt.colorbar(im, ax=ax, label="Convergence rate (%)")

    fig.suptitle(
        f"KRAB convergence rate (%) for relative error < {TARGET:.0e}\n"
        f"as a function of Q and ε, per n_qubits",
        fontsize=11,
    )
    fig.tight_layout()
    savefig(fig, out_dir, "heatmap_Q_eps.png")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Produce KRAB scaling plots from data/results.csv."
    )
    p.add_argument("--data_dir", default="data",           help="Directory containing results.csv.")
    p.add_argument("--out_dir",  default="scaling_plots",  help="Output directory for plots.")
    args = p.parse_args()

    print(f"Loading results from {args.data_dir}/results.csv ...")
    df = load(args.data_dir)
    print(f"  {len(df)} rows, {df['converged'].sum()} converged "
          f"({100*df['converged'].mean():.1f}%)")
    print(f"  n_qubits: {sorted(df['n_qubits'].unique())}")
    print(f"  Q:        {sorted(df['Q'].unique())}")
    print(f"  epsilon:  {sorted(df['epsilon'].unique())}")
    print()

    print("Producing plots ...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plot_iter_vs_nqubits(df, args.out_dir)
        plot_iter_vs_Q(df, args.out_dir)
        plot_iter_vs_epsilon(df, args.out_dir)
        plot_subspace_vs_nqubits(df, args.out_dir)
        plot_walltime_vs_nqubits(df, args.out_dir)
        plot_convergence_rate(df, args.out_dir)
        plot_heatmap_Q_eps(df, args.out_dir)

    print(f"\nAll plots written to {args.out_dir}/")


if __name__ == "__main__":
    main()
