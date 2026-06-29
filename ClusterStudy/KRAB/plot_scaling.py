"""
Comparative analysis of the KRAB-vs-SKQD cluster study.

Reads data/comparison.csv (produced by run_experiment.py) and answers the three
questions the study is meant to settle:

  Q1  Does KRAB beat SKQD?
        - head_to_head.png        paired KRAB vs SKQD cost (states to target)
        - win_rate_by_param.png   fraction of cases KRAB wins, per hyperparameter

  Q2  Where does it work / not work (the dependencies)?
        - convergence_rate.png    KRAB vs SKQD reach-target rate per hyperparameter
        - region_heatmaps.png     win-rate & KRAB-converged-rate over parameter pairs

  Q3  How do the computational resources scale?
        - cost_scaling.png        states-to-target vs n_qubits (KRAB / SKQD / random)
        - subspace_scaling.png    KRAB subspace size vs n_qubits vs full dim 2ⁿ
        - walltime_scaling.png    KRAB vs SKQD wall time vs n_qubits

  Plus the KRAB-specific knobs:
        - krab_hyperparams.png    KRAB reach-rate & cost vs Q and ε

Usage
-----
    python plot_scaling.py [--data_dir data] [--out_dir analysis]
"""

import argparse
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TARGET = 1e-3

KRAB_COL = "tab:red"
SKQD_COL = "tab:purple"
RAND_COL = "steelblue"


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def load(data_dir: str, data_file: str = "comparison.csv") -> pd.DataFrame:
    path = os.path.join(data_dir, data_file)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Results file not found: {path}")
    df = pd.read_csv(path)
    for c in ("krab_native_converged", "skqd_any_converged", "krab_beats_skqd"):
        if c in df:
            df[c] = df[c].astype(str).str.lower().isin(["true", "1", "1.0"])
    return df


def savefig(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def fmt_val(v):
    if isinstance(v, float) and v < 1e-2:
        return f"{v:.0e}"
    return str(v)


def label_for(param, v):
    """Axis-tick label for a parameter value (Q_frac shown as a percentage)."""
    if param == "Q_frac":
        return f"{v*100:g}%"
    return fmt_val(v)


def median_iqr(s: pd.Series):
    s = s.dropna()
    s = s[np.isfinite(s)]
    if s.empty:
        return np.nan, np.nan, np.nan
    return float(s.median()), float(s.quantile(0.25)), float(s.quantile(0.75))


# ===========================================================================
# Q1 — Does KRAB beat SKQD?
# ===========================================================================

def plot_head_to_head(df, out_dir):
    """Paired scatter of KRAB vs SKQD states-to-target (log-log)."""
    k = df["krab_n_states_to_target"].to_numpy(dtype=float)
    s = df["skqd_best_n_states_to_target"].to_numpy(dtype=float)
    dim = df["dim"].to_numpy(dtype=float)
    nq = df["n_qubits"].to_numpy(dtype=float)

    # Non-convergers are placed just past the full Hilbert dimension so they're visible.
    cap = dim * 1.6
    kk = np.where(np.isnan(k), cap, k)
    ss = np.where(np.isnan(s), cap, s)

    fig, (ax, axb) = plt.subplots(1, 2, figsize=(14, 6),
                                  gridspec_kw={"width_ratios": [1.4, 1]})

    nqs = sorted(df["n_qubits"].unique())
    cmap = plt.cm.viridis
    colors = {q: cmap(i / max(len(nqs) - 1, 1)) for i, q in enumerate(nqs)}
    for q in nqs:
        m = nq == q
        ax.scatter(ss[m], kk[m], s=28, alpha=0.55, color=colors[q],
                   edgecolor="k", linewidth=0.3, label=f"{int(q)}q")

    lim_hi = np.nanmax([np.nanmax(kk), np.nanmax(ss)]) * 1.1
    diag = np.array([1, lim_hi])
    ax.plot(diag, diag, "k--", linewidth=1, alpha=0.7)
    ax.fill_between(diag, diag, lim_hi, color="red", alpha=0.05)
    ax.text(0.05, 0.92, "KRAB cheaper", transform=ax.transAxes,
            color="darkred", fontsize=10)
    ax.text(0.55, 0.06, "SKQD cheaper", transform=ax.transAxes,
            color="purple", fontsize=10)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("SKQD (best t): basis states to target")
    ax.set_ylabel("KRAB: basis states to target")
    ax.set_title("Head-to-head cost (points past full dim = never reached target)")
    ax.legend(fontsize=8, title="system")
    ax.grid(True, alpha=0.3, which="both")

    # Win/loss summary bar
    both = (~np.isnan(k)) & (~np.isnan(s))
    krab_only = (~np.isnan(k)) & (np.isnan(s))
    skqd_only = (np.isnan(k)) & (~np.isnan(s))
    neither = (np.isnan(k)) & (np.isnan(s))
    krab_win = int(np.sum(both & (k < s)) + np.sum(krab_only))
    skqd_win = int(np.sum(both & (s < k)) + np.sum(skqd_only))
    tie = int(np.sum(both & (k == s)))
    cats = ["KRAB wins", "tie", "SKQD wins", "neither\nconverges"]
    vals = [krab_win, tie, skqd_win, int(np.sum(neither))]
    bar_colors = [KRAB_COL, "gray", SKQD_COL, "lightgray"]
    axb.bar(cats, vals, color=bar_colors, alpha=0.85)
    total = len(df)
    for i, v in enumerate(vals):
        axb.text(i, v, f"{v}\n({100*v/total:.0f}%)", ha="center", va="bottom", fontsize=9)
    axb.set_ylabel("Number of experiments")
    axb.set_title(f"Outcome over all {total} experiments\n(fewer states to reach rel.err<{TARGET:.0e})")
    axb.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    savefig(fig, out_dir, "head_to_head.png")


def plot_win_rate_by_param(df, out_dir):
    params = ["n_qubits", "gs_sparsity", "ham_sparsity", "overlap", "Q_frac", "epsilon"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for ax, p in zip(axes.flat, params):
        grp = df.groupby(p)["krab_beats_skqd"].mean() * 100
        grp = grp.sort_index()
        ax.bar([label_for(p, v) for v in grp.index], grp.values, color=KRAB_COL, alpha=0.8)
        ax.axhline(50, color="black", linestyle="--", linewidth=0.8)
        ax.set_xlabel(p); ax.set_ylabel("KRAB win rate (%)")
        ax.set_ylim(0, 105)
        ax.tick_params(axis="x", rotation=30)
        ax.grid(True, alpha=0.3, axis="y")
    fig.suptitle("Q1: fraction of experiments where KRAB reaches the target with "
                 "fewer basis states than the best SKQD time step", fontsize=12)
    fig.tight_layout()
    savefig(fig, out_dir, "win_rate_by_param.png")


# ===========================================================================
# Q2 — Where does it work?
# ===========================================================================

def plot_convergence_rate(df, out_dir):
    params = ["n_qubits", "gs_sparsity", "ham_sparsity", "overlap", "Q_frac", "epsilon"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for ax, p in zip(axes.flat, params):
        gk = df.groupby(p)["krab_native_converged"].mean() * 100
        gs = df.groupby(p)["skqd_any_converged"].mean() * 100
        idx = sorted(df[p].unique())
        x = np.arange(len(idx))
        w = 0.4
        ax.bar(x - w/2, [gk.get(i, 0) for i in idx], w, color=KRAB_COL, alpha=0.85, label="KRAB")
        ax.bar(x + w/2, [gs.get(i, 0) for i in idx], w, color=SKQD_COL, alpha=0.85, label="SKQD")
        ax.set_xticks(x); ax.set_xticklabels([label_for(p, i) for i in idx], rotation=30)
        ax.set_xlabel(p); ax.set_ylabel("Reach-target rate (%)")
        ax.set_ylim(0, 105); ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")
    fig.suptitle(f"Q2: fraction reaching rel.err<{TARGET:.0e}  "
                 f"(KRAB native convergence vs SKQD over its time-step sweep)", fontsize=12)
    fig.tight_layout()
    savefig(fig, out_dir, "convergence_rate.png")


def _heatmap(ax, df, row_param, col_param, value_fn, title, cmap, vmin, vmax, fmt):
    rows = sorted(df[row_param].unique(), reverse=True)
    cols = sorted(df[col_param].unique())
    M = np.full((len(rows), len(cols)), np.nan)
    for ri, rv in enumerate(rows):
        for ci, cv in enumerate(cols):
            cell = df[(df[row_param] == rv) & (df[col_param] == cv)]
            if len(cell):
                M[ri, ci] = value_fn(cell)
    im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels([fmt_val(c) for c in cols], rotation=30)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([fmt_val(r) for r in rows])
    ax.set_xlabel(col_param); ax.set_ylabel(row_param)
    ax.set_title(title, fontsize=10)
    for ri in range(len(rows)):
        for ci in range(len(cols)):
            v = M[ri, ci]
            if not np.isnan(v):
                ax.text(ci, ri, fmt(v), ha="center", va="center", fontsize=8,
                        color="black")
    return im


def plot_region_heatmaps(df, out_dir):
    """Where KRAB works: win-rate and KRAB reach-rate over parameter pairs."""
    pairs = [("gs_sparsity", "overlap"),
             ("gs_sparsity", "ham_sparsity"),
             ("overlap", "ham_sparsity")]

    fig, axes = plt.subplots(2, 3, figsize=(17, 9))

    for ci, (rp, cp) in enumerate(pairs):
        im1 = _heatmap(
            axes[0, ci], df, rp, cp,
            value_fn=lambda c: 100 * c["krab_beats_skqd"].mean(),
            title=f"KRAB win-rate (%)  [{rp} × {cp}]",
            cmap="RdYlGn", vmin=0, vmax=100, fmt=lambda v: f"{v:.0f}",
        )
        plt.colorbar(im1, ax=axes[0, ci], fraction=0.046, pad=0.04)

        im2 = _heatmap(
            axes[1, ci], df, rp, cp,
            value_fn=lambda c: 100 * c["krab_native_converged"].mean(),
            title=f"KRAB reach-target rate (%)  [{rp} × {cp}]",
            cmap="RdYlGn", vmin=0, vmax=100, fmt=lambda v: f"{v:.0f}",
        )
        plt.colorbar(im2, ax=axes[1, ci], fraction=0.046, pad=0.04)

    fig.suptitle("Q2: regions where KRAB works — top row: KRAB beats SKQD; "
                 "bottom row: KRAB reaches the target at all\n"
                 "(marginalized over the remaining parameters)", fontsize=12)
    fig.tight_layout()
    savefig(fig, out_dir, "region_heatmaps.png")


# ===========================================================================
# Q3 — Computational scaling
# ===========================================================================

def _scaling_curve(ax, df, col, label, color, nqs):
    med, lo, hi = [], [], []
    for q in nqs:
        m, l, h = median_iqr(df[df["n_qubits"] == q][col])
        med.append(m); lo.append(l); hi.append(h)
    med = np.array(med); lo = np.array(lo); hi = np.array(hi)
    ax.plot(nqs, med, marker="o", color=color, linewidth=1.8, label=label)
    ax.fill_between(nqs, lo, hi, color=color, alpha=0.18)


def plot_cost_scaling(df, out_dir):
    nqs = sorted(df["n_qubits"].unique())
    fig, ax = plt.subplots(figsize=(8, 6))
    _scaling_curve(ax, df, "krab_n_states_to_target", "KRAB", KRAB_COL, nqs)
    _scaling_curve(ax, df, "skqd_best_n_states_to_target", "SKQD (best t)", SKQD_COL, nqs)
    _scaling_curve(ax, df, "rand_median_n_states_to_target", "Random (median)", RAND_COL, nqs)
    ax.plot(nqs, [2**q for q in nqs], "k--", linewidth=1, alpha=0.6, label="Full dim 2ⁿ")
    ax.set_yscale("log")
    ax.set_xlabel("n_qubits"); ax.set_ylabel("Basis states to reach target (median, IQR band)")
    ax.set_xticks(nqs)
    ax.set_title(f"Q3: how the sampling cost scales with system size\n"
                 f"(states needed to reach rel.err<{TARGET:.0e}; converged runs only)")
    ax.legend(); ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    savefig(fig, out_dir, "cost_scaling.png")


def plot_subspace_scaling(df, out_dir):
    nqs = sorted(df["n_qubits"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    _scaling_curve(ax, df, "krab_final_subspace", "KRAB final subspace", KRAB_COL, nqs)
    _scaling_curve(ax, df, "krab_subspace_at_target", "KRAB subspace @ target", "darkorange", nqs)
    ax.plot(nqs, [2**q for q in nqs], "k--", linewidth=1, alpha=0.6, label="Full dim 2ⁿ")
    ax.set_yscale("log"); ax.set_xlabel("n_qubits"); ax.set_xticks(nqs)
    ax.set_ylabel("Subspace size (median, IQR band)")
    ax.set_title("KRAB classical subspace size vs system size")
    ax.legend(); ax.grid(True, alpha=0.3, which="both")

    # subspace as a fraction of full dim — does it stay sparse as n grows?
    ax = axes[1]
    df2 = df.copy()
    df2["frac"] = df2["krab_final_subspace"] / df2["dim"]
    _scaling_curve(ax, df2, "frac", "KRAB final / 2ⁿ", KRAB_COL, nqs)
    ax.set_xlabel("n_qubits"); ax.set_xticks(nqs)
    ax.set_ylabel("Subspace fraction of full Hilbert space")
    ax.set_title("Does KRAB stay sub-exponential? (smaller is better)")
    ax.grid(True, alpha=0.3); ax.legend()
    fig.tight_layout()
    savefig(fig, out_dir, "subspace_scaling.png")


def plot_walltime_scaling(df, out_dir):
    nqs = sorted(df["n_qubits"].unique())
    fig, ax = plt.subplots(figsize=(8, 6))
    _scaling_curve(ax, df, "krab_walltime_s", "KRAB", KRAB_COL, nqs)
    _scaling_curve(ax, df, "skqd_walltime_s", "SKQD sweep", SKQD_COL, nqs)
    ax.set_yscale("log")
    ax.set_xlabel("n_qubits"); ax.set_ylabel("Wall time (s, median, IQR band)")
    ax.set_xticks(nqs)
    ax.set_title("Q3: classical wall time vs system size\n"
                 "(KRAB run vs full SKQD time-step sweep + path reconstruction)")
    ax.legend(); ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    savefig(fig, out_dir, "walltime_scaling.png")


def plot_krab_hyperparams(df, out_dir):
    """KRAB-specific knobs: reach-rate and cost vs Q and epsilon, per n_qubits."""
    nqs = sorted(df["n_qubits"].unique())
    cmap = plt.cm.viridis
    colors = {q: cmap(i / max(len(nqs) - 1, 1)) for i, q in enumerate(nqs)}

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # reach-rate vs Q
    ax = axes[0, 0]
    for q in nqs:
        g = df[df["n_qubits"] == q].groupby("Q_frac")["krab_native_converged"].mean() * 100
        ax.plot(g.index * 100, g.values, marker="o", color=colors[q], label=f"{int(q)}q")
    ax.set_xlabel("Q (% of 2ⁿ)"); ax.set_ylabel("KRAB reach-target rate (%)")
    ax.set_title("KRAB convergence vs Q"); ax.set_ylim(0, 105)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # reach-rate vs epsilon
    ax = axes[0, 1]
    for q in nqs:
        g = df[df["n_qubits"] == q].groupby("epsilon")["krab_native_converged"].mean() * 100
        ax.plot(g.index, g.values, marker="o", color=colors[q], label=f"{int(q)}q")
    ax.set_xscale("log")
    ax.set_xlabel("epsilon"); ax.set_ylabel("KRAB reach-target rate (%)")
    ax.set_title("KRAB convergence vs ε"); ax.set_ylim(0, 105)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, which="both")

    # cost (states to target) vs Q
    ax = axes[1, 0]
    for q in nqs:
        sub = df[df["n_qubits"] == q]
        g = sub.groupby("Q_frac")["krab_n_states_to_target"].median()
        ax.plot(g.index * 100, g.values, marker="o", color=colors[q], label=f"{int(q)}q")
    ax.set_xlabel("Q (% of 2ⁿ)"); ax.set_ylabel("KRAB states to target (median)")
    ax.set_title("KRAB cost vs Q (converged runs)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # final subspace vs epsilon
    ax = axes[1, 1]
    for q in nqs:
        sub = df[df["n_qubits"] == q]
        g = sub.groupby("epsilon")["krab_final_subspace"].median()
        ax.plot(g.index, g.values, marker="o", color=colors[q], label=f"{int(q)}q")
    ax.set_xscale("log")
    ax.set_xlabel("epsilon"); ax.set_ylabel("KRAB final subspace (median)")
    ax.set_title("KRAB final subspace vs ε")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, which="both")

    fig.suptitle("KRAB hyperparameter dependence (Q, ε)", fontsize=12)
    fig.tight_layout()
    savefig(fig, out_dir, "krab_hyperparams.png")


# ===========================================================================
# Driver
# ===========================================================================

def print_summary(df):
    n = len(df)
    print(f"  {n} experiments")
    print(f"  KRAB reaches target : {100*df['krab_native_converged'].mean():.1f}%")
    print(f"  SKQD reaches target : {100*df['skqd_any_converged'].mean():.1f}%")
    print(f"  KRAB beats SKQD     : {100*df['krab_beats_skqd'].mean():.1f}%")
    both = df.dropna(subset=["krab_n_states_to_target", "skqd_best_n_states_to_target"])
    if len(both):
        ratio = both["skqd_best_n_states_to_target"] / both["krab_n_states_to_target"]
        print(f"  Where both converge ({len(both)}): median SKQD/KRAB state ratio "
              f"= {ratio.median():.2f}× (>1 ⇒ KRAB cheaper)")
    print(f"  n_qubits: {sorted(df['n_qubits'].unique())}")
    print()


def main():
    p = argparse.ArgumentParser(description="Comparative KRAB-vs-SKQD analysis plots.")
    p.add_argument("--data_dir",  default="data")
    p.add_argument("--data_file", default="comparison.csv",
                   help="CSV inside --data_dir (use comparison_decay.csv for the "
                        "decaying-Q study).")
    p.add_argument("--out_dir",   default="analysis",
                   help="Output dir for plots (use analysis_decay for the "
                        "decaying-Q study).")
    args = p.parse_args()

    print(f"Loading {args.data_dir}/{args.data_file} ...")
    df = load(args.data_dir, args.data_file)
    print_summary(df)

    print("Producing plots ...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Q1
        plot_head_to_head(df, args.out_dir)
        plot_win_rate_by_param(df, args.out_dir)
        # Q2
        plot_convergence_rate(df, args.out_dir)
        plot_region_heatmaps(df, args.out_dir)
        # Q3
        plot_cost_scaling(df, args.out_dir)
        plot_subspace_scaling(df, args.out_dir)
        plot_walltime_scaling(df, args.out_dir)
        # KRAB knobs
        plot_krab_hyperparams(df, args.out_dir)

    print(f"\nAll plots written to {args.out_dir}/")


if __name__ == "__main__":
    main()
