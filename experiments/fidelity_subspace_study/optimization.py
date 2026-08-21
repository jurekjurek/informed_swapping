"""Parameter optimisation: the best SKQD / BARK setting per (Hamiltonian, fidelity).

The study runs a full grid -- SKQD over ``(dt, shots)``, BARK over
``(score_mode, selection_strategy, keep_states)`` -- and records, for every run
and every target fidelity, the subspace dimension at which the target was first
met. This module turns that raw table into the answer the study is actually
after:

    for this Hamiltonian, this initial overlap and this target fidelity, which
    parameters reach the target with the *smallest normalised subspace
    dimension*?

Aggregation over repeats uses the median, because SKQD sampling is stochastic
and a single lucky shot record should not win the grid. A configuration is only
eligible if it reaches the target in at least ``min_reach_fraction`` of its
repeats; ties on subspace dimension are broken by total wall-clock time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


SKQD_PARAM_COLUMNS = ["skqd_dt", "skqd_shots"]
BARK_PARAM_COLUMNS = [
    "bark_score_mode",
    "bark_selection_strategy",
    "bark_keep_states",
]

CASE_COLUMNS = [
    "ham_id",
    "num_qubits",
    "hilbert_dim",
    "max_interactions",
    "max_interactions_label",
    "seed",
    "initial_spec",
    "initial_overlap",
]

GROUP_COLUMNS = CASE_COLUMNS + ["target_fidelity", "method"]

# Runs that never reach the target are ranked as if they needed the entire
# Hilbert space; the reported dimension stays NaN so plots do not invent data.
UNREACHED_PENALTY = np.inf


def aggregate_over_repeats(
    convergence: pd.DataFrame,
    min_reach_fraction: float = 0.5,
) -> pd.DataFrame:
    """Collapse repeats, giving one row per (case, target, method, config)."""
    df = convergence.copy()
    df["_penalised_fraction"] = df["subspace_fraction_at_target"].where(
        df["reached"], UNREACHED_PENALTY
    )

    keys = GROUP_COLUMNS + ["config_label"]
    grouped = df.groupby(keys, dropna=False, observed=True)

    aggregated = grouped.agg(
        n_repeats=("reached", "size"),
        n_reached=("reached", "sum"),
        median_subspace_fraction=("_penalised_fraction", "median"),
        mean_subspace_fraction=("_penalised_fraction", "mean"),
        min_subspace_fraction=("subspace_fraction_at_target", "min"),
        max_subspace_fraction=("subspace_fraction_at_target", "max"),
        median_subspace_dim=("subspace_dim_at_target", "median"),
        median_iterations=("iteration_at_target", "median"),
        median_shots=("shots_at_target", "median"),
        median_t_algorithm=("t_algorithm_at_target", "median"),
        median_t_solve=("t_solve_at_target", "median"),
        median_t_total=("t_total_at_target", "median"),
        best_final_fidelity=("final_fidelity", "max"),
    ).reset_index()

    aggregated["reach_fraction"] = aggregated["n_reached"] / aggregated["n_repeats"]
    aggregated["eligible"] = aggregated["reach_fraction"] >= min_reach_fraction

    # Carry the parameter columns through for readability.
    param_cols = [
        c for c in SKQD_PARAM_COLUMNS + BARK_PARAM_COLUMNS if c in convergence.columns
    ]
    if param_cols:
        params = (
            convergence.groupby(keys, dropna=False, observed=True)[param_cols]
            .first()
            .reset_index()
        )
        aggregated = aggregated.merge(params, on=keys, how="left")

    return aggregated


def optimal_configurations(
    convergence: pd.DataFrame,
    min_reach_fraction: float = 0.5,
) -> pd.DataFrame:
    """Pick the best configuration per (case, target fidelity, method).

    Returns one row per group with the winning configuration, its normalised
    subspace dimension, and a ``feasible`` flag that is ``False`` when no
    configuration in the grid ever reached the target.
    """
    aggregated = aggregate_over_repeats(convergence, min_reach_fraction)
    if aggregated.empty:
        return aggregated

    winners = []
    for _, group in aggregated.groupby(GROUP_COLUMNS, dropna=False, observed=True):
        eligible = group[group["eligible"] & np.isfinite(group["median_subspace_fraction"])]
        if not eligible.empty:
            ranked = eligible.sort_values(
                ["median_subspace_fraction", "median_t_total", "median_iterations"],
                kind="stable",
            )
            winner = ranked.iloc[0].copy()
        else:
            # Nothing met the reach criterion: report the closest attempt so the
            # analysis can still show *that* the target was out of reach.
            ranked = group.sort_values(
                ["reach_fraction", "best_final_fidelity"],
                ascending=[False, False],
                kind="stable",
            )
            winner = ranked.iloc[0].copy()
        winner["reached_any"] = bool(winner["n_reached"] > 0)
        winners.append(winner)

    result = pd.DataFrame(winners).reset_index(drop=True)
    result = result.rename(
        columns={
            "config_label": "best_config_label",
            "median_subspace_fraction": "best_subspace_fraction",
            "median_subspace_dim": "best_subspace_dim",
            "median_t_algorithm": "best_t_algorithm",
            "median_t_solve": "best_t_solve",
            "median_t_total": "best_t_total",
            "median_iterations": "best_iterations",
            "median_shots": "best_shots",
        }
    )
    # A winner only counts as feasible if the median over repeats is an actual
    # dimension. Configurations that reached the target in a minority of repeats
    # keep an infinite median, which is reported as "not reached" rather than as
    # a number that would be misleading in a plot.
    result["feasible"] = np.isfinite(result["best_subspace_fraction"])
    result.loc[~result["feasible"], ["best_subspace_fraction", "best_subspace_dim"]] = (
        np.nan
    )
    return result


def method_comparison(optimal: pd.DataFrame) -> pd.DataFrame:
    """Wide table: one row per case+target with SKQD and BARK side by side."""
    index_cols = CASE_COLUMNS + ["target_fidelity"]
    value_cols = [
        "best_subspace_fraction",
        "best_subspace_dim",
        "best_t_algorithm",
        "best_t_total",
        "best_config_label",
        "feasible",
    ]
    wide = optimal.pivot_table(
        index=index_cols,
        columns="method",
        values=[c for c in value_cols if c in optimal.columns],
        aggfunc="first",
    )
    wide.columns = [f"{value}__{method}" for value, method in wide.columns]
    wide = wide.reset_index()

    skqd = "best_subspace_fraction__SKQD"
    bark = "best_subspace_fraction__BARK"
    if skqd in wide.columns and bark in wide.columns:
        # >1 means BARK needs a smaller subspace than SKQD.
        wide["subspace_advantage_bark"] = wide[skqd] / wide[bark]
    return wide
