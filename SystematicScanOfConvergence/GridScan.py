import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from MakeHam import make_sparse_ground_state_hamiltonian_from_qubits
from JBARK import BARK


class GridScan:
    def __init__(
        self,
        n_qubits: list[int],
        sparsity_values: list[float | int],
        overlaps: list[float],
        seeds: list[int],
        keepstates: list[int] = [1],
    ):
        """
        Parameter scan for BARK experiments.

        Parameters
        ----------
        n_qubits : list[int]
            List of qubit counts to scan.
        sparsity_values : list[float | int]
            Requested ground-state sparsities passed to the Hamiltonian generator.
        overlaps : list[float]
            Target overlap probabilities used to choose a starting index from psi.
        seeds : list[int]
            Random seeds for Hamiltonian generation.
        keepstates : list[int], default=[1]
            Passed to BARK.
        """
        self.n_qubits = list(n_qubits)
        self.sparsity_values = list(sparsity_values)
        self.wanted_overlaps = list(overlaps)
        self.seeds = list(seeds)
        self.keepstates = keepstates

        self.results: pd.DataFrame | None = None

    @staticmethod
    def _compute_stopping_time(samples, support) -> float:
        """
        Return the first time step t+1 at which all support states appear in a sample.
        Returns np.nan if the support is never fully observed.
        """
        support_set = set(support)

        for t, sublist in enumerate(samples):
            if support_set.issubset(set(sublist)):
                return len(sublist)

        return np.nan

    @staticmethod
    def _choose_start_index(psi: np.ndarray, wanted_overlap: float) -> tuple[int, float]:
        """
        Choose the index whose probability |psi_i|^2 is closest to wanted_overlap.

        Returns
        -------
        idx : int
            Chosen basis-state index.
        real_overlap : float
            Actual probability at that index.
        """
        amps = np.abs(psi) ** 2
        idx = int(np.argmin(np.abs(amps - wanted_overlap)))
        real_overlap = float(amps[idx])
        return idx, real_overlap

    def run(self, ground_energy: float = -5.0, gap: float = 1.0) -> pd.DataFrame:
        """
        Execute the full scan and return results as a pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            One row per (n_qubits, sparsity, seed, wanted_overlap).
        """
        rows: list[dict] = []

        for n_qubits in self.n_qubits:
            hilbert_dim = 2 ** n_qubits

            for requested_sparsity in self.sparsity_values:
                for seed in self.seeds:
                    H, psi, support = make_sparse_ground_state_hamiltonian_from_qubits(
                        n_qubits=n_qubits,
                        ground_state_sparsity=requested_sparsity,
                        seed=seed,
                        ground_energy=ground_energy,
                        gap=gap,
                        return_sparse=True,
                    )

                    support = list(support)
                    support_size = len(support)
                    real_sparsity = support_size / hilbert_dim

                    for wanted_overlap in self.wanted_overlaps:
                        start_index, real_overlap = self._choose_start_index(
                            psi, wanted_overlap
                        )

                        for keep in self.keepstates:
                            bark = BARK(H, start_index, keepstates=keep)
                            bark.run()

                            stopping_time = self._compute_stopping_time(
                                bark.samples, support
                            )

                            if stopping_time<support_size:
                                print(
                                    f"Warning: stopping_time {stopping_time} is less than support_size {support_size}. This may indicate an issue with the experiment or the stopping time calculation."
                                )

                            rows.append(
                                {
                                    "n_qubits": n_qubits,
                                    "hilbert_dim": hilbert_dim,
                                    "requested_sparsity": requested_sparsity,
                                    "real_sparsity": real_sparsity,
                                    "keepstates": keep,
                                    "seed": seed,
                                    "wanted_overlap": wanted_overlap,
                                    "real_overlap": real_overlap,
                                    "start_index": start_index,
                                    "support_size": support_size,
                                    "stopping_time": stopping_time,
                                    "relative_stopping_time": stopping_time / hilbert_dim,
                                }
                            )

        self.results = pd.DataFrame(rows)

        if not self.results.empty:
            self.results = self.results.sort_values(
                by=["n_qubits", "requested_sparsity", "seed", "wanted_overlap", "keepstates"]
            ).reset_index(drop=True)

        return self.results

    def get_results(self) -> pd.DataFrame:
        """
        Return the full results DataFrame.
        """
        if self.results is None:
            raise RuntimeError("No results available yet. Call run() first.")
        return self.results.copy()

    def summary(
        self,
        groupby: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Aggregate results with mean/std/count over stopping_time and averages of realized values.

        Parameters
        ----------
        groupby : list[str] | None
            Columns to group by. If None, a sensible default is used.

        Returns
        -------
        pd.DataFrame
        """
        if self.results is None:
            raise RuntimeError("No results available yet. Call run() first.")

        if groupby is None:
            groupby = ["n_qubits", "requested_sparsity", "wanted_overlap"]

        summary_df = (
            self.results.groupby(groupby, dropna=False)
            .agg(
                mean_stopping_time=("stopping_time", "mean"),
                std_stopping_time=("stopping_time", "std"),
                min_stopping_time=("stopping_time", "min"),
                max_stopping_time=("stopping_time", "max"),
                count=("stopping_time", "count"),
                mean_real_overlap=("real_overlap", "mean"),
                mean_real_sparsity=("real_sparsity", "mean"),
            )
            .reset_index()
        )

        return summary_df

    def filter_results(self, **filters) -> pd.DataFrame:
        """
        Filter the results DataFrame by exact matches.

        Example
        -------
        scan.filter_results(n_qubits=8, seed=2)
        """
        if self.results is None:
            raise RuntimeError("No results available yet. Call run() first.")

        df = self.results.copy()
        for col, value in filters.items():
            if col not in df.columns:
                raise KeyError(f"Unknown column: {col}")
            df = df[df[col] == value]

        return df.reset_index(drop=True)
    
    def plot_keepstates_effect(
        self,
        n_qubits: int,
        wanted_overlap: float,
        use_relative_stopping_time: bool = False,
        use_real_overlap: bool = False,
        use_real_sparsity: bool = False,
        overlap_tol: float = 1e-8,
        figsize: tuple[int, int] = (8, 5),
    ) -> None:
        """
        Plot mean stopping time vs sparsity for different keepstates values
        at fixed n_qubits and wanted_overlap.

        The plot aggregates over seeds (and any repeated runs) by taking the mean.

        Parameters
        ----------
        n_qubits : int
            Fixed qubit number to plot.
        wanted_overlap : float
            Target overlap used for filtering.
        use_relative_stopping_time : bool, default=False
            If True, plot relative_stopping_time instead of stopping_time.
        use_real_overlap : bool, default=False
            If True, include the mean realized overlap in the title.
        use_real_sparsity : bool, default=False
            If True, use real_sparsity on the x-axis instead of requested_sparsity.
        overlap_tol : float, default=1e-8
            Tolerance for matching wanted_overlap.
        figsize : tuple[int, int], default=(8, 5)
            Figure size.
        """
        if self.results is None:
            raise RuntimeError("No results available yet. Call run() first.")

        df = self.results.copy()

        required_cols = {"n_qubits", "wanted_overlap", "requested_sparsity", "real_sparsity", "keepstates", "stopping_time"}
        missing = required_cols - set(df.columns)
        if missing:
            raise KeyError(
                f"Missing required columns in results: {sorted(missing)}. "
                "Did you include 'keepstates' when building the results table?"
            )

        ycol = "relative_stopping_time" if use_relative_stopping_time else "stopping_time"
        if ycol not in df.columns:
            raise KeyError(f"Column '{ycol}' not found in results.")

        xcol = "real_sparsity" if use_real_sparsity else "requested_sparsity"
        xlabel = "Real sparsity" if use_real_sparsity else "Requested sparsity"
        ylabel = "Relative stopping time" if use_relative_stopping_time else "Stopping time"

        filtered_df = df[
            (df["n_qubits"] == n_qubits)
            & (np.isclose(df["wanted_overlap"], wanted_overlap, atol=overlap_tol, rtol=0.0))
        ].copy()

        if filtered_df.empty:
            raise ValueError(
                f"No rows found for n_qubits={n_qubits} and wanted_overlap≈{wanted_overlap}."
            )

        # Aggregate over seeds / repeated runs for cleaner lines
        plot_df = (
            filtered_df.groupby(["keepstates", xcol], as_index=False)
            .agg(
                mean_y=(ycol, "mean"),
                std_y=(ycol, "std"),
                mean_real_overlap=("real_overlap", "mean"),
                count=(ycol, "count"),
            )
            .sort_values(["keepstates", xcol])
        )

        plt.figure(figsize=figsize)

        for keepstates, subdf in plot_df.groupby("keepstates"):
            plt.plot(
                subdf[xcol],
                subdf["mean_y"],
                marker="o",
                label=f"keepstates={keepstates}",
            )

        if use_real_overlap:
            title_overlap = f"mean real overlap ≈ {filtered_df['real_overlap'].mean():.4f}"
        else:
            title_overlap = f"wanted overlap = {wanted_overlap:.4f}"

        plt.title(
            f"Effect of keepstates on {ylabel.lower()}\n"
            f"n_qubits={n_qubits}, {title_overlap}"
        )
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.tight_layout()
        plt.show()


    def plot_mean_stopping_time_vs_overlap(
        self,
        qubit_col: str = "n_qubits",
        overlap_col: str = "wanted_overlap",
        use_real_overlap: bool = False,
        use_relative_stopping_time: bool = False,
        figsize: tuple[int, int] = (8, 5),
    ) -> None:
        """
        Plot mean stopping time vs overlap, grouped by qubit count.

        Parameters
        ----------
        qubit_col : str
            Usually "n_qubits".
        overlap_col : str
            Usually "wanted_overlap".
        use_real_overlap : bool
            If True, plot against real_overlap instead of wanted_overlap.
        use_relative_stopping_time : bool
            If True, plot relative stopping time.
        """
        if self.results is None:
            raise RuntimeError("No results available yet. Call run() first.")

        xcol = "real_overlap" if use_real_overlap else overlap_col

        if use_relative_stopping_time:
            ycol = "relative_stopping_time"
        else:
            ycol = "stopping_time"

        summary_df = (
            self.results.groupby([qubit_col, xcol], dropna=False)[ycol]
            .mean()
            .reset_index()
            .sort_values([qubit_col, xcol])
        )

        plt.figure(figsize=figsize)

        for q, subdf in summary_df.groupby(qubit_col):
            plt.plot(
                subdf[xcol],
                subdf[ycol],
                marker="o",
                label=f"{q} qubits",
            )

        plt.xlabel(xcol)
        plt.ylabel("Mean stopping time")
        plt.title("Mean stopping time vs overlap")
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_mean_stopping_time_vs_sparsity(
        self,
        use_real_sparsity: bool = True,
        use_relative_stopping_time: bool = False,
        figsize: tuple[int, int] = (8, 5),
    ) -> None:
        """
        Plot mean stopping time vs sparsity, grouped by n_qubits.
        """
        if self.results is None:
            raise RuntimeError("No results available yet. Call run() first.")

        xcol = "real_sparsity" if use_real_sparsity else "requested_sparsity"

        if use_relative_stopping_time:
            ycol = "relative_stopping_time"
        else:
            ycol = "stopping_time"

        summary_df = (
            self.results.groupby(["n_qubits", xcol], dropna=False)[ycol]
            .mean()
            .reset_index()
            .sort_values(["n_qubits", xcol])
        )

        plt.figure(figsize=figsize)

        for q, subdf in summary_df.groupby("n_qubits"):
            plt.plot(
                subdf[xcol],
                subdf[ycol],
                marker="o",
                label=f"{q} qubits",
            )

        plt.xlabel(xcol)
        plt.ylabel("Mean stopping time")
        plt.title("Mean stopping time vs sparsity")
        plt.legend()
        plt.tight_layout()
        plt.show()

    def scatter_stopping_time(
        self,
        x: str = "real_overlap",
        y: str = "stopping_time",
        groupby: str = "n_qubits",
        figsize: tuple[int, int] = (8, 5),
        alpha: float = 0.7,
    ) -> None:
        """
        Scatter plot of raw results, grouped by one column.
        """
        if self.results is None:
            raise RuntimeError("No results available yet. Call run() first.")

        if x not in self.results.columns:
            raise KeyError(f"Unknown x column: {x}")
        if y not in self.results.columns:
            raise KeyError(f"Unknown y column: {y}")
        if groupby not in self.results.columns:
            raise KeyError(f"Unknown groupby column: {groupby}")

        plt.figure(figsize=figsize)

        for group_value, subdf in self.results.groupby(groupby):
            plt.scatter(subdf[x], subdf[y], alpha=alpha, label=str(group_value))

        plt.xlabel(x)
        plt.ylabel(y)
        plt.title(f"{y} vs {x}")
        plt.legend(title=groupby)
        plt.tight_layout()
        plt.show()

    def pivot_table(
        self,
        index: str = "wanted_overlap",
        columns: str = "n_qubits",
        values: str = "stopping_time",
        aggfunc: str = "mean",
    ) -> pd.DataFrame:
        """
        Convenience wrapper around pandas pivot_table for quick inspection.
        """
        if self.results is None:
            raise RuntimeError("No results available yet. Call run() first.")

        return pd.pivot_table(
            self.results,
            index=index,
            columns=columns,
            values=values,
            aggfunc=aggfunc,
        )