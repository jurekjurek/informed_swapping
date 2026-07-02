"""Plotting helpers used across the whole project.

These are the plots that keep getting re-written inline in notebooks and
cluster scripts: the structure of a Hamiltonian, its spectrum, the amplitude
profile of a (sparse) ground state, and energy-vs-#states convergence paths for
comparing orderings from different methods.

All functions accept an optional ``ax``/``axes`` so they compose into larger
figures, and return the Matplotlib object they drew on. Nothing here calls
``plt.show()`` — that is left to the caller / notebook.
"""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Optional, Sequence, Union

import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

ArrayLike = Union[np.ndarray, Sequence[float]]
HamLike = Union[np.ndarray, sp.spmatrix]


def _to_dense(H: HamLike) -> np.ndarray:
    """Return a dense ndarray view of a dense or scipy-sparse matrix."""
    if sp.issparse(H):
        return np.asarray(H.toarray())
    return np.asarray(H)


def plot_hamiltonian(
    H: HamLike,
    *,
    axes=None,
    show_spectrum: bool = True,
    log_magnitude: bool = True,
    title: Optional[str] = None,
    cmap: str = "viridis",
):
    """Visualise a Hamiltonian: sparsity pattern, |H| heatmap and spectrum.

    Parameters
    ----------
    H
        Square Hamiltonian, dense ``ndarray`` or any ``scipy.sparse`` matrix.
    axes
        Optional sequence of Matplotlib axes to draw on. If ``None`` a new
        figure is created. Needs 3 axes when ``show_spectrum`` else 2.
    show_spectrum
        Also compute and plot the sorted eigenvalues (dense diagonalisation —
        keep this off for very large ``H``).
    log_magnitude
        Colour the magnitude heatmap on a log scale (``log10|H_ij|``), which is
        usually far more informative for these Hamiltonians.
    title
        Optional suptitle.
    cmap
        Matplotlib colormap name for the magnitude heatmap.

    Returns
    -------
    matplotlib.figure.Figure
    """
    dense = _to_dense(H)
    n = dense.shape[0]

    n_panels = 3 if show_spectrum else 2
    if axes is None:
        fig, axes = plt.subplots(1, n_panels, figsize=(5.2 * n_panels, 4.6))
    else:
        fig = np.asarray(axes).ravel()[0].get_figure()
    axes = np.asarray(axes).ravel()

    # (1) sparsity pattern
    ax = axes[0]
    ax.spy(sp.csr_matrix(dense), markersize=max(0.5, 400.0 / n), color="black")
    ax.set_title(f"Sparsity pattern (dim={n}, nnz={int(np.count_nonzero(dense))})")
    ax.set_xlabel("column"); ax.set_ylabel("row")

    # (2) magnitude heatmap
    ax = axes[1]
    mag = np.abs(dense)
    if log_magnitude:
        floor = mag[mag > 0].min() if np.any(mag > 0) else 1.0
        shown = np.log10(np.where(mag > 0, mag, floor * 1e-3))
        cbar_label = r"$\log_{10}|H_{ij}|$"
    else:
        shown = mag
        cbar_label = r"$|H_{ij}|$"
    im = ax.imshow(shown, cmap=cmap, aspect="equal")
    ax.set_title("Matrix-element magnitude")
    ax.set_xlabel("column"); ax.set_ylabel("row")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=cbar_label)

    # (3) spectrum
    if show_spectrum:
        ax = axes[2]
        evals = np.linalg.eigvalsh(0.5 * (dense + dense.conj().T))
        ax.plot(np.arange(n), evals, marker="o", markersize=3, linewidth=1.0)
        ax.axhline(evals[0], color="red", linestyle="--", linewidth=1.0,
                   label=f"E₀={evals[0]:.4f}")
        if n > 1:
            ax.set_title(f"Spectrum (gap={evals[1] - evals[0]:.4f})")
        else:
            ax.set_title("Spectrum")
        ax.set_xlabel("eigenvalue index"); ax.set_ylabel("energy")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    if title:
        fig.suptitle(title, fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
    else:
        fig.tight_layout()
    return fig


def plot_ground_state(
    psi: ArrayLike,
    *,
    ax=None,
    n_qubits: Optional[int] = None,
    top: Optional[int] = None,
    title: str = "Ground-state amplitude profile",
):
    """Bar plot of ``|psi_i|^2`` over the computational basis.

    Parameters
    ----------
    psi
        State vector (length ``2**n_qubits``).
    ax
        Optional Matplotlib axis.
    n_qubits
        If given, x-tick labels use ``n_qubits``-wide bitstrings; inferred from
        ``len(psi)`` otherwise.
    top
        If set, only the ``top`` largest-probability basis states are shown
        (useful for sparse states in a large Hilbert space).
    title
        Axis title.

    Returns
    -------
    matplotlib.axes.Axes
    """
    psi = np.asarray(psi).ravel()
    probs = np.abs(psi) ** 2
    dim = probs.size
    if n_qubits is None:
        n_qubits = int(round(np.log2(dim))) if dim > 1 else 1

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4))

    if top is not None and top < dim:
        idx = np.argsort(probs)[::-1][:top]
        idx = np.sort(idx)
        labels = [format(i, f"0{n_qubits}b") for i in idx]
        ax.bar(range(len(idx)), probs[idx], color="steelblue")
        ax.set_xticks(range(len(idx)))
        ax.set_xticklabels(labels, rotation=90, fontsize=7)
        ax.set_xlabel(f"basis state (top {top} by probability)")
    else:
        ax.bar(range(dim), probs, color="steelblue")
        ax.set_xlabel("basis index")

    support = int(np.count_nonzero(probs > 1e-12))
    ax.set_ylabel(r"$|\psi_i|^2$")
    ax.set_title(f"{title}  (support={support}/{dim})")
    ax.grid(True, alpha=0.3, axis="y")
    return ax


def plot_convergence_paths(
    paths: Mapping[str, ArrayLike],
    *,
    true_energy: Optional[float] = None,
    ax=None,
    rel_error_band: Optional[float] = 1e-3,
    title: str = "Energy vs. number of sampled basis states",
):
    """Overlay several energy-vs-#states convergence paths.

    Each entry maps a label to a path array, where ``path[i]`` is the
    ground-energy estimate using the first ``i+1`` sampled basis states (exactly
    what :func:`subspace_search.paths.get_one_path` returns). This is the
    canonical way to compare orderings produced by KRAB, BARK, SKQD and random
    baselines on the same axes.

    Parameters
    ----------
    paths
        ``{label: path}``. A "path" that is a 2-D array (several orderings) is
        drawn as a faint bundle under a single legend entry — handy for random
        baselines.
    true_energy
        If given, draws the reference line and (with ``rel_error_band``) the
        target band.
    ax
        Optional Matplotlib axis.
    rel_error_band
        Relative-error band drawn around ``true_energy`` (e.g. ``1e-3``). Set
        ``None`` to omit.
    title
        Axis title.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    for label, path in paths.items():
        arr = np.asarray(path, dtype=float)
        if arr.ndim == 2:  # bundle of orderings, e.g. random baseline
            for row in arr:
                ax.plot(np.arange(1, row.size + 1), row,
                        alpha=0.08, color="steelblue", linewidth=0.6)
            ax.plot([], [], color="steelblue", alpha=0.5, linewidth=1.5, label=label)
        else:
            ax.plot(np.arange(1, arr.size + 1), arr, linewidth=1.8, label=label)

    if true_energy is not None:
        ax.axhline(true_energy, color="black", linestyle=":", linewidth=1.3,
                   label=f"True E₀={true_energy:.4f}")
        if rel_error_band is not None:
            band = abs(true_energy) * rel_error_band
            ax.axhspan(true_energy - band, true_energy + band,
                       color="green", alpha=0.12,
                       label=f"±{rel_error_band:.0e} rel. band")

    ax.set_xlabel("number of sampled basis states (in order)")
    ax.set_ylabel("ground-state energy estimate")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    return ax
