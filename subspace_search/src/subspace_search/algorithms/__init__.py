"""Subspace-search algorithms.

This is the home for the classical Krylov-style ground-state search algorithms
that are compared against SKQD:

* **KRAB** — :func:`selected_krylov_ground_state` (selected-subspace Krylov with
  scheduled per-iteration budget ``Q``, candidate pruning, coefficient pruning
  and an optional hard subspace cap). Returns a :class:`SelectedKrylovResult`
  with rich per-iteration diagnostics.
* **BARK** — :class:`BarkBarkBark` (*Bitstring Algorithm for Recursive Krylov*),
  a Pauli-string-propagation classical analogue of SKQD.

**Adding a new algorithm** (e.g. a new dog-named variant): drop a
``my_algorithm.py`` module in this package, expose its public entry point, and
re-export it here in ``__all__``. See ``algorithms/README.md`` for the
conventions (accept a scipy CSR/qiskit ``H``, return an ordering of basis
indices and/or a result object) so it plugs into ``subspace_search.paths`` and
the cluster studies unchanged.
"""

from .krab import (
    selected_krylov_ground_state,
    SelectedKrylovResult,
    plot_selected_krylov_diagnostics,
)
from .bark import BarkBarkBark

__all__ = [
    "selected_krylov_ground_state",
    "SelectedKrylovResult",
    "plot_selected_krylov_diagnostics",
    "BarkBarkBark",
]
