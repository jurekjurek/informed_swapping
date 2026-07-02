"""subspace_search
================

Reusable building blocks for the *informed-swapping* study: can classical,
subspace-selecting heuristics reproduce or beat SKQD (Subspace-based Krylov
Quantum Diagonalization) at finding sparse ground states?

Subpackages / modules
---------------------
``subspace_search.hamiltonians``
    Generators for computational-basis Hamiltonians with a controlled-sparsity
    exact ground state.
``subspace_search.skqd``
    The SKQD reference routine (``do_skqd``) and the power-iteration sampler.
``subspace_search.algorithms``
    Classical Krylov-style search algorithms: KRAB
    (``selected_krylov_ground_state``) and BARK (``BarkBarkBark``). New
    algorithms go here.
``subspace_search.paths``
    Subspace projection and energy-vs-#states convergence paths — the common
    currency used to compare orderings from any method.
``subspace_search.plotting``
    Plotting helpers for Hamiltonians, spectra, ground states and convergence
    paths.

Typical use
-----------
>>> from subspace_search.hamiltonians import make_controlled_sparse_ground_state_hamiltonian_fast
>>> from subspace_search.skqd import do_skqd
>>> from subspace_search.algorithms import selected_krylov_ground_state
>>> from subspace_search.paths import get_one_path
>>> from subspace_search.plotting import plot_hamiltonian
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
