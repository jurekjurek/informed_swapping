"""SKQD (Subspace-based Krylov Quantum Diagonalization) reference routine.

``do_skqd`` is the classical simulation of SKQD used as the reference method in
every study: it time-evolves the state with ``U = exp(-iHt)`` and samples the
highest-amplitude basis states at each step to build an ordering of the
computational basis. ``do_power`` is the analogous power-iteration sampler
(uses the inverse of a shifted ``H`` instead of a real-time propagator).
"""

from .skqd import do_skqd, get_exponential
from .power import do_power

__all__ = ["do_skqd", "get_exponential", "do_power"]
