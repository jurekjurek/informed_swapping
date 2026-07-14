"""Random spin models with controllable disorder.

Implements the Hamiltonian

    H = - sum_{i<j} ( J^x_{ij} S^x_i S^x_j
                    + J^y_{ij} S^y_i S^y_j
                    + J^z_{ij} S^z_i S^z_j )
        - sum_i  ( B^x_i S^x_i + B^y_i S^y_i + B^z_i S^z_i ),

where the couplings ``J^alpha_{ij}`` and the local fields ``B^alpha_i`` are
sampled randomly with tunable properties:

* ``max_interactions`` bounds the interaction graph: each site couples to at
  most this many partners, so every row of each ``J^alpha`` matrix has at most
  ``max_interactions`` non-zero entries.
* ``J_max`` / ``B_max`` bound the maximal amplitude of the sampled couplings
  and fields.
* ``J_components`` / ``B_components`` select which spin components are active.

Spins are spin-1/2: ``S^alpha = spin * sigma^alpha`` with ``spin = 1/2`` by
default (set ``spin=1.0`` to work directly with Pauli operators).
"""

import numpy as np
from qiskit.quantum_info import SparsePauliOp

_PAULI = {"x": "X", "y": "Y", "z": "Z"}


def sample_interaction_graph(
    num_sites: int,
    max_interactions: int | None,
    rng: np.random.Generator,
):
    """
    Sample an undirected interaction graph with bounded degree.

    Every site participates in at most ``max_interactions`` bonds. Edges are
    added greedily from a random permutation of all candidate pairs, so the
    result is a random graph respecting the degree cap (not necessarily
    regular). ``max_interactions=None`` returns the complete graph.

    Args:
        num_sites: number of spins.
        max_interactions: maximum number of bonds per site, or None for
            all-to-all coupling.
        rng: NumPy random generator.

    Returns:
        edges: sorted list of ``(i, j)`` tuples with ``i < j``.
    """
    all_pairs = [
        (i, j) for i in range(num_sites) for j in range(i + 1, num_sites)
    ]

    if max_interactions is None or max_interactions >= num_sites - 1:
        return all_pairs

    if max_interactions <= 0:
        return []

    order = rng.permutation(len(all_pairs))
    degree = np.zeros(num_sites, dtype=int)
    edges: list[tuple[int, int]] = []

    for idx in order:
        i, j = all_pairs[idx]
        if degree[i] < max_interactions and degree[j] < max_interactions:
            edges.append((i, j))
            degree[i] += 1
            degree[j] += 1

    edges.sort()
    return edges


def _sample_amplitude(size, max_amplitude, distribution, rng):
    """Sample couplings/fields with a bounded maximal amplitude."""
    if max_amplitude == 0.0:
        return np.zeros(size)

    if distribution == "uniform":
        # Uniform in [-max_amplitude, max_amplitude]: |value| <= max_amplitude.
        return rng.uniform(-max_amplitude, max_amplitude, size=size)

    if distribution == "normal":
        # Gaussian with std max_amplitude, then hard-clipped to the cap.
        vals = rng.normal(0.0, max_amplitude, size=size)
        return np.clip(vals, -max_amplitude, max_amplitude)

    if distribution == "bimodal":
        # +-max_amplitude with equal probability (Ising-style disorder).
        return max_amplitude * rng.choice([-1.0, 1.0], size=size)

    raise ValueError(
        f"Unknown distribution {distribution!r}; "
        "expected 'uniform', 'normal', or 'bimodal'."
    )


def make_random_spin_hamiltonian(
    num_sites: int,
    max_interactions: int | None = None,
    J_max: float = 1.0,
    B_max: float = 1.0,
    J_components: tuple[str, ...] = ("x", "y", "z"),
    B_components: tuple[str, ...] = ("x", "y", "z"),
    coupling_distribution: str = "uniform",
    field_distribution: str = "uniform",
    spin: float = 0.5,
    seed: int | None = None,
):
    """
    Build a random spin-1/2 Hamiltonian with controllable disorder.

        H = - sum_{i<j} sum_alpha J^alpha_{ij} S^alpha_i S^alpha_j
            - sum_i sum_alpha B^alpha_i S^alpha_i

    Args:
        num_sites: number of spins (qubits).
        max_interactions: maximum number of bonds per site. Each row of every
            ``J^alpha`` matrix then has at most this many non-zero entries.
            ``None`` (default) means all-to-all coupling.
        J_max: maximal amplitude of the sampled couplings.
        B_max: maximal amplitude of the sampled fields.
        J_components: which coupling components are active, subset of
            ``("x", "y", "z")``.
        B_components: which field components are active, subset of
            ``("x", "y", "z")``.
        coupling_distribution: how ``J`` is drawn -- "uniform", "normal", or
            "bimodal" (see ``_sample_amplitude``).
        field_distribution: how ``B`` is drawn, same options.
        spin: spin length scaling ``S^alpha = spin * sigma^alpha``. Default 0.5
            gives physical spin-1/2 operators; 1.0 uses bare Pauli operators.
        seed: seed for the NumPy random generator.

    Returns:
        H: SparsePauliOp for the Hamiltonian.
        info: dict with the sampled couplings, fields, interaction graph, and
            metadata.
    """
    rng = np.random.default_rng(seed)

    for comp in tuple(J_components) + tuple(B_components):
        if comp not in _PAULI:
            raise ValueError(
                f"Unknown spin component {comp!r}; expected 'x', 'y', or 'z'."
            )

    edges = sample_interaction_graph(num_sites, max_interactions, rng)

    # J[alpha] is a symmetric coupling matrix; B[alpha] is a field vector.
    J = {a: np.zeros((num_sites, num_sites)) for a in ("x", "y", "z")}
    B = {a: np.zeros(num_sites) for a in ("x", "y", "z")}

    sparse_terms: list[tuple[str, list[int], float]] = []

    # Coupling terms: -J^alpha_{ij} S^alpha_i S^alpha_j.
    for alpha in J_components:
        if not edges:
            continue
        values = _sample_amplitude(
            len(edges), J_max, coupling_distribution, rng
        )
        pauli = _PAULI[alpha]
        for (i, j), val in zip(edges, values):
            J[alpha][i, j] = val
            J[alpha][j, i] = val
            coeff = -val * spin * spin
            if coeff != 0.0:
                sparse_terms.append((pauli + pauli, [i, j], coeff))

    # Field terms: -B^alpha_i S^alpha_i.
    for alpha in B_components:
        values = _sample_amplitude(num_sites, B_max, field_distribution, rng)
        pauli = _PAULI[alpha]
        for i, val in enumerate(values):
            B[alpha][i] = val
            coeff = -val * spin
            if coeff != 0.0:
                sparse_terms.append((pauli, [i], coeff))

    if sparse_terms:
        H = SparsePauliOp.from_sparse_list(
            sparse_terms, num_qubits=num_sites
        ).simplify(atol=1e-12)
    else:
        H = SparsePauliOp.from_list([("I" * num_sites, 0.0)])

    degrees = np.zeros(num_sites, dtype=int)
    for i, j in edges:
        degrees[i] += 1
        degrees[j] += 1

    info = {
        "num_sites": num_sites,
        "edges": edges,
        "degrees": degrees,
        "max_degree": int(degrees.max()) if num_sites else 0,
        "max_interactions": max_interactions,
        "J": J,
        "B": B,
        "J_components": tuple(J_components),
        "B_components": tuple(B_components),
        "J_max": J_max,
        "B_max": B_max,
        "spin": spin,
        "coupling_distribution": coupling_distribution,
        "field_distribution": field_distribution,
        "num_pauli_terms": len(H.paulis),
    }

    return H, info
