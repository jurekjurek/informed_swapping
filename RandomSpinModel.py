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

import math

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
    N_target: int | None = None,          
    penalty_strength: float = 0.0,
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
        N_target: target number of excitations (for penalty term).
        penalty_strength: strength of the penalty term for enforcing the target excitation number.
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

    # Particle Number Penalty
    if N_target is not None and penalty_strength != 0.0:

        penalty_strength *= 1024/(2**num_sites)  # Scale penalty strength with system size

        a = num_sites / 2 - N_target

        # the following terms come from squaring the operator: 
        # (N-N_target)^2 = constant + single-Z terms + ZZ terms
        # constant term
        sparse_terms.append(
            ("I", [0], penalty_strength * (a**2 + num_sites / 4))
        )

        # single-Z terms
        for i in range(num_sites):
            sparse_terms.append(
                ("Z", [i], -penalty_strength * a)
            )

        # ZZ terms
        for i in range(num_sites):
            for j in range(i + 1, num_sites):
                sparse_terms.append(
                    ("ZZ", [i, j], penalty_strength / 2)
                )

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
        "N_target": N_target,
        "penalty_strength": penalty_strength,
    }

    return H, info

# ---------------------------------------------------------------------------
# Deterministic XXZ-Heisenberg model (Eq. (1) of Entropy 2026, 28, 367)
# ---------------------------------------------------------------------------


def most_square_rectangle(num_sites: int):
    """
    Map ``num_sites`` qubits onto the most-square open rectangle r x c.

    Implements the divisor-search heuristic of the reference paper,

        (r, c) = argmin_{rc=N, r<=c} |r - c|,
        r = max{ q <= sqrt(N) | N mod q == 0 },

    i.e. the search starts at floor(sqrt(N)) and steps downward until the
    first divisor is found; ``c = N / r`` follows. For prime ``N`` the loop
    ends at ``r = 1`` and the layout degenerates to a ``1 x N`` strip.

    Args:
        num_sites: number of spins.

    Returns:
        (rows, cols) with ``rows <= cols`` and ``rows * cols == num_sites``.
    """
    if num_sites < 1:
        raise ValueError("num_sites must be >= 1.")

    for rows in range(math.isqrt(num_sites), 0, -1):
        if num_sites % rows == 0:
            return rows, num_sites // rows

    return 1, num_sites  # unreachable, kept for clarity


def lattice_edges(num_sites: int, dimension: int):
    """
    Nearest-neighbour bonds with open boundary conditions.

    ``dimension=1`` gives a linear chain ``0-1-2-...-(N-1)``.
    ``dimension=2`` gives the most-square open rectangle from
    ``most_square_rectangle``, with row-major site labelling
    ``i = row * cols + col`` and bonds along both axes.

    Args:
        num_sites: number of spins.
        dimension: 1 for a chain, 2 for a square lattice.

    Returns:
        edges: sorted list of ``(i, j)`` tuples with ``i < j``.
    """
    if dimension == 1:
        return [(i, i + 1) for i in range(num_sites - 1)]

    if dimension == 2:
        rows, cols = most_square_rectangle(num_sites)
        edges: list[tuple[int, int]] = []
        for r in range(rows):
            for c in range(cols):
                site = r * cols + c
                if c + 1 < cols:                      # horizontal bond
                    edges.append((site, site + 1))
                if r + 1 < rows:                      # vertical bond
                    edges.append((site, site + cols))
        edges.sort()
        return edges

    raise ValueError(f"Unknown dimension {dimension!r}; expected 1 or 2.")


def _particle_number_penalty_terms(
    num_sites: int, N_target: int, penalty_strength: float
):
    """
    Sparse Pauli terms for ``penalty_strength * (N - N_target)^2``.

    With ``N = sum_i (I - Z_i) / 2`` the square expands into a constant, a
    single-Z part and a ZZ part. Same convention (and the same size scaling
    ``1024 / 2**num_sites``) as in ``make_random_spin_hamiltonian``.
    """
    terms: list[tuple[str, list[int], float]] = []

    penalty_strength *= 1024 / (2**num_sites)  # scale with system size
    a = num_sites / 2 - N_target

    # constant term
    terms.append(("I", [0], penalty_strength * (a**2 + num_sites / 4)))

    # single-Z terms
    for i in range(num_sites):
        terms.append(("Z", [i], -penalty_strength * a))

    # ZZ terms
    for i in range(num_sites):
        for j in range(i + 1, num_sites):
            terms.append(("ZZ", [i, j], penalty_strength / 2))

    return terms


def make_heisenberg_hamiltonian(
    num_sites: int,
    dimension: int = 1,
    J: float = 1.0,
    delta: float = 1.0,
    h: tuple[float, float, float] = (0.0, 0.0, 0.0),
    spin: float = 0.5
):
    """
    Build the uniform XXZ-Heisenberg Hamiltonian on a 1D or 2D lattice.

        H = J sum_<i,j> ( S^x_i S^x_j + S^y_i S^y_j + delta * S^z_i S^z_j )
            - sum_i ( h^x S^x_i + h^y S^y_i + h^z S^z_i )

    Note the sign convention: this follows Eq. (1) of the reference paper, so
    the coupling enters with a *plus* sign (``J > 0`` is antiferromagnetic,
    ``J < 0`` ferromagnetic) while the field enters with a *minus* sign. This
    differs from ``make_random_spin_hamiltonian``, where both terms carry a
    minus sign; flip the sign of ``J`` if you need the other convention.

    The sum runs over nearest-neighbour bonds only, with open boundary
    conditions in both geometries.

    Args:
        num_sites: number of spins (qubits).
        dimension: 1 for a linear chain, 2 for the most-square open rectangle
            (see ``most_square_rectangle``) with four-nearest-neighbour
            connectivity.
        J: exchange coupling, identical on every bond.
        delta: anisotropy parameter. ``delta = 1`` is the isotropic XXX model,
            ``delta != 1`` the anisotropic XXZ model (``delta >> 1`` is
            Ising-like, ``delta < 1`` easy-plane).
        h: magnetic field as a 3-vector ``(h^x, h^y, h^z)``, identical on
            every site.
        spin: spin length scaling ``S^alpha = spin * sigma^alpha``. Default
            0.5 gives physical spin-1/2 operators; 1.0 uses bare Pauli
            operators (the convention of Eq. (1) in the paper).

    Returns:
        H: SparsePauliOp for the Hamiltonian.
        info: dict with the lattice, couplings, field and metadata.
    """
    if num_sites < 1:
        raise ValueError("num_sites must be >= 1.")

    h = tuple(float(x) for x in h)
    if len(h) != 3:
        raise ValueError(
            "h must be a 3-component vector (h^x, h^y, h^z); "
            f"got {len(h)} component(s)."
        )

    edges = lattice_edges(num_sites, dimension)
    shape = (1, num_sites) if dimension == 1 else most_square_rectangle(num_sites)

    sparse_terms: list[tuple[str, list[int], float]] = []

    # Coupling terms: +J S^alpha_i S^alpha_j, with delta on the zz component.
    bond_couplings = {"x": J, "y": J, "z": J * delta}
    for alpha, coupling in bond_couplings.items():
        coeff = coupling * spin * spin
        if coeff == 0.0:
            continue
        pauli = _PAULI[alpha]
        for i, j in edges:
            sparse_terms.append((pauli + pauli, [i, j], coeff))

    # Field terms: -h^alpha S^alpha_i.
    for alpha, field in zip(("x", "y", "z"), h):
        coeff = -field * spin
        if coeff == 0.0:
            continue
        pauli = _PAULI[alpha]
        for i in range(num_sites):
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

    # Dense coupling matrices, for compatibility with the random-model info.
    J_matrices = {a: np.zeros((num_sites, num_sites)) for a in ("x", "y", "z")}
    for alpha, coupling in bond_couplings.items():
        for i, j in edges:
            J_matrices[alpha][i, j] = coupling
            J_matrices[alpha][j, i] = coupling

    info = {
        "num_sites": num_sites,
        "dimension": dimension,
        "lattice_shape": shape,
        "edges": edges,
        "degrees": degrees,
        "max_degree": int(degrees.max()) if num_sites else 0,
        "J": J,
        "delta": delta,
        "h": h,
        "J_matrices": J_matrices,
        "spin": spin,
        "num_pauli_terms": len(H.paulis),
    }

    return H, info
