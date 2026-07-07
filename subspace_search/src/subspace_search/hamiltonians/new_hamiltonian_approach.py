import itertools
import numpy as np
from qiskit.quantum_info import SparsePauliOp


def single_qubit_dyad_paulis(bra_bit: int, ket_bit: int):
    """
    Return Pauli decomposition of |bra_bit><ket_bit| on one qubit.

    Conventions:
        |0><0| = (I + Z)/2
        |1><1| = (I - Z)/2
        |0><1| = (X + iY)/2
        |1><0| = (X - iY)/2
    """
    if bra_bit == 0 and ket_bit == 0:
        return [("I", 0.5), ("Z", 0.5)]
    if bra_bit == 1 and ket_bit == 1:
        return [("I", 0.5), ("Z", -0.5)]
    if bra_bit == 0 and ket_bit == 1:
        return [("X", 0.5), ("Y", 0.5j)]
    if bra_bit == 1 and ket_bit == 0:
        return [("X", 0.5), ("Y", -0.5j)]


def dyad_to_sparse_pauli_op(bra: int, ket: int, num_qubits: int, coeff=1.0):
    """
    Convert coeff * |bra><ket| into a SparsePauliOp.

    Note:
        Qiskit Pauli strings are written left-to-right as qubits n-1 ... 0.
        This code uses integer bit order consistently and then reverses labels
        for Qiskit.
    """
    bra_bits = [(bra >> q) & 1 for q in range(num_qubits)]
    ket_bits = [(ket >> q) & 1 for q in range(num_qubits)]

    local_terms = [
        single_qubit_dyad_paulis(bra_bits[q], ket_bits[q])
        for q in range(num_qubits)
    ]

    pauli_terms = []
    for product_term in itertools.product(*local_terms):
        label_q0_first = "".join(p for p, _ in product_term)
        label_qiskit = label_q0_first[::-1]
        c = coeff * np.prod([c for _, c in product_term])
        pauli_terms.append((label_qiskit, c))

    return SparsePauliOp.from_list(pauli_terms).simplify(atol=1e-12)


def planted_sparse_projector(
    num_qubits: int,
    support: list[int],
    amplitudes: np.ndarray | None = None,
):
    """
    Build |g><g| as SparsePauliOp, where |g> has support only on `support`.

    Args:
        num_qubits: number of qubits.
        support: list of computational-basis indices.
        amplitudes: optional complex amplitudes on support.

    Returns:
        projector: SparsePauliOp for |g><g|.
        g_dense: dense statevector, useful for diagnostics.
    """
    dim = 2**num_qubits
    support = list(support)

    if amplitudes is None:
        rng = np.random.default_rng()
        amplitudes = rng.normal(size=len(support)) + 1j * rng.normal(size=len(support))

    amplitudes = np.asarray(amplitudes, dtype=complex)
    amplitudes = amplitudes / np.linalg.norm(amplitudes)

    g_dense = np.zeros(dim, dtype=complex)
    for idx, amp in zip(support, amplitudes):
        g_dense[idx] = amp

    projector = SparsePauliOp.from_list([("I" * num_qubits, 0.0)])

    for a, ca in zip(support, amplitudes):
        for b, cb in zip(support, amplitudes):
            # |g><g| = sum_ab c_a c_b^* |a><b|
            projector += dyad_to_sparse_pauli_op(
                bra=a,
                ket=b,
                num_qubits=num_qubits,
                coeff=ca * np.conj(cb),
            )

    return projector.simplify(atol=1e-12), g_dense


def random_pauli_background(
    num_qubits: int,
    pauli_density: float,
    rng: np.random.Generator,
    offdiag_only: bool = True,
    normalize: bool = True,
):
    """
    Random Hermitian Pauli background R.

    pauli_density controls the fraction of possible non-identity Pauli strings used.
    If offdiag_only=True, only strings containing X or Y are used, so they create
    off-diagonal couplings in the computational basis.

    Returns:
        SparsePauliOp
    """
    labels = []

    for chars in itertools.product("IXYZ", repeat=num_qubits):
        label = "".join(chars)

        if set(label) == {"I"}:
            continue

        if offdiag_only and not any(p in label for p in "XY"):
            continue

        labels.append(label)

    max_terms = len(labels)
    num_terms = max(1, int(round(pauli_density * max_terms)))
    num_terms = min(num_terms, max_terms)

    chosen = rng.choice(labels, size=num_terms, replace=False)

    coeffs = rng.normal(size=num_terms)

    if normalize:
        # Keeps ||R|| from blowing up too fast as the number of terms grows.
        coeffs = coeffs / np.sqrt(num_terms)

    return SparsePauliOp.from_list(list(zip(chosen, coeffs))).simplify(atol=1e-12)

def make_sparse_ground_state_with_initial_overlap(
    num_qubits: int,
    ground_support_size: int,
    rng: np.random.Generator,
    initial_bitstring: int | None = None,
    target_initial_overlap: float | None = None,
    support: list[int] | None = None,
):
    """
    Construct a sparse planted ground state |g>.

    If initial_bitstring is given and target_initial_overlap is given, then
    |<initial_bitstring|g>|^2 is set approximately to target_initial_overlap.

    Args:
        num_qubits: number of qubits.
        ground_support_size: number of nonzero computational-basis amplitudes.
        rng: NumPy random generator.
        initial_bitstring: optional computational-basis index to force into support.
        target_initial_overlap: desired squared overlap with initial_bitstring.
        support: optional explicit support.

    Returns:
        support: list of basis indices in support.
        amplitudes: normalized amplitudes on support.
        suggested_initial_bitstring: basis state with largest planted overlap.
        suggested_initial_overlap: squared overlap of that bitstring.
    """
    dim = 2**num_qubits

    if support is None:
        if initial_bitstring is not None:
            remaining = [x for x in range(dim) if x != initial_bitstring]
            extra = rng.choice(
                remaining,
                size=ground_support_size - 1,
                replace=False,
            ).tolist()
            support = [initial_bitstring] + extra
        else:
            support = rng.choice(
                dim,
                size=ground_support_size,
                replace=False,
            ).tolist()
    else:
        support = list(support)
        ground_support_size = len(support)

        if initial_bitstring is not None and initial_bitstring not in support:
            support[0] = initial_bitstring

    if target_initial_overlap is not None:
        if initial_bitstring is None:
            raise ValueError(
                "target_initial_overlap requires initial_bitstring to be set."
            )

        if not (0.0 <= target_initial_overlap <= 1.0):
            raise ValueError("target_initial_overlap must be between 0 and 1.")

        if initial_bitstring not in support:
            raise ValueError("initial_bitstring must be in support.")

        amplitudes = np.zeros(ground_support_size, dtype=complex)

        init_pos = support.index(initial_bitstring)

        # Set desired amplitude magnitude.
        phase = np.exp(1j * 2 * np.pi * rng.random())
        amplitudes[init_pos] = np.sqrt(target_initial_overlap) * phase

        # Randomly distribute remaining norm over the other support states.
        other_positions = [i for i in range(ground_support_size) if i != init_pos]

        if other_positions:
            rest = rng.normal(size=len(other_positions)) + 1j * rng.normal(
                size=len(other_positions)
            )
            rest = rest / np.linalg.norm(rest)
            rest *= np.sqrt(1.0 - target_initial_overlap)

            for pos, amp in zip(other_positions, rest):
                amplitudes[pos] = amp

    else:
        amplitudes = rng.normal(size=ground_support_size) + 1j * rng.normal(
            size=ground_support_size
        )
        amplitudes = amplitudes / np.linalg.norm(amplitudes)

    probs = np.abs(amplitudes) ** 2
    best_pos = int(np.argmax(probs))

    suggested_initial_bitstring = support[best_pos]
    suggested_initial_overlap = probs[best_pos]

    return (
        support,
        amplitudes,
        suggested_initial_bitstring,
        suggested_initial_overlap,
    )

def make_planted_hamiltonian(
    num_qubits: int,
    ground_support_size: int,
    pauli_density: float,
    Delta: float = 10.0,
    lam: float = 0.1,
    seed: int | None = None,
    support: list[int] | None = None,
    initial_bitstring: int | None = None,
    target_initial_overlap: float | None = None,
):
    """
    Build

        H = -Delta |g><g| + lam R,

    where |g> is sparse in the computational basis, and R has controllable
    Pauli density.

    Additional controls:
        initial_bitstring:
            Force this bitstring to have nonzero planted ground-state overlap.

        target_initial_overlap:
            Set |<initial_bitstring|g>|^2 to this value.
    """
    rng = np.random.default_rng(seed)

    (
        support,
        amplitudes,
        suggested_initial_bitstring,
        suggested_initial_overlap,
    ) = make_sparse_ground_state_with_initial_overlap(
        num_qubits=num_qubits,
        ground_support_size=ground_support_size,
        rng=rng,
        initial_bitstring=initial_bitstring,
        target_initial_overlap=target_initial_overlap,
        support=support,
    )

    P_g, g_dense = planted_sparse_projector(
        num_qubits=num_qubits,
        support=support,
        amplitudes=amplitudes,
    )

    R = random_pauli_background(
        num_qubits=num_qubits,
        pauli_density=pauli_density,
        rng=rng,
        offdiag_only=True,
        normalize=True,
    )

    H = (-Delta * P_g + lam * R).simplify(atol=1e-12)

    info = {
        "support": support,
        "amplitudes": amplitudes,
        "planted_ground_state": g_dense,
        "projector": P_g,
        "background": R,
        "num_pauli_terms_H": len(H.paulis),
        "num_pauli_terms_R": len(R.paulis),
        "Delta": Delta,
        "lambda": lam,
        "pauli_density": pauli_density,
        "suggested_initial_bitstring": suggested_initial_bitstring,
        "suggested_initial_bitstring_binary": format(
            suggested_initial_bitstring, f"0{num_qubits}b"
        ),
        "suggested_initial_overlap": suggested_initial_overlap,
    }

    return H, info


def diagnostics(H: SparsePauliOp, planted_g: np.ndarray):
    H_dense = H.to_matrix()
    evals, evecs = np.linalg.eigh(H_dense)

    gs = evecs[:, 0]
    E0 = evals[0]

    fidelity = abs(np.vdot(planted_g, gs)) ** 2
    ipr = np.sum(np.abs(gs) ** 4)
    effective_support = 1.0 / ipr

    return {
        "E0": E0,
        "fidelity_with_planted_state": fidelity,
        "IPR": ipr,
        "effective_support_size": effective_support,
    }