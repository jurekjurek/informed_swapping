import numpy as np
import scipy.sparse as sp
from qiskit.quantum_info import Operator, SparsePauliOp


def _build_ground_state_amps(rng, k, max_amplitude, zero_tol=1e-12):
    """
    Build k complex amplitudes for the ground state, unit-normalized.

    If max_amplitude is None: complex-normal amplitudes, normalized.

    If max_amplitude = p is given, it is interpreted as a PROBABILITY:
    one chosen amplitude satisfies |c_i|^2 = p exactly (with a random phase),
    and the remaining k-1 amplitudes have random phases and magnitudes whose
    squared moduli sum to exactly 1 - p. The dominant amplitude is kept the
    strict maximum in modulus; |c_i|^2 = p is never altered.

    Feasibility: 1/k <= p <= 1, since the dominant probability cannot fall
    below the equal-weight value 1/k while remaining the largest.
    """
    if max_amplitude is None:
        amps = rng.standard_normal(k) + 1j * rng.standard_normal(k)
        amps /= np.linalg.norm(amps)
        return amps

    p = float(max_amplitude)
    if not (0.0 < p <= 1.0):
        raise ValueError("max_amplitude (a probability |c|^2) must be in (0, 1].")

    if k == 1:
        if abs(p - 1.0) > 1e-9:
            raise ValueError("With k=1 the only amplitude must have |c|^2 = 1.")
        return np.array([1.0 + 0j])

    lower = 1.0 / k
    if p < lower - 1e-12:
        raise ValueError(
            f"max_amplitude={p:.4g} is too small for k={k}; "
            f"need max_amplitude >= 1/k = {lower:.4g} to remain the maximum."
        )

    dom_mod = np.sqrt(p)

    # Random probabilities for the rest, summing to exactly (1 - p),
    # each capped below p so the dominant one stays the strict maximum.
    remaining = 1.0 - p
    # Dirichlet gives a random split of `remaining` over k-1 parts.
    probs = rng.dirichlet(np.ones(k - 1)) * remaining

    # Enforce each rest-probability < p (strict maximum). Clip and redistribute.
    # With p >= 1/k this is always feasible.
    for _ in range(64):
        over = probs >= p
        if not np.any(over):
            break
        excess = np.sum(probs[over] - (p * (1 - 1e-9)))
        probs[over] = p * (1 - 1e-9)
        under = ~over
        if np.any(under):
            probs[under] += excess * probs[under] / np.sum(probs[under])
        else:
            break

    rest_mod = np.sqrt(probs)
    rest_phase = np.exp(1j * 2 * np.pi * rng.random(k - 1))
    rest = rest_mod * rest_phase

    dom = dom_mod * np.exp(1j * 2 * np.pi * rng.random())

    amps = np.concatenate([[dom], rest])
    rng.shuffle(amps)
    return amps


def make_controlled_sparse_ground_state_hamiltonian_from_qubits(
    n_qubits: int,
    ground_state_sparsity: float | int,
    hamiltonian_sparsity: float,
    seed: int = 42,
    ground_energy: float = 0.0,
    gap: float = 1.0,
    random_strength: float = 1.0,
    return_info: bool = True,
    force_nondiagonal: bool = True,
    zero_tol: float = 1e-12,
    max_amplitude: float | None = None,
):
    """
    Returns
    -------
    H_csr : scipy.sparse.csr_matrix
        Hamiltonian in the computational basis.

    H_pauli : qiskit.quantum_info.SparsePauliOp
        Pauli-basis representation of the same Hamiltonian.

    psi : np.ndarray
        Exact ground-state vector.

    support : np.ndarray
        Computational-basis support of psi.

    info : dict, optional
        Diagnostics.

    Notes
    -----
    max_amplitude : float | None
        If given, the largest contributing basis-state amplitude has this
        modulus; the remaining amplitudes are random and the full vector is
        normalized to unit norm. Requires 1/sqrt(k) <= max_amplitude <= 1,
        where k is the ground-state support size.
    """

    if not isinstance(n_qubits, int) or n_qubits < 1:
        raise ValueError("n_qubits must be a positive integer.")

    if not (0.0 <= float(hamiltonian_sparsity) <= 1.0):
        raise ValueError("hamiltonian_sparsity must be in [0, 1].")

    if gap <= 0:
        raise ValueError("gap must be strictly positive.")

    if random_strength <= 0:
        raise ValueError("random_strength must be strictly positive.")

    rng = np.random.default_rng(seed)
    dim = 2**n_qubits

    # -------------------------------------------------------------------------
    # Interpret ground-state sparsity.
    # 0 -> one nonzero basis amplitude
    # 1 -> full computational-basis support
    # -------------------------------------------------------------------------
    if isinstance(ground_state_sparsity, (int, np.integer)) and not isinstance(
        ground_state_sparsity, bool
    ):
        k = int(ground_state_sparsity)
    else:
        s = float(ground_state_sparsity)
        if not (0.0 <= s <= 1.0):
            raise ValueError("Float ground_state_sparsity must be in [0, 1].")
        k = 1 + int(round(s * (dim - 1)))

    if not (1 <= k <= dim):
        raise ValueError(f"Ground-state support size must satisfy 1 <= k <= {dim}.")

    # -------------------------------------------------------------------------
    # Build ground state.
    # -------------------------------------------------------------------------
    support = np.sort(rng.choice(dim, size=k, replace=False))
    support_set = set(int(x) for x in support)
    complement = np.array([i for i in range(dim) if i not in support_set])

    amps = _build_ground_state_amps(rng, k, max_amplitude, zero_tol=zero_tol)

    psi = np.zeros(dim, dtype=np.complex128)
    psi[support] = amps

    # -------------------------------------------------------------------------
    # Build sparse annihilator A with A @ psi = 0.
    # Then K = A^† A is PSD and K @ psi = 0.
    # -------------------------------------------------------------------------
    row_indices = []
    col_indices = []
    data = []
    n_rows = 0

    def add_row(entries, weight=1.0):
        nonlocal n_rows

        if weight <= 0:
            return

        scale = np.sqrt(weight)
        cleaned = []

        for col, val in entries:
            if abs(val) > zero_tol:
                cleaned.append((int(col), complex(scale * val)))

        if len(cleaned) == 0:
            return

        for col, val in cleaned:
            row_indices.append(n_rows)
            col_indices.append(col)
            data.append(val)

        n_rows += 1

    def add_support_edge(a, b, weight=1.0):
        add_row(
            [
                (a, psi[b]),
                (b, -psi[a]),
            ],
            weight=weight,
        )

    def add_complement_singleton(c, weight=1.0):
        add_row([(c, 1.0)], weight=weight)

    def add_complement_edge(a, b, weight=1.0):
        z1 = rng.standard_normal() + 1j * rng.standard_normal()
        z2 = rng.standard_normal() + 1j * rng.standard_normal()
        add_row([(a, z1), (b, z2)], weight=weight)

    def add_cross_edge(s, c, anchor, weight=1.0):
        beta = rng.standard_normal() + 1j * rng.standard_normal()
        add_row(
            [
                (s, psi[anchor]),
                (anchor, -psi[s]),
                (c, beta),
            ],
            weight=weight,
        )

    def build_K():
        A = sp.csr_matrix(
            (
                np.asarray(data, dtype=np.complex128),
                (row_indices, col_indices),
            ),
            shape=(n_rows, dim),
            dtype=np.complex128,
        )
        K = A.getH() @ A
        K = 0.5 * (K + K.getH())
        K.eliminate_zeros()
        return K.tocsr()

    def count_offdiag_pairs(M):
        M = M.tocoo()
        pairs = set()
        for i, j, val in zip(M.row, M.col, M.data):
            if i != j and abs(val) > zero_tol:
                pairs.add(tuple(sorted((int(i), int(j)))))
        return len(pairs)

    # -------------------------------------------------------------------------
    # Minimal valid construction.
    # -------------------------------------------------------------------------
    if k >= 2:
        support_order = support.copy()
        rng.shuffle(support_order)

        for idx in range(1, k):
            a = int(support_order[idx - 1])
            b = int(support_order[idx])
            add_support_edge(a, b, weight=1.0)

    for c in complement:
        add_complement_singleton(int(c), weight=1.0)

    if force_nondiagonal and k == 1 and len(complement) >= 2:
        add_complement_edge(int(complement[0]), int(complement[1]), weight=1.0)

    K = build_K()
    min_pairs = count_offdiag_pairs(K)

    # -------------------------------------------------------------------------
    # Target off-diagonal pair count.
    # -------------------------------------------------------------------------
    n_support_pairs = k * (k - 1) // 2
    n_complement_pairs = len(complement) * (len(complement) - 1) // 2
    n_cross_pairs = k * len(complement) if k >= 2 else 0

    max_pairs = n_support_pairs + n_complement_pairs + n_cross_pairs

    target_pairs = min_pairs + int(
        round(float(hamiltonian_sparsity) * max(0, max_pairs - min_pairs))
    )

    # -------------------------------------------------------------------------
    # Add optional sparsity-pattern elements.
    # -------------------------------------------------------------------------
    candidates = []

    for ii in range(k):
        for jj in range(ii + 1, k):
            candidates.append(("support", int(support[ii]), int(support[jj])))

    for ii in range(len(complement)):
        for jj in range(ii + 1, len(complement)):
            candidates.append(("complement", int(complement[ii]), int(complement[jj])))

    if k >= 2:
        anchor0 = int(support[0])
        anchor1 = int(support[1])

        for c in complement:
            c = int(c)
            for s_idx in support:
                s_idx = int(s_idx)
                anchor = anchor0 if s_idx != anchor0 else anchor1
                candidates.append(("cross", s_idx, c, anchor))

    rng.shuffle(candidates)

    current_pairs = min_pairs

    for candidate in candidates:
        if current_pairs >= target_pairs:
            break

        weight = random_strength * (0.5 + rng.random())
        kind = candidate[0]

        if kind == "support":
            _, a, b = candidate
            add_support_edge(a, b, weight=weight)

        elif kind == "complement":
            _, a, b = candidate
            add_complement_edge(a, b, weight=weight)

        elif kind == "cross":
            _, s_idx, c, anchor = candidate
            add_cross_edge(s_idx, c, anchor, weight=weight)

        K = build_K()
        current_pairs = count_offdiag_pairs(K)

    # -------------------------------------------------------------------------
    # Scale PSD term to enforce the requested spectral gap.
    # -------------------------------------------------------------------------
    K_dense = K.toarray()
    K_dense = 0.5 * (K_dense + K_dense.conj().T)

    evals_K = np.linalg.eigvalsh(K_dense)
    positive_evals = evals_K[evals_K > 100 * zero_tol]

    if len(positive_evals) == 0:
        raise RuntimeError("Construction failed: PSD part has no positive spectrum.")

    min_positive = float(np.min(positive_evals))
    scale = gap / min_positive

    H_dense = ground_energy * np.eye(dim, dtype=np.complex128) + scale * K_dense
    H_dense = 0.5 * (H_dense + H_dense.conj().T)
    H_dense[np.abs(H_dense) < zero_tol] = 0.0

    # -------------------------------------------------------------------------
    # Required output 1: scipy.sparse._csr.csr_matrix
    # -------------------------------------------------------------------------
    H_csr = sp.csr_matrix(H_dense)
    H_csr.eliminate_zeros()

    # -------------------------------------------------------------------------
    # Required output 2: Qiskit SparsePauliOp
    # -------------------------------------------------------------------------
    H_pauli = SparsePauliOp.from_operator(Operator(H_dense))

    # Optional simplification: removes tiny Pauli coefficients from numerical
    # conversion noise.
    H_pauli = H_pauli.simplify(atol=zero_tol)

    # -------------------------------------------------------------------------
    # Diagnostics.
    # -------------------------------------------------------------------------
    ham_nnz = int(np.count_nonzero(np.abs(H_dense) > zero_tol))
    gs_nnz = int(np.count_nonzero(np.abs(psi) > zero_tol))

    offdiag = H_dense - np.diag(np.diag(H_dense))
    offdiag_nnz = int(np.count_nonzero(np.abs(offdiag) > zero_tol))

    evals = np.linalg.eigvalsh(H_dense)
    actual_gap = float(evals[1] - evals[0]) if dim > 1 else np.inf

    gs_abs = np.abs(psi[support])
    info = {
        "n_qubits": n_qubits,
        "dim": dim,
        "ground_state_support_size": gs_nnz,
        "ground_state_density": gs_nnz / dim,
        "ground_state_control_sparsity": (gs_nnz - 1) / (dim - 1),
        "ground_state_max_amplitude": float(np.max(gs_abs)) if gs_nnz else 0.0,
        "requested_max_amplitude": max_amplitude,
        "hamiltonian_nnz": ham_nnz,
        "hamiltonian_density": ham_nnz / (dim * dim),
        "hamiltonian_offdiag_nnz": offdiag_nnz,
        "hamiltonian_offdiag_pairs": offdiag_nnz // 2,
        "target_offdiag_pairs": target_pairs,
        "min_offdiag_pairs": min_pairs,
        "max_offdiag_pairs_possible": max_pairs,
        "num_pauli_terms": len(H_pauli),
        "ground_energy": ground_energy,
        "requested_gap": gap,
        "actual_gap": actual_gap,
        "is_nondiagonal": offdiag_nnz > 0,
        "csr_type": type(H_csr),
        "pauli_type": type(H_pauli),
    }

    if return_info:
        return H_csr, H_pauli, psi, support, info

    return H_csr, H_pauli, psi, support


# Faster Alternative

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh


def make_controlled_sparse_ground_state_hamiltonian_fast(
    n_qubits: int,
    ground_state_sparsity: float | int,
    hamiltonian_sparsity: float,
    seed: int = 42,
    ground_energy: float = 0.0,
    gap: float = 1.0,
    random_strength: float = 1.0,
    force_nondiagonal: bool = True,
    zero_tol: float = 1e-12,
    dense_eig_threshold: int = 1024,
    make_pauli_op: bool = True,
    pauli_chop_tol: float = 1e-12,
    return_info: bool = True,
    max_amplitude: float | None = None,
):
    """
    Efficient construction of a computational-basis sparse Hamiltonian with a
    controlled-sparsity exact ground state.

    Returns
    -------
    H_csr : scipy.sparse.csr_matrix
        Computational-basis Hamiltonian.

    H_pauli : qiskit.quantum_info.SparsePauliOp | None
        Pauli representation of the same Hamiltonian. Returned as None if
        make_pauli_op=False.

    psi : np.ndarray
        Exact ground-state vector.

    support : np.ndarray
        Computational-basis support of psi.

    info : dict, optional
        Diagnostics.

    Notes
    -----
    max_amplitude : float | None
        If given, the largest contributing basis-state amplitude has this
        modulus; the remaining amplitudes are random and the full vector is
        normalized to unit norm. Requires 1/sqrt(k) <= max_amplitude <= 1,
        where k is the ground-state support size.
    """

    if not isinstance(n_qubits, int) or n_qubits < 1:
        raise ValueError("n_qubits must be a positive integer.")

    hamiltonian_sparsity = float(hamiltonian_sparsity)
    if not (0.0 <= hamiltonian_sparsity <= 1.0):
        raise ValueError("hamiltonian_sparsity must be in [0, 1].")

    if gap <= 0:
        raise ValueError("gap must be strictly positive.")

    if random_strength <= 0:
        raise ValueError("random_strength must be strictly positive.")

    rng = np.random.default_rng(seed)
    dim = 2**n_qubits

    # ---------------------------------------------------------------------
    # Interpret ground-state sparsity.
    #
    # Float convention:
    #   0 -> one nonzero computational-basis amplitude
    #   1 -> full computational-basis support
    #
    # Integer convention:
    #   exact number of nonzero amplitudes.
    # ---------------------------------------------------------------------
    if isinstance(ground_state_sparsity, (int, np.integer)) and not isinstance(
        ground_state_sparsity, bool
    ):
        k = int(ground_state_sparsity)
    else:
        s = float(ground_state_sparsity)
        if not (0.0 <= s <= 1.0):
            raise ValueError("Float ground_state_sparsity must be in [0, 1].")
        k = 1 + int(round(s * (dim - 1)))

    if not (1 <= k <= dim):
        raise ValueError(f"Ground-state support size must satisfy 1 <= k <= {dim}.")

    # ---------------------------------------------------------------------
    # Construct sparse ground state psi.
    # ---------------------------------------------------------------------
    support = np.sort(rng.choice(dim, size=k, replace=False))

    in_support = np.zeros(dim, dtype=bool)
    in_support[support] = True
    complement = np.flatnonzero(~in_support)

    amps = _build_ground_state_amps(rng, k, max_amplitude, zero_tol=zero_tol)

    psi = np.zeros(dim, dtype=np.complex128)
    psi[support] = amps

    # ---------------------------------------------------------------------
    # Build sparse annihilator A such that A @ psi = 0.
    #
    # Then K = A^† A is PSD and satisfies K @ psi = 0.
    # Finally H = ground_energy * I + scale * K.
    # ---------------------------------------------------------------------
    row_indices: list[int] = []
    col_indices: list[int] = []
    data: list[complex] = []
    row_id = 0

    selected_pairs: set[tuple[int, int]] = set()

    def mark_pair(a: int, b: int):
        if a != b:
            selected_pairs.add(tuple(sorted((int(a), int(b)))))

    def add_row(entries, weight: float = 1.0):
        nonlocal row_id

        if weight <= 0:
            return

        scale = np.sqrt(weight)
        added = False

        for col, value in entries:
            value = scale * complex(value)
            if abs(value) > zero_tol:
                row_indices.append(row_id)
                col_indices.append(int(col))
                data.append(value)
                added = True

        if added:
            row_id += 1

    def add_support_edge(a: int, b: int, weight: float = 1.0):
        """
        Row proportional to:
            psi[b] |a> - psi[a] |b>

        This annihilates psi exactly.
        """
        add_row(
            [
                (a, psi[b]),
                (b, -psi[a]),
            ],
            weight=weight,
        )
        mark_pair(a, b)

    def add_complement_singleton(c: int, weight: float = 1.0):
        """
        Since psi[c] = 0 outside the support, this row annihilates psi.
        """
        add_row([(c, 1.0)], weight=weight)

    def add_complement_edge(a: int, b: int, weight: float = 1.0):
        """
        Adds non-diagonal structure entirely inside the excited subspace.
        """
        z1 = rng.standard_normal() + 1j * rng.standard_normal()
        z2 = rng.standard_normal() + 1j * rng.standard_normal()

        add_row(
            [
                (a, z1),
                (b, z2),
            ],
            weight=weight,
        )
        mark_pair(a, b)

    def add_cross_edge(s_idx: int, c: int, anchor: int, weight: float = 1.0):
        """
        Adds support-complement coupling while preserving A @ psi = 0.

        Requires at least two support indices.

        Row:
            psi[anchor] |s_idx>
          - psi[s_idx] |anchor>
          + beta |c>
        """
        beta = rng.standard_normal() + 1j * rng.standard_normal()

        add_row(
            [
                (s_idx, psi[anchor]),
                (anchor, -psi[s_idx]),
                (c, beta),
            ],
            weight=weight,
        )

        mark_pair(s_idx, c)
        mark_pair(anchor, c)
        mark_pair(s_idx, anchor)

    # ---------------------------------------------------------------------
    # Minimal valid construction.
    # ---------------------------------------------------------------------

    # Make psi the unique zero mode inside its support using a random tree.
    if k >= 2:
        support_order = support.copy()
        rng.shuffle(support_order)

        for idx in range(1, k):
            a = int(support_order[idx - 1])
            b = int(support_order[idx])
            add_support_edge(a, b, weight=1.0)

    # Penalize every computational-basis state outside the support.
    for c in complement:
        add_complement_singleton(int(c), weight=1.0)

    # If psi is a single basis state, it cannot be coupled to the rest while
    # remaining an exact eigenstate. But the excited subspace can be made
    # non-diagonal if it has dimension at least 2.
    if force_nondiagonal and k == 1 and len(complement) >= 2:
        add_complement_edge(int(complement[0]), int(complement[1]), weight=1.0)

    min_offdiag_pairs = len(selected_pairs)

    # ---------------------------------------------------------------------
    # Candidate rows for increasing Hamiltonian sparsity.
    #
    # Unlike the previous version, we do not rebuild K after every candidate.
    # We choose a fraction of all available candidates, build A once, then
    # build K once.
    # ---------------------------------------------------------------------
    candidates = []

    # Support-support edges.
    for ii in range(k):
        for jj in range(ii + 1, k):
            a = int(support[ii])
            b = int(support[jj])
            pair = tuple(sorted((a, b)))
            if pair not in selected_pairs:
                candidates.append(("support", a, b))

    # Complement-complement edges.
    for ii in range(len(complement)):
        for jj in range(ii + 1, len(complement)):
            a = int(complement[ii])
            b = int(complement[jj])
            pair = tuple(sorted((a, b)))
            if pair not in selected_pairs:
                candidates.append(("complement", a, b))

    # Support-complement edges.
    if k >= 2:
        anchor0 = int(support[0])
        anchor1 = int(support[1])

        for c in complement:
            c = int(c)

            for s_idx in support:
                s_idx = int(s_idx)
                anchor = anchor0 if s_idx != anchor0 else anchor1
                candidates.append(("cross", s_idx, c, anchor))

    rng.shuffle(candidates)

    n_select = int(round(hamiltonian_sparsity * len(candidates)))
    selected_candidates = candidates[:n_select]

    for candidate in selected_candidates:
        kind = candidate[0]
        weight = random_strength * (0.5 + rng.random())

        if kind == "support":
            _, a, b = candidate
            add_support_edge(a, b, weight=weight)

        elif kind == "complement":
            _, a, b = candidate
            add_complement_edge(a, b, weight=weight)

        elif kind == "cross":
            _, s_idx, c, anchor = candidate
            add_cross_edge(s_idx, c, anchor, weight=weight)

    max_candidate_rows = len(candidates)

    # ---------------------------------------------------------------------
    # Build K = A^† A once.
    # ---------------------------------------------------------------------
    A = sp.csr_matrix(
        (
            np.asarray(data, dtype=np.complex128),
            (row_indices, col_indices),
        ),
        shape=(row_id, dim),
        dtype=np.complex128,
    )

    K = A.getH() @ A
    K = 0.5 * (K + K.getH())
    K.eliminate_zeros()
    K = K.tocsr()

    # ---------------------------------------------------------------------
    # Scale K so that the first excited energy is ground_energy + gap.
    # ---------------------------------------------------------------------
    if dim <= dense_eig_threshold:
        K_dense = K.toarray()
        K_dense = 0.5 * (K_dense + K_dense.conj().T)
        evals_K = np.linalg.eigvalsh(K_dense)
    else:
        # Sparse low-spectrum calculation.
        # k=2 should return the zero mode and the first positive mode.
        try:
            evals_K = eigsh(
                K,
                k=2,
                which="SA",
                return_eigenvectors=False,
                tol=1e-9,
                maxiter=20_000,
            )
            evals_K = np.sort(np.real(evals_K))
        except Exception:
            # Conservative fallback. This may be expensive for large dim.
            K_dense = K.toarray()
            K_dense = 0.5 * (K_dense + K_dense.conj().T)
            evals_K = np.linalg.eigvalsh(K_dense)

    positive_evals = evals_K[evals_K > 100 * zero_tol]

    # If sparse eigsh only found the null space numerically, ask for more.
    if len(positive_evals) == 0 and dim > 4:
        evals_K = eigsh(
            K,
            k=min(8, dim - 1),
            which="SA",
            return_eigenvectors=False,
            tol=1e-9,
            maxiter=20_000,
        )
        evals_K = np.sort(np.real(evals_K))
        positive_evals = evals_K[evals_K > 100 * zero_tol]

    if len(positive_evals) == 0:
        raise RuntimeError("Could not find a positive eigenvalue for gap scaling.")

    min_positive = float(np.min(positive_evals))
    scale = gap / min_positive

    H_csr = (scale * K).tocsr()

    if ground_energy != 0.0:
        H_csr = H_csr + ground_energy * sp.identity(
            dim,
            format="csr",
            dtype=np.complex128,
        )

    H_csr.data[np.abs(H_csr.data) < zero_tol] = 0.0
    H_csr.eliminate_zeros()

    # ---------------------------------------------------------------------
    # Convert to Qiskit SparsePauliOp once.
    #
    # This is usually the remaining expensive step.
    # ---------------------------------------------------------------------
    H_pauli = None

    if make_pauli_op:
        from qiskit.quantum_info import Operator, SparsePauliOp

        H_pauli = SparsePauliOp.from_operator(Operator(H_csr.toarray()))
        H_pauli = H_pauli.simplify(atol=pauli_chop_tol)

    # ---------------------------------------------------------------------
    # Diagnostics.
    # ---------------------------------------------------------------------
    coo = H_csr.tocoo()
    offdiag_mask = coo.row != coo.col
    offdiag_nnz = int(np.count_nonzero(offdiag_mask))

    gs_nnz = int(np.count_nonzero(np.abs(psi) > zero_tol))
    gs_abs = np.abs(psi[support])

    info = {
        "n_qubits": n_qubits,
        "dim": dim,
        "ground_state_support_size": gs_nnz,
        "ground_state_density": gs_nnz / dim,
        "ground_state_control_sparsity": (gs_nnz - 1) / (dim - 1),
        "ground_state_max_amplitude": float(np.max(gs_abs)) if gs_nnz else 0.0,
        "requested_max_amplitude": max_amplitude,
        "hamiltonian_nnz": int(H_csr.nnz),
        "hamiltonian_density": H_csr.nnz / (dim * dim),
        "hamiltonian_offdiag_nnz": offdiag_nnz,
        "hamiltonian_offdiag_pairs_lower_bound": min_offdiag_pairs,
        "num_annihilator_rows": row_id,
        "num_selected_candidate_rows": n_select,
        "num_available_candidate_rows": max_candidate_rows,
        "ground_energy": ground_energy,
        "requested_gap": gap,
        "gap_scaling_factor": scale,
        "is_nondiagonal": offdiag_nnz > 0,
        "csr_type": type(H_csr),
        "pauli_type": type(H_pauli) if H_pauli is not None else None,
        "num_pauli_terms": len(H_pauli) if H_pauli is not None else None,
    }

    if force_nondiagonal and offdiag_nnz == 0:
        info["warning"] = (
            "Could not make H non-diagonal. This occurs for the one-qubit case "
            "with a one-basis-state ground state, where there is no excited "
            "subspace large enough to couple internally."
        )

    if return_info:
        return H_csr, H_pauli, psi, support, info

    return H_csr, H_pauli, psi, support