import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp

from scipy.sparse.linalg import eigsh, expm_multiply, LinearOperator
from scipy.linalg import eigh, eigh_tridiagonal
from itertools import combinations
from scipy.sparse import csr_matrix, diags

def make_unitary_operator(H: sp.csr_matrix, dt: float) -> LinearOperator:
    """
    Return a matrix-free operator U = exp(-i H dt), applied through expm_multiply.
    This avoids constructing the full exponential explicitly.
    """
    n = H.shape[0]
    A = (-1j * dt) * H

    def matvec(x: np.ndarray) -> np.ndarray:
        return expm_multiply(A, x)

    return LinearOperator((n, n), matvec=matvec, dtype=np.complex128)

def make_hermitian_sparse_random(size: int, density: float, seed: int = 42) -> sp.csr_matrix:
    rng = np.random.default_rng(seed)
    H = sp.random(size, size, density=density, format="csr", dtype=np.complex128, random_state=rng)
    H = H + H.conj().T
    return H.tocsr()


def make_random_state(size: int, seed: int = 123) -> np.ndarray:
    rng = np.random.default_rng(seed)
    psi = rng.random(size) + 1j * rng.random(size)
    psi = psi.astype(np.complex128)
    psi /= np.linalg.norm(psi)
    return psi


def zero_charge_basis(N: int):
    """
    Basis of the zero-charge sector:
    all bitstrings of length N with exactly N/2 ones.
    """
    if N % 2 != 0:
        raise ValueError("Zero-charge sector with equal numbers of 0 and 1 requires even N.")

    basis = []
    for occ in combinations(range(N), N // 2):
        s = 0
        for i in occ:
            s |= (1 << i)
        basis.append(s)

    index = {s: i for i, s in enumerate(basis)}
    return basis, index


def state_to_z(state: int, N: int):
    """
    Convert integer bitstring to Z eigenvalues.
    bit 0 -> |0> with Z=+1
    bit 1 -> |1> with Z=-1
    """
    bits = np.array([(state >> n) & 1 for n in range(N)], dtype=np.int8)
    z = 1 - 2 * bits
    return z


def build_kinetic_term(N: int, x: float, basis=None, index=None) -> csr_matrix:
    """
    Build the kinetic term in the zero-charge sector:

        W_kin = (x/2) sum_{n=0}^{N-2} (X_n X_{n+1} + Y_n Y_{n+1})

    In the computational basis:
        (X_i X_j + Y_i Y_j) swaps 01 <-> 10 with matrix element 2,
    so the prefactor x/2 gives amplitude x per allowed nearest-neighbor swap.
    """
    if basis is None or index is None:
        basis, index = zero_charge_basis(N)

    rows = []
    cols = []
    data = []

    for col, s in enumerate(basis):
        for n in range(N - 1):
            b_n = (s >> n) & 1
            b_np1 = (s >> (n + 1)) & 1

            if b_n != b_np1:
                flipped = s ^ ((1 << n) | (1 << (n + 1)))
                row = index[flipped]
                rows.append(row)
                cols.append(col)
                data.append(x)

    dim = len(basis)
    return csr_matrix((data, (rows, cols)), shape=(dim, dim))


def diagonal_energy(state: int, N: int, x: float, m_lat: float, g: float, l0: float) -> float:
    """
    Diagonal part of the Hamiltonian in the restricted sector, with lambda dropped.
    """
    z = state_to_z(state, N).astype(np.int64)

    mass = (m_lat / g) * np.sqrt(x) * np.sum(((-1) ** np.arange(N)) * z)

    zz = 0.0
    prefix = z[0]
    for k in range(1, N):
        zz += 0.5 * (N - k - 1) * z[k] * prefix
        prefix += z[k]

    linear = 0.0
    for n in range(N - 1):
        coeff = N / 4.0 - 0.5 * ((n + 1) // 2) + l0 * (N - n - 1)
        linear += coeff * z[n]

    const = l0**2 * (N - 1) + 0.5 * l0 * N + 0.125 * N**2

    return mass + zz + linear + const


def build_full_hamiltonian(
    N: int,
    x: float,
    m_lat: float,
    g: float,
    l0: float,
    basis=None,
    index=None,
):
    """
    Build both:
      - kinetic term as csr_matrix
      - full Hamiltonian as csr_matrix
    """
    if basis is None or index is None:
        basis, index = zero_charge_basis(N)

    H_kin = build_kinetic_term(N, x, basis=basis, index=index)

    diag = np.array(
        [diagonal_energy(s, N, x, m_lat, g, l0) for s in basis],
        dtype=np.float64
    )
    H_diag = diags(diag, offsets=0, format="csr")

    H_full = H_kin + H_diag
    return H_kin, H_full, basis, index

def make_cooked_unitary_operator(H: sp.csr_matrix, dt: float) -> LinearOperator:
    """
    Return a matrix-free operator U = exp(-i H dt), applied through expm_multiply.
    This version is "cooked" to be faster by avoiding some overhead, at the cost of
    being less general (e.g. it assumes the input vector is dense and of the correct size).
    """
    n = H.shape[0]
    A = (-1j * dt) * (-H + 1/2* H@H)

    def matvec(x: np.ndarray) -> np.ndarray:
        # Directly call expm_multiply without the LinearOperator overhead
        return expm_multiply(A, x)

    return LinearOperator((n, n), matvec=matvec, dtype=np.complex128)


def lanczos_ground_energy(
    H: sp.csr_matrix,
    initial_state: np.ndarray,
    max_iter: int = 50,
    tol: float = 1e-12,
    reorthogonalize: bool = True,
    eig_eval_every: int = 1,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Lanczos on Hermitian H.

    Returns
    -------
    energies : np.ndarray
        Ritz ground energy estimates.
    eval_iters : np.ndarray
        Iterations at which energies were evaluated.
    """
    q = np.array(initial_state, dtype=np.complex128, copy=True)
    q /= np.linalg.norm(q)

    n = H.shape[0]
    Q = np.zeros((n, max_iter + 1), dtype=np.complex128)
    alpha = np.zeros(max_iter, dtype=np.float64)
    beta = np.zeros(max_iter, dtype=np.float64)

    Q[:, 0] = q
    q_prev = np.zeros(n, dtype=np.complex128)
    b_prev = 0.0

    energies = []
    eval_iters = []

    for j in range(max_iter):
        z = H @ Q[:, j]
        if j > 0:
            z -= b_prev * q_prev

        a = np.vdot(Q[:, j], z).real
        z -= a * Q[:, j]

        if reorthogonalize:
            for k in range(j + 1):
                coeff = np.vdot(Q[:, k], z)
                z -= coeff * Q[:, k]

        b = np.linalg.norm(z)

        alpha[j] = a
        if j < max_iter - 1:
            beta[j] = b

        do_eval = (j == 0) or ((j + 1) % eig_eval_every == 0) or (b < tol) or (j == max_iter - 1)
        if do_eval:
            if j == 0:
                e0 = alpha[0]
            else:
                e0 = eigh_tridiagonal(
                    alpha[: j + 1],
                    beta[:j],
                    select="i",
                    select_range=(0, 0),
                )[0][0]

            energies.append(float(e0))
            eval_iters.append(j + 1)

            if verbose:
                print(f"Lanczos iter {j + 1:3d}: E = {e0:.15e}, beta = {b:.3e}")

        if b < tol or j == max_iter - 1:
            break

        q_prev = Q[:, j].copy()
        Q[:, j + 1] = z / b
        b_prev = b

    return np.array(energies, dtype=float), np.array(eval_iters, dtype=int)



def arnoldi_ground_energy(
    H: sp.csr_matrix,
    U: LinearOperator,
    initial_state: np.ndarray,
    max_iter: int = 50,
    tol: float = 1e-12,
    eig_eval_every: int = 1,
    orth_tol: float = 1e-8,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Arnoldi on a general operator U, with Ritz energies extracted from the
    projected Hamiltonian Q^* H Q.

    Additional diagnostics:
    - beta detects near-dependence of the new vector on the current subspace
    - Gram-matrix check detects loss of orthogonality among accepted basis vectors
    """
    q0 = np.array(initial_state, dtype=np.complex128, copy=True)
    q0 /= np.linalg.norm(q0)

    n = H.shape[0]
    Q = np.zeros((n, max_iter + 1), dtype=np.complex128)
    HQ = np.zeros((n, max_iter), dtype=np.complex128)
    Hproj = np.zeros((max_iter, max_iter), dtype=np.complex128)

    Q[:, 0] = q0

    energies = []
    eval_iters = []

    for j in range(max_iter):
        v = U @ Q[:, j]

        # Full Arnoldi orthogonalization
        for i in range(j + 1):
            hij = np.vdot(Q[:, i], v)
            v -= hij * Q[:, i]

        beta = np.linalg.norm(v)

        # Cache H @ q_j once
        HQ[:, j] = H @ Q[:, j]

        # Incrementally update the small projected Hamiltonian
        for i in range(j + 1):
            val = np.vdot(Q[:, i], HQ[:, j])
            Hproj[i, j] = val
            Hproj[j, i] = np.conj(val)

        # Check orthogonality of existing basis vectors
        G = Q[:, :j+1].conj().T @ Q[:, :j+1]
        offdiag = G - np.eye(j + 1, dtype=np.complex128)
        max_offdiag = np.max(np.abs(offdiag))

        do_eval = (
            (j == 0)
            or ((j + 1) % eig_eval_every == 0)
            or (beta < tol)
            or (j == max_iter - 1)
            or (max_offdiag > orth_tol)
        )

        if do_eval:
            small = Hproj[: j + 1, : j + 1]
            small = 0.5 * (small + small.conj().T)
            e0 = eigh(small, eigvals_only=True)[0].real

            energies.append(float(e0))
            eval_iters.append(j + 1)

            if verbose:
                print(
                    f"Arnoldi iter {j + 1:3d}: "
                    f"E = {e0:.15e}, beta = {beta:.3e}, "
                    f"max|Q*Q-I| = {max_offdiag:.3e}"
                )

        if max_offdiag > orth_tol and verbose:
            print(
                f"Warning: loss of orthogonality detected at iter {j+1}, "
                f"max|q_i^* q_k - delta_ik| = {max_offdiag:.3e}"
            )
            break

        if beta < tol or j == max_iter - 1:
            break

        Q[:, j + 1] = v / beta

    return np.array(energies, dtype=float), np.array(eval_iters, dtype=int)


def compare_methods(
    H: sp.csr_matrix,
    initial_state: np.ndarray,
    time_steps: list[float],
    max_iter: int = 50,
    tol: float = 1e-12,
    eig_eval_every_lanczos: int = 1,
    eig_eval_every_arnoldi: int = 1,
    verbose: bool = True,
):
    """
    Run Lanczos on H and Arnoldi on U(dt)=exp(-i H dt) for several dt values.
    """
    exact_energy, exact_groundstate = eigsh(H, k=1, which="SA", tol=1e-10)
    exact_energy = exact_energy[0].real
    print(f"Exact ground state energy: {exact_energy:.15e}")
    #Overlap between initial state and exact ground state can affect convergence, so we print it for reference.
    overlap = np.abs(np.vdot(exact_groundstate[:, 0], initial_state))
    print(f"Overlap of initial state with exact ground state: {overlap:.3e}")
    # Sparsity of ground state
    sparsity = np.count_nonzero(exact_groundstate[:, 0]) / len(exact_groundstate)
    print(f"Sparsity of exact ground state: {sparsity:.3e}")

    lanczos_energies, lanczos_iters = lanczos_ground_energy(
        H,
        initial_state,
        max_iter=max_iter,
        tol=tol,
        reorthogonalize=True,
        eig_eval_every=eig_eval_every_lanczos,
        verbose=verbose,
    )

    arnoldi_results = []
    cooked_results = []
    for dt in time_steps:
        print(f"\n--- Arnoldi with dt = {dt} ---")
        U = make_unitary_operator(H, dt)
        U_cooked = make_cooked_unitary_operator(H, dt)
        energies, iters = arnoldi_ground_energy(
            H,
            U,
            initial_state,
            max_iter=max_iter,
            tol=tol,
            eig_eval_every=eig_eval_every_arnoldi,
            verbose=verbose,
        )
        arnoldi_results.append((dt, energies, iters))
        energies_cooked, iters_cooked = arnoldi_ground_energy(
            H,
            U_cooked,
            initial_state,
            max_iter=max_iter,
            tol=tol,
            eig_eval_every=eig_eval_every_arnoldi,
            verbose=verbose,
        )
        cooked_results.append((dt, energies_cooked, iters_cooked))

    return exact_energy, (lanczos_energies, lanczos_iters), arnoldi_results, cooked_results


def plot_results(
    exact_energy: float,
    lanczos_result,
    arnoldi_results,
    cooked_results,

    title: str = "Arnoldi vs Lanczos convergence",
):
    lanczos_energies, lanczos_iters = lanczos_result

    plt.figure(figsize=(9, 6))

    lanczos_err = np.abs(lanczos_energies - exact_energy)
    lanczos_err = np.maximum(lanczos_err, 1e-16)
    plt.plot(lanczos_iters, lanczos_err, marker="v", label="Lanczos on H", alpha=len(lanczos_iters)/max_iter)

    for dt, energies, iters in arnoldi_results:
        err = np.abs(energies - exact_energy)
        err = np.maximum(err, 1e-16)
        plt.plot(iters, err, marker="o", label=f"Arnoldi on exp(-iHt), dt={dt}", alpha=len(iters)/max_iter)

    for dt, energies_cooked, iters_cooked in cooked_results:
        err_cooked = np.abs(energies_cooked - exact_energy)
        err_cooked = np.maximum(err_cooked, 1e-16)
        plt.plot(iters_cooked, err_cooked, marker="s", label=f"Arnoldi on exp(-it*(-H + 1/2* H^2)), dt={dt}", alpha=len(iters_cooked)/max_iter)

    plt.yscale("log")
    # plt.xscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("Absolute energy error")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    np.random.seed(42)

    max_iter = 100
    tol = 1e-12
    time_steps = [0.1, 0.2, 0.3, 0.4]

    _, H, _, _ = build_full_hamiltonian(
        N=12,
        x=1.0,
        m_lat=10,
        g=1.0,
        l0=0.0,
    )
    initial_state = make_random_state(size=H.shape[0], seed=123)

    exact_energy, lanczos_result, arnoldi_results, cooked_results  = compare_methods(
        H,
        initial_state,
        time_steps=time_steps,
        max_iter=max_iter,
        tol=tol,
        eig_eval_every_lanczos=1,
        eig_eval_every_arnoldi=1,
        verbose=True,
    )

    plot_results(
        exact_energy,
        lanczos_result,
        arnoldi_results,
        cooked_results,
        title="Ground-state convergence: Lanczos vs Arnoldi",
    )