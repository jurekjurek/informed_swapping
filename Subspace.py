"""
Shared subspace machinery for the BARK and SKQD protocols.

Both protocols repeatedly diagonalize the Hamiltonian restricted to a pool of
basis states that only ever grows. The obvious implementation rebuilds that
block from scratch every iteration::

    projected = H[pool][:, pool].toarray()

which costs two scipy sparse fancy-index passes (the column one is internally a
sparse mat-mat product) plus a k x k dense allocation, *per iteration*. A run
that grows a pool to size P therefore spends O(P^3) memory traffic merely
assembling matrices whose entries were already known one iteration earlier.

``GrowingProjection`` keeps the block in a capacity-doubling buffer and writes
only the rows and columns a new batch of states actually adds, which brings the
cost of assembling the whole sequence down to O(P * nnz_per_row).

The pool is held in insertion order rather than sorted order. Reordering the
pool permutes the rows and columns of the projected block, which leaves its
spectrum -- and the fidelity computed from the resulting eigenvector -- exactly
unchanged, as long as the pool order and the block order agree. They do here by
construction.
"""

import numpy as np
from scipy.linalg import eigh
from scipy.sparse import csr_matrix, issparse


def lowest_eigenpair(block):
    """
    Lowest eigenvalue and eigenvector of a small dense Hermitian block.

    ``eigh(block)`` computes *all* k eigenvalues and all k eigenvectors and then
    the caller throws away all but one. Asking LAPACK for the single lowest pair
    via ``subset_by_index`` with the ``evr`` driver returns the identical
    eigenpair for a fraction of the work.
    """
    size = block.shape[0]
    if size < 3:
        # ``evr`` with a subset is not worth its setup cost on 1x1 and 2x2
        # blocks, and older LAPACK builds dislike the degenerate range.
        eigenvalues, eigenvectors = eigh(block)
        index = int(np.argmin(eigenvalues))
        return float(np.real(eigenvalues[index])), eigenvectors[:, index]

    eigenvalues, eigenvectors = eigh(block, check_finite=False,
                                     subset_by_index=[0, 0], driver="evr")
    return float(np.real(eigenvalues[0])), eigenvectors[:, 0]


class GrowingProjection:
    """
    The Hamiltonian projected onto a monotonically growing pool of basis states.

    Usage::

        projection = GrowingProjection(hamiltonian)
        projection.extend([initial_state_index])
        while ...:
            projection.extend(newly_sampled_states)   # duplicates are ignored
            value, vector = lowest_eigenpair(projection.block)

    ``block`` is a view of the internal buffer, valid until the next ``extend``.
    ``reset`` rewinds to an empty pool without giving up the allocation, so one
    object can serve thousands of runs.
    """

    def __init__(self, hamiltonian, initial_capacity: int = 64):
        if issparse(hamiltonian):
            matrix = hamiltonian.tocsr()
            if matrix is hamiltonian:
                # tocsr() is a no-op on a CSR input, and the canonicalization
                # below mutates in place -- copy so the caller's matrix is left
                # alone (two protocols share one Hamiltonian).
                matrix = matrix.copy()
        else:
            matrix = csr_matrix(np.asarray(hamiltonian))
        matrix.sum_duplicates()
        # Explicitly stored zeros would show up as extra "connected" basis states
        # when BARK reads a row, so they are dropped here as they were before.
        matrix.eliminate_zeros()

        self.matrix = matrix
        self.dimension = matrix.shape[0]
        self.dtype = matrix.dtype
        self.size = 0

        self._indptr = matrix.indptr
        self._indices = matrix.indices
        self._data = matrix.data

        capacity = int(min(max(initial_capacity, 1), self.dimension))
        self._block = np.zeros((capacity, capacity), dtype=self.dtype)
        self._pool = np.zeros(capacity, dtype=np.int64)
        # position[state] is the row of ``block`` holding that state, or -1.
        self._position = np.full(self.dimension, -1, dtype=np.int64)

    # -- buffer management --------------------------------------------------

    def _reserve(self, needed: int) -> None:
        capacity = self._block.shape[0]
        if needed <= capacity:
            return
        while capacity < needed:
            capacity *= 2
        capacity = int(min(capacity, self.dimension))

        block = np.zeros((capacity, capacity), dtype=self.dtype)
        block[: self.size, : self.size] = self._block[: self.size, : self.size]
        self._block = block

        pool = np.zeros(capacity, dtype=np.int64)
        pool[: self.size] = self._pool[: self.size]
        self._pool = pool

    def reset(self) -> None:
        """Rewind to an empty pool, keeping the allocated buffer."""
        if self.size:
            self._position[self._pool[: self.size]] = -1
        self.size = 0

    # -- growing ------------------------------------------------------------

    def extend(self, states) -> np.ndarray:
        """
        Add every state in ``states`` that is not in the pool yet.

        Returns the newly added states, in the order they were appended (empty
        if the batch contributed nothing, which is how both protocols detect a
        stagnating pool).
        """
        states = np.atleast_1d(np.asarray(states, dtype=np.int64)).ravel()
        if states.size == 0:
            return states[:0]

        # ``unique`` also de-duplicates within the batch, which matters because
        # a shot distribution routinely samples the same basis state twice.
        candidates = np.unique(states)
        fresh = candidates[self._position[candidates] < 0]
        if fresh.size == 0:
            return fresh

        start = self.size
        stop = start + fresh.size
        self._reserve(stop)

        self._pool[start:stop] = fresh
        self._position[fresh] = np.arange(start, stop, dtype=np.int64)
        self.size = stop

        block = self._block
        position = self._position

        # The buffer is reused across runs, so the new rows and columns must be
        # cleared before the sparse entries are scattered into them -- only the
        # non-zeros of H are written below, and anything left over from an
        # earlier run would silently survive as a bogus matrix element.
        block[start:stop, :stop] = 0
        block[:stop, start:stop] = 0

        for offset in range(fresh.size):
            state = int(fresh[offset])
            row = start + offset
            begin, end = self._indptr[state], self._indptr[state + 1]
            targets = position[self._indices[begin:end]]
            keep = targets >= 0
            targets = targets[keep]
            if targets.size == 0:
                continue
            values = self._data[begin:end][keep]
            block[row, targets] = values
            # H is Hermitian, so the mirrored entry is the conjugate. Writing it
            # here means a state added later in this same batch already finds its
            # column populated. (Two fresh states write each other's entry twice,
            # with identical values -- H[a,b] and conj(H[b,a]) agree.)
            block[targets, row] = np.conj(values)

        return fresh

    # -- views --------------------------------------------------------------

    @property
    def block(self) -> np.ndarray:
        """The projected Hamiltonian, in pool order. Invalidated by ``extend``."""
        return self._block[: self.size, : self.size]

    @property
    def pool(self) -> np.ndarray:
        """The pooled basis-state indices, in block order."""
        return self._pool[: self.size]

    def contains(self, state: int) -> bool:
        return bool(self._position[state] >= 0)
