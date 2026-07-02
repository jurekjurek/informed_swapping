# subspace_search.algorithms

Classical, subspace-selecting ground-state search algorithms — the candidates
that are pitted against SKQD. This is where new algorithms (the next BARK/KRAB)
should live.

## What's here

| Algorithm | Entry point | Idea |
|-----------|-------------|------|
| **KRAB** | `krab.py` → `selected_krylov_ground_state` | *Selected-subspace Krylov.* Repeatedly applies `H` to the current sparse state, keeps up to `Q` new candidate basis states (scored by residual / PT2), diagonalises in the growing subspace, then prunes small coefficients. Returns a `SelectedKrylovResult` with full per-iteration diagnostics. |
| **BARK** | `bark.py` → `BarkBarkBark` | *Bitstring Algorithm for Recursive Krylov.* Propagates the state as a set of computational-basis bitstrings via Pauli-string application — a classical analogue of SKQD that never forms a dense state vector. |

### KRAB — key parameters

`selected_krylov_ground_state(H, initial_bitstring, Q, delta, epsilon, n_iterations, ...)`

- `Q` — new basis states kept per application. Accepts an `int`, a
  `list`/`tuple` (per-application), or a **callable** `app -> int` (e.g. a
  decaying schedule). See `resolve_Q`.
- `delta` — candidate residual threshold (keep candidates with `|r_a| >= delta`).
- `epsilon` — coefficient-pruning threshold after diagonalisation (keep states
  with `|c_b|² >= epsilon`).
- `score_mode` — `"residual"` (default) or `"pt2"`.
- `max_subspace_dim` — optional hard cap on the active subspace.

The result exposes `final_energy`, `final_indices` (basis states in discovery
order — feed this to `subspace_search.paths.get_one_path`), and
`iteration_diagnostics` (energy, subspace size, residual norm, participation
ratio, ... per iteration). `plot_selected_krylov_diagnostics(result)` renders it.

## Adding a new algorithm

The studies and plotting glue only assume an algorithm can produce an **ordering
of basis-state indices** (and optionally a final energy). To add one:

1. **Create the module** `algorithms/my_algo.py`. Depend only on
   `numpy`/`scipy`/`qiskit` and, if useful, other `subspace_search` modules —
   do **not** add `sys.path` hacks; this is an installed package.

2. **Accept a standard Hamiltonian.** Take `H` as a `scipy.sparse` CSR matrix
   (use `subspace_search.algorithms.krab.as_csr_hamiltonian` if you want to also
   accept dense / qiskit inputs) and an integer/bitstring initial state.

3. **Return an ordering** of basis indices (and/or a small result object). That
   ordering plugs straight into `subspace_search.paths.get_one_path(H, order)`
   to get an energy-vs-#states path, and into
   `subspace_search.plotting.plot_convergence_paths` for comparison against SKQD.

4. **Re-export it** in `algorithms/__init__.py` (add to the imports and
   `__all__`).

5. **Wire up a study** by copying `cluster_studies/BARK` or
   `cluster_studies/KRAB` and swapping in your algorithm.

6. **Document + log it.** Add a row to the table above, then run
   "Update Documentation" so the change is propagated to the other READMEs and
   recorded in `docs/LOGBOOK.md`.

Keeping the "produces an ordering of basis indices" contract is what lets every
method share the same `paths` / `plotting` / cluster-study machinery.
