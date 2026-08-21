# experiments

Scratch space for **quick, local** test and exploration scripts — anything you
want to run on your own machine (not the cluster) to sanity-check an idea,
eyeball a plot, or prototype before scaling up to a full
[`cluster_studies/`](../cluster_studies/) run.

Everything here imports from the installed `subspace_search` package, so make
sure it's installed first:

```bash
source ../.BARK/bin/activate
pip install ../subspace_search
```

> **Not editable.** The repo lives in iCloud Drive, which sets the macOS
> `UF_HIDDEN` flag on files inside the venv; CPython's `site` module silently
> skips hidden `.pth` files, so `pip install -e` registers an editable install
> that never takes effect. Re-run the plain `pip install` after editing anything
> under `subspace_search/src/`. See
> [`fidelity_subspace_study/README.md`](fidelity_subspace_study/README.md) for
> the full story.

Then run any script directly, e.g.:

```bash
python compare_methods.py
```

## Conventions

- **Import from the package**, never via `sys.path` hacks:
  `from subspace_search.algorithms import selected_krylov_ground_state`.
- Keep scripts self-contained and cheap (small `n_qubits`); promote anything
  that needs a parameter sweep to `cluster_studies/`.
- Write throwaway outputs to a local `out/` (gitignored) or the system temp dir,
  not into the repo.

## What's here

| File | Purpose |
|------|---------|
| `compare_methods.py` | End-to-end example: build a controlled-sparsity Hamiltonian, run KRAB / SKQD / a random baseline, and plot their convergence paths on one axis. Also shows the newer planted Pauli Hamiltonian (`make_planted_hamiltonian` → `SparsePauliOp`, converted to CSR) with its `diagnostics`. A good starting template to copy. |
| `compare_bark_variants_vs_skqd.py` | Compares six BARK settings (pool/best-first x amplitude/two-dimensional/perturbative) against a sweep of SKQD timesteps on random spin Hamiltonians, one plot per Hamiltonian and BARK setting. |
| [`fidelity_subspace_study/`](fidelity_subspace_study/) | Systematic BARK vs SKQD scan: how large a subspace (normalised by `2**n`) each method needs to reach a target fidelity, with SKQD's `(dt, shots)` and BARK's `(score_mode, selection_strategy, keep_states)` optimised per Hamiltonian. Writes parquet tables with full per-stage runtime statistics and renders the comparison figures. Has its own README. |
