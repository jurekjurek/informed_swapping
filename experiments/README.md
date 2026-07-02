# experiments

Scratch space for **quick, local** test and exploration scripts — anything you
want to run on your own machine (not the cluster) to sanity-check an idea,
eyeball a plot, or prototype before scaling up to a full
[`cluster_studies/`](../cluster_studies/) run.

Everything here imports from the installed `subspace_search` package, so make
sure it's installed first:

```bash
source ../.SKQD/bin/activate
pip install -e ../subspace_search
```

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
| `compare_methods.py` | End-to-end example: build a controlled-sparsity Hamiltonian, run KRAB / SKQD / a random baseline, and plot their convergence paths on one axis. A good starting template to copy. |
