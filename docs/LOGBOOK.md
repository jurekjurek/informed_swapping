# Logbook

A running, append-only record of what changed in this repository and how the
documentation was brought back in sync. New entries are added to the **top** by
the "Update Documentation" routine (see [`../CLAUDE.md`](../CLAUDE.md)). Newest
first.

---

## 2026-07-08 — Planted Hamiltonian + SKQD energy proxy

**What happened**

- New Hamiltonian family
  `hamiltonians/new_hamiltonian_approach.py` (commit `895bb38`): builds
  `H = -Δ|g⟩⟨g| + λ R` from a sparse planted ground state `|g⟩` plus a random
  off-diagonal Pauli background `R`. Public API `make_planted_hamiltonian(...)`
  (returns a `SparsePauliOp` + an `info` dict with support, amplitudes, dense
  planted state, and a `suggested_initial_bitstring`) and `diagnostics(H, g)`
  (`E0`, planted-state fidelity, IPR, effective support). Both re-exported from
  `subspace_search.hamiltonians`.
- SKQD energy tracking: new `skqd/energy_tracking.py`
  (`update_ground_state_proxy`, `EnergyTrackingStep`) and a new
  `do_skqd_with_energy_tracking(...)` in `skqd/skqd.py` that maintains an
  iterative variational ground-state proxy over the sampled states and returns
  `(ordering, [EnergyTrackingStep, ...])`. The original `do_skqd(...)` is kept
  backward-compatible (ordering only); the sampling loop also gained a
  zero-probability / empty-leftover guard. New symbols re-exported from
  `subspace_search.skqd`.
- `experiments/compare_methods.py` gained a demo of the planted Pauli
  Hamiltonian (convert `SparsePauliOp` → CSR, print `diagnostics`).
- New top-level `AGENTS.md` mirroring `CLAUDE.md` (already documents the
  energy-tracking API).

**Verification**

- `.SKQD` venv: `make_planted_hamiltonian` + `diagnostics` on a 5-qubit system
  recover the planted state (E₀ ≈ −9.99, fidelity 1.000).
- `do_skqd_with_energy_tracking` runs end-to-end: returns the length-`dim`
  ordering plus one `EnergyTrackingStep` per step (proxy energy improves from
  −4.40 to −9.99 on step 0); `isinstance(step, EnergyTrackingStep)` holds.
- All new symbols import from `subspace_search.hamiltonians` / `.skqd`; package
  version unchanged at `0.1.0`.

**Docs updated:** `subspace_search/README.md` (package map + hamiltonians &
skqd API sections), `subspace_search/src/subspace_search/__init__.py` overview
docstring, `experiments/README.md` (compare_methods description), root
`README.md` (layout comment). Algorithms / cluster_studies / BARK READMEs
reviewed — no code changes there, left as-is.

---

## 2026-07-02 — Repository restructure & documentation rebuild

**What happened**

- The old, sprawling top-level contents (notebooks, loose scripts, `ClusterStudy/`,
  `new_approach/`, `Permutations/`, `SystematicScanOfConvergence/`,
  `UnitaryVsPower/`, PDFs, PNGs, etc.) were moved wholesale into
  [`../backup/`](../backup/). Nothing was deleted; the `.git`, `.gitignore` and
  `.SKQD/` virtualenv stayed in place.
- A new pip-installable package **`subspace_search`** was created (src layout,
  `pyproject.toml`), consolidating the routines that are needed everywhere:
  - `hamiltonians/controlled_sparsity.py` ← `new_approach/Hamiltonian_controlled_sparsity.py`
  - `skqd/skqd.py` ← `Permutations/SKQD.py`; `skqd/power.py` ← `Permutations/PowerSampling.py`
  - `algorithms/krab.py` ← `new_approach/krab_4.py` (KRAB)
  - `algorithms/bark.py` ← `Permutations/Bark_2_0.py` (BARK)
  - `paths.py` ← `Permutations/Helpers.py`
  - `plotting.py` — **new** shared plotting helpers (`plot_hamiltonian`,
    `plot_ground_state`, `plot_convergence_paths`).
- New run areas:
  - [`../experiments/`](../experiments/) for quick local scripts, with a working
    `compare_methods.py` example.
  - [`../cluster_studies/`](../cluster_studies/) with `KRAB/` and `BARK/` studies,
    copied from the old `ClusterStudy/` **scripts only** (outputs regenerate) and
    re-pointed to import from the `subspace_search` package instead of
    `sys.path` hacks. Slurm `cd` paths updated `ClusterStudy` → `cluster_studies`
    (and the BARK slurm `cd` fixed to include the `/BARK` subfolder).
- README files written/rewritten at every level: repo root, package, the
  `algorithms` subpackage (with a "how to add a new algorithm" guide),
  `experiments/`, `cluster_studies/` (+ `KRAB/` and `BARK/`).
- Added `CLAUDE.md` defining the **"Update Documentation"** routine and this
  logbook, plus a `/update-documentation` slash command wrapping it.

**Verification**

- `pip install -e subspace_search` succeeds in the `.SKQD` venv.
- Smoke test: build Hamiltonian → KRAB and SKQD both recover `E₀ = -5.0`;
  `plot_hamiltonian` renders.
- `experiments/compare_methods.py` runs end-to-end (KRAB reaches `E₀`).
- Both cluster-study `run_experiment.py` / `run_chunk.py` import cleanly against
  the installed package.

**Docs updated:** all READMEs (created fresh in this restructure).
