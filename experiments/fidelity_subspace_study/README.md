# fidelity_subspace_study

Systematic scan of **BARK vs SKQD** on random spin Hamiltonians, built around one
question:

> How large a subspace — normalised by the Hilbert space dimension `2**n` — does
> each method need in order to reach a given fidelity with the exact ground
> state, once its own parameters are chosen optimally?

Everything is arranged so the answer can be read off **one plot** that carries
every qubit count, interaction cap, Hamiltonian sample and parameter setting at
once.

## What the scan does

For every combination of

| axis | values (default preset) |
|------|------------------------|
| qubit number | `--num-qubits 6 8 10` |
| interaction cap | `--max-interactions 1 2 -1` (`-1` = all-to-all) |
| Hamiltonian sample | `--num-hamiltonians 3` (seeds `0,1,2`) |
| **initial overlap** | `--initial-overlaps max 0.05` |
| target fidelity | `--target-fidelities 0.5 0.9 0.99` |

it runs

* **SKQD** over the full `(dt, shots)` grid, repeated `--skqd-repeats` times, and
* **BARK** over the full `(score_mode, selection_strategy, keep_states)` grid,

then picks, per Hamiltonian and per target fidelity, the parameter setting that
reaches that fidelity with the **smallest normalised subspace dimension**. That
is the "optimisation step" for SKQD's `dt` and shot count, and the equivalent
selection for BARK.

### Shared initial state

The initial overlap is a scanned parameter, not a fixed choice. `max` selects the
basis state with the largest ground-state weight; a number like `0.05` selects
the basis state whose weight is closest to it *in log space*. **SKQD and BARK
always start from the same basis state**, so the comparison is not confounded by
the starting point. The realised overlap is stored on every row
(`initial_overlap`).

### One diagonalisation per iteration

Both methods diagonalise the accumulated subspace exactly **once per
iteration** — never once per bitstring:

* **SKQD** (`skqd_runner.py`) implements the real protocol: propagate with
  `exp(-i H dt)`, draw `shots` measurements from `|<i|psi_k>|^2`, union the
  unique outcomes into the subspace, diagonalise. This is *not*
  `subspace_search.skqd.do_skqd`, which returns an ordering of the whole Hilbert
  space and has no shot count.
* **BARK** (`bark_runner.py`) subclasses
  `subspace_search.algorithms.bark_best_first_baab.BarkBarkBark` and re-expresses
  its two selection strategies as generators, yielding one round per Hamiltonian
  application. The packaged algorithm itself is untouched.

An iteration whose subspace did not change is not re-diagonalised, and is
flagged with `diagonalized = False`.

## Fidelity

`fidelity = || P_gs |v> ||^2`, where `|v>` is the lowest eigenvector of the
projected Hamiltonian embedded back into the full space and `P_gs` projects onto
the exact ground **eigenspace**. Using the eigenspace projector rather than a
single eigenvector keeps the metric meaningful under degeneracy (`degeneracy` is
stored per Hamiltonian, so degenerate cases can be filtered out).

`captured_weight` is the companion upper bound: the ground-state weight the
subspace contains, which no diagonalisation inside it can exceed.

## Files

| File | Role |
|------|------|
| `study_config.py` | `StudyConfig` — the complete scan definition, plus the `smoke` / `default` / `full` presets. |
| `hamiltonian_cases.py` | Samples one random spin Hamiltonian, computes the exact low spectrum and every static statistic, and resolves initial-overlap specs into basis states. |
| `subspace_metrics.py` | `timed` context manager and `SubspaceEvaluator` (project → diagonalise once → score). Shared by both methods so the timing columns mean the same thing. |
| `skqd_runner.py` | Shot-based SKQD with a per-`dt` cached propagator. |
| `bark_runner.py` | `InstrumentedBark` + the per-round driver. |
| `optimization.py` | Collapses repeats and picks the best parameters per (Hamiltonian, initial overlap, target fidelity, method). |
| `run_study.py` | CLI driver; writes the parquet tables. |
| `analyze_study.py` | CLI; turns the tables into the figures and their table views. |

## Running

```bash
# from the repository root, with the project venv
.BARK/bin/python experiments/fidelity_subspace_study/run_study.py --preset smoke
.BARK/bin/python experiments/fidelity_subspace_study/analyze_study.py \
    --results-dir experiments/fidelity_subspace_study/results
```

`--dry-run` prints the scan size without executing anything — always worth doing
before `--preset full`.

Long scans:

```bash
# split across processes / array tasks, resume safely
python run_study.py --preset full --chunk 3 --num-chunks 16 --resume
python run_study.py --merge-only          # merge shards + redo the optimisation
```

Useful knobs: `--max-subspace-fraction 0.5` caps how far a run is allowed to grow
(the diagonalisations dominate the cost at large `n`), and
`--min-reach-fraction` sets how many repeats a configuration must succeed in
before it is eligible to win the grid.

### Cost

`--dry-run` reports the run count; on this machine a run costs roughly 0.1 s at
`n = 6` and ~1 s at `n = 10`, and the subspace diagonalisations dominate. As a
reference point, `--preset smoke` is ~1 500 runs (seconds), `--preset default`
is ~3 900 runs (tens of minutes) and `--preset full` is ~32 600 runs — that one
is meant to be chunked. Runs that must reach a high target fidelity on a large
Hilbert space are the expensive ones, so raising `--max-subspace-fraction`
costs superlinearly.

### Caveats worth knowing

* Fidelity is measured against the exact ground space, so the scan is limited to
  Hamiltonians that can be diagonalised exactly (`n <= 12` comfortably).
* A run that stops at `--max-subspace-fraction` or at
  `--skqd-max-iterations` / `--bark-max-iterations` before hitting a target is
  recorded as *not reached* (`feasible = False`, `NaN` dimension) rather than
  extrapolated. Check `reached_any` in `optimal.parquet` to tell "no
  configuration ever managed it" from "only a minority of repeats managed it".
* The headline envelopes stop at the largest subspace the runs actually
  reached; they are not extended flat to dimension 1.

## Output tables (parquet, via pandas)

Written to `--output-dir` (default `results/`), with per-Hamiltonian shards in
`results/shards/` so a scan can be resumed or parallelised.

| Table | Grain | Contents |
|-------|-------|----------|
| `hamiltonians.parquet` | one Hamiltonian | Hamiltonian sparsity (`ham_nnz`, `ham_density`, row-`nnz`, Pauli-term count, Frobenius norm), interaction-graph degrees, ground-state sparsity (`gs_support_fraction`, `gs_ipr`, `gs_participation_fraction`, `gs_entropy_normalized`, `gs_dim_fraction_for_{50,90,99}pct`), `ground_energy`, `gap`, `degeneracy`, `spectral_range`, and the construction timings. |
| `iterations.parquet` | one iteration of one run | The big table. Subspace size and fraction, new indices, shots, energy, fidelity, captured weight, plus the full timing breakdown and its cumulative sums. |
| `runs.parquet` | one run | Per-run totals for every timing stage and the final state of the run. |
| `convergence.parquet` | one run × target fidelity | Whether the target was reached, and the subspace dimension, iteration, shot count and cumulative runtimes at which it first was. |
| `optimal.parquet` | one (Hamiltonian, overlap, target, method) | The winning configuration and its normalised subspace dimension. |
| `method_comparison.parquet` | one (Hamiltonian, overlap, target) | SKQD and BARK side by side, with `subspace_advantage_bark = frac_SKQD / frac_BARK`. |

### Timing columns

Every iteration carries: `t_propagator_setup`, `t_propagate`, `t_sample`,
`t_select`, `t_expand`, `t_ground_update`, `t_subspace_update`, `t_project`,
`t_diagonalize`, `t_fidelity`, and the roll-ups `t_algorithm` (method-specific
work), `t_solve` (`t_project + t_diagonalize`) and `t_iteration`.

Two things to know before summing them:

* `t_fidelity` is **evaluation-only** — it compares against the exact ground
  space, which no real run has. Exclude it from any cost comparison;
  `t_algorithm + t_solve` is the honest total.
* For BARK with `score_mode` in `{coupling, perturbative}`, `t_ground_update` is
  a full-subspace diagonalisation of the *same* subspace that `t_solve`
  diagonalises for the energy readout. Both are recorded unmodified and
  `internal_diagonalization` flags the affected rows; a production
  implementation would share the work, so do not naively add the two columns for
  those modes.

## Figures

`analyze_study.py` writes to `results/figures/`, each with a `.csv` table view
next to it.

* **`fidelity_vs_subspace.png` — the headline.** One axis, fidelity vs
  normalised subspace dimension (log). Each thin line is the *best-parameter
  envelope* of one (Hamiltonian, initial overlap, repeat): the highest fidelity
  any scanned parameter setting reached at each subspace size, made monotone by
  a running maximum. Reading the plot at a fidelity therefore gives exactly the
  optimised quantity. Bold lines are the per-method median, bands the IQR.
* `subspace_scaling.png` — normalised subspace needed vs qubit count, one panel
  per target fidelity.
* `initial_overlap_effect.png` — the same against the shared initial overlap.
* `sparsity_effect.png` — against the ground-state participation fraction.
* `runtime_breakdown.png` — share of wall-clock time per stage, per method and
  qubit count.

Colour encodes **method identity only** (SKQD / BARK); qubit count and
interaction cap are read from the supporting panels, which is what keeps the
headline figure legible with hundreds of curves on it. The hexes are slots 1–2
(and 1–6 for the runtime stages) of the validated categorical palette in the
`dataviz` skill's `references/palette.md`; that file documents those slots as
passing the CVD/contrast gates in both modes, and no re-validation was run here
(the validator needs `node`, which is not installed on this machine).
`--theme dark` renders the dark-mode steps of the same palette.

## Environment note (macOS + iCloud)

The repository lives in iCloud Drive, and iCloud sets the macOS `UF_HIDDEN` flag
on files inside `.BARK/`. CPython's `site` module **silently skips hidden `.pth`
files**, so `pip install -e subspace_search` registers an editable install that
never takes effect (`import subspace_search` then falls back to an empty
namespace package, and `subspace_search.algorithms` fails to import).
`chflags nohidden` is undone by iCloud within seconds.

The package is therefore installed **non-editable**:

```bash
.BARK/bin/pip install ./subspace_search
```

Re-run that after editing anything under `subspace_search/src/`. Moving the venv
outside iCloud would restore editable installs.
