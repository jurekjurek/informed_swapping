# cluster_studies

Systematic, Slurm-driven studies that run one of the algorithms against **SKQD**
(and a random baseline) across a hyperparameter grid, then aggregate the results
into analysis plots. These are the heavy runs — for quick local prototyping use
[`experiments/`](../experiments/) instead.

Each study is self-contained in its own subfolder but shares the same shape:

```
<STUDY>/
├── run_experiment.py   # one experiment -> one figure (+ one CSV row)
├── run_chunk.py        # run a slice of the param file sequentially
├── generate_jobs.py    # write the parameter grid (params/*.txt)
├── run_array.slurm     # Slurm array job over the chunks
├── params/             # generated parameter grids
└── SlurmOut/           # Slurm logs
```

Outputs (`figures/`, `data/`, `analysis/`) are created on the fly when you run a
study; they are **not** committed. The original outputs from before the
2026-07-02 restructure live under [`../backup/ClusterStudy/`](../backup/ClusterStudy/).

## Studies

| Study | Algorithm vs SKQD | README |
|-------|-------------------|--------|
| [`KRAB/`](KRAB/) | **KRAB** (selected-subspace Krylov), incl. a decaying-`Q` variant | [KRAB/README.md](KRAB/README.md) |
| [`BARK/`](BARK/) | **BARK** (bitstring recursive Krylov) | [BARK/README.md](BARK/README.md) |

## Common workflow

```bash
source ../.SKQD/bin/activate
pip install -e ../subspace_search        # once; scripts import from the package

cd KRAB                                   # or BARK
python generate_jobs.py --preset full     # write params/param_grid.txt
sbatch run_array.slurm                     # submit the array (edit account/paths first)
python plot_scaling.py                     # aggregate results -> analysis/  (KRAB)
```

All scripts import the core routines from the installed `subspace_search`
package — there are no `sys.path` hacks and nothing depends on the working
directory beyond the study folder itself.

## Adding a study for a new algorithm

Copy `BARK/` or `KRAB/`, swap the imported algorithm in `run_experiment.py`
(`from subspace_search.algorithms import ...`), adjust the parameter grid in
`generate_jobs.py`, and update this table. Then run "Update Documentation".
