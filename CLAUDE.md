# CLAUDE.md — guide for Claude Code in this repository

## What this project is

Research code asking whether **classical, subspace-selecting heuristics** (the
dog-named **KRAB** and **BARK**) can reproduce or beat **SKQD** (Subspace-based
Krylov Quantum Diagonalization) at finding sparse ground states. See
[`README.md`](README.md) for the full picture.

## Structure (post-2026-07-02 restructure)

- **`subspace_search/`** — the pip-installable package (src layout). The reusable
  core: `hamiltonians/`, `skqd/`, `algorithms/` (KRAB, BARK + new ones),
  `paths.py`, `plotting.py`. Install with `pip install -e subspace_search` inside
  the `.SKQD` venv.
- **`experiments/`** — quick local test/prototype scripts.
- **`cluster_studies/`** — Slurm grid studies (`KRAB/`, `BARK/`).
- **`docs/LOGBOOK.md`** — running log of documentation updates.
- **`backup/`** — everything from before the restructure (do not edit; reference
  only).
- **`.SKQD/`** — the project virtualenv (gitignored).

## Working conventions

- Code imports from the installed package (`from subspace_search... import ...`);
  **never** add `sys.path` hacks in new code.
- New algorithms go in `subspace_search/src/subspace_search/algorithms/` and are
  re-exported in that package's `__init__.py`. See its README for the contract
  (produce an ordering of basis indices so it plugs into `paths`/`plotting`).
- Keep the `.SKQD` venv as the interpreter for running/testing.
- Don't commit generated outputs (figures, CSVs, `results/`, `analysis/`).

## Routine: "Update Documentation"

When the user says **"Update Documentation"** (or runs `/update-documentation`),
execute this routine end to end:

1. **Find what changed.** Determine the changes since the last logbook entry:
   - read the top (most recent) entry in `docs/LOGBOOK.md` to get the last date;
   - run `git status` and `git log` / `git diff` since then, and note new,
     moved, renamed, or deleted files — especially anything under
     `subspace_search/`, `experiments/`, `cluster_studies/`, and any new
     algorithm modules or changed public APIs.

2. **Review every README and doc.** Check each of these against the actual code
   and fix anything now stale (paths, import examples, function/parameter names,
   file tables, layout diagrams, run instructions):
   - `README.md` (root)
   - `subspace_search/README.md`
   - `subspace_search/src/subspace_search/algorithms/README.md`
   - `experiments/README.md`
   - `cluster_studies/README.md`, `cluster_studies/KRAB/README.md`,
     `cluster_studies/BARK/README.md`
   - any README in a newly added directory (create one if a new dir lacks it)
   - package docstrings / `__init__.py` overviews if the API changed.

3. **Verify claims.** Before writing that something "works", confirm it (imports
   resolve, example snippets are valid, referenced files/flags exist). Update
   version numbers if the public API changed meaningfully.

4. **Append a logbook entry.** Add a new dated section to the **top** of
   `docs/LOGBOOK.md` (newest first) with:
   - `## YYYY-MM-DD — <short title>`
   - **What happened** — bullet summary of the code/structure changes;
   - **Verification** — what was checked and the result;
   - **Docs updated** — which README/doc files were changed.
   Use today's real date. Keep entries append-only (never rewrite old ones).

5. **Report back** concisely: what changed, which docs were updated, and the new
   logbook entry title.

Keep documentation accurate over exhaustive — if something is uncertain or
unverified, say so rather than asserting it.
