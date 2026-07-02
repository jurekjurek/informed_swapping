---
description: Review the repo for changes, update every README/doc to match, and append a dated logbook entry.
---

Run the **"Update Documentation"** routine defined in `CLAUDE.md`.

Steps:

1. Read the most recent entry in `docs/LOGBOOK.md` to find the last update date,
   then inspect what has changed since (`git status`, `git log`, `git diff`) —
   new/moved/renamed/deleted files, new algorithms, changed public APIs.
2. Check and fix every README and package docstring so it matches the current
   code: root `README.md`, `subspace_search/README.md`, the `algorithms`
   README, `experiments/README.md`, and `cluster_studies/` (+ `KRAB/`, `BARK/`).
   Create a README for any new directory that lacks one.
3. Verify claims (imports resolve, example snippets valid, referenced
   files/flags exist) before asserting anything works.
4. Append a new dated section to the **top** of `docs/LOGBOOK.md` (newest first):
   `## YYYY-MM-DD — <title>`, with **What happened**, **Verification**, and
   **Docs updated**. Use today's real date; never rewrite old entries.
5. Report back: what changed, which docs were updated, and the new entry title.

$ARGUMENTS
