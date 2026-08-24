#!/bin/bash
# ---------------------------------------------------------------------------
# SLURM submit script for ClusterStudy.py
#
# Run it directly and it submits itself as a job array plus a dependent merge
# job. Run it under SLURM (which is what the submission does) and it executes
# one array task.
#
#   ./submit_study.sh                       # submit with the settings below
#   NUM_JOBS=40 ./submit_study.sh           # override any setting from the env
#   PARTITION=long TIME=48:00:00 ./submit_study.sh
#   ./submit_study.sh --plan                # just print the split, submit nothing
#
# Because every array task gets the same mix of system sizes, one TIME and one
# MEM setting fits every task -- that is the whole point of the stratified split.
# ---------------------------------------------------------------------------
#SBATCH --job-name=debunk-sqkd
#SBATCH --ntasks=1
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

# ----------------------------- configuration -------------------------------
NUM_HAMILTONIANS=${NUM_HAMILTONIANS:-10}
NUM_SITES=${NUM_SITES:-"6 8 10 12 14"}
MAX_INTERACTIONS=${MAX_INTERACTIONS:-"1 2 3"}
FIDELITIES=${FIDELITIES:-"0.8 0.85 0.9 0.95 0.99"}
NUM_JOBS=${NUM_JOBS:-40}

PARTITION=${PARTITION:-}                 # empty -> cluster default
TIME=${TIME:-24:00:00}
MEM=${MEM:-16G}
CPUS=${CPUS:-1}
THROTTLE=${THROTTLE:-}                   # e.g. 10 -> at most 10 tasks at once

SHARD_DIR=${SHARD_DIR:-shards}
OUTPUT=${OUTPUT:-systematic_study_results.csv}
PYTHON=${PYTHON:-python}
EXTRA_ARGS=${EXTRA_ARGS:-}               # e.g. "--dense" or "--overwrite"
# ---------------------------------------------------------------------------

STUDY_ARGS=(
  --num-hamiltonians "$NUM_HAMILTONIANS"
  --num-sites $NUM_SITES
  --max-interactions $MAX_INTERACTIONS
  --fidelities $FIDELITIES
  --num-jobs "$NUM_JOBS"
  --shard-dir "$SHARD_DIR"
)

# ------------------------------ worker mode --------------------------------
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  cd "${SLURM_SUBMIT_DIR:-$PWD}"
  # One thread per core, so CPUS=1 does not have BLAS oversubscribing the node.
  export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
  export MKL_NUM_THREADS="$OMP_NUM_THREADS"
  export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"

  echo "host=$(hostname) task=$SLURM_ARRAY_TASK_ID threads=$OMP_NUM_THREADS"
  exec "$PYTHON" ClusterStudy.py run "${STUDY_ARGS[@]}" \
       --job-index "$SLURM_ARRAY_TASK_ID" $EXTRA_ARGS
fi

# ---------------------------- submission mode ------------------------------
SCRIPT="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"

# Show the split and stop, if asked.
if [[ "${1:-}" == "--plan" ]]; then
  exec "$PYTHON" ClusterStudy.py plan "${STUDY_ARGS[@]}" --fidelities $FIDELITIES
fi

mkdir -p logs "$SHARD_DIR"

echo "=== planned split ==="
"$PYTHON" ClusterStudy.py plan "${STUDY_ARGS[@]}"
echo

ARRAY_SPEC="0-$((NUM_JOBS - 1))"
[[ -n "$THROTTLE" ]] && ARRAY_SPEC="${ARRAY_SPEC}%${THROTTLE}"

SBATCH_COMMON=(--time="$TIME" --mem="$MEM" --cpus-per-task="$CPUS")
[[ -n "$PARTITION" ]] && SBATCH_COMMON+=(--partition="$PARTITION")

ARRAY_JOB=$(sbatch --parsable --array="$ARRAY_SPEC" "${SBATCH_COMMON[@]}" "$SCRIPT")
echo "submitted array job $ARRAY_JOB ($ARRAY_SPEC)"

# Merge once every task has succeeded. afterok means a failed task blocks the
# merge, so partial results are never silently written out as if complete.
MERGE_JOB=$(sbatch --parsable \
  --dependency="afterok:${ARRAY_JOB}" \
  --job-name=debunk-sqkd-merge \
  --time=00:20:00 --mem=8G --cpus-per-task=1 \
  ${PARTITION:+--partition="$PARTITION"} \
  --output="logs/merge_%j.out" --error="logs/merge_%j.err" \
  --wrap="cd '$PWD' && $PYTHON ClusterStudy.py merge ${STUDY_ARGS[*]} --fidelities $FIDELITIES --output '$OUTPUT'")
echo "submitted merge job $MERGE_JOB (runs after the array succeeds)"
echo
echo "watch:   squeue -j $ARRAY_JOB,$MERGE_JOB"
echo "logs:    logs/"
echo "result:  $OUTPUT"
echo
echo "If some tasks time out, raise TIME and re-run ./submit_study.sh -- finished"
echo "shards in $SHARD_DIR are skipped, so only the missing work is redone."
