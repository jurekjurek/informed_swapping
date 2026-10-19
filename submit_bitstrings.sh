#!/bin/bash
# ---------------------------------------------------------------------------
# SLURM submit script for EqualNumberOfBitstrings.py
#
# Same shape as submit_study.sh: run it directly and it submits itself as a job
# array plus a dependent merge job, which also draws the figures. Run it under
# SLURM (which is what the submission does) and it executes one array task.
# In front of the array runs a plan job, which selects the Hamiltonians of
# --balanced; that takes too long for the login node.
#
#   ./submit_bitstrings.sh                        # submit with the settings below
#   NUM_JOBS=40 ./submit_bitstrings.sh            # override any setting from the env
#   PARTITION=long TIME=48:00:00 ./submit_bitstrings.sh
#   ./submit_bitstrings.sh --plan                 # just print the split, submit nothing
#                                                 # (selects on the login node if
#                                                 # the plan job has not yet)
#
# Because every array task gets the same estimated load, one TIME and one MEM
# setting fits every task -- that is the whole point of the cost-balanced split.
# Unlike the fixed-target study, these runs grow a pool proportional to the
# Hilbert space, so MAX_FRACTION is the knob that decides the cost: halving it
# takes roughly an order of magnitude off the largest cells.
# ---------------------------------------------------------------------------
#SBATCH --job-name=debunk-sqkd-bits
#SBATCH --ntasks=1
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

module load Python/3.10.4-GCCcore-11.3.0 && source /home/erosanow_hpc/informed_swapping/.SKQD/bin/activate

# ----------------------------- configuration -------------------------------
NUM_HAMILTONIANS=${NUM_HAMILTONIANS:-20}
NUM_SITES=${NUM_SITES:-"6 8 10 12"}
DIMENSIONS=${DIMENSIONS:-"1 2"}               # 1 = chain, 2 = open rectangle
DELTAS=${DELTAS:-"0 0.5 1 2 5 10"}            # XXZ anisotropies
DISTANCES=${DISTANCES:-"1 2 3"}               # largest bond distances; 1 = nearest neighbours
NUM_INITIAL_STATES=${NUM_INITIAL_STATES:-6}   # from each end of the overlap distribution
N_REPEATS=${N_REPEATS:-3}                     # SKQD trajectories per run
GRID_POINTS=${GRID_POINTS:-30}                # budgets at which the curves are read
MAX_FRACTION=${MAX_FRACTION:-0.5}             # largest budget, as a fraction of 2^N
NUM_JOBS=${NUM_JOBS:-40}

PARTITION=${PARTITION:-intelsr_long}                 # empty -> cluster default
TIME=${TIME:-07-00:00:00}
MEM=${MEM:-40G}
# As in the other study: a very large number of small-to-medium dense LAPACK
# calls plus single-threaded sampling, not a few big ones. Few threads per task,
# spend the freed cores on a larger array instead.
CPUS=${CPUS:-4}
THROTTLE=${THROTTLE:-}                   # e.g. 10 -> at most 10 tasks at once
PLAN_CPUS=${PLAN_CPUS:-16}               # the plan job selects one group per core
PLAN_TIME=${PLAN_TIME:-02:00:00}

BALANCE=${BALANCE:-cost}                 # cost (load-balanced) or stratified
DENSE_LIMIT=${DENSE_LIMIT:-4096}         # dimension up to which SKQD gets the
                                         # full eigendecomposition; 0 disables
SHARD_DIR=${SHARD_DIR:-bitstring_shards_heisenberg_balanced_log_dense}
OUTPUT=${OUTPUT:-equal_bitstrings_heisenberg_results_balanced_log_dense.csv}
OUTPUT_ROOT=${OUTPUT_ROOT:-equal_bitstrings_plots_heisenberg_balanced_log_dense}
PYTHON=${PYTHON:-python}
EXTRA_ARGS=${EXTRA_ARGS:-}               # e.g. "--overwrite" or "--no-resume"
# ---------------------------------------------------------------------------

STUDY_ARGS=(
  --num-hamiltonians "$NUM_HAMILTONIANS"
  --num-sites $NUM_SITES
  --dimensions $DIMENSIONS
  --deltas $DELTAS
  --distances $DISTANCES
  --num-initial-states "$NUM_INITIAL_STATES"
  --n-repeats "$N_REPEATS"
  --grid-points "$GRID_POINTS"
  --max-fraction "$MAX_FRACTION"
  --num-jobs "$NUM_JOBS"
  --shard-dir "$SHARD_DIR"
  --balance "$BALANCE"
  --dense-limit "$DENSE_LIMIT"
  --balanced
)

# ------------------------------ worker mode --------------------------------
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  cd "${SLURM_SUBMIT_DIR:-$PWD}"
  # One thread per core, so CPUS=1 does not have BLAS oversubscribing the node.
  export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
  export MKL_NUM_THREADS="$OMP_NUM_THREADS"
  export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"

  echo "host=$(hostname) task=$SLURM_ARRAY_TASK_ID threads=$OMP_NUM_THREADS"
  exec "$PYTHON" EqualNumberOfBitstrings.py run "${STUDY_ARGS[@]}" \
       --job-index "$SLURM_ARRAY_TASK_ID" $EXTRA_ARGS
fi

# ---------------------------- submission mode ------------------------------
SCRIPT="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"

# Show the split and stop, if asked.
if [[ "${1:-}" == "--plan" ]]; then
  exec "$PYTHON" EqualNumberOfBitstrings.py plan "${STUDY_ARGS[@]}"
fi

mkdir -p logs "$SHARD_DIR"

# Plan in a job of its own rather than here on the login node: with --balanced
# it first selects the Hamiltonians, into $SHARD_DIR where every array task
# reads them, and that takes too long here. The split goes to its log. Once the
# selection is on disk, a resubmission just reads it back.
PLAN_JOB=$(sbatch --parsable \
  --job-name=debunk-sqkd-bits-plan \
  --time="$PLAN_TIME" --mem=16G --cpus-per-task="$PLAN_CPUS" \
  ${PARTITION:+--partition="$PARTITION"} \
  --output="logs/plan_%j.out" --error="logs/plan_%j.err" \
  --wrap="cd '$PWD' && $PYTHON EqualNumberOfBitstrings.py plan ${STUDY_ARGS[*]}")
echo "submitted plan job $PLAN_JOB (split in logs/plan_$PLAN_JOB.out)"

ARRAY_SPEC="0-$((NUM_JOBS - 1))"
[[ -n "$THROTTLE" ]] && ARRAY_SPEC="${ARRAY_SPEC}%${THROTTLE}"

SBATCH_COMMON=(--time="$TIME" --mem="$MEM" --cpus-per-task="$CPUS")
[[ -n "$PARTITION" ]] && SBATCH_COMMON+=(--partition="$PARTITION")

ARRAY_JOB=$(sbatch --parsable --dependency="afterok:${PLAN_JOB}" --array="$ARRAY_SPEC" \
  "${SBATCH_COMMON[@]}" "$SCRIPT")
echo "submitted array job $ARRAY_JOB ($ARRAY_SPEC, runs after the plan job succeeds)"

# Merge once every task has succeeded, then draw the figures from the merged
# CSV. afterok means a failed task blocks both, so partial results are never
# silently written out -- or plotted -- as if complete.
MERGE_JOB=$(sbatch --parsable \
  --dependency="afterok:${ARRAY_JOB}" \
  --job-name=debunk-sqkd-bits-merge \
  --time=02:00:00 --mem=8G --cpus-per-task=1 \
  ${PARTITION:+--partition="$PARTITION"} \
  --output="logs/merge_%j.out" --error="logs/merge_%j.err" \
  --wrap="cd '$PWD' && $PYTHON EqualNumberOfBitstrings.py merge ${STUDY_ARGS[*]} --output '$OUTPUT' && $PYTHON EqualNumberOfBitstrings.py plot ${STUDY_ARGS[*]} --output '$OUTPUT' --output-root '$OUTPUT_ROOT'")
echo "submitted merge job $MERGE_JOB (runs after the array succeeds)"
echo
echo "watch:   squeue -j $PLAN_JOB,$ARRAY_JOB,$MERGE_JOB"
echo "logs:    logs/"
echo "result:  $OUTPUT"
echo "figures: $OUTPUT_ROOT/"
echo
echo "If some tasks time out, raise TIME (or lower MAX_FRACTION) and re-run"
echo "./submit_bitstrings.sh -- finished shards in $SHARD_DIR are skipped, and a"
echo "task that died mid-shard resumes from its .partial file, so only the"
echo "missing cells are redone."
