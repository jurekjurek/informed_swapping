#!/bin/bash
# ---------------------------------------------------------------------------
# Local counterpart of submit_bitstrings.sh: the same study on one machine.
#
# The study is split into NUM_JOBS shards, more than there are workers, and
# xargs hands them out WORKERS at a time as workers free up. A fixed split into
# exactly WORKERS slices would leave the machine waiting on whichever slice the
# cost model misjudged; small shards balance themselves.
#
#   ./run_bitstrings_local.sh                 # run (or resume) the whole study
#   WORKERS=4 ./run_bitstrings_local.sh       # leave more of the machine free
#   ./run_bitstrings_local.sh --plan          # just print the split
#
# Interrupting is safe: finished shards are skipped on the next call, and a
# shard that was cut off resumes from its .partial file, so only the cell that
# was running is redone. Progress: tail -f logs/local/job_*.log
# ---------------------------------------------------------------------------

set -euo pipefail
cd "$(dirname "$0")"

# ----------------------------- configuration -------------------------------
# Same study as submit_bitstrings.sh; only the distribution differs.
NUM_HAMILTONIANS=${NUM_HAMILTONIANS:-15}
NUM_SITES=${NUM_SITES:-"6 8 10 12"}
DIMENSIONS=${DIMENSIONS:-"1 2"}
DELTAS=${DELTAS:-"0 0.5 1 10 100"}
NUM_INITIAL_STATES=${NUM_INITIAL_STATES:-6}
N_REPEATS=${N_REPEATS:-3}
GRID_POINTS=${GRID_POINTS:-30}
MAX_FRACTION=${MAX_FRACTION:-0.5}
SKQD_MAX_ITERATIONS=${SKQD_MAX_ITERATIONS:-10000}
DENSE_LIMIT=${DENSE_LIMIT:-4096}

# Parallel processes, one BLAS thread each. The default is the number of
# performance cores; efficiency cores would only slow the last shards down.
WORKERS=${WORKERS:-$(sysctl -n hw.perflevel0.physicalcpu 2>/dev/null || echo 4)}
NUM_JOBS=${NUM_JOBS:-60}

# A new shard folder and result file, so nothing is mixed with the runs made
# under the old SKQD stopping rule.
SHARD_DIR=${SHARD_DIR:-bitstring_shards_heisenberg_v2}
OUTPUT=${OUTPUT:-equal_bitstrings_heisenberg_results_v2.csv}
OUTPUT_ROOT=${OUTPUT_ROOT:-equal_bitstrings_plots_heisenberg_v2}
PYTHON=${PYTHON:-python}
LOG_DIR=${LOG_DIR:-logs/local}
# ---------------------------------------------------------------------------

STUDY_ARGS=(
  --num-hamiltonians "$NUM_HAMILTONIANS"
  --num-sites $NUM_SITES
  --dimensions $DIMENSIONS
  --deltas $DELTAS
  --num-initial-states "$NUM_INITIAL_STATES"
  --n-repeats "$N_REPEATS"
  --grid-points "$GRID_POINTS"
  --max-fraction "$MAX_FRACTION"
  --skqd-max-iterations "$SKQD_MAX_ITERATIONS"
  --num-jobs "$NUM_JOBS"
  --shard-dir "$SHARD_DIR"
  --balance cost
  --dense-limit "$DENSE_LIMIT"
)

if [[ "${1:-}" == "--plan" ]]; then
  exec "$PYTHON" EqualNumberOfBitstrings.py plan "${STUDY_ARGS[@]}"
fi

# One thread per process: many small dense solves plus single-threaded sampling
# gain little from threads, and WORKERS processes would oversubscribe the cores.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

mkdir -p "$LOG_DIR" "$SHARD_DIR"
echo "$(date '+%F %T')  $NUM_JOBS shards on $WORKERS workers -> $SHARD_DIR (logs in $LOG_DIR)"

# caffeinate keeps the machine from idle-sleeping for exactly as long as the
# study runs. Closing the lid still sleeps it, so keep it open and plugged in.
export PYTHON LOG_DIR
seq 0 $((NUM_JOBS - 1)) | caffeinate -i xargs -P "$WORKERS" -I{} bash -c '
  "$PYTHON" -W ignore EqualNumberOfBitstrings.py run "$@" --job-index {} \
      > "$LOG_DIR/job_{}.log" 2>&1 \
    && echo "$(date "+%F %T")  shard {} done" \
    || echo "$(date "+%F %T")  shard {} FAILED, see $LOG_DIR/job_{}.log"
' _ "${STUDY_ARGS[@]}"

"$PYTHON" EqualNumberOfBitstrings.py merge "${STUDY_ARGS[@]}" --output "$OUTPUT"
"$PYTHON" EqualNumberOfBitstrings.py plot "${STUDY_ARGS[@]}" --output "$OUTPUT" \
    --output-root "$OUTPUT_ROOT"
echo "$(date '+%F %T')  done: $OUTPUT, figures in $OUTPUT_ROOT/"
echo "paper figures: $PYTHON PaperFigures.py --data $OUTPUT --output PaperFigures_v2"
