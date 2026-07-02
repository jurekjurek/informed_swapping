#!/bin/bash
# =============================================================================
# Slurm array job: BARK vs SKQD systematic study
#
# 30 array jobs, each running ~1/30 of the parameter grid sequentially.
# Each job may run for up to 24 hours.
#
# Workflow
# --------
# 1. Generate the parameter file (once):
#       cd ClusterStudy
#       python generate_jobs.py --preset full
#
# 2. Submit all 30 chunks:
#       sbatch run_array.slurm
#
# 3. If some jobs fail or time out, re-submit the same script — experiments
#    whose output PNG already exists are skipped automatically.
#
# Changing the grid size
# ----------------------
# If you add more parameter combinations, update N_CHUNKS below and in the
# --array range.  The chunk_slice logic in run_chunk.py distributes evenly
# regardless of how many lines are in param_grid.txt.
# =============================================================================

#SBATCH --job-name=BARKvsSKQD

#SBATCH --export=NONE

#SBATCH --account=ag_hiskp_funcke

#SBATCH --partition=intelsr_medium

# 30 array jobs, 0-indexed
#SBATCH --array=0-29

# Up to 24 hours per chunk
#SBATCH --time=24:00:00

# Memory: eigsh on a 1024×1024 matrix is light, but 100 random paths can
# accumulate intermediate matrices.  32 GB is a safe ceiling.
#SBATCH --mem=32G

# Single-threaded per job; parallelism comes from the array.
#SBATCH --cpus-per-task=4

# Log files: one per array task
#SBATCH -o ./SlurmOut/%A_%a_%x.out

# =============================================================================
# Environment
# =============================================================================

# Adapt these two lines to your cluster setup.
source /home/erosanow_hpc/informed_swapping/.SKQD/bin/activate
cd /home/erosanow_hpc/informed_swapping/ClusterStudy

# =============================================================================
# Run this chunk
# =============================================================================

echo "=================================================="
echo "Array job  : $SLURM_ARRAY_JOB_ID"
echo "Task (chunk): $SLURM_ARRAY_TASK_ID / 29"
echo "Node       : $(hostname)"
echo "Started    : $(date)"
echo "=================================================="

python -u run_chunk.py \
    --chunk          "$SLURM_ARRAY_TASK_ID" \
    --n_chunks       30                     \
    --param_file     params/param_grid.txt  \
    --n_random_paths 100                    \
    --seed           42                     \
    --output_dir     results

echo "=================================================="
echo "Finished   : $(date)"
echo "=================================================="
