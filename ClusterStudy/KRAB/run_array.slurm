#!/bin/bash
# =============================================================================
# Slurm array job: KRAB systematic study
#
# Workflow
# --------
# 1. Generate the parameter file (once):
#       cd ClusterStudy/KRAB
#       python generate_jobs.py --preset full
#
# 2. Submit 30 array jobs:
#       sbatch run_array.slurm
#
# 3. Re-submit safely if some jobs time out — completed experiments are
#    skipped automatically (figure-existence check in run_chunk.py).
#
# After all jobs finish, aggregate results and produce scaling plots:
#       python plot_scaling.py
# =============================================================================

#SBATCH --job-name=KRAB

#SBATCH --export=NONE

#SBATCH --account=ag_hiskp_funcke

#SBATCH --partition=intelsr_medium

#SBATCH --array=0-29

# KRAB is cheap per iteration (no random paths), but n_qubits=10 with
# many iterations can still take a while when many experiments are chained.
#SBATCH --time=24:00:00

#SBATCH --mem=16G

#SBATCH --cpus-per-task=2

#SBATCH -o ./SlurmOut/%A_%a_%x.out

# =============================================================================
# Environment  — adapt these two lines
# =============================================================================
source /home/erosanow_hpc/informed_swapping/.SKQD/bin/activate
cd /home/erosanow_hpc/informed_swapping/ClusterStudy/KRAB

# =============================================================================
# Run chunk
# =============================================================================
echo "=================================================="
echo "Array job   : $SLURM_ARRAY_JOB_ID"
echo "Chunk       : $SLURM_ARRAY_TASK_ID / 29"
echo "Node        : $(hostname)"
echo "Started     : $(date)"
echo "=================================================="

python -u run_chunk.py \
    --chunk      "$SLURM_ARRAY_TASK_ID" \
    --n_chunks   30                     \
    --param_file params/param_grid.txt  \
    --seed       42                     \
    --figure_dir results                \
    --data_dir   data

echo "=================================================="
echo "Finished    : $(date)"
echo "=================================================="
