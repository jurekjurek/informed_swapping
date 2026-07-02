#!/bin/bash
# =============================================================================
# Slurm array job: KRAB-vs-SKQD systematic study — DECAYING-Q variant
#
# Identical to run_array.slurm but KRAB uses a geometric decaying-Q schedule
#     Q(app) = max(1%·2ⁿ, int(20%·2ⁿ · 0.75**app))
# instead of a constant Q. Outputs go to separate locations so the fixed-Q
# study is left completely untouched:
#     figures -> figures_decay/
#     CSV     -> data/comparison_decay.csv
#
# Workflow
# --------
# 1. Generate the parameter file (once):
#       cd ClusterStudy/KRAB
#       python generate_jobs.py --preset full --mode decay
#
# 2. Submit 30 array jobs:
#       sbatch run_array_decay.slurm
#
# 3. Re-submit safely if some jobs time out — completed experiments are
#    skipped automatically (figure-existence check in run_chunk.py).
#
# After all jobs finish, produce the comparative analysis plots:
#       python plot_scaling.py \
#           --data_file comparison_decay.csv --out_dir analysis_decay
# =============================================================================

#SBATCH --job-name=KRABdecay

#SBATCH --export=NONE

#SBATCH --account=ag_hiskp_funcke

#SBATCH --partition=intelsr_medium

#SBATCH --array=0-29

# The SKQD sweep + path reconstruction dominates the runtime; n_qubits=10
# experiments take minutes each, so chained chunks need the full budget.
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
echo "Chunk       : $SLURM_ARRAY_TASK_ID / 29  (decaying-Q study)"
echo "Node        : $(hostname)"
echo "Started     : $(date)"
echo "=================================================="

python -u run_chunk.py \
    --chunk          "$SLURM_ARRAY_TASK_ID"      \
    --n_chunks       30                          \
    --param_file     params/param_grid_decay.txt \
    --Q_mode         decay                        \
    --seed           42                          \
    --figure_dir     figures_decay               \
    --data_dir       data                        \
    --n_random_paths 20

echo "=================================================="
echo "Finished    : $(date)"
echo "=================================================="
