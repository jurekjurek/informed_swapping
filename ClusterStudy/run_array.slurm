#!/bin/bash
# =============================================================================
# Slurm array job: BARK vs SKQD systematic study
#
# Usage
# -----
# 1. Generate the parameter file:
#       cd ClusterStudy
#       python generate_jobs.py --preset full
#
# 2. Check the total number of jobs printed by generate_jobs.py, then submit:
#       sbatch --array=0-<N-1> run_array.slurm
#
#    Or submit a limited number of simultaneous jobs (e.g. 50 at a time):
#       sbatch --array=0-<N-1>%50 run_array.slurm
#
# =============================================================================

#SBATCH --job-name=BARKvsSKQD

#SBATCH --export=NONE

#SBATCH --account=ag_hiskp_funcke

#SBATCH --partition=intelsr_medium

# Wall time: set generously; n_qubits=10 with 100 random paths can take >1h.
#SBATCH --time=04:00:00

# Memory: dominated by storing the full Hilbert space (dim x dim matrices for
# n_qubits=10 are 1024x1024 = ~8 MB each, but eigsh and path computation
# can spike to several GB). Adjust if jobs OOM.
#SBATCH --mem=32G

# Single-threaded per job — parallelism comes from the array.
#SBATCH --cpus-per-task=4

# Log files: one per array task, stored in SlurmOut/
#SBATCH -o ./SlurmOut/%A_%a_%x.out

# =============================================================================
# Environment setup
# =============================================================================

# Adapt the path below to wherever the virtual environment lives on your cluster.
source /home/erosanow_hpc/informed_swapping/.SKQD/bin/activate

# Move to the ClusterStudy directory so relative paths resolve correctly.
cd /home/erosanow_hpc/informed_swapping/ClusterStudy

# =============================================================================
# Read hyperparameters for this array task
# =============================================================================

PARAM_FILE="params/param_grid.txt"

# $SLURM_ARRAY_TASK_ID is 0-based; sed line numbers are 1-based.
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$PARAM_FILE")

if [ -z "$LINE" ]; then
    echo "ERROR: no parameter line for task ID $SLURM_ARRAY_TASK_ID" >&2
    exit 1
fi

# Parse the five whitespace-separated fields.
read -r N_QUBITS GS_SPARSITY HAM_SPARSITY OVERLAP KEEP_STATES <<< "$LINE"

echo "==================================================="
echo "Task ID      : $SLURM_ARRAY_TASK_ID"
echo "n_qubits     : $N_QUBITS"
echo "gs_sparsity  : $GS_SPARSITY"
echo "ham_sparsity : $HAM_SPARSITY"
echo "overlap      : $OVERLAP"
echo "keep_states  : $KEEP_STATES"
echo "==================================================="

# =============================================================================
# Run the experiment
# =============================================================================

python -u run_experiment.py \
    --n_qubits       "$N_QUBITS"     \
    --gs_sparsity    "$GS_SPARSITY"  \
    --ham_sparsity   "$HAM_SPARSITY" \
    --overlap        "$OVERLAP"      \
    --keep_states    "$KEEP_STATES"  \
    --n_random_paths 100             \
    --seed           42              \
    --output_dir     results
