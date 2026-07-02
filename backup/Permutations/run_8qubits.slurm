#!/bin/bash

#SBATCH --job-name=Permutations

#SBATCH --export=NONE

#SBATCH --account=ag_hiskp_funcke

#SBATCH --partition=intelsr_medium

#SBATCH --time=01-00:00:00

#SBATCH --mem=900G

#SBATCH --cpus-per-task=96

#SBATCH -o ./SlurmOut/%j_%x.out

# Optional: load your Python module here (adapt to your cluster)
# module load python/3.10

source /home/erosanow_hpc/informed_swapping/.SKQD/bin/activate

# Move to project directory
cd /home/erosanow_hpc/informed_swapping

# Run the 8-qubit SKQD script
python -u Permutations/Test_8qubits.py
