#!/bin/bash
#SBATCH --job-name=paper_figures
#SBATCH --partition=intelsr_long
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/paper_figures_%j.log
module load Python/3.10.4-GCCcore-11.3.0
source /home/erosanow_hpc/informed_swapping/.SKQD/bin/activate
python PaperFigures.py --data equal_bitstrings_heisenberg_results_balanced_runs.csv --output equal_bitstrings_plots_heisenberg_balanced_runs/PaperFigures
