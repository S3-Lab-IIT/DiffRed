#!/bin/bash
#SBATCH --job-name=spectral_plots
#SBATCH --partition=hm
#SBATCH --cpus-per-task=32

# PASS THE DATASET NAME AS COMMAND LINE ARGUMENT

python3 compute_spectral_plots.py -d $1