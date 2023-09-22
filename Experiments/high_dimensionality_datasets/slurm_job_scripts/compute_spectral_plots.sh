#!/bin/bash
#SBATCH --job-name=spectral_plots
#SBATCH --partition=hm
#SBATCH --cpus-per-task=32

python3 compute_spectral_plots.py -d DIV2k