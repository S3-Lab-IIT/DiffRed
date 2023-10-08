#!/bin/bash
#SBATCH --job-name=stable_rank_plots
#SBATCH --partition=hm
#SBATCH --cpus-per-task=32

# PASS THE DATASET NAME AS COMMAND LINE ARGUMENT

python3 plot_stable_rank.py -d $1 -c $2 --all_in_one $3