#!/bin/sh

#SBATCH --job-name=dist_matrix
#SBATCH --partition=hm
#SBATCH --cpus-per-task=32

# PASS THE DATASET NAME AS COMMAND LINE ARGUMENT
python3 compute_distance_matrix.py -d $1