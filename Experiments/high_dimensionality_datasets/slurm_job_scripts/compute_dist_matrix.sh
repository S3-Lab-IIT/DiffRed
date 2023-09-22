#!/bin/sh

#SBATCH --job-name=dist_matrix
#SBATCH --partition=hm
#SBATCH --cpus-per-task=32

python3 compute_distance_matrix.py -d DIV2k