#!/bin/bash
#SBATCH --job-name=embeddings
#SBATCH --cpus-per-task=32
#SBATCH --partition=hm
# PASS THE DATASET NAME AS COMMAND LINE ARGUMENT

python3 compute_embeddings.py -d $1 --k1 0 1 2 3 4 5 6 7 0 2 3 4 5 8 10 12 15 18 0 2 3 5 8 10 12 15 18 20 25 27 0 2 4 5 8 10 15 16 20 25 30 35 11 --k2 10 9 8 7 6 5 4 3 20 18 17 16 15 12 10 8 5 2 30 28 27 25 22 20 18 15 12 10 5 3 40 38 36 35 32 30 25 24 20 15 10 5 19 --max_iter_list False --max_iter 100