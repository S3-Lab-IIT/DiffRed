#!/bin/bash
#SBATCH --job-name=stable_rank
#SBATCH --partition=hm

# PASS THE DATASET NAME AS COMMAND LINE ARGUMENT
python3 compute_stable_rank.py -d $1