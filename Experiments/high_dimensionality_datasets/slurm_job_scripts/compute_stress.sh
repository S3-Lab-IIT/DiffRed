#!/bin/bash
#SBATCH --job-name=embeddings
#SBATCH --cpus-per-task=32
#SBATCH --partition=hm

# PASS THE DATASET NAME AS COMMAND LINE ARGUMENT
# PASS K1 as CLI ARGUMENT $2
# PASS K2 as CLI ARGUMENT $3

python3 compute_stress.py -d $1 --k1 $2 --k2 $3 --max_iter_list False --max_iter 100 -f "${1}_stress_results" --dr_tech DiffRed