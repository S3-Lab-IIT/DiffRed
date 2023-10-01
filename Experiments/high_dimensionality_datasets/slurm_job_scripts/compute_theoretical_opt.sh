#!/bin/bash
#SBATCH --job-name=theoretical_opt
#SBATCH --cpus-per-task=32
#SBATCH --partition=hm
# PASS THE DATASET NAME AS COMMAND LINE ARGUMENT
python3 compute_theoretical_opt.py -d $1 --target_dims $2
# python3 compute_bound_values.py -d DIV2k --target_dims $1