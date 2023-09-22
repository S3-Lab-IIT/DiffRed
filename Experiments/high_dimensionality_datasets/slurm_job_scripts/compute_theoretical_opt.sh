#!/bin/bash
#SBATCH --job-name=theoretical_opt
#SBATCH --cpus-per-task=32
#SBATCH --partition=hm

python3 compute_theoretical_opt.py -d DIV2k --target_dims $1
# python3 compute_bound_values.py -d DIV2k --target_dims $1