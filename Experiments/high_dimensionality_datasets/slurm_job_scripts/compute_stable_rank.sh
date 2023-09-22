#!/bin/bash
#SBATCH --job-name=stable_rank
#SBATCH --partition=hm

python3 compute_stable_rank.py -d DIV2k