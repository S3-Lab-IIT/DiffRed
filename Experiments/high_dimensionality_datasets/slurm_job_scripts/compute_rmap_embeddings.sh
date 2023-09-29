#!/bin/bash
#SBATCH --job-name=embeddings
#SBATCH --cpus-per-task=32
#SBATCH --partition=hm

python3 compute_RMap.py -d DIV2k --target_dims 10 20 30 40 --eta 20