#!/bin/bash
#SBATCH --job-name=embeddings
#SBATCH --cpus-per-task=32
#SBATCH --partition=hm

# $1: Dataset name
# $2: By default, "10 20 30 40"

python3 compute_PCA.py -d $1 --target_dims $2