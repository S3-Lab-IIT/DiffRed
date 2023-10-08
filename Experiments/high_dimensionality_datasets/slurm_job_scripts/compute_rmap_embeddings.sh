#!/bin/bash
#SBATCH --job-name=embeddings
#SBATCH --cpus-per-task=32
#SBATCH --partition=hm
# $1: Dataset name
# $2: Target dimensions: Typically "10 20 30 40"
# $3: Number of RMaps to compute: typically 20


python3 compute_RMap.py -d $1 --target_dims $2 --eta $3