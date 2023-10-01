#!/bin/sh

#SBATCH --job-name=normalize_div2k
#SBATCH --partition=compute
# PASS THE DATASET NAME AS COMMAND LINE ARGUMENT
module load python 
python3 normalize_datasets.py -d $1 