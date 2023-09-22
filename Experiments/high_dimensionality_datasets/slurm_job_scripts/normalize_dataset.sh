#!/bin/sh

#SBATCH --job-name=normalize_div2k
#SBATCH --partition=compute

module load python 
python3 normalize_datasets.py -d DIV2k 