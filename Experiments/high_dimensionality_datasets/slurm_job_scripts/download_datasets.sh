#!/bin/sh

#SBATCH --job-name=div2k
#SBATCH --partition=compute

# PASS THE DATASET NAME AS COMMAND LINE ARGUMENT
module load python 
python3 get_datasets.py -d $1