#!/bin/sh

#SBATCH --job-name=div2k
#SBATCH --partition=compute

module load python 
python3 div2k.py -d DIV2k 