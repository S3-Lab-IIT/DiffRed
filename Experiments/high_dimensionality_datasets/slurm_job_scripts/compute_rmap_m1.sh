#!/bin/bash
#SBATCH --job-name=embeddings
#SBATCH --cpus-per-task=32
#SBATCH --partition=hm

python3 compute_m1.py -d DIV2k --dr_tech RMap --target_dims 10 20 30 40 --embed_dir ./rmap_embeddings/ --file_name rmap_m1_results --dr_args 20