#!/bin/bash
#SBATCH --job-name=embeddings
#SBATCH --cpus-per-task=32
#SBATCH --partition=hm

python3 compute_stress.py -d DIV2k --embed_dir ./rmap_embeddings --file_name rmap_stress_results --k1 None --k2 None --max_iter_list None --target_dims 10 20 30 40 --dr_tech RMap --dr_args 20