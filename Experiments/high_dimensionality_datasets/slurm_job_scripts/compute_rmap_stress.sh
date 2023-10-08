#!/bin/bash
#SBATCH --job-name=embeddings
#SBATCH --cpus-per-task=32
#SBATCH --partition=hm
# $1: Dataset
# $2: Target dimensions, typically "10 20 30 40"
# $3: File name
# $4: Eta value typically 20
python3 compute_stress.py -d $1 --embed_dir ./rmap_embeddings --file_name $3 --k1 None --k2 None --max_iter_list None --target_dims $2 --dr_tech RMap --dr_args $4