#!/bin/bash
#PASS DATASET NAME AS CLI ARGUMENT $1
# PASS K1 values as CLI ARGUMENT $2
# PASS K2 values as CLI ARGUMENT $3
# PASS max_iter values as CLI ARGUMENT $4

python3 compute_m1.py -d $1 --save_dir ./results/M1_results/ -f $2 --k1 $3 --k2 $4 --max_iter_list False --max_iter $5