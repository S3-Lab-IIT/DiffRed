#!/bin/bash

#ARGUMENTS:
# $1: Dataset name 
# $2: Target dimensions for computing theoretical optimum
# 

python3 compute_spectral_plot.py -d $1 
echo "Spectral plots computed!"
python3 compute_stable_rank.py -d $1
echo "Stable rank computed!"
python3 compute_theoretical_opt.py -d $1 --target_dims $2
echo "Theoretical optimum computed"
python3 plot_stable_rank.py -d $1

