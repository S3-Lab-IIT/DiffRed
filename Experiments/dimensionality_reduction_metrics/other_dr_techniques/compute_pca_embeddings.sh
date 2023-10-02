#!/bin/bash

#PASS DATASET NAME AS CLI ARGUMENT $1
# PASS target dimension values as CLI ARGUMENT $2

python3 compute_PCA.py -d $1 -t $2


