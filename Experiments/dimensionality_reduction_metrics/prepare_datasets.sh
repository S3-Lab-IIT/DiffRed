#!/bin/bash

# PASS THE DATASET NAME AS THE CLI ARGUMENT

python3 get_datasets.py -d $1 
python3 normalized_datasets.py -d $1