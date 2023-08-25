#!/bin/bash

echo "Computing Cifar10"

python3 compute_PCA.py -d Cifar10 -t 15 20 30 40 100

echo "Computing FMNIST"

python3 compute_PCA.py -d FMnist -t 10 20 30 40 

echo "Computing Reuters"

python3 compute_PCA.py -d Reuters30k -t 10 20 30 40

echo "Computing Bank"

python3 compute_PCA.py -d Bank -t 1 2 3 5 6 7 8 10

echo "Done"
