#!/bin/bash
#PASS DATASET NAME AS CLI ARGUMENT $1
# PASS target dimension values as CLI ARGUMENT $2


echo "Computing Cifar10"

python3 compute_stress.py -s ../results/other_dr_techniques/pca -e ./pca_embeddings -d Cifar10 --dr_tech pca --target_dims 10 15 20 30 40 100 -f stress_results_PCA

echo "Computing FMnist"

python3 compute_stress.py -s ../results/other_dr_techniques/pca -e ./pca_embeddings -d FMnist --dr_tech pca --target_dims 10 20 30 40 -f stress_results_PCA

echo "Computing Reuters30k"

python3 compute_stress.py -s ../results/other_dr_techniques/pca -e ./pca_embeddings -d Reuters30k --dr_tech pca --target_dims 10 20 30 40 -f stress_results_PCA

echo "Computing Bank"
python3 compute_stress.py -s ../results/other_dr_techniques/pca -e ./pca_embeddings -d Bank --dr_tech pca --target_dims 1 2 3 5 6 7 8 10 -f stress_results_PCA

echo "Computing hatespeech"
python3 compute_stress.py -s ../results/other_dr_techniques/PCA -e ./pca_embeddings -d hatespeech --dr_tech PCA --target_dims 10 20 30 40 -f stress_results_PCA