#!/bin/bash

# $1: Space seperated datasets, eg: "Bank Cifar10 FMnist Reuters30k geneRNASeq"
# $2: K1 values seperated by space
# $3: K2 values corresponding to the K1 values seperated by space
# See below for typical values

datasets=$1
k1=$2
k2=$3



#Typical k1,k2:

# k1="0 2 3 4 5 6 7 0 2 4 5 6 8 10 12 15 18 0 3 5 8 10 12 15 18 20 25 27 0 4 5 8 10 15 16 20 25 30 35"
# k2="10 8 7 6 5 4 3 20 18 16 15 14 12 10 8 5 2 30 27 25 22 20 18 15 12 10 5 3 40 36 35 32 30 25 24 20 15 10 5"

# FOR CIFAR:
# k1="0 2 3 4 5 6 7 0 3 5 6 7 8 9 10 12 0 2 4 5 8 10 12 15 18 0 3 5 8 12 15 18 20 25 27 0 4 5 8 10 15 16 20 25 30 35"
# k2="10 8 7 6 5 4 3 15 12 10 9 8 7 6 5 3 20 18 16 15 12 10 8 5 2 30 27 25 22 18 15 12 10 5 3 40 36 35 32 30 25 24 20 15 10 5"

#FOR BANK:
# k1="0 0 1 0 1 2 0 1 2 3 4 0 1 2 3 4 5 0 1 2 3 4 5 6 0 2 3 4 5 6 7 0 2 4 5 7 8"
# k2="1 2 1 3 2 1 5 4 3 2 1 6 5 4 3 2 1 7 6 5 4 3 2 1 8 6 5 4 3 2 1 10 8 6 5 3 2"

total_iterations=$(( ${#datasets[@]}))

update_progress_description() {
    echo -ne "r\ \nComputing M1 $1..."
}


for dataset in $datasets; do

    update_progress_description $dataset

    if [ $dataset == "Cifar10" ]; then
        python3 compute_m1.py --file_name "${dataset}" --dataset "${dataset}" --max_iter_list False --k1 $k1 --k2 $k2 --max_iter 100
    
    else 
        python3 compute_m1.py --file_name "${dataset}" --dataset "${dataset}" --max_iter_list False --k1 $k1 --k2 $k2 --max_iter 100
    fi

    progress_percentage=$((progress_percentage+1))

    progress_bar_length=$((progress_percentage*40/total_iterations))
        printf "[%-40s] %d%%" $(printf "#%.0s" {1..40}) $progress_percentage

    sleep 0.1

done

echo ""
echo "All computations finished"
echo ""
