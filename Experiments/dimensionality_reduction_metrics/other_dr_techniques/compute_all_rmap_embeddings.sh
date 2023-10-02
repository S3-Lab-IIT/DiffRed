#!/bin/bash

datasets="Cifar10 Bank FMnist Reuters30k geneRNASeq hatespeech"

common_t_value="10 20 30 40 50"
bank_t_value="1 2 3 5 6 7 8 10"

total_iterations=$(( ${#datasets[@]}))

update_progress_description() {
    echo -ne "\rComputing RMap $1..."
}

for dataset in $datasets; do

    update_progress_description $dataset

    if [ $dataset == "Bank" ]; then
        python3 compute_rmap.py -d $dataset --target_dims $bank_t_value --eta 20
    else
        python3 compute_rmap.py -d $dataset --target_dims $common_t_value --eta 20
    fi

    progress_percentage=$((progress_percentage+1))

    progress_bar_length=$((progress_percentage*40/total_iterations))
        printf "[%-40s] %d%%" $(printf "#%.0s" {1..40}) $progress_percentage

    sleep 0.1

done

echo ""
echo "All computations finished"
echo ""

