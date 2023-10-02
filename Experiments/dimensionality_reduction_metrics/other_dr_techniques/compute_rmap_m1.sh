#!/bin/bash

datasets="Cifar10 FMnist Reuters30k geneRNASeq hatespeech"

common_t_value="10 20 30 40 50"
bank_t_value="1 2 3 5 6 7 8 10"


total_iterations=$(( ${#datasets[@]}))

update_progress_description() {
    echo -ne "r\ \nComputing M1 $1..."
}

save_dir="../results/other_dr_techniques/RMap"
embed_dir="./rmap_embeddings"

for dataset in $datasets; do

    update_progress_description $dataset

        if [ $dataset == "Bank" ]; then
        python3 compute_m1.py --save_dir $save_dir --embed_dir $embed_dir --file_name m1_results_RMap --dataset $dataset --dr_tech RMap --setting def --target_dims $bank_t_value --dr_args 20
    else
       python3 compute_m1.py --save_dir $save_dir --embed_dir $embed_dir --file_name m1_results_RMap --dataset $dataset --dr_tech RMap --setting def --target_dims $common_t_value --dr_args 20
    fi

    progress_percentage=$((progress_percentage+1))

    progress_bar_length=$((progress_percentage*40/total_iterations))
        printf "[%-40s] %d%%" $(printf "#%.0s" {1..40}) $progress_percentage

    sleep 0.1

done

echo ""
echo "All computations finished"
echo ""