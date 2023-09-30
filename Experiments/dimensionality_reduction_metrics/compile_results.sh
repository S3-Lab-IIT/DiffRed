#!/bin/bash

datasets="Bank Cifar10 FMnist Reuters30k geneRNASeq"
metrics="M1 Stress"

common_t_value="10 20 30 40"
bank_t_value="1 2 3 5 6 7 8 10"

total_iterations=$(( ${#datasets[@]} * ${#metrics[@]} ))

update_progress_description() {
    echo -ne "\rComputing $1 $2..."
}

for dataset in $datasets; do
    for metric in $metrics; do
        update_progress_description $dataset $metric

        if [ $dataset == "Bank" ]; then
            python3 compile_results.py -m $metric --dr_tech all -d $dataset --target_dims $bank_t_value
        else
            python3 compile_results.py -m $metric --dr_tech all -d $dataset --target_dims $common_t_value
        fi
        progress_percentage=$((progress_percentage + 1))
        progress_bar_length=$((progress_percentage * 40 / total_iterations))
        printf "[%-40s] %d%%" $(printf "#%.0s" {1..40}) $progress_percentage
        
        sleep 0.1
    done
    echo "{$dataset} done"
done
echo ""
echo "All computations finished."

echo ""

