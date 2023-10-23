#!/bin/bash
# datasets="Bank Cifar10 FMnist Reuters30k geneRNASeq hatespeech"
datasets=$1

# algorithms="UMap T-SNE S-PCA K-PCA"
algorithms=$2

# common_t_value="10 20 30 40 50"
common_t_value="10 20 30 40"
bank_t_value="1 2 3 5 6 7 8 10"

save_dir="../results/other_dr_techniques/"
embed_dir="./embeddings/"

total_iterations=$(( ${#datasets[@]} * ${#algorithms[@]} ))

update_progress_description() {
    echo -ne "Computing Stress $1 $2..."
}

for dataset in $datasets; do
    for algorithm in $algorithms; do
        update_progress_description $dataset $algorithm 

        if [ ! -d "${save_dir}${algorithm}" ]; then
            echo "Creating save_dir for ${algorithm}"
            mkdir -p "${save_dir}${algorithm}"
        fi
        if [ ! $algorithm == "T-SNE" ]; then
            if [ $dataset == "Bank " ]; then
                python3 compute_stress.py --save_dir "${save_dir}${algorithm}" --embed_dir $embed_dir --file_name "stress_results_${algorithm}" --dataset $dataset --dr_tech $algorithm --setting all --target_dims $bank_t_value
            else
                python3 compute_stress.py --save_dir "${save_dir}${algorithm}" --embed_dir $embed_dir --file_name "stress_results_${algorithm}" --dataset $dataset --dr_tech $algorithm --setting all --target_dims $common_t_value
            fi
        else 

            if [ $dataset == "Bank " ]; then
                python3 compute_stress.py --save_dir "${save_dir}${algorithm}" --embed_dir $embed_dir --file_name "stress_results_${algorithm}" --dataset $dataset --dr_tech $algorithm  --setting all --target_dims 2
            
            else
                python3 compute_stress.py --save_dir "${save_dir}${algorithm}" --embed_dir $embed_dir --file_name "stress_results_${algorithm}" --dataset $dataset --dr_tech $algorithm --setting all --target_dims 2
            fi
        fi

        progress_percentage=$((progress_percentage+1))
        progress_bar_length=$((progress_percentage*40/total_iterations))
        printf "[%-40s] %d%%" $(printf "#%.0s" {1..40}) $progress_percentage
        
        sleep 0.1
    done

    echo "{$dataset} done"
done

echo ""
echo "All computations finished"

echo ""


    