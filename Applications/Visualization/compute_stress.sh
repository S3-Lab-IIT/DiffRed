#!/bin/bash
# datasets="Bank Cifar10 FMnist hatespeech geneRNASeq Reuters30k"
datasets="Reuters30k geneRNASeq hatespeech FMnist"
algorithms="DiffRed PCA"
common_t_value="10 20 30 40"
bank_t_value="3 5 6 7 8 10"


declare -A opt_k1
opt_k1["Bank"]="2 4 5 6 7 7"
opt_k1["Cifar10"]="3 8 15 20"
opt_k1["FMnist"]="5 8 15 15"
opt_k1["geneRNASeq"]="5 8 10 10"
opt_k1["hatespeech"]="1 5 12 20"
opt_k1["Reuters30k"]="0 2 2 4"


embed_dir=$1 # Default value for tsne, for tsne2, it is ./tsne2_embeddings/
save_dir=$2 # Default value for tsne, for tsne2, it is ./results/m1_matched_stress_results/
dist_dir="../../Experiments/dimensionality_reduction_metrics/norm_dist_matrices/"

total_iterations=$(( ${#datasets[@]} * ${#algorithms[@]} ))

update_progress_description() {
    echo -ne "Computing Stress $1 $2..."
}

for dataset in $datasets; do
    for algorithm in $algorithms; do
        update_progress_description $dataset $algorithm 

        # if [ ! -d "${save_dir}${algorithm}" ]; then
        #     echo "Creating save_dir for ${algorithm}"
        #     mkdir -p "${save_dir}${algorithm}"
        # fi
        k1="${opt_k1[$dataset]}"
        if [ $dataset == "Cifar10" ] || [ $dataset == "FMnist" ]; then
            python3 compute_stress.py -d $dataset --dist_dir $dist_dir --k1 $k1 --max_iter 100 --max_iter_list False --dr_tech $algorithm -t $common_t_value -u True --save_dir $2 --embed_dir $1
        
        elif [ $dataset == "Bank" ]; then
            python3 compute_stress.py -d $dataset --dist_dir $dist_dir --k1 $k1 --max_iter 100 --max_iter_list False --dr_tech $algorithm -t $bank_t_value --save_dir $2 --embed_dir $1
        else
           python3 compute_stress.py -d $dataset --dist_dir $dist_dir --k1 $k1 --max_iter 100 --max_iter_list False --dr_tech $algorithm -t $common_t_value --save_dir $2 --embed_dir $1
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