#!/bin/bash
# datasets="Cifar10 FMnist Reuters30k geneRNASeq hatespeech"
datasets=$1
# algorithms="UMap T-SNE S-PCA K-PCA"
algorithms=$2

# common_t_value="10 20 30 40 50"
common_t_value="10 20 30 40"
bank_t_value="1 2 3 5 6 7 8 10"

datetime=$(date +"%Y%m%d_%H%M%S")
# log_file="./logs/log_${datetime}.txt"

total_iterations=$(( ${#datasets[@]} * ${#algorithms[@]} ))

update_progress_description() {
    echo -ne "\rComputing $1 $2..."
}

for dataset in $datasets; do
    for algorithm in $algorithms; do
        update_progress_description $dataset $algorithm
        # echo "python3 compute_embeddings.py -d \"$dataset\" -t \"$common_t_value\" --dr_tech \"$algorithm\""
        if [ $algorithm == "T-SNE" ]; then
                    if [ $dataset == "Reuters30k" ]; then
                        #using MultiCoreTSNE for Reuters30k because it is a big dataset
                        python3 compute_embeddings.py --dataset $dataset -t 2 --dr_tech M-TSNE --setting all
                    else
                        python3 compute_embeddings.py --dataset $dataset -t 2 --dr_tech $algorithm --setting all
                    fi  
        else
            python3 compute_embeddings.py --dataset $dataset -t $common_t_value --dr_tech $algorithm --setting all 
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
