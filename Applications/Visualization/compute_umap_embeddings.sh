datasets="Bank Cifar10 FMnist hatespeech geneRNASeq Reuters30k"
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

#choosing best settings
declare -A setting
setting["Bank"]="setting8"
setting["Cifar10"]="setting3"
setting["FMnist"]="setting5"
setting["geneRNASeq"]="setting5"
setting["hatespeech"]="setting8"
setting["Reuters30k"]="setting8"


pca_data_dir='../../Experiments/dimensionality_reduction_metrics/other_dr_techniques/pca_embeddings/'
dr_data_dir='../../Experiments/dimensionality_reduction_metrics/embeddings/'
max_iter='100'
total_iterations=$(( ${#datasets[@]} * ${#algorithms[@]} ))

update_progress_description() {
    echo -ne "Computing UMap embeddings $1 $2..."
}


for dataset in $datasets; do
    for algorithm in $algorithms; do
        update_progress_description $dataset $algorithm 
        k1="${opt_k1[$dataset]}"
        setting="${setting[$dataset]}"
        if [ $dataset == 'Bank' ]; then

            if [ $algorithm == 'PCA' ]; then
            
                python3 compute_umap_embeddings.py -d $dataset --data_dir $pca_data_dir -t $bank_t_value --dr_tech $algorithm --setting $setting 
            else
                python3 compute_umap_embeddings.py -d $dataset --data_dir $dr_data_dir -t $bank_t_value --k1 $k1 --dr_tech $algorithm --setting $setting --max_iter $max_iter --max_iter_list 'False'
            fi
        else 
            
            if [ $algorithm == 'PCA' ]; then
            
                python3 compute_umap_embeddings.py -d $dataset --data_dir $pca_data_dir -t $common_t_value --dr_tech $algorithm --setting $setting 
            else
                python3 compute_umap_embeddings.py -d $dataset --data_dir $dr_data_dir -t $common_t_value --k1 $k1 --dr_tech $algorithm --setting $setting --max_iter $max_iter --max_iter_list 'False'
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