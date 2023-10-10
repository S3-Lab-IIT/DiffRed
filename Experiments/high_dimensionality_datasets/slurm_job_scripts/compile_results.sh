#!/bin/bash
#SBATCH --job-name=compile_results
#SBATCH --cpus-per-task=3
#SBATCH --partition=hm
# ENTER DATASETS AS THE FIRST CLI ARGUMENT. SEPERATED BY SPACE. 
# For example, in case of Reuters30k and Bank, write compile_results.sh "Reuters30k Bank"
# Second CLI argument is for specifying which metrics to compile. For both stress and m1, pass "Stress M1" as argument
# Third CLI is for specifying the dr_techniques to use. Specify "all" for all, else one at a time. 
# Use fourth CLI argument to specify the target dimensions
datasets=$1
metrics=$2
dr_tech=$3


total_iterations=$(( ${#datasets[@]} * ${#metrics[@]} ))

update_progress_description() {
    echo -ne "Compiling $1 $2..."
}

for dataset in $datasets; do
    for metric in $metrics; do
        update_progress_description $dataset $metric
        
        python3 compile_results.py -m $metric --dr_tech $3 -d $dataset --target_dims $4
        
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