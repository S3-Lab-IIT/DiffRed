# <i>DiffRed</i>: Dimensionality Reduction guided by stable rank
<p>
    <a href="https://badge.fury.io/py/diffred">
        <img src="https://badge.fury.io/py/diffred.svg">
    </a>
    <a href="https://www.python.org/">
        <img alt="Build" src="https://img.shields.io/badge/Python-3.9-3776AB.svg?style=flat&logo=python&logoColor=white">
    </a>
    <a href="https://creativecommons.org/licenses/by-nc-sa/4.0/">
        <img alt="Build" src="https://img.shields.io/badge/License-CC_BY--NC--SA_4.0-lightgrey.svg">
    </a>
    <a href="https://arxiv.org/abs/2403.05882">
        <img src="https://img.shields.io/badge/cs.ML-2403.05882-b31b1b?logo=arxiv&logoColor=red">
    </a>
</p>

This is the official repository containing the code for the experiments of our AISTATS 2024 paper [<b><i>DiffRed</i>: Dimensionality Reduction guided by stable rank</b>](https://arxiv.org/abs/2403.05882)


## Setup
The <i>DiffRed</i> package maybe installed either from PyPI or from the source.

PyPI Installation:

```bash
  pip install diffred 
```

Installation from source:

```bash
git clone https://github.com/S3-Lab-IIT/DiffRed.git

cd DiffRed

pip install -r requirements.txt

pip install -e .
```

## Using the parallel stress package
Currently, the parallel stress implementation can only be used by installing from source.

```bash
git clone https://github.com/S3-Lab-IIT/DiffRed.git

cd DiffRed
```

After cloning, parallel stress can be imported like a regular package. 

### Example Usage

```python
from parallel_stress import stress as pstress
from share_array.share_array import get_shared_array, make_shared_array
import numpy as np
from Experiments.dimensionality_reduction_metrics.metrics import distance_matrix
from DiffRed import DiffRed

n=100
D=50

data=np.random.normal(size=(n,D))
dist_matrix=distance_matrix(data,None,None,None)
dr=DiffRed(k1=5,k2=5)
embedding_matrix=dr.fit_transform(data)

make_shared_array(dist_matrix, 'dist_matrix')
make_shared_array(embedding_matrix, name='embedding_matrix')

stress=pstress('dist_matrix', 'embedding_matrix')

print(f"Stress: {stress}")

```

## Reproducing Experiment Results

For reproducing our experiment results, refer to the scripts in the `Experiments/` directory. 

### Low and Moderate Dimensional Datasets

For datasets having low to moderate dimensionality, the code can be run with relatively less memory and GPU. The datasets used in our experiments, which fall under this category are: 

| **Dataset** 	| $\mathbf{D}$ 	| $\mathbf{n}$ 	| **Stable Rank** 	| **Domain** 	|
|-------------	|--------------	|--------------	|-----------------	|------------	|
| Bank        	| 17           	| 45K          	| 1.48            	| Finance    	|
| hatespeech  	| 100          	| 3.2K         	| 11.00           	| NLP        	|
| FMNIST     	| 784          	| 60K          	| 2.68            	| Image      	|
| Cifar10     	| 3072         	| 50K          	| 6.13            	| Image      	|
| geneRNASeq  	| 20.53K       	| 801          	| 1.12            	| Biology    	|
| Reuters30k  	| 30.92K       	| 10.7K        	| 14.50           	| NLP        	|


The experiments related to these datasets can be run from the scripts available in the `Experiments/dimensionality_reduction_metrics` directory. 

### Grid Search Experiments

<b>Note:</b> Before proceeding, make sure that all bash scripts have executable permission. Use the following command:

```bash
chmod u+x script-name.sh
```

To run the grid search experiments for <i>Stress</i> and <i>M1</i> metrics, follow these steps:

Create the required subdirectories by running :

```bash
./create_subdirectories
```

Now download and preprocess the dataset:

```bash
./prepare_datasets dataset
```
Here, a list of datasets may also be provided as CLI argument (dataset names seperated by space). Ensure that the dataset names are the same as the names in the table above (case-sensitive). 

Next, compute the distance matrices of the datasets:

```bash
./compute_distance_matrix dataset
```

Now, compute the <i>DiffRed</i> embeddings:

```bash
./compute_embeddings [dataset-name] [list of k1 values seperated by space] [list of k2 values seperated by space] 100
```

Now, compute <i>Stress</i> and <i>M1</i> distortion using:

```bash
./compute_stress [dataset-name] [save-as] [list of k1 values seperated by space] [list of k2 values seperated by space] 100
```

```bash
./compute_m1 [dataset-name] [save-as] [list of k1 values seperated by space] [list of k2 values seperated by space] 100
```

For using the same $k_1$ and $k_2$ that we used in the paper, refer to the excel sheet in the `results/M1_results/` and the `results/Stress_results` directories. 

### Comparison with other dimensionality reduction algorithms

The scripts to compute the stress using other dimensionality reduction algorithms (PCA, RMap, Kernel PCA, Sparse PCA, t-SNE, UMap) are in the `Experiments/dimensionality_reduction_metrics/other_dr_techniques` directory. 

The `compile_results` script compiles the best values of the grid search results into an excel file for all dimensionality reduction techniques (including <i>DiffRed</i>). 

### Running custom experiments/Extending Research

The repository was developed to allow adding new datasets and dimensionality reduction algorithms, and to provide customizability for extending our research. For running experiments with custom datasets/settings, the python scripts can be run by specifying the datasets/other settings via CLI arguments. To view the utility of a particular script, use the help option of the python script in the command line:

```bash
python3 <script-name>.py --help
```

#### Adding a new dataset

A new dataset, may be added to the repository by adding a corresponding data class (inherited from the `Datasets.Dataset`) to the `Datasets.py` file. Then, the `get_datasets.py` file needs to be updated by adding the download url and the data class object to the `url` and `objs` dictionary.

#### Adding a new dimensionality reduction algorithm

A new dimensionality reduction algorithm can be added to the repository by implementing it as a function in `other_dr_techniques/dr_techniques.py` and adding the initial values of hyperparameters to `other_dr_techniques.settings.SETTINGS`.

### High and Very High Dimensionality Datasets

For datasets having high and very high dimensionality, more memory and GPU may be required. For such datasets, we used a shared commodity cluster. The following datasets from our paper fall in this category:

| **Dataset** 	| $\mathbf{D}$ 	| $\mathbf{n}$ 	| **Stable Rank** 	| **Domain**     	|
|-------------	|--------------	|--------------	|-----------------	|----------------	|
| APTOS 2019  	| 509K         	| 13K          	| 1.32            	| Healthcare     	|
| DIV2K       	| 6.6M         	| 800          	| 8.39            	| High Res Image 	|


The experiment scripts for these datasets can be found at `Experiments/high_dimensionality_datasets/`. Slurm job scripts have been provided to facilitate usage in HPC environments. The usage is similar to what is described above for low dimensionality datasets.

### Reproducing Plots

To reproduce the plots provided in the paper and the supplementary material, use the `make_plots.ipynb` at `Experiments/dimensionality_reduction_metrics`. For the stable rank plots (Figure 11, Figure 13) and the spectral value plots (Figure 12), use the `plot_stable_rank.py` and the `compute_spectral_plot.py` scripts. 


## Experiment Results

The results of our experiments may be obtained from `Experiments/dimensionality_reduction_metrics/results` and `Experiments/high_dimensionality_datasets/results/` directories. For the low and moderate dimensionality datasets, refer to the the subdirectories `Stress_results` and `M1_results` for the full grid search results. For the high dimensionality datasets, refer to the excel files `aptos2019_M1_results.xlsx`, `aptos2019_Stress_results.xlsx`, `DIV2k_M1_results.xlsx`, and `DIV2k_Stress_results.xlsx` for the full grid search results. For both, refer to the subdirectory `other_dr_techniques` for results of other dimensionality reduction algorithms and refer to the `compiled_results` subdirectory for a comparitive summary of the grid search experiments.

## Citations
Please cite the paper and star this repository if you use <i>DiffRed</i> and/or find it interesting. Queries regarding the paper or the code may be directed to [prarabdhshukla@iitbhilai.ac.in](mailto:prarabdhshukla@iitbhilai.ac.in). Alternatively, you may also open an issue on this repository.

```bibtex
@misc{shukla2024diffred,
      title={DiffRed: Dimensionality Reduction guided by stable rank}, 
      author={Prarabdh Shukla and Gagan Raj Gupta and Kunal Dutta},
      year={2024},
      eprint={2403.05882},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```



