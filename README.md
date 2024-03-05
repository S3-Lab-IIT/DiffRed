# <i>DiffRed</i>: Dimensionality Reduction guided by stable rank

This is the official repository containing the code for the experiments of our AISTATS 2024 paper <b><i>DiffRed</i>: Dimensionality Reduction guided by stable rank</b> 
<!-- This repo contains the python package for DiffRed implmentation as well as the code for all the experiments in the paper. 

### <b>To use the package:</b>
1. Create a conda environment: <br>
`conda create -n <env_name>`

2. Install the required libraries from requirements.txt:<br>
`pip install -r requirements.txt`

3. Import the DiffRed package and start using it!<br>
`from DiffRed import DiffRed`

An example of how to use the package is in the `example.ipynb` file. 

<b>NOTE:</b> When you want to fit and transform on the same dataset, it is advisable to use the `DiffRed.fit_transform()` method and the `DiffRed.fit()` and the `DiffRed.transform()` methods should seperately be used only when you are fitting one dataset and transforming on another. If you use `DiffRed.fit()` and `DiffRed.transform()` seperately on the same dataset, you would get the correct metric values, however the attributes of the DiffRed object will be stored under different names than while using `DiffRed.fit_transform()` (usually the same name but with a 'trans_' prefix). -->