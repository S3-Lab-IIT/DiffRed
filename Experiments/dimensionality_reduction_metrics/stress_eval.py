import sys, os
sys.path.append('../..')

from DiffRed import DiffRed as dr
from DiffRed.utils import stress,opt_dimensions
import numpy as np
import math as m
from utils import data_loader,data_scaler
import os
import pandas as pd
from tqdm import tqdm 

datasets=os.listdir('./datasets')

exceptions=['Reuters80k', 'ElectricityLoadDiagrams']

datasets=[item for item in datasets if item not in exceptions]

for dataset in tqdm(datasets):
    x,y=data_loader(dataset)
    A=data_scaler(x)
    k1,k2=opt_dimensions(A,)