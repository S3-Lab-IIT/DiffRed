import sys,os
sys.path.append('../..')

from DiffRed import DiffRed as dr
from DiffRed.utils import stress
import numpy as np
import math as m
from utils import data_loader,data_scaler
from metrics import *
import os
import pandas as pd
from tqdm import tqdm

datasets=os.listdir('./datasets')

Stress=[]

for dataset in tqdm(datasets, desc='Processing'):
    x,y=data_loader(dataset)
    A=data_scaler(x)
    DiffRed=dr(k1=1,k2=1,opt_metric='stress')
    Z=DiffRed.fit_transform(A)
    Stress.append(espadoto_stress(A,Z))
    # print(f'{dataset} done ')

results=pd.DataFrame(list(zip(Stress)),index=datasets,columns=['Stress'])

results.to_excel('./results/espadoto_stress.xlsx')