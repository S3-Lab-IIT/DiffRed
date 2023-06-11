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


datasets= os.listdir('./datasets')

M_T=[] #Trustworthiness
M_C=[] # Continuity
M_NH=[] #Neighborhood hit
M_S=[] #Shepard Goodness
M_sig=[] #Stress

for dataset in tqdm(datasets):
    x,y=data_loader(dataset)
    A=data_scaler(x)
    DiffRed=dr(1,1)
    Z=DiffRed.fit_transform(A)
    M_T.append(metric_trustworthiness(A,Z))
    M_C.append(continuity(A,Z,7,None))
    M_NH.append(neighborhood_hit(Z,y,7))
    M_S.append(shepard_goodness(A,Z))
    M_sig.append(stress(A,Z))
    # print('{} done '.format(dataset))

results=pd.DataFrame(list(zip(M_T,M_C,M_NH,M_S,M_sig)),index=datasets,columns=['Trustworthiness','Continuity','Neighborhood Hit','Shepard Goodness', 'Stress'])

results.to_excel('./results/scalar_metric_results.xlsx')
