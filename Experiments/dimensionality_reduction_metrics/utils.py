import numpy as np
import os
from sklearn.preprocessing import StandardScaler, normalize

def data_scaler(A):
    scaler=StandardScaler()
    a=scaler.fit_transform(A)
    a=normalize(a,norm='l2')
    return a 

def data_loader(dataset: str):
    x=np.load(os.path.join('./datasets',dataset,'X.npy'))
    y=np.load(os.path.join('./datasets',dataset,'y.npy'))
    return x,y

