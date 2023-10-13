import sys
sys.path.append('../../')
from DiffRed import DiffRed
import os
import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler, normalize
import math as m

def pseudoinverse(A:np.ndarray):
    return (LA.inv(A.T@A))@A.T


def scale_data(X:np.ndarray)->tuple[StandardScaler, np.ndarray]:
    scaler=StandardScaler()
    B=scaler.fit_transform(X.T)
    B=B.T
    norms=LA.norm(B,axis=1)
    B=normalize(B, norm='l2')
    return scaler, norms, B

def inverse_norm(X:np.ndarray, norms:np.ndarray):
    return (X.T*norms).T

def inverse_scale(scaler:StandardScaler, B:np.ndarray):
    
    X=scaler.inverse_transform(B.T)
    X=X.T
    return X

def load_dataset(save_dir:str,dataset:str):
    X=np.load(os.path.join(save_dir, dataset, 'X.npy'))
    return X

def mse(A:np.ndarray, A_res:np.ndarray):
    return np.mean((A-A_res)**2)