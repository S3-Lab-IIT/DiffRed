import numpy as np
import numpy.linalg as LA
import math as m
import os
import argparse
from tqdm import tqdm
from multiprocessing import cpu_count,Pool, Lock
from umap import UMAP
from tsnecuda import TSNE
from sklearn.decomposition import SparsePCA, KernelPCA
from MulticoreTSNE import MulticoreTSNE as mTSNE



def umap(X:np.ndarray, target_dim:int, setting:dict)->np.ndarray:

    reducer=UMAP(n_components=target_dim)

    for param,value in setting.items():
        setattr(reducer,param,value)
    
    return reducer.fit_transform(X)

def tsne(X:np.ndarray,setting:dict)->np.ndarray:

    # target_dim is hardcoded here because tsnecuda only supports target dimension 2
    tsne=TSNE(n_components=2)

    for param,value in setting.items():
        setattr(tsne, param, value)
    
    return tsne.fit_transform(X)

def s_pca(X:np.ndarray, target_dim: int, setting:dict)->np.ndarray:

    spca=SparsePCA(n_components=target_dim)

    for param,value in setting.items():
        setattr(spca,param,value)
    
    return spca.fit_transform(X)

def k_pca(X:np.ndarray, target_dim:int, setting:dict)->np.ndarray:

    kpca=KernelPCA(n_components=target_dim)

    for param,value in setting.items():
        setattr(kpca,param,value)
    
    return kpca.fit_transform(X)

def mtsne(X:np.ndarray,setting:dict)->np.ndarray:

    tsne=mTSNE(n_components=2)

    for param,value in setting.items():
        setattr(tsne, param, value)
    
    return tsne.fit_transform(X)


DR_MAPS={'UMap': umap, 'T-SNE': tsne, 'S-PCA': s_pca, 'K-PCA': k_pca, 'M-TSNE':mtsne}








