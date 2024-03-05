import numpy as np
import numpy.linalg as LA
import math as m
from tqdm import tqdm



def distance_matrix(A:np.ndarray,sample:int, worker_id:int, dataset: str=None) -> np.ndarray :
    m,n=A.shape
    examples= sample if sample else m
    D=np.zeros((examples,m))
    progress_bar_id = f"Computing {dataset}"
    dataset_length = examples
    progress_bar = tqdm(total=dataset_length, desc=progress_bar_id, position=worker_id)
    for i in range(examples):
        for j in range(m):
            D[i,j]=LA.norm(A[i,:]-A[j,:])
        progress_bar.update(1)
    progress_bar.close()
    return D







