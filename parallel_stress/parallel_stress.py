import numpy as np
from multiprocessing import Pool, cpu_count
from share_array.share_array import get_shared_array, make_shared_array
from time import time
import math as m

def compute_partial_sum(dist_matrix:str, embedding:str, start, end):
    array1=get_shared_array(dist_matrix)
    array2=get_shared_array(embedding)
    partial_sum = 0.0
    for i in range(start, end):
        for j in range(i):
            partial_sum += (array1[i][j] - np.linalg.norm(array2[i] - array2[j]))**2
    return partial_sum

def stress(dist_matrix:str, embedding:str):

    num_cores=cpu_count()

    Z=get_shared_array(embedding)
    Dist_matrix=get_shared_array(dist_matrix)
    n=Z.shape[0]
    
    if n%num_cores==0:
        chunk_size = n // (num_cores)
        chunks = [(i * chunk_size, (i + 1) * chunk_size) for i in range(num_cores)]
    else:
        chunk_size = n // (num_cores)
        chunks = [(i * chunk_size, (i + 1) * chunk_size) for i in range(num_cores)]

        chunks.append((chunks[-1][1], n))

    pool = Pool(processes=num_cores)

 
    partial_sums = pool.starmap(
        compute_partial_sum,
        [(dist_matrix, embedding,start, end) for start, end in chunks],
    )


    pool.close()
    pool.join()

    final_sum = sum(partial_sums)
    sum_sq=np.sum(Dist_matrix**2)
    stress=m.sqrt(final_sum/sum_sq)
    
    return stress