import numpy as np
import numpy.linalg as LA
import math as m
import os
import argparse
from metrics import *
from DiffRed.utils import opt_dimensions, stress
from tqdm import tqdm 
from multiprocessing import cpu_count, Pool

def parse_arguments():
    parser=argparse.ArgumentParser(description='Evaluate metrics',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--save_dir','-s',help='Dir where results are stored', default='./results')
    parser.add_argument('--datasets','-d',nargs='+',help='List of datasets')
    parser.add_argument('--target_dims', nargs='+', help='List of target dimensions')
    parser.add_argument('--stress_only', type=bool, default=True,help='Whether only stress will be calculated')
    parser.add_argument('--data_dir', default='./datasets', help='Directory of the dataset')
    parser.add_argument('--distance_matrix_cache',default='./distance_matrices',help='Dir where distance matrices are cached')

    args=parser.parse_args()
    return args


def stress(dist_path:str,embeddings:np.ndarray,worker_id:int,dataset:str):

    D=np.load(dist_path)
    n=embeddings.shape[0]
    stress=0
    sum_sq=0
    progress_bar_id = f"Computing Stress ({dataset})"
    dataset_length = n
    progress_bar=tqdm(total=dataset_length,desc=progress_bar_id,position=worker_id)
    for i in tqdm(range(n)):
        for j in range(i):
            d2ij=embeddings[i,:]-embeddings[j,:]
            dij=D[i,j]
            d=(dij-LA.norm(d2ij))**2
            sum_sq+=D[i,j]**2
            stress+=d
        progress_bar.update(1)
    progress_bar.close()
    stress/=sum_sq
    return np.sqrt(stress)



def main():
    args=parse_arguments()
    SAVE_DIR=args.save_dir
    DATA_DIR=args.data_dir
    datasets={}
    for i in range(len(args.datasets)):
        datasets[datasets[i]]=int(args.target_dims[i])

    num_cores=cpu_count()
    pool=Pool(processes=num_cores)

    progress_bars=[]

    results=[pool.apply_async(stress,args=(os.path.join(DATA_DIR,f'{dataset}','X.npy')))]

    



    