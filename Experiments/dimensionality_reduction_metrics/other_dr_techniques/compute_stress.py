import numpy as np
import numpy.linalg as LA
import math as m
import os
import argparse
import pandas as pd
from time import time
from datetime import datetime
from tqdm import tqdm
from multiprocessing import cpu_count,Pool, Lock
from share_array.share_array import get_shared_array, make_shared_array

lock=Lock()

def parse_arguments():

    parser=argparse.ArgumentParser(description="Compute Stress from emebddings")

    parser.add_argument('--save_dir','-s', help='Path to the directory where the results are to be saved')
    parser.add_argument('--embed_dir','-e',help='Directory where embeddings are stored')
    parser.add_argument('--file_name', '-f', help='Name of the excel file where results are to be updated/File name of the excel file to be created', default='stress_results')
    parser.add_argument('--dataset','-d',help='Name of the dataset whose stress is to be computed')
    parser.add_argument('--data_dir', help='Directory where the datasets are stored', default='../normalized_data' )
    parser.add_argument('--dist_dir', help='Directory where distance matrices are saved', default='../norm_dist_matrices')
    parser.add_argument('--dr_tech', help='Name of the DR technique')
    parser.add_argument('--setting', help='Which setting is being used', default='def')
    parser.add_argument('--dr_args', help='Arguments of the dr technique', default=None)
    parser.add_argument('--target_dims', help='List of target dimension values', nargs='+')


    args=parser.parse_args()
    return args

def stress(dist_matrix:np.ndarray,Z:np.ndarray, worker_desc:str,worker_id:int):
    n=Z.shape[0]
    c=0
    stress=0
    p=n*(n-1)*0.5

    pbar = tqdm(total=p, desc=f'Stress {worker_desc}',position=worker_id)
    for i in range(n):
        for j in range(i):
            d2ij=Z[i]-Z[j]
            stress+=(dist_matrix[i,j]-LA.norm(d2ij))**2
            c+=1
            pbar.update(1)
    
    sum_sq=np.sum(dist_matrix**2)
    return m.sqrt(stress/sum_sq)


def compute_stress(dataset:str, DIST_DIR:str, SAVE_DIR:str, EMBED_DIR:str, file_name:str, worker_id:int, target_dim:int,dr_technique:str,setting:str):

    global lock

    dist_matrix=get_shared_array('dist_matrix')

    if setting=='def':
        Z=np.load(os.path.join(EMBED_DIR, dataset, f'{dataset}_{target_dim}_{dr_technique}.npy'))
    else:
        Z=np.load(os.path.join(EMBED_DIR, dataset, dr_technique, f'{dataset}_{target_dim}_{setting}.npy'))

    worker_desc=f'{dataset}_{setting}_{target_dim}'

    stress_val=stress(dist_matrix, Z, worker_desc, worker_id)

    lock.acquire()
    if not os.path.exists(os.path.join(SAVE_DIR,f'{file_name}.xlsx')):

        column_names=['Timestamp', 'Dataset', 'Setting', 'Target Dimension', 'Stress']

        df=pd.DataFrame(columns=column_names)
        df.to_excel(os.path.join(SAVE_DIR,f'{file_name}.xlsx'), index=False)
    
    new_row=[datetime.now().strftime("%Y-%m-%d %H:%M:%S"), dataset, setting, target_dim,stress_val]

    result_sheet=pd.read_excel(os.path.join(SAVE_DIR,f'{file_name}.xlsx'))

    result_sheet.loc[len(result_sheet.index)]=new_row
    result_sheet.to_excel(os.path.join(SAVE_DIR,f'{file_name}.xlsx'), index=False)
    lock.release()


def main():

    args=parse_arguments()
    target_dims=[int(x) for x in args.target_dims]

    dist_matrix=np.load(os.path.join(args.dist_dir, f'{args.dataset}.npy'))

    make_shared_array(dist_matrix,'dist_matrix')

    num_cores=cpu_count()
    pool=Pool(processes=num_cores)

    results=[pool.apply_async(compute_stress, args=(args.dataset, args.dist_dir,args.save_dir, args.embed_dir, args.file_name,i, target_dims[i], args.dr_tech, args.setting)) for i in range(len(target_dims))]

    for result in results:
        result.wait()
    
    pool.close()
    pool.join()

if __name__=="__main__":
    main()

