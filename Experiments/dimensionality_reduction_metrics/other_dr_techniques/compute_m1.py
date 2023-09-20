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
from itertools import product
from settings import SETTINGS
import sys

lock= Lock()

def parse_arguments():

    parser=argparse.ArgumentParser(description='Compute M1 metric for other dr techniques')

    parser.add_argument('--save_dir','-s', help='Path to the directory where the results are to be saved')
    parser.add_argument('--embed_dir','-e',help='Directory where embeddings are stored')
    parser.add_argument('--file_name', '-f', help='Name of the excel file where results are to be updated/File name of the excel file to be created', default='m1_results')
    parser.add_argument('--dataset','-d',help='Name of the dataset whose M1 is to be computed')
    parser.add_argument('--data_dir', help='Directory where the datasets are stored', default='../normalized_data' )
    # parser.add_argument('--dist_dir', help='Directory where distance matrices are saved', default='../norm_dist_matrices')
    parser.add_argument('--dr_tech', help='Name of the DR technique')
    parser.add_argument('--setting', help='Which setting is being used', default='def')
    parser.add_argument('--dr_args', help='Arguments of the dr technique', default=None)
    parser.add_argument('--target_dims', help='List of target dimension values', nargs='+')


    args=parser.parse_args()
    return args


def fr_sq(A:np.ndarray):
    return LA.norm(A, ord='fro')**2

def compute_m1(dataset:str, DATA_DIR:str, SAVE_DIR: str, EMBED_DIR: str, file_name: str, target_dim:int, dr_technique:str, setting:str):
    
    global lock
    if dr_technique=='PCA':
        if setting=='def':
            Z=np.load(os.path.join(EMBED_DIR, dataset, f'{dataset}_{target_dim}_pca.npy'))
        else:
            Z=np.load(os.path.join(EMBED_DIR, dataset, dr_technique, f'{dataset}_{target_dim}_{setting}.npy'))
    else:
        Z=np.load(os.path.join(EMBED_DIR,dataset,dr_technique,f'{dataset}_{target_dim}_{setting}.npy'))
    A=np.load(os.path.join(DATA_DIR,dataset, 'X.npy'))

    m1_val= abs(1-(fr_sq(Z)/fr_sq(A)))

    save_file=os.path.join(SAVE_DIR, f'{file_name}.xlsx')

    lock.acquire()
    if not os.path.exists(save_file):

        column_names=['Timestamp', 'Dataset', 'Setting', 'Target Dimension', 'M1']

        df= pd.DataFrame(columns=column_names)
        df.to_excel(save_file, index=False)
    
    new_row=[datetime.now().strftime("%Y-%m-%d %H:%M:%S"),dataset,setting, target_dim, m1_val]

    result_sheet=pd.read_excel(save_file)
    
    result_sheet.loc[len(result_sheet.index)]=new_row

    result_sheet.to_excel(save_file, index=False)

    lock.release()


def main():
    args=parse_arguments()
    target_dims=[int(x) for x in args.target_dims]

    num_cores=cpu_count()
    pool=Pool(processes=num_cores)

    if not args.setting=='all':

        results=[pool.apply_async(compute_m1, args=(args.dataset, args.data_dir,args.save_dir, args.embed_dir, args.file_name, target_dims[i], args.dr_tech, args.setting)) for i in range(len(target_dims))]
    else:

        all_settings=[k for k in SETTINGS[args.dr_tech].keys()]
        combinations=product(target_dims, all_settings)

        results=[pool.apply_async(compute_m1, args=(args.dataset,args.data_dir, args.save_dir, args.embed_dir, args.file_name, target_dim, args.dr_tech, setting)) for target_dim, setting in combinations]

    for result in results:
        result.wait()
    
    pool.close()
    pool.join()

if __name__=="__main__":
    main()