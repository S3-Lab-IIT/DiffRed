import numpy as np
import numpy.linalg as LA
import math as m
import os
import argparse
import pandas as pd
from time import time
from datetime import datetime
from metrics import distance_matrix
from tqdm import tqdm
from multiprocessing import cpu_count,Pool, Lock

lock=Lock()

def parse_arguments():

    parser=argparse.ArgumentParser(description='Compute the stable rank of a dataset')

    parser.add_argument('--datasets', '-d', nargs='+', help='List of datasets')
    parser.add_argument('--sing_dir', help='Directory where singular values are stored', default='./norm_singular_values')
    parser.add_argument('--save_dir', help='Directory where results are to be saved', default='./results')
    parser.add_argument('--file_name', help='Name of the excel file (without extension)', default='stable_rank')

    args=parser.parse_args()
    return args


def stable_rank(sigma:np.ndarray):
    return np.sum(sigma**2)/sigma[0]**2

def compute_stable_rank(dataset:str, SING_DIR:str, SAVE_DIR:str, file_name:str):

    global lock

    
    if not os.path.exists(os.path.join(SING_DIR,f'{dataset}.npy')):
        print(f"Singular values for {dataset} are not pre-computed, please run compute_spectral_plots.py first")
    
    sigma=np.load(os.path.join(SING_DIR,f'{dataset}.npy'))

    sr=stable_rank(sigma)
    
    lock.acquire()
    if not os.path.exists(os.path.join(SAVE_DIR, f'{file_name}.xlsx')):

        column_names=['Timestamp', 'Dataset', 'Stable Rank']

        df=pd.DataFrame(columns=column_names)

        df.to_excel(os.path.join(SAVE_DIR, f'{file_name}.xlsx'), index=False)

    new_row=[datetime.now().strftime("%Y-%m-%d %H:%M:%S"), dataset, sr]

    result_sheet=pd.read_excel(os.path.join(SAVE_DIR, f'{file_name}.xlsx'))

    result_sheet.loc[len(result_sheet.index)]=new_row

    result_sheet.to_excel(os.path.join(SAVE_DIR, f'{file_name}.xlsx'), index=False)
    lock.release()

    print(f'{dataset} done!')


def main():
    args=parse_arguments()
    num_cores=cpu_count()
    pool=Pool(processes=num_cores)

    results=[pool.apply_async(compute_stable_rank, args=(args.datasets[i],args.sing_dir,args.save_dir,args.file_name)) for i in range(len(args.datasets))]

    for result in results:
        result.wait()

    pool.close()
    pool.join()


if __name__=="__main__":
    main()
    