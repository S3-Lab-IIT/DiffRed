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
from compute_theoretical_opt import calculate_stress_bound

lock=Lock()

def parse_arguments():
    
    parser = argparse.ArgumentParser(description='Compute the bound values and store them in a file')

    parser.add_argument('--dataset', '-d', help='Name of the dataset')

    parser.add_argument('--sing_dir',help='Directory where the singular values are stored', default='./norm_singular_values')

    parser.add_argument('--target_dims', nargs='+', help='Target Dimensions at which the bound values are to be calculated')

    parser.add_argument('--metric','-m', help='Metric whose theoretical bound is to be calculated', choices=['stress', 'm1'], default='stress')

    parser.add_argument('--save_dir', help='Directory where results are to be saved')

    # parser.add_argument('--file_name', help='File name of the CSV file in which results are stored (without extension)')

    args=parser.parse_args()
    return args


def compute_stress_vals(dataset:str, SAVE_DIR:str, SING_DIR:str, target_dim:int, worker_id:int):

    global lock

    if not os.path.exists(os.path.join(SING_DIR,f'{dataset}.npy')):
        print(f"Singular values for {dataset} are not pre-computed, please run compute_spectral_plots.py first")
    sigma=np.load(os.path.join(SING_DIR,f'{dataset}.npy'))

    bound_vals=[]
    k1_vals=[]

    for k1 in tqdm(range(target_dim), desc=f'Stress Bound Computing ({target_dim})',position=worker_id):

        if target_dim-k1!=0:
           bound_vals.append(calculate_stress_bound(sigma,target_dim,k1))

           k1_vals.append(k1)
        
    
    
    
    if not os.path.exists(os.path.join(SAVE_DIR,f'{dataset}')):
        lock.acquire()
        os.mkdir(os.path.join(SAVE_DIR,f'{dataset}'))
        lock.release()
    
    if not os.path.exists(os.path.join(SAVE_DIR, f'{dataset}', 'stress')):
        lock.acquire()
        os.mkdir(os.path.join(SAVE_DIR,f'{dataset}','stress'))
        lock.release()
    
    data={'k1': k1_vals, 'stress_bound': bound_vals}
    column_names=['k1', 'stress_bound']
    df=pd.DataFrame(data,columns=column_names)

    df.to_excel(os.path.join(SAVE_DIR,f'{dataset}','stress',f'bound_{target_dim}.xlsx'))


def main():

    args=parse_arguments()

    target_dims=[int(x) for x in args.target_dims]

    if args.metric=='stress':

        num_cores=cpu_count()
        pool=Pool(processes=num_cores)

        results=[pool.apply_async(compute_stress_vals, args=(args.dataset, args.save_dir,args.sing_dir,target_dims[i],i)) for i in range(len(target_dims))]

        for result in results:
            result.wait()
        
        pool.close()
        pool.join()
    
    elif args.metric=='m1':
        print('Code for M1 not written yet')
    
    else:
        print('Wrong metric, choose from \'stress\' or \'m1\'')

if __name__=="__main__":
    main()


    
    

