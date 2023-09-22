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

lock=Lock()

def parse_arguments():

    parser=argparse.ArgumentParser(description='Compute the theoretical minimum k1 k2 for a dataset')

    parser.add_argument('--dataset','-d', help='Dataset')
    parser.add_argument('--sing_dir', help='Directory where singular values are stored', default='./norm_singular_values')
    parser.add_argument('--target_dims', nargs='+', help='Target dimensions at which theoretical optimum is to be calculated')
    parser.add_argument('--metric', '-m', help='Metric whose theoretical optimum is needed', choices=['stress', 'm1'], default='stress')
    parser.add_argument('--save_dir', help='Directory where results are to be saved', default='./results')
    parser.add_argument('--file_name', help='Name of the excel file (without extension)', default='theoretical_optimum')


    
    args=parser.parse_args()
    return args

def calculate_p(sigma:np.ndarray, k1:int):
    return np.sum(sigma[:k1]**2)/np.sum(sigma**2)

def calculate_stress_bound(sigma:np.ndarray, target_dim:int, k1:int):
    p=calculate_p(sigma,k1)
    return m.sqrt((1-p)/(target_dim-k1))

def stress_opt(dataset:str, SAVE_DIR:str, file_name:str, SING_DIR:str, target_dim:int, worker_id:int):

    global lock
    
    if not os.path.exists(os.path.join(SING_DIR,f'{dataset}.npy')):
        print(f"Singular values for {dataset} are not pre-computed, please run compute_spectral_plots.py first")
    sigma=np.load(os.path.join(SING_DIR,f'{dataset}.npy'))

    bound_vals=[]
    hyper_params=[]
    for k1 in tqdm(range(target_dim), desc=f'Stress Grid Search [d= {target_dim}]',position=worker_id):

        if target_dim-k1!=0:
            bound_vals.append(calculate_stress_bound(sigma,target_dim,k1))

            hyper_params.append((k1,target_dim-k1))
    
    min_bound_val=min(bound_vals)
    min_params=hyper_params[min((j,i) for i, j in enumerate(bound_vals))[1]]
    lock.acquire()

    if not os.path.exists(os.path.join(SAVE_DIR,f'{file_name}.xlsx')):

        column_names=['Timestamp', 'Dataset','Metric', 'Target Dimension', 'opt_k1', 'opt_k2', 'opt_bound_val']

        df=pd.DataFrame(columns=column_names)

        df.to_excel(os.path.join(SAVE_DIR,f'{file_name}.xlsx'), index=False)

    
    new_row=[datetime.now().strftime("%Y-%m-%d %H:%M:%S"),dataset,'stress',target_dim,min_params[0], min_params[1], min_bound_val]

    result_sheet=pd.read_excel(os.path.join(SAVE_DIR, f'{file_name}.xlsx'))

    result_sheet.loc[len(result_sheet.index)]=new_row

    result_sheet.to_excel(os.path.join(SAVE_DIR, f'{file_name}.xlsx'),index=False)

    lock.release()    

def main():

    args=parse_arguments()

    target_dims=[int(x) for x in args.target_dims]

    if args.metric=='stress':

        num_cores=cpu_count()
        pool=Pool(processes=num_cores)

        results=[pool.apply_async(stress_opt, args=(args.dataset,args.save_dir,args.file_name,args.sing_dir,target_dims[i],i)) for i in range(len(target_dims))]

        for result in results:
            result.wait()
        
        pool.close()
        pool.join()
    
    elif args.metric=='m1':
        print('Code for M1 yet to be written')
    
    else:
        print('Wrong metric, choose from \'stress\' or \'m1\'')
    

if __name__=="__main__":
    main()



