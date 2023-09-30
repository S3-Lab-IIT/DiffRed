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
import sys


lock=Lock()

def parse_arguments():
    parser=argparse.ArgumentParser(description='Compute M1 metric')
    parser.add_argument('--save_dir','-s', help='Path to the directory where the results are to be saved', default='./results')
    parser.add_argument('--embed_dir','-e',help='Directory where embeddings are stored',default='./embeddings')
    parser.add_argument('--file_name', '-f', help='Name of the excel file where results are to be updated/File name of the excel file to be created', default='m1_results')
    parser.add_argument('--dataset','-d',help='Name of the dataset whose stress is to be computed')
    parser.add_argument('--data_dir', help='Directory where the datasets are stored', default='./normalized_data' )
    # parser.add_argument('--dist_dir', help='Directory where distance matrices are saved', default='./norm_dist_matrices')
    parser.add_argument('--dr_tech', help='Name of the DR technique', choices=['DiffRed', 'PCA', 'RMap'])
    parser.add_argument('--setting', help='Which setting is being used', default='def')
    parser.add_argument('--dr_args', help='Arguments of the dr technique', default=None)
    parser.add_argument('--target_dims', help='List of target dimension values', nargs='+', default=None)
    parser.add_argument('--max_iter_list', help='True if max_iter is a list. If false then the single value provided is assumed to be for all the target dimensions.', default='True')
    parser.add_argument('--k1', help='List of k1 values seperated by space', nargs='+')
    parser.add_argument('--k2', help='List of corresponding k2 values seperated by space', nargs='+')
    parser.add_argument('--max_iter', help='List of max iter values corresponding to k1 and k2', nargs='+')

    args=parser.parse_args()
    return args

def fr_sq(A:np.ndarray):
    return LA.norm(A, ord='fro')**2

def compute_m1(dataset:str, DATA_DIR:str, SAVE_DIR: str, EMBED_DIR: str, file_name: str,target_dim:int,dr_technique:str, setting:str, k1:int, k2:int, max_iter:int,dr_args:str):
    
    global lock
    if dr_technique=='DiffRed':
        Z= np.load(os.path.join(EMBED_DIR,dataset, f'{k1}_{k2}_{max_iter}.npy'))
        A=np.load(os.path.join(DATA_DIR,dataset, 'X.npy'))

        m1_val= abs(1-(fr_sq(Z)/fr_sq(A)))

        save_file=os.path.join(SAVE_DIR, f'{file_name}.xlsx')

        lock.acquire()
        if not os.path.exists(save_file):

            column_names=['Timestamp', 'Dataset', 'Max_iter', 'Target Dimension', 'k1', 'k2', 'M1']

            df= pd.DataFrame(columns=column_names)
            df.to_excel(save_file, index=False)
        
        new_row=[datetime.now().strftime("%Y-%m-%d %H:%M:%S"),dataset,max_iter,k1+k2,k1,k2,m1_val]

        result_sheet=pd.read_excel(save_file)
        
        result_sheet.loc[len(result_sheet.index)]=new_row

        result_sheet.to_excel(save_file, index=False)

        lock.release()
    
    elif dr_technique=='PCA':
        if setting=='def':
            Z=np.load(os.path.join(EMBED_DIR, dataset, f'{dataset}_{target_dim}_pca.npy'))
        else:
            Z=np.load(os.path.join(EMBED_DIR, dataset, dr_technique, f'{dataset}_{target_dim}_{setting}.npy'))
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
    else:
        A=np.load(os.path.join(DATA_DIR,dataset, 'X.npy'))
        eta=int(dr_args)

        for i in tqdm(range(eta), desc=f'RMap M1 {dataset} Target Dim: {target_dim}...'):
            Z=np.load(os.path.join(EMBED_DIR, dataset, str(target_dim), f'{i}.npy'))
            m1_val=abs(1-(fr_sq(Z)/fr_sq(A)))
            lock.acquire()
            if not os.path.exists(os.path.join(SAVE_DIR, dataset)):
                os.mkdir(os.path.join(SAVE_DIR, dataset))
            if not os.path.exists(os.path.join(SAVE_DIR, dataset,f'{file_name}_{target_dim}.xlsx')):

                column_names=['Timestamp', 'Dataset', 'Target Dimension', 'Instance', 'M1']

                df=pd.DataFrame(columns=column_names)
                df.to_excel(os.path.join(SAVE_DIR,dataset,f'{file_name}_{target_dim}.xlsx'), index=False)
            
            new_row=[datetime.now().strftime("%Y-%m-%d %H:%M:%S"), dataset, target_dim, i, m1_val]
            result_sheet=pd.read_excel(os.path.join(SAVE_DIR, dataset, f'{file_name}_{target_dim}.xlsx'))

            result_sheet.loc[len(result_sheet.index)]=new_row
            result_sheet.to_excel(os.path.join(SAVE_DIR,dataset,f'{file_name}_{target_dim}.xlsx'), index=False)
            lock.release()

def main():

    args=parse_arguments()
    num_cores=cpu_count()
    pool=Pool(processes=num_cores)

    if args.dr_tech=='DiffRed':
        k1=[int(x) for x in args.k1]
        k2=[int(x) for x in args.k2]
        if args.max_iter_list=='True':
            max_iter=[int(x) for x in args.max_iter]
        else:
            max_iter=[int(args.max_iter[0]) for x in range(len(k1))]  
        dr_args=list(zip(k1,k2,max_iter))      
        results=[pool.apply_async(compute_m1, args=(args.dataset, args.data_dir,args.save_dir, args.embed_dir, args.file_name,None,args.dr_tech,args.setting, dr_args[i][0], dr_args[i][1], dr_args[i][2], args.dr_args )) for i in range(len(dr_args))]
    else:
        target_dims=[int(x) for x in args.target_dims]
        results=[pool.apply_async(compute_m1, args=(args.dataset, args.data_dir,args.save_dir, args.embed_dir, args.file_name,target_dims[i],args.dr_tech,args.setting, None, None, None, args.dr_args)) for i in range(len(target_dims))]

    for result in results:
        result.wait()
    
    pool.close()
    pool.join()


if __name__=="__main__":
    main()
    
    

