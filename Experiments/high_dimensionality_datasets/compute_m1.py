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
    parser.add_argument('--max_iter_list', help='True if max_iter is a list. If false then the single value provided is assumed to be for all the target dimensions.', default='True')
    parser.add_argument('--k1', help='List of k1 values seperated by space', nargs='+')
    parser.add_argument('--k2', help='List of corresponding k2 values seperated by space', nargs='+')
    parser.add_argument('--max_iter', help='List of max iter values corresponding to k1 and k2', nargs='+')

    args=parser.parse_args()
    return args

def fr_sq(A:np.ndarray):
    return LA.norm(A, ord='fro')**2

def compute_m1(dataset:str, DATA_DIR:str, SAVE_DIR: str, EMBED_DIR: str, file_name: str, k1:int, k2:int, max_iter:int):
    
    global lock
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

def main():

    args=parse_arguments()
    k1=[int(x) for x in args.k1]
    k2=[int(x) for x in args.k2]
    if args.max_iter_list=='True':
        max_iter=[int(x) for x in args.max_iter]
    else:
        max_iter=[int(args.max_iter[0]) for x in range(len(k1))]
    

    
    
    dr_args=list(zip(k1,k2,max_iter))
    num_cores=cpu_count()
    pool=Pool(processes=num_cores)

    results=[pool.apply_async(compute_m1, args=(args.dataset, args.data_dir,args.save_dir, args.embed_dir, args.file_name,dr_args[i][0], dr_args[i][1], dr_args[i][2] )) for i in range(len(dr_args))]

    for result in results:
        result.wait()
    
    pool.close()
    pool.join()


if __name__=="__main__":
    main()
    
    

