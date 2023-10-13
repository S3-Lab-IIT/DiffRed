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
import sys

lock=Lock()

def parse_arguments():

    # Here, the k1, k2 and max_iter values are
    # used to uniquely identify the file where 
    # embeddings are saved. 

    parser=argparse.ArgumentParser(description='Compute and save embedding matrices')
    parser.add_argument('--save_dir','-s', help='Path to the directory where the results are to be saved', default='./results/stress_results/')
    parser.add_argument('--embed_dir','-e',help='Directory where embeddings are stored',default='./tsne_embeddings')
    parser.add_argument('--file_name', '-f', help='Name of the excel file where results are to be updated/File name of the excel file to be created', default='stress_results')
    parser.add_argument('--dataset','-d',help='Name of the dataset whose stress is to be computed')
    # parser.add_argument('--data_dir', help='Directory where the datasets are stored', default='./normalized_data' )
    parser.add_argument('--dist_dir', help='Directory where distance matrices are saved')
    parser.add_argument('--k1', help='List of k1 values seperated by space (corresponding to target dimensions)', nargs='+')
    # parser.add_argument('--k2', help='List of corresponding k2 values seperated by space', nargs='+')
    parser.add_argument('--max_iter', help='List of max iter values corresponding to k1 and k2', nargs='+')
    parser.add_argument('--max_iter_list', help='True if max_iter is a list. If false then the single value provided is assumed to be for all the target dimensions.', default='True')
    parser.add_argument('--target_dims', '-t', help='List of target dimensions seperated by space', nargs='+', default='na')
    parser.add_argument('--dr_tech', help='DR Technique whose stress is to be calculated', default='DiffRed', choices=['DiffRed', 'PCA'])

    parser.add_argument('--use_memoized_embeddings', '-u', type=bool, help='Use memoized embeddings (T) or compute fresh(F)', default=True)

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




def compute_stress(dataset:str, SAVE_DIR:str, EMBED_DIR:str, file_name:str,k1:int, k2:int, max_iter:int, worker_id:int, dr_tech:str, target_dim:int):
    global lock
    # dist_matrix= np.load(os.path.join(DIST_DIR,f'{dataset}.npy'))
    dist_matrix=get_shared_array('dist_matrix')
    
    if dr_tech=='DiffRed':
        EMBED_DIR=os.path.join(EMBED_DIR, 'DiffRed')
        Z= np.load(os.path.join(EMBED_DIR,dataset, f'{dataset}_{target_dim}_{k1}_{k2}'))
        worker_desc=f'{dataset}({k1},{k2})'

    
        stress_val=stress(dist_matrix,Z,worker_desc,worker_id)
        
        lock.acquire()
        if not os.path.exists(os.path.join(SAVE_DIR,f'{file_name}.xlsx')):

            column_names=['Timestamp', 'Dataset','Max_iter','Target Dimension', 'k1', 'k2', 'Stress' ]

            df=pd.DataFrame(columns=column_names)
            df.to_excel(os.path.join(SAVE_DIR, f'{file_name}.xlsx'), index=False)
        

        new_row=[datetime.now().strftime("%Y-%m-%d %H:%M:%S"),dataset,max_iter,k1+k2,k1,k2,stress_val]
        # else:
        #     new_row=[datetime.now().strftime("%Y-%m-%d %H:%M:%S"),dataset,dr_tech,target_dim,stress_val]

        result_sheet=pd.read_excel(os.path.join(SAVE_DIR, f'{file_name}.xlsx'))

        result_sheet.loc[len(result_sheet.index)]=new_row

        result_sheet.to_excel(os.path.join(SAVE_DIR, f'{file_name}.xlsx'), index=False)
        lock.release()
    elif dr_tech=='PCA':
        EMBED_DIR=os.path.join(EMBED_DIR, dr_tech)
        Z=np.load(os.path.join(EMBED_DIR, dataset, f'{dataset}_{target_dim}.npy'))
        worker_desc=f'{dataset}({target_dim})'

    
        stress_val=stress(dist_matrix,Z,worker_desc,worker_id)
        
        lock.acquire()
        if not os.path.exists(os.path.join(SAVE_DIR,f'{file_name}.xlsx')):

            if dr_tech=='DiffRed':
                column_names=['Timestamp', 'Dataset','Max_iter','Target Dimension', 'k1', 'k2', 'Stress' ]
            else:
                column_names=['Timestamp', 'Dataset', 'DR Technique','Target Dimension', 'Stress' ]


            df=pd.DataFrame(columns=column_names)
            df.to_excel(os.path.join(SAVE_DIR, f'{file_name}.xlsx'), index=False)
        

        new_row=[datetime.now().strftime("%Y-%m-%d %H:%M:%S"),dataset,dr_tech,target_dim,stress_val]

        result_sheet=pd.read_excel(os.path.join(SAVE_DIR, f'{file_name}.xlsx'))

        result_sheet.loc[len(result_sheet.index)]=new_row

        result_sheet.to_excel(os.path.join(SAVE_DIR, f'{file_name}.xlsx'), index=False)
        lock.release()
    # elif dr_tech=='RMap':
    #     alpha=int(dr_args)

    #     for i in range(alpha):
    #         Z=np.load(os.path.join(EMBED_DIR, dataset, str(target_dim), f'{i}.npy'))
    #         worker_desc=f'{dataset}_{dr_tech}_{target_dim}'
    #         stress_val=stress(dist_matrix, Z, worker_desc, worker_id*i)
    #         lock.acquire()
    #         if not os.path.exists(os.path.join(SAVE_DIR, dataset)):
    #             os.mkdir(os.path.join(SAVE_DIR, dataset))
    #         if not os.path.exists(os.path.join(SAVE_DIR, dataset,f'{file_name}_{target_dim}.xlsx')):

    #             column_names=['Timestamp', 'Dataset', 'Target Dimension', 'Instance', 'Stress']

    #             df=pd.DataFrame(columns=column_names)
    #             df.to_excel(os.path.join(SAVE_DIR,dataset,f'{file_name}_{target_dim}.xlsx'), index=False)
            
    #         new_row=[datetime.now().strftime("%Y-%m-%d %H:%M:%S"), dataset, target_dim, i, stress_val]
    #         result_sheet=pd.read_excel(os.path.join(SAVE_DIR, dataset, f'{file_name}_{target_dim}.xlsx'))

    #         result_sheet.loc[len(result_sheet.index)]=new_row
    #         result_sheet.to_excel(os.path.join(SAVE_DIR,dataset,f'{file_name}_{target_dim}.xlsx'), index=False)
    #         lock.release()
   
    




def main():


    args=parse_arguments()
    

    dist_matrix= np.load(os.path.join(args.dist_dir,f'{args.dataset}.npy'))
    make_shared_array(dist_matrix,name='dist_matrix')

    # lock=Lock()
    if args.use_memoized_embeddings:
        target_dims=[int(x) for x in args.target_dims]
        # for arg in dr_args:
        #     compute_stress(args.dataset,args.dist_dir,args.save_dir,args.embed_dir,args.file_name,arg[0],arg[1],arg[2])
        num_cores=cpu_count()
        pool=Pool(processes=num_cores)

        if args.dr_tech=='DiffRed':
            k1=[int(x) for x in args.k1]
            k2=[target_dims[x]-k1[x] for x in range(len(target_dims))]
            if args.max_iter_list=='True':
                max_iter=[int(x) for x in args.max_iter]
            else:
                max_iter=[int(args.max_iter[0]) for x in range(len(k1))]
            dr_args=list(zip(k1,k2,max_iter))

            results=[pool.apply_async(compute_stress, args=(args.dataset,args.save_dir,args.embed_dir,args.file_name,dr_args[i][0],dr_args[i][1],dr_args[i][2],i, args.dr_tech, target_dims[i])) for i in range(len(dr_args))]
        else:
            results=[pool.apply_async(compute_stress, args=(args.dataset,args.save_dir,args.embed_dir,args.file_name,None,None,None,i, args.dr_tech, target_dims[i])) for i in range(len(target_dims))]

        for result in results:
            result.wait()

        pool.close()
        pool.join()
    else:
        print('Code for this part has not been written yet')

if __name__=="__main__":
    main()



