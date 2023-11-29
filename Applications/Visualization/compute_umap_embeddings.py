#BANK---> Setting8
#

import numpy as np
import numpy.linalg as LA
import math as m
import os
import argparse
from tqdm import tqdm
from multiprocessing import cpu_count,Pool, Lock
from itertools import product
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
import sys 
sys.path.append('../../Experiments/dimensionality_reduction_metrics/')
from other_dr_techniques.dr_techniques import DR_MAPS
from other_dr_techniques.settings import SETTINGS

lock=Lock()

def parse_arguments():

    parser=argparse.ArgumentParser(description='Compute and save embedding matrices')

    parser.add_argument('--save_dir', '-s', help='Path to the directory where embeddings are to be saved', default='./umap_embeddings/')
    parser.add_argument('--dataset', '-d', help='Dataset')
    parser.add_argument('--data_dir', help='Directory where embeddings of dr_tech are stored')
    parser.add_argument('--target_dims', '-t', help='List of target dimensions seperated by space', nargs='+')
    parser.add_argument('--dr_tech', help='Preprocessing dimensionality reduction technique', choices=['DiffRed', 'PCA'])
    # parser.add_argument('--tsne', help=' Implementation of tsne to use', choices=['multicore', 'cuda'], default='cuda')
    parser.add_argument('--k1', help='List of k1 values seperated by space', nargs='+')
    # parser.add_argument('--k2', help='List of corresponding k2 values seperated by space', nargs='+')
    parser.add_argument('--max_iter', help='List of max iter values corresponding to k1 and k2', nargs='+')
    parser.add_argument('--max_iter_list', help='True if max_iter is a list. If false then the single value provided is assumed to be for all the target dimensions.', default='True')


    parser.add_argument('--setting', help='Which setting to use for the dr technique. Enter \'def\' for default or \'setting1\', \'setting2\' for custom settings stored in settings.py and \'all\' for computing on all settings', default='def')

    args=parser.parse_args()
    return args


def compute_embeddings(dataset:str, DATA_DIR:str,SAVE_DIR:str, dr_tech:str, target_dim:int, k1:int, k2:int, max_iter:int, setting:dict):

    if dr_tech=='DiffRed':
        embed_path=os.path.join(DATA_DIR, dataset, f'{k1}_{k2}_{max_iter}.npy')
        X=np.load(embed_path)
        if not os.path.exists(os.path.join(SAVE_DIR, dr_tech)):
            lock.acquire()
            os.mkdir(os.path.join(SAVE_DIR, dr_tech))
            lock.release()

            
        if not os.path.exists(os.path.join(SAVE_DIR, dr_tech, dataset)):
            lock.acquire()
            os.mkdir(os.path.join(SAVE_DIR,dr_tech,dataset))
            lock.release()
        if os.path.exists(os.path.join(SAVE_DIR, dr_tech, dataset, f'{dataset}_{target_dim}_{setting}.npy')):

            print(f'Embeddings already saved for dataset: {dataset}, dr_tech: {dr_tech}, target dimension: {target_dim}, setting: {setting}')
            return
        
        Z=DR_MAPS['UMap'](X,SETTINGS['UMap'][setting])       
        np.save(os.path.join(SAVE_DIR, dr_tech,dataset, f'{dataset}_{target_dim}_{k1}_{k2}.npy'), Z)
    
    elif dr_tech=='PCA':
        embed_path=os.path.join(DATA_DIR,dataset,f'{dataset}_{target_dim}_pca.npy')
        X=np.load(embed_path)
        
        if not os.path.exists(os.path.join(SAVE_DIR, dr_tech)):
            lock.acquire()
            os.mkdir(os.path.join(SAVE_DIR, dr_tech))
            lock.release()

            
        if not os.path.exists(os.path.join(SAVE_DIR, dr_tech, dataset)):
            lock.acquire()
            os.mkdir(os.path.join(SAVE_DIR,dr_tech,dataset))
            lock.release()

        if os.path.exists(os.path.join(SAVE_DIR, dr_tech, dataset, f'{dataset}_{target_dim}_{setting}.npy')):

            print(f'Embeddings already saved for dataset: {dataset}, dr_tech: {dr_tech}, target dimension: {target_dim}, setting: {setting}')
            return
        
        
        Z=DR_MAPS['UMap'](X,SETTINGS['UMap'][setting])   

        np.save(os.path.join(SAVE_DIR, dr_tech,dataset, f'{dataset}_{target_dim}.npy'), Z)
    
    else:
        print("Invalid dr_tech provided")

def main():

    args=parse_arguments()
    num_cores=cpu_count()
    pool=Pool(processes=num_cores)
    target_dims=[int(x) for x in args.target_dims]

    if args.dr_tech=='DiffRed':
            k1=[int(x) for x in args.k1]
            k2=[target_dims[x]-k1[x] for x in range(len(target_dims))]
            if args.max_iter_list=='True':
                max_iter=[int(x) for x in args.max_iter]
            else:
                max_iter=[int(args.max_iter[0]) for x in range(len(k1))]
            dr_args=list(zip(k1,k2,max_iter))

            results=[pool.apply_async(compute_embeddings, args=(args.dataset, args.data_dir, args.save_dir,args.dr_tech,target_dims[i], k1[i], k2[i], max_iter[i], args.setting )) for i in range(len(k1))]

    else:

        results=[pool.apply_async(compute_embeddings, args=(args.dataset, args.data_dir, args.save_dir, args.dr_tech, target_dims[i], None, None, None, args.setting)) for i in range(len(target_dims)) ]

    

    for result in results:
        result.wait()
    
    pool.close()
    pool.join()

if __name__=="__main__":
    main()
