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
lock=Lock()

def parse_arguments():
    parser=argparse.ArgumentParser(description='Renormalize UMap for stress computation as described in the paper (UMap2). ')

    parser.add_argument('--dataset', '-d', help='Dataset for which renormalization is to be done' )
    parser.add_argument('--data_dir', help='Directory where the normalized datasets are stored', default='../normalized_data/')
    parser.add_argument('--embed_dir', help='Directory where UMap embeddings are stored', default='./embeddings')
    parser.add_argument('--target_dims', '-t',nargs='+', help='List of target dimensions (seperated by space) for which the renormalization is to be done')
    parser.add_argument('--settings', '-s', nargs='+',help='List of settings(seperated by space) corresponding to target dimension for which the renormalization is to be done')
    parser.add_argument('--setting_list', help='Whether a list of settings is provided or a single setting is provided which is to be used all target dimensions', default='True')
    parser.add_argument('--save_dir', help='Directory where to save the re-normalized embeddings', default='./embeddings')

    args=parser.parse_args()
    return args

def renormalize_embeddings(dataset:str, embed_dir:str, target_dim:int, setting:str, save_dir:str, data_dir:str):

    global lock
    X=np.load(os.path.join(embed_dir, dataset, 'UMap', f'{dataset}_{target_dim}_{setting}.npy'))
    
    lock.acquire()
    A=np.load(os.path.join(data_dir,dataset, 'X.npy'))
    lock.release()
    constant=LA.norm(A, ord='fro')/LA.norm(X, ord='fro')
    X=constant*X

    
    save_path=os.path.join(save_dir,dataset, 'UMap2')
    lock.acquire()
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    lock.release()
    

    np.save(os.path.join(save_dir, dataset,'UMap2', f'{dataset}_{target_dim}_{setting}.npy'),X)

def main():
    args=parse_arguments()

    target_dims=[int(x) for x in args.target_dims]

    if args.setting_list=='True':
        settings=args.settings
    else:
        settings=[args.settings[0] for i in range(len(target_dims))]
    
    combinations =product(target_dims,settings)
    num_cores=cpu_count()
    pool=Pool(processes=num_cores)

    results=[pool.apply_async(renormalize_embeddings, args=(args.dataset, args.embed_dir,target_dim, setting, args.save_dir, args.data_dir)) for target_dim,setting in combinations]

    for result in results:
        result.wait()
    
    pool.close()
    pool.join()


if __name__=="__main__":
    main()